import torch
import torch.nn as nn
import torch.distributed as dist
from flash_attn import flash_attn_with_kvcache, flash_attn_varlen_func


class BlockManager:
    def __init__(self, num_blocks, block_size, num_layers, num_kv_heads, head_dim, device="cuda", dtype=torch.float16):
        self.block_size = block_size
        tp_size = dist.get_world_size() if dist.is_initialized() else 1
        self.num_kv_heads_per_gpu = num_kv_heads // tp_size

        cache_shape = (num_layers, num_blocks, block_size, self.num_kv_heads_per_gpu, head_dim)
        self.k_cache = torch.zeros(cache_shape, device=device, dtype=dtype)
        self.v_cache = torch.zeros(cache_shape, device=device, dtype=dtype)
        self.free_blocks = list(range(num_blocks))
        self.request_to_blocks = {}

    def allocate_blocks_for_request(self, request_id, total_tokens):
        needed_blocks = (total_tokens + self.block_size - 1) // self.block_size
        if request_id not in self.request_to_blocks:
            self.request_to_blocks[request_id] = []
        while len(self.request_to_blocks[request_id]) < needed_blocks:
            if not self.free_blocks: raise MemoryError("KV Cache OOM!")
            self.request_to_blocks[request_id].append(self.free_blocks.pop(0))

    def free_request(self, request_id):
        if request_id in self.request_to_blocks:
            self.free_blocks.extend(self.request_to_blocks[request_id])
            del self.request_to_blocks[request_id]


class AttentionPaged(nn.Module):
    def __init__(self, hidden, num_heads, num_kv_heads, layer_idx, rope):
        super().__init__()
        tp_size = dist.get_world_size() if dist.is_initialized() else 1
        self.num_heads = num_heads // tp_size
        self.num_kv_heads = num_kv_heads // tp_size
        self.head_dim = hidden // num_heads
        self.scale = self.head_dim ** -0.5
        self.layer_idx = layer_idx
        self.rope = rope

        self.qkv = nn.Linear(hidden, (self.num_heads + 2 * self.num_kv_heads) * self.head_dim, bias=True)
        self.o = nn.Linear(self.num_heads * self.head_dim, hidden, bias=False)

    def forward(self, x, request_ids, pos_list, block_mgr, is_prefill=False, cu_seqlens=None, max_seqlen=None):
        # 1. 强制类型统一
        x = x.to(torch.float16)
        B_total = x.shape[0]
        qkv = self.qkv(x)

        q, k, v = qkv.split([self.num_heads * self.head_dim,
                             self.num_kv_heads * self.head_dim,
                             self.num_kv_heads * self.head_dim], dim=-1)

        q = q.view(B_total, self.num_heads, self.head_dim).to(torch.float16)
        k = k.view(B_total, self.num_kv_heads, self.head_dim).to(torch.float16)
        v = v.view(B_total, self.num_kv_heads, self.head_dim).to(torch.float16)

        q, k = self.rope.apply_rope(q, k, pos_list)

        if is_prefill:
            # 2. Prefill 阶段
            output = flash_attn_varlen_func(
                q.half(), k.half(), v.half(),
                cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen,
                dropout_p=0.0, softmax_scale=self.scale, causal=True
            )
            self._write_to_paged_cache(k, v, request_ids, pos_list, block_mgr)
        else:
            # 3. Decode 阶段
            q, k, v = q.unsqueeze(1), k.unsqueeze(1), v.unsqueeze(1)
            max_blocks = max(len(block_mgr.request_to_blocks[rid]) for rid in request_ids)
            block_table = torch.full((B_total, max_blocks), -1, device=x.device, dtype=torch.int32)
            for i, rid in enumerate(request_ids):
                blocks = block_mgr.request_to_blocks[rid]
                block_table[i, :len(blocks)] = torch.tensor(blocks, dtype=torch.int32)

            output = flash_attn_with_kvcache(
                q=q.half(), k=k.half(), v=v.half(),
                k_cache=block_mgr.k_cache[self.layer_idx].half(),
                v_cache=block_mgr.v_cache[self.layer_idx].half(),
                cache_seqlens=pos_list.to(torch.int32) + 1,
                block_table=block_table,
                softmax_scale=self.scale, causal=True
            )

        # 4. 修正缩进后的输出逻辑
        output = self.o(output.view(B_total, -1))
        if dist.is_initialized() and dist.get_world_size() > 1:
            dist.all_reduce(output)
        return output

    def _write_to_paged_cache(self, k, v, request_ids, pos_list, block_mgr):
        for i in range(len(request_ids)):
            rid = request_ids[i]
            pos = pos_list[i].item()
            block_list = block_mgr.request_to_blocks[rid]
            b_idx = block_list[pos // block_mgr.block_size]
            b_offset = pos % block_mgr.block_size
            block_mgr.k_cache[self.layer_idx, b_idx, b_offset] = k[i].half()
            block_mgr.v_cache[self.layer_idx, b_idx, b_offset] = v[i].half()
