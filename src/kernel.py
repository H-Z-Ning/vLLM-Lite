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
        
        # 这里的计算必须严格遵循模型配置
        self.num_heads = num_heads // tp_size
        self.num_kv_heads = num_kv_heads // tp_size
        self.head_dim = hidden // num_heads # 注意：有的模型 head_dim 是独立定义的
        self.scale = self.head_dim ** -0.5
        self.layer_idx = layer_idx
        self.rope = rope

        # 关键修正：QKV 的总输出维度
        # 必须是 (n_heads + 2 * n_kv_heads) * head_dim
        qkv_out_features = (self.num_heads + 2 * self.num_kv_heads) * self.head_dim
        self.qkv = nn.Linear(hidden, qkv_out_features, bias=True) # Qwen 通常有 Bias
        self.o = nn.Linear(self.num_heads * self.head_dim, hidden, bias=False)

    def forward(self, x, request_ids, pos_list, block_mgr, is_prefill=False, cu_seqlens=None, max_seqlen=None):
        # 1. 类型安全检查
        x = x.to(torch.float16)
        B_total = x.shape[0]
        
        qkv = self.qkv(x)
        # Qwen3 在大参数模型下可能使用不同的头部布局，这里确保 split 正确
        q, k, v = qkv.split([self.num_heads * self.head_dim,
                             self.num_kv_heads * self.head_dim,
                             self.num_kv_heads * self.head_dim], dim=-1)

        q = q.view(B_total, self.num_heads, self.head_dim)
        k = k.view(B_total, self.num_kv_heads, self.head_dim)
        v = v.view(B_total, self.num_kv_heads, self.head_dim)

        # 2. 应用适配 Qwen3 的 RoPE
        q, k = self.rope.apply_rope(q, k, pos_list)

        if is_prefill:
            # Prefill 阶段：Qwen3 建议开启 causal=True 并注意窗口大小
            output = flash_attn_varlen_func(
                q.half(), k.half(), v.half(),
                cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen,
                dropout_p=0.0, softmax_scale=self.scale, causal=True
            )
            self._write_to_paged_cache(k, v, request_ids, pos_list, block_mgr)
        else:
            # Decode 阶段：动态构建 block_table
            # 注意：若 Qwen3 采用 MLA，此处的 k_cache 维度需要重构，
            # 当前代码假设 Qwen3 仍保持 GQA 结构（Lite版最通用做法）
            q = q.unsqueeze(1) 
            k = k.unsqueeze(1)
            v = v.unsqueeze(1)
            
            max_blocks = max(len(block_mgr.request_to_blocks[rid]) for rid in request_ids)
            
            # 使用 -1 填充未使用的 block_table 槽位，有些算子支持，
            # 若不支持则保持 0 但必须确保分配逻辑正确
            block_table = torch.zeros((B_total, max_blocks), device=x.device, dtype=torch.int32)
            
            for i, rid in enumerate(request_ids):
                blocks = block_mgr.request_to_blocks[rid]
                block_table[i, :len(blocks)] = torch.tensor(blocks, device=x.device, dtype=torch.int32)

            # 注意：flash_attn_with_kvcache 会自动将当前步的 k,v 写入 k_cache/v_cache
            output = flash_attn_with_kvcache(
                q=q.half(), k=k.half(), v=v.half(),
                k_cache=block_mgr.k_cache[self.layer_idx], 
                v_cache=block_mgr.v_cache[self.layer_idx],
                cache_seqlens=pos_list.to(torch.int32) + 1,
                block_table=block_table,
                softmax_scale=self.scale, 
                causal=True
            )

        return self.o(output.view(B_total, -1))
    # def _write_to_paged_cache(self, k, v, request_ids, pos_list, block_mgr):
    #     for i in range(len(request_ids)):
    #         rid = request_ids[i]
    #         pos = pos_list[i].item()
    #         block_list = block_mgr.request_to_blocks[rid]
    #         b_idx = block_list[pos // block_mgr.block_size]
    #         b_offset = pos % block_mgr.block_size
    #         block_mgr.k_cache[self.layer_idx, b_idx, b_offset] = k[i].half()
    #         block_mgr.v_cache[self.layer_idx, b_idx, b_offset] = v[i].half()
    def _write_to_paged_cache(self, k, v, request_ids, pos_list, block_mgr):
        """
        高性能向量化写入：一次性将整个 Batch 的 KV 写入 Paged Cache
        """
        if len(request_ids) == 0:
            return

        # 1. 准备索引数据
        # pos_list 形状为 [B_total], 存储了每个 token 在各自 sequence 中的位置
        # block_size 是每个块的大小
        pos_tensor = pos_list.to(torch.long)
        
        # 计算每个 token 属于该 request 的第几个 block
        block_table_idx = pos_tensor // block_mgr.block_size
        # 计算每个 token 在 block 内部的偏移量
        block_offsets = pos_tensor % block_mgr.block_size

        # 2. 获取每个请求对应的物理块 ID
        # 我们需要从 block_mgr.request_to_blocks 中提取物理 block_id
        # 为了向量化，我们需要构造一个物理块索引数组
        physical_block_ids = []
        for i, rid in enumerate(request_ids):
            req_blocks = block_mgr.request_to_blocks[rid]
            # 确保索引不越界
            idx = block_table_idx[i].item()
            if idx >= len(req_blocks):
                # 动态追加分配逻辑（或者在 prefill 前确保分配够了）
                raise RuntimeError(f"Request {rid} needs more blocks than allocated!")
            physical_block_ids.append(req_blocks[idx])
        
        physical_block_ids = torch.tensor(physical_block_ids, device=k.device, dtype=torch.long)

        # 3. 执行向量化写入 (原地操作)
        # k 的形状: [B_total, num_kv_heads, head_dim]
        # k_cache 形状: [num_layers, num_blocks, block_size, num_kv_heads, head_dim]
        
        # 使用高级索引：一次性写入所有 token 的 KV
        # 这里 layer_idx 是固定的，physical_block_ids 和 block_offsets 对应 B_total
        block_mgr.k_cache[self.layer_idx, physical_block_ids, block_offsets] = k.half()
        block_mgr.v_cache[self.layer_idx, physical_block_ids, block_offsets] = v.half()
