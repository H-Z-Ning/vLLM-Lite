
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from kernel import AttentionPaged


def get_tp_size():
    return dist.get_world_size() if dist.is_initialized() else 1


def get_tp_rank():
    return dist.get_rank() if dist.is_initialized() else 0


class Rope():
    def __init__(self, base=1000000.0):
        self.base = base

    def apply_rope(self, q, k, pos):
        dim = q.size(-1)
        device = q.device
        inv_freq = 1.0 / (self.base ** (torch.arange(0, dim, 2, device=device).float() / dim))
        t = pos.to(device).float()
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().unsqueeze(1)
        sin = emb.sin().unsqueeze(1)

        def rotate_half(x):
            x1, x2 = x.chunk(2, dim=-1)
            return torch.cat((-x2, x1), dim=-1)

        return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        # 强制在 fp32 计算保证精度，但返回 fp16 满足 FlashAttention
        x_fp32 = x.float()
        variance = x_fp32.pow(2).mean(-1, keepdim=True)
        x_fp32 = x_fp32 * torch.rsqrt(variance + self.eps)
        return (x_fp32 * self.weight.float()).half()


class MLP(nn.Module):
    def __init__(self, hidden, intermediate):
        super().__init__()
        tp_size = get_tp_size()
        self.gate_up = nn.Linear(hidden, (intermediate * 2) // tp_size, bias=False)
        self.down = nn.Linear(intermediate // tp_size, hidden, bias=False)

    def forward(self, x):
        gate, up = self.gate_up(x).chunk(2, dim=-1)
        x = self.down(F.silu(gate) * up)
        if dist.is_initialized() and dist.get_world_size() > 1:
            dist.all_reduce(x)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, hidden, heads, kv_heads, intermediate, idx, rope):
        super().__init__()
        self.ln1 = RMSNorm(hidden)
        self.ln2 = RMSNorm(hidden)
        self.attn = AttentionPaged(hidden, heads, kv_heads, idx, rope)
        self.mlp = MLP(hidden, intermediate)

    def forward(self, x, request_ids, pos, block_mgr, is_prefill=False, cu_seqlens=None, max_seqlen=None):
        h = x + self.attn(self.ln1(x), request_ids, pos, block_mgr, is_prefill, cu_seqlens, max_seqlen)
        return h + self.mlp(self.ln2(h))


class VLLMLite(nn.Module):
    def __init__(self, hf_config):
        super().__init__()
        cfg = hf_config
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.rope = Rope(cfg.rope_theta)
        self.layers = nn.ModuleList([
            DecoderLayer(cfg.hidden_size, cfg.num_attention_heads, cfg.num_key_value_heads, cfg.intermediate_size, i,
                         self.rope)
            for i in range(cfg.num_hidden_layers)
        ])
        self.norm = RMSNorm(cfg.hidden_size)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

    def load_weights_tp(self, hf_sd):
        rank = get_tp_rank()
        tp_size = get_tp_size()

        self.embedding.weight.data.copy_(hf_sd["model.embed_tokens.weight"])
        self.lm_head.weight.data.copy_(hf_sd["lm_head.weight"])
        self.norm.weight.data.copy_(hf_sd["model.norm.weight"])

        for i, layer in enumerate(self.layers):
            p = f"model.layers.{i}"
            # 注意：chunk 在 tp_size=1 时返回原列表[W]，索引[0]即原权重，完美兼容单卡
            Wq = hf_sd[f"{p}.self_attn.q_proj.weight"].chunk(tp_size, dim=0)[rank]
            Wk = hf_sd[f"{p}.self_attn.k_proj.weight"].chunk(tp_size, dim=0)[rank]
            Wv = hf_sd[f"{p}.self_attn.v_proj.weight"].chunk(tp_size, dim=0)[rank]
            layer.attn.qkv.weight.data.copy_(torch.cat([Wq, Wk, Wv], dim=0))

            if f"{p}.self_attn.q_proj.bias" in hf_sd:
                Bq = hf_sd[f"{p}.self_attn.q_proj.bias"].chunk(tp_size, dim=0)[rank]
                Bk = hf_sd[f"{p}.self_attn.k_proj.bias"].chunk(tp_size, dim=0)[rank]
                Bv = hf_sd[f"{p}.self_attn.v_proj.bias"].chunk(tp_size, dim=0)[rank]
                layer.attn.qkv.bias.data.copy_(torch.cat([Bq, Bk, Bv], dim=0))

            layer.attn.o.weight.data.copy_(hf_sd[f"{p}.self_attn.o_proj.weight"].chunk(tp_size, dim=1)[rank])

            gate_w = hf_sd[f"{p}.mlp.gate_proj.weight"].chunk(tp_size, dim=0)[rank]
            up_w = hf_sd[f"{p}.mlp.up_proj.weight"].chunk(tp_size, dim=0)[rank]
            layer.mlp.gate_up.weight.data.copy_(torch.cat([gate_w, up_w], dim=0))

            layer.mlp.down.weight.data.copy_(hf_sd[f"{p}.mlp.down_proj.weight"].chunk(tp_size, dim=1)[rank])

            layer.ln1.weight.data.copy_(hf_sd[f"{p}.input_layernorm.weight"])
            layer.ln2.weight.data.copy_(hf_sd[f"{p}.post_attention_layernorm.weight"])

    # 在 VLLMLite 的 forward 里面加上 .to(dtype)
    def forward(self, input_ids, request_ids, pos, block_mgr, is_prefill=False, cu_seqlens=None, max_seqlen=None):
        # 显式转换入口数据
        x = self.embedding(input_ids).half()
        for layer in self.layers:
            x = layer(x, request_ids, pos, block_mgr, is_prefill, cu_seqlens, max_seqlen)
        return self.lm_head(self.norm(x))
