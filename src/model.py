import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from kernel import AttentionPaged

def get_tp_size():
    return dist.get_world_size() if dist.is_initialized() else 1

def get_tp_rank():
    return dist.get_rank() if dist.is_initialized() else 0

# --- 1. RoPE 适配层 ---
class Qwen3RoPE(nn.Module):
    def __init__(self, dim, base=1000000.0, scaling_factor=1.0):
        super().__init__()
        self.dim = dim
        self.base = base
        self.scaling_factor = scaling_factor

    def apply_rope(self, q, k, pos):
        dim = q.size(-1)
        device = q.device
        inv_freq = 1.0 / (self.base ** (torch.arange(0, dim, 2, device=device).float() / dim))
        t = pos.to(device).float() / self.scaling_factor
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos, sin = emb.cos().unsqueeze(1), emb.sin().unsqueeze(1)

        def rotate_half(x):
            x1, x2 = x.chunk(2, dim=-1)
            return torch.cat((-x2, x1), dim=-1)

        return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)

# --- 2. 归一化层 ---
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        x_fp32 = x.float()
        variance = x_fp32.pow(2).mean(-1, keepdim=True)
        x_fp32 = x_fp32 * torch.rsqrt(variance + self.eps)
        return (x_fp32 * self.weight.float()).to(x.dtype)

# --- 3. 门控多层感知机 ---
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

# --- 4. 解码层 (必须在 VLLMLite 之前定义) ---
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

# --- 5. 主模型类 ---
class VLLMLite(nn.Module):
    def __init__(self, hf_config):
        super().__init__()
        cfg = hf_config
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        
        # 修复后的 RoPE Scaling 获取逻辑
        rope_st = getattr(cfg, "rope_scaling", {})
        if rope_st is None: rope_st = {}
        scaling_factor = rope_st.get("factor", 1.0)

        self.rope = Qwen3RoPE(
            cfg.hidden_size // cfg.num_attention_heads, 
            base=cfg.rope_theta, 
            scaling_factor=scaling_factor
        )
        
        # 现在调用 DecoderLayer 就不会报错了，因为上面已经定义了
        self.layers = nn.ModuleList([
            DecoderLayer(cfg.hidden_size, cfg.num_attention_heads, cfg.num_key_value_heads, 
                         cfg.intermediate_size, i, self.rope)
            for i in range(cfg.num_hidden_layers)
        ])
        self.norm = RMSNorm(cfg.hidden_size)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

    def load_weights_tp(self, hf_sd):
        rank = get_tp_rank()
        tp_size = get_tp_size()

        self.embedding.weight.data.copy_(hf_sd["model.embed_tokens.weight"])
        
        # 适配 Tied Weights
        if "lm_head.weight" in hf_sd:
            self.lm_head.weight.data.copy_(hf_sd["lm_head.weight"])
        else:
            self.lm_head.weight = self.embedding.weight 

        self.norm.weight.data.copy_(hf_sd["model.norm.weight"])

        for i, layer in enumerate(self.layers):
            p = f"model.layers.{i}"
            
            # 1. 提取原始权重并进行 TP 切分
            # 注意：Q, K, V 必须分别切分后再合并
            Wq = hf_sd[f"{p}.self_attn.q_proj.weight"].chunk(tp_size, dim=0)[rank]
            Wk = hf_sd[f"{p}.self_attn.k_proj.weight"].chunk(tp_size, dim=0)[rank]
            Wv = hf_sd[f"{p}.self_attn.v_proj.weight"].chunk(tp_size, dim=0)[rank]
            
            # 打印一下维度，方便调试（可选）
            # if rank == 0 and i == 0:
            #     print(f"DEBUG: Wq={Wq.shape}, Wk={Wk.shape}, Wv={Wv.shape}")
            
            # 2. 合并
            combined_qkv = torch.cat([Wq, Wk, Wv], dim=0)
            
            # 3. 写入（确保 layer.attn.qkv 的定义与之匹配）
            layer.attn.qkv.weight.data.copy_(combined_qkv)
    
            # 4. 同样处理 Bias (Qwen2.5/3 默认有 bias)
            if f"{p}.self_attn.q_proj.bias" in hf_sd:
                Bq = hf_sd[f"{p}.self_attn.q_proj.bias"].chunk(tp_size, dim=0)[rank]
                Bk = hf_sd[f"{p}.self_attn.k_proj.bias"].chunk(tp_size, dim=0)[rank]
                Bv = hf_sd[f"{p}.self_attn.v_proj.bias"].chunk(tp_size, dim=0)[rank]
                layer.attn.qkv.bias.data.copy_(torch.cat([Bq, Bk, Bv], dim=0))

    def forward(self, input_ids, request_ids, pos, block_mgr, is_prefill=False, cu_seqlens=None, max_seqlen=None):
        x = self.embedding(input_ids).half()
        for layer in self.layers:
            x = layer(x, request_ids, pos, block_mgr, is_prefill, cu_seqlens, max_seqlen)
        return self.lm_head(self.norm(x))
