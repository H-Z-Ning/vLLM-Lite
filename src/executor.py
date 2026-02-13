import os
import torch
import torch.distributed as dist
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
from kernel import BlockManager
from model import VLLMLite

class Sequence:
    def __init__(self, request_id, prompt_ids, max_gen_len=512):
        self.request_id = request_id
        self.prompt_ids = prompt_ids
        self.generated_ids = prompt_ids.copy()
        self.max_gen_len = max_gen_len
        self.finished = False

class Engine:
    def __init__(self, model, tokenizer, block_mgr, config, result_queue):
        self.model = model
        self.tokenizer = tokenizer
        self.block_mgr = block_mgr
        self.result_queue = result_queue # 传入队列
        self.running_batch = []
        self.waiting_queue = []

        self.max_batch = config['engine_config'].get('max_batch_size', 20)
        self.max_gen_len = config['engine_config'].get('max_gen_len', 512)
        self.extra_slots = config['cache_config'].get('extra_token_slot', 128)
        self.rank = dist.get_rank() if dist.is_initialized() else 0

    def add_request(self, req_id, prompt):
        ids = self.tokenizer.encode(prompt)
        self.waiting_queue.append(Sequence(req_id, ids, self.max_gen_len))

    def step(self):
        if self.waiting_queue and len(self.running_batch) < self.max_batch:
            new_seqs = []
            while self.waiting_queue and len(self.running_batch) < self.max_batch:
                new_seqs.append(self.waiting_queue.pop(0))
            self._handle_prefill(new_seqs)
            self.running_batch.extend(new_seqs)

        if not self.running_batch:
            return False

        self._handle_decode()
        return True

    def _handle_prefill(self, seqs):
        if not seqs: return
        all_ids, cu_seqlens, pos_list = [], [0], []
        for seq in seqs:
            needed_token_space = len(seq.prompt_ids) + self.extra_slots
            self.block_mgr.allocate_blocks_for_request(seq.request_id, needed_token_space)
            all_ids.extend(seq.prompt_ids)
            for p in range(len(seq.prompt_ids)):
                pos_list.append(p)
            cu_seqlens.append(len(all_ids))

        input_tensor = torch.tensor(all_ids, device="cuda")
        cu_seqlens_tensor = torch.tensor(cu_seqlens, device="cuda", dtype=torch.int32)
        pos_tensor = torch.tensor(pos_list, device="cuda", dtype=torch.int32)
        req_ids_flat = [s.request_id for s in seqs for _ in range(len(s.prompt_ids))]

        with torch.inference_mode(), torch.amp.autocast('cuda', dtype=torch.float16):
            logits = self.model(input_tensor, req_ids_flat, pos_tensor, self.block_mgr,
                               is_prefill=True, cu_seqlens=cu_seqlens_tensor,
                               max_seqlen=max(len(s.prompt_ids) for s in seqs))

        last_indices = cu_seqlens_tensor[1:] - 1
        next_tokens = logits[last_indices].argmax(dim=-1)
        for i, seq in enumerate(seqs):
            seq.generated_ids.append(next_tokens[i].item())

    def _handle_decode(self):
        if not self.running_batch: return
        input_tokens = torch.tensor([seq.generated_ids[-1] for seq in self.running_batch], device="cuda")
        req_ids = [seq.request_id for seq in self.running_batch]
        pos_list = torch.tensor([len(seq.generated_ids) - 1 for seq in self.running_batch],
                                device="cuda", dtype=torch.int32)

        with torch.inference_mode(), torch.amp.autocast('cuda', dtype=torch.float16):
            logits = self.model(input_tokens, req_ids, pos_list, self.block_mgr, is_prefill=False)
        
        next_tokens = logits.argmax(dim=-1)

        for i, seq in enumerate(self.running_batch):
            token_id = next_tokens[i].item()
            seq.generated_ids.append(token_id)
            
            # 停止判定
            if token_id == self.tokenizer.eos_token_id or len(seq.generated_ids) >= seq.max_gen_len:
                seq.finished = True
                self.block_mgr.free_request(seq.request_id)
                
                if self.rank == 0:
                    # 【核心修改】：只取 prompt 长度之后的部分
                    prompt_len = len(seq.prompt_ids)
                    ai_only_ids = seq.generated_ids[prompt_len:]
                    
                    # 解码 AI 生成的内容
                    ai_response = self.tokenizer.decode(ai_only_ids, skip_special_tokens=True)
                    self.result_queue.put((seq.request_id, ai_response.strip()))

        self.running_batch = [s for s in self.running_batch if not s.finished]

def worker(rank, world_size, config, prompts, result_queue):
    if world_size > 1:
        os.environ['MASTER_ADDR'] = config['dist_config'].get('master_addr', '127.0.0.1')
        os.environ['MASTER_PORT'] = config['dist_config'].get('master_port', '29505')
        dist.init_process_group("nccl", rank=rank, world_size=world_size)

    torch.cuda.set_device(rank)
    tokenizer = AutoTokenizer.from_pretrained(config['model_path'])
    hf_model = AutoModelForCausalLM.from_pretrained(config['model_path'], torch_dtype=torch.float16, device_map="cpu")
    
    model = VLLMLite(hf_model.config).to("cuda").to(torch.float16).eval()
    model.load_weights_tp(hf_model.state_dict())
    del hf_model

    attn_conf = model.layers[0].attn
    block_mgr = BlockManager(
        num_blocks=config['cache_config']['num_blocks'],
        block_size=config['cache_config']['block_size'],
        num_layers=len(model.layers),
        num_kv_heads=attn_conf.num_kv_heads * world_size,
        head_dim=attn_conf.head_dim
    )

    # 引擎接收 result_queue
    engine = Engine(model, tokenizer, block_mgr, config, result_queue)

    for i, p_content in enumerate(prompts):
        chat_prompt = tokenizer.apply_chat_template([{"role": "user", "content": p_content}], 
                                                   tokenize=False, add_generation_prompt=True)
        engine.add_request(f"REQ_{i}", chat_prompt)

    while True:
        work_done = engine.step()
        if not work_done and not engine.waiting_queue:
            break
        # 给 CPU 喘息时间
        time.sleep(0.001)

    # 为了让 main 知道全部结束，放一个结束标志
    if rank == 0:
        result_queue.put(("DONE", None))

    if world_size > 1:
        dist.destroy_process_group()
