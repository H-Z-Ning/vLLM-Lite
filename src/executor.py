import os
import torch
import torch.distributed as dist
import time
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from kernel import BlockManager
from model import VLLMLite

class Sequence:
    def __init__(self, request_id, prompt_ids, max_gen_len=2048, sampling_params=None):
        self.request_id = request_id
        self.prompt_ids = prompt_ids
        self.generated_ids = prompt_ids.copy()
        self.max_gen_len = max_gen_len
        # 默认采样参数
        self.sampling_params = sampling_params or {
            "repetition_penalty": 1.1,
            "temperature": 0.7,
            "top_p": 0.8
        }
        self.finished = False

def sample_logits(logits, sequence_ids, repetition_penalty=1.1, temperature=0.7, top_p=0.8):
    logits = logits.clone()

    if repetition_penalty != 1.0:
        score = torch.gather(logits, 1, sequence_ids)
        score = torch.where(score > 0, score / repetition_penalty, score * repetition_penalty)
        logits.scatter_(1, sequence_ids, score)

    # 防止 temperature 为 0 导致除以 0 错误
    temperature = max(temperature, 1e-5)
    logits = logits / temperature

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = float('-inf')

    probs = torch.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    return next_token.squeeze(-1)

class Engine:
    def __init__(self, model, tokenizer, block_mgr, config, result_queue):
        self.model = model
        self.tokenizer = tokenizer
        self.block_mgr = block_mgr
        self.result_queue = result_queue
        self.running_batch = []
        self.waiting_queue = []
        self.max_batch = config['engine_config'].get('max_batch_size', 20)
        self.extra_slots = config['cache_config'].get('extra_token_slot', 4)

    def add_request(self, req_id, prompt_ids, max_gen_len, sampling_params):
        self.waiting_queue.append(Sequence(req_id, prompt_ids, max_gen_len, sampling_params))

    def step(self):
        new_seqs_to_prefill = []
        while self.waiting_queue and len(self.running_batch) < self.max_batch:
            seq = self.waiting_queue.pop(0)
            try:
                needed_space = len(seq.prompt_ids) + self.extra_slots
                self.block_mgr.allocate_blocks_for_request(seq.request_id, needed_space)
                new_seqs_to_prefill.append(seq)
            except MemoryError:
                self.waiting_queue.insert(0, seq)
                break

        if new_seqs_to_prefill:
            self._handle_prefill(new_seqs_to_prefill)
            self.running_batch.extend(new_seqs_to_prefill)

        if not self.running_batch:
            return False

        self._handle_decode()
        return True

    def _handle_prefill(self, seqs):
        if not seqs: return
        all_ids, cu_seqlens, pos_list = [], [0], []
        for seq in seqs:
            all_ids.extend(seq.prompt_ids)
            for p in range(len(seq.prompt_ids)):
                pos_list.append(p)
            cu_seqlens.append(len(all_ids))

        input_tensor = torch.tensor(all_ids, device="cuda")
        cu_seqlens_tensor = torch.tensor(cu_seqlens, device="cuda", dtype=torch.int32)
        pos_tensor = torch.tensor(pos_list, device="cuda", dtype=torch.int32)
        req_ids_flat = [s.request_id for s in seqs for _ in range(len(s.prompt_ids))]

        with torch.inference_mode():
            logits = self.model(input_tensor, req_ids_flat, pos_tensor, self.block_mgr, 
                               is_prefill=True, cu_seqlens=cu_seqlens_tensor, 
                               max_seqlen=max(len(s.prompt_ids) for s in seqs))
        
        last_indices = cu_seqlens_tensor[1:] - 1
        # Prefill 阶段简单使用 argmax 或应用第一个 seq 的采样（此处为简化统一使用 argmax）
        next_tokens = logits[last_indices].argmax(dim=-1)
        for i, seq in enumerate(seqs):
            seq.generated_ids.append(next_tokens[i].item())

    def _handle_decode(self):
        if not self.running_batch: return
        
        for seq in self.running_batch:
            current_len = len(seq.generated_ids)
            self.block_mgr.allocate_blocks_for_request(seq.request_id, current_len + 1)

        input_tokens = torch.tensor([s.generated_ids[-1] for s in self.running_batch], device="cuda")
        req_ids = [s.request_id for s in self.running_batch]
        pos_list = torch.tensor([len(s.generated_ids)-1 for s in self.running_batch], device="cuda", dtype=torch.int32)

        with torch.inference_mode():
            logits = self.model(input_tokens, req_ids, pos_list, self.block_mgr, is_prefill=False)
        
        next_tokens = []
        for i, seq in enumerate(self.running_batch):
            seq_history = torch.tensor([seq.generated_ids], device=logits.device)
            single_logits = logits[i:i+1] 
            
            # 使用 seq 自带的动态参数
            token_id = sample_logits(
                single_logits, 
                seq_history, 
                **seq.sampling_params
            )
            if isinstance(token_id, torch.Tensor):
                token_id = token_id.item()
            next_tokens.append(token_id)
        
        stop_tokens = {
            self.tokenizer.eos_token_id, 
            151643, 151645, 
            self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        }

        for i, seq in enumerate(self.running_batch):
            token_id = next_tokens[i]
            seq.generated_ids.append(token_id)
            
            if token_id in stop_tokens or len(seq.generated_ids) >= seq.max_gen_len:
                seq.finished = True
                self.block_mgr.free_request(seq.request_id)
                response = self.tokenizer.decode(seq.generated_ids[len(seq.prompt_ids):], skip_special_tokens=True)
                self.result_queue.put((seq.request_id, response))
        
        self.running_batch = [s for s in self.running_batch if not s.finished]

def profile_memory(model, config, world_size):
    torch.cuda.empty_cache()
    total_gpu_mem = torch.cuda.get_device_properties(0).total_memory
    used_mem = torch.cuda.memory_allocated()
    available_mem = total_gpu_mem * config['cache_config']['gpu_memory_utilization'] - used_mem
    
    attn_conf = model.layers[0].attn
    num_layers = len(model.layers)
    block_size = config['cache_config']['block_size']
    num_kv_heads = attn_conf.num_kv_heads
    head_dim = attn_conf.head_dim
    
    one_block_cache_size = num_layers * 2 * num_kv_heads * head_dim * block_size * 2
    num_blocks = int(available_mem // one_block_cache_size)
    return max(num_blocks, 64)

def worker(rank, world_size, config, input_queue, result_queue):
    if world_size > 1:
        os.environ['MASTER_ADDR'] = config['dist_config']['master_addr']
        os.environ['MASTER_PORT'] = config['dist_config']['master_port']
        dist.init_process_group("nccl", rank=rank, world_size=world_size)

    torch.cuda.set_device(rank)
    tokenizer = AutoTokenizer.from_pretrained(config['model_path'])
    hf_cfg = AutoConfig.from_pretrained(config['model_path'])
    model = VLLMLite(hf_cfg).half() 
    
    if rank == 0: print("⏳ Loading weights...")
    hf_model = AutoModelForCausalLM.from_pretrained(config['model_path'], torch_dtype=torch.float16, device_map="cpu")
    model.load_weights_tp(hf_model.state_dict())
    del hf_model
    gc.collect()
    
    model = model.to("cuda").eval()
    num_blocks = profile_memory(model, config, world_size)
    block_mgr = BlockManager(num_blocks, config['cache_config']['block_size'], len(model.layers), model.layers[0].attn.num_kv_heads * world_size, model.layers[0].attn.head_dim)
    engine = Engine(model, tokenizer, block_mgr, config, result_queue)

    while True:
        requests_to_add = []
        if rank == 0:
            try:
                while not input_queue.empty() and len(requests_to_add) < 16: 
                    requests_to_add.append(input_queue.get_nowait())
            except: pass
        if world_size > 1:
            requests_to_add = broadcast_object_list(requests_to_add, src=0)

        for req_id, prompt_ids, max_len, sampling_params in requests_to_add:
            engine.add_request(req_id, prompt_ids, max_len, sampling_params)
        
        work_done = engine.step()
        if not work_done and not requests_to_add:
            time.sleep(0.005)

def broadcast_object_list(obj_list, src=0):
    container = [obj_list]
    dist.broadcast_object_list(container, src=src)
    return container[0]
