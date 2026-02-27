import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uuid
import torch.multiprocessing as mp
import torch
import yaml
import time
import asyncio
from contextlib import asynccontextmanager
from transformers import AutoTokenizer
from executor import worker

config = yaml.safe_load(open("config.yaml", "r"))
tokenizer = AutoTokenizer.from_pretrained(config['model_path'])

pending_events = {}
pending_results = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    handler_task = asyncio.create_task(result_handler())
    yield
    handler_task.cancel()

app = FastAPI(lifespan=lifespan)

async def result_handler():
    while True:
        while not result_queue.empty():
            try:
                req_id, text = result_queue.get_nowait()
                if req_id in pending_events:
                    pending_results[req_id] = text
                    pending_events[req_id].set()
            except: break
        await asyncio.sleep(0.01)

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    data = await request.json()
    messages = data.get("messages", [])
    model_name = data.get("model", "vllm-lite")
    
    # --- 新增：提取采样参数 ---
    sampling_params = {
        "temperature": data.get("temperature", 0.7),
        "top_p": data.get("top_p", 0.8),
        "repetition_penalty": data.get("repetition_penalty", 1.1)
    }
    max_gen_len = data.get("max_tokens", config['engine_config']['max_gen_len'])
    # ------------------------

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt_ids = tokenizer.encode(prompt)
    req_id = f"chatcmpl-{uuid.uuid4()}"
    
    event = asyncio.Event()
    pending_events[req_id] = event
    
    # 这里的 input_queue 增加发送采样参数
    input_queue.put((req_id, prompt_ids, max_gen_len, sampling_params))
    
    try:
        await asyncio.wait_for(pending_events[req_id].wait(), timeout=120)
        response_text = pending_results.pop(req_id)
    except asyncio.TimeoutError:
        return JSONResponse({"error": "Inference Timeout"}, status_code=504)
    finally:
        pending_events.pop(req_id, None)

    return {
        "id": req_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_name,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": response_text},
            "finish_reason": "stop"
        }]
    }

if __name__ == "__main__":
    ctx = mp.get_context('spawn')
    input_queue = ctx.Queue()
    result_queue = ctx.Queue()
    
    gpu_count = torch.cuda.device_count()
    processes = []
    for rank in range(gpu_count):
        p = ctx.Process(target=worker, args=(rank, gpu_count, config, input_queue, result_queue))
        p.start()
        processes.append(p)
    
    try:
        uvicorn.run(app, host="0.0.0.0", port=8000)
    finally:
        for p in processes:
            p.terminate()
            p.join()
