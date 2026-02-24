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

# --- 全局配置加载 ---
config = yaml.safe_load(open("config.yaml", "r"))
tokenizer = AutoTokenizer.from_pretrained(config['model_path'])

# 用于管理请求状态的全局字典
pending_events = {}
pending_results = {}

# --- 1. 使用 lifespan 管理后台任务和进程清理 ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动后台结果监听协程
    handler_task = asyncio.create_task(result_handler())
    yield
    # 停止后的清理逻辑（可选）
    handler_task.cancel()

app = FastAPI(lifespan=lifespan)

# --- 2. 异步结果监听器 ---
async def result_handler():
    while True:
        # 这里 result_queue 会在 main 中定义并传入
        while not result_queue.empty():
            try:
                # 使用 get_nowait 避免阻塞协程
                req_id, text = result_queue.get_nowait()
                if req_id in pending_events:
                    pending_results[req_id] = text
                    pending_events[req_id].set()
            except:
                break
        await asyncio.sleep(0.01)

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    data = await request.json()
    messages = data.get("messages", [])
    model_name = data.get("model", "vllm-lite")
    
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt_ids = tokenizer.encode(prompt)
    
    req_id = f"chatcmpl-{uuid.uuid4()}"
    max_gen_len = data.get("max_tokens", config['engine_config']['max_gen_len'])
    
    event = asyncio.Event()
    pending_events[req_id] = event
    
    # 将任务放入输入队列
    input_queue.put((req_id, prompt_ids, max_gen_len))
    
    try:
        await asyncio.wait_for(event.wait(), timeout=120)
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
    # --- 重要：统一使用 spawn 上下文 ---
    ctx = mp.get_context('spawn')
    
    # 所有的 Queue 必须从 ctx 创建
    input_queue = ctx.Queue()
    result_queue = ctx.Queue()
    
    gpu_count = torch.cuda.device_count()
    print(f"🚀 Starting VLLMLite with {gpu_count} GPUs using 'spawn' context.")

    processes = []
    for rank in range(gpu_count):
        # 所有的 Process 必须从 ctx 创建
        p = ctx.Process(
            target=worker, 
            args=(rank, gpu_count, config, input_queue, result_queue)
        )
        p.start()
        processes.append(p)
    
    try:
        uvicorn.run(app, host="0.0.0.0", port=8000)
    finally:
        # 确保退出时关闭子进程
        for p in processes:
            p.terminate()
            p.join()
