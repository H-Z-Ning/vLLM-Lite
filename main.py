import torch
import torch.multiprocessing as mp
import yaml
import time
from executor import worker

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def main():
    config = load_config()
    gpu_count = torch.cuda.device_count()
    
    test_prompts = [
        "请介绍一下你自己",
        "The moon is",
        "Beijing is the capital of"
    ]
    num_requests = len(test_prompts)

    print(f"🚀 Starting VLLMLite with {gpu_count} GPUs.")
    print(f"📢 Total requests: {num_requests}\n")

    ctx = mp.get_context('spawn')
    result_queue = ctx.Queue()

    if gpu_count > 1:
        processes = mp.spawn(
            worker,
            args=(gpu_count, config, test_prompts, result_queue),
            nprocs=gpu_count,
            join=False
        )
    else:
        p = ctx.Process(target=worker, args=(0, 1, config, test_prompts, result_queue))
        p.start()

    # --- 监听并只显示 AI 的回答 ---
    completed = 0
    while completed < num_requests:
        if not result_queue.empty():
            req_id, ai_text = result_queue.get()
            
            if req_id == "DONE":
                break
            
            completed += 1
            # 格式化输出：去掉换行符以便在预览中查看
            clean_text = ai_text.replace('\n', ' ')
            print(f"✅ {req_id} 完成！")
            print(f"🤖 AI 回答: {clean_text[:200]}...")
            print("-" * 60)
        else:
            time.sleep(0.1)

    print(f"\n✨ 所有 {completed} 个任务已处理完毕。")

    if gpu_count > 1:
        processes.join()

if __name__ == "__main__":
    main()
