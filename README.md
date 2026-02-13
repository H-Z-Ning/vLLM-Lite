# vLLM-Lite: A Minimalist Implementation of High-Performance LLM Inference

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)
[![License-MIT](https://img.shields.io/badge/license-MIT-green.svg)](./LICENSE)

**vLLM-Lite** 是一个专为教育和快速原型开发设计的轻量级大语言模型（LLM）推理框架。它在不足千行的纯 Python 代码中，复现了工业级推理引擎的核心技术：**PagedAttention**、**Tensor Parallelism (TP)** 和 **Continuous Batching**。

如果你觉得官方 vLLM 的源码（包含大量 C++/CUDA 混编）过于庞大难以拆解，那么 **vLLM-Lite** 是你深度理解现代推理引擎工作原理的最佳实践指南。

---

## ✨ 核心特性

* **🧩 极简 PagedAttention**: 纯 Python 实现的物理/逻辑块映射机制，直观展示 KV Cache 如何消除显存碎片。
* **🚀 连续批处理 (Continuous Batching)**: 实现了 Prefill 和 Decode 阶段的动态混合调度，提升系统吞吐。
* **🔥 算子级加速**: 核心集成 `flash_attn_with_kvcache`，在保持代码可读性的同时，获得接近原生的推理速度。
* **📡 多卡张量并行 (TP)**: 基于 `torch.distributed` 实现，支持模型权重自动切分与多 GPU 协同推理。
* **🛠️ 易于扩展**: 架构解耦，支持分钟级适配新模型或实验新的调度算法。



---

## 🛠️ 技术架构

vLLM-Lite 的设计遵循“核心功能模块化”原则：

| 模块文件 | 核心职责 |
| :--- | :--- |
| **`kernel.py`** | 包含 `BlockManager` (显存池管理) 与分页注意力算子逻辑 |
| **`executor.py`** | 推理引擎大脑，负责请求队列管理、状态机切换与 Token 解码 |
| **`model.py`** | 标准 Transformer 架构，支持分布式权重加载与 TP 通讯 |
| **`config.yaml`** | 集中化管理模型路径、显存块大小及调度策略 |



---

## 🚀 快速开始

### 1. 环境准备
确保你的环境中安装了 CUDA 12.x 以及 Flash Attention 2.0+：
```bash
pip install torch transformers pyyaml flash-attn --no-build-isolation
```
### 2. 配置参数
在 config.yaml 中设置你的模型路径（默认支持 Qwen2 / Llama 等主流架构）：
```yaml
model_path: "Qwen/Qwen2.5-1.5B-Instruct"
cache_config:
  num_blocks: 1024  # KV Cache 总块数
  block_size: 256   # 每个块容纳的 Token 数量
```

### 3. 启动推理
```bash
python main.py
```


---

## 💡 为什么这个项目值得关注？

* **透视 PagedAttention**: 将复杂的 C++ 显存分页逻辑简化为 `BlockManager` 类中的 Python `list` 操作，让你一眼看穿 vLLM 的核心秘密。
* **工程化参考**: 完整展示了如何处理多进程 `spawn`、多卡 NCCL 通讯以及生产环境中的结果异步队列（Async Queue）。
* **零重构迁移**: 采用与 HuggingFace 完全兼容的权重加载方式，方便开发者快速测试自己的私有模型。

---

## 📊 推理流程解析

vLLM-Lite 实现了精简化的 **迭代式调度策略**：

1.  **Prefill 阶段**: 将新请求聚合，利用 `flash_attn_varlen_func` 进行高效的 Prompt 预处理，并为 KV Cache 分配物理块。
2.  **Decode 阶段**: 针对正在运行的请求，从 `BlockManager` 获取物理块索引，利用分页缓存（Paged Cache）进行高效的单 Token 迭代。
3.  **资源回收**: 一旦序列触发停止词（EOS）或达到长度上限，立即将对应的物理块退还至 `free_blocks` 队列，实现显存的动态复用。



---

## 🤝 贡献与讨论

期待你的加入！你可以通过以下方式参与建设：
* 优化 `BlockManager` 的分配算法以减少碎片。
* 适配 `DeepSeek`、`Mixtral` 等更多模型架构。
* 改进多卡通讯效率或引入算子融合。

欢迎提交 **Pull Request** 或开 **Issue** 交流！

---

## 📄 开源协议

本项目采用 **[MIT](./LICENSE)** 协议。
