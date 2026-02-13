# vLLM-Lite: A Minimalist Implementation of High-Performance LLM Inference
[English](./README.en.md) | [简体中文](./README.md)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)
[![License-MIT](https://img.shields.io/badge/license-MIT-green.svg)](./LICENSE)

**vLLM-Lite** is a lightweight large language model (LLM) inference framework designed for education and rapid prototyping. In less than a thousand lines of pure Python code, it reproduces the core technologies of industrial-grade inference engines: **PagedAttention**, **Tensor Parallelism (TP)**, and **Continuous Batching**.

If you find the official vLLM codebase (with extensive C++/CUDA mixed programming) too large and complex to dissect, **vLLM-Lite** is your best hands-on guide to deeply understanding the working principles of modern inference engines.

---

## ✨ Key Features

* **🧩 Minimalist PagedAttention**: Pure Python implementation of physical/logical block mapping, intuitively demonstrating how KV Cache eliminates memory fragmentation.
* **🚀 Continuous Batching**: Implements dynamic hybrid scheduling of Prefill and Decode phases to boost system throughput.
* **🔥 Operator-Level Acceleration**: Core integration of `flash_attn_with_kvcache` achieves near-native inference speeds while maintaining code readability.
* **📡 Multi-GPU Tensor Parallelism (TP)**: Built on `torch.distributed`, supports automatic model weight sharding and collaborative multi-GPU inference.
* **🛠️ Easy to Extend**: Decoupled architecture supports adapting new models or experimenting with new scheduling algorithms in minutes.

---

## 🛠️ Technical Architecture

vLLM-Lite's design follows the principle of "core functionality modularization":

| Module File | Core Responsibility |
| :--- | :--- |
| **`kernel.py`** | Contains `BlockManager` (memory pool management) and paged attention operator logic |
| **`executor.py`** | The brain of the inference engine, responsible for request queue management, state machine transitions, and token decoding |
| **`model.py`** | Standard Transformer architecture, supporting distributed weight loading and TP communication |
| **`config.yaml`** | Centralized management of model paths, memory block sizes, and scheduling policies |

---

## 🚀 Quick Start

### 1. Environment Setup
Ensure your environment has CUDA 12.x and Flash Attention 2.0+ installed:
```bash
pip install torch transformers pyyaml flash-attn --no-build-isolation
```

### 2. Configuration
Set your model path in `config.yaml` (supports mainstream architectures like Qwen2 / Llama by default):
```yaml
model_path: "Qwen/Qwen2.5-1.5B-Instruct"
cache_config:
  num_blocks: 1024  # Total number of KV Cache blocks
  block_size: 256   # Number of tokens per block
```

### 3. Start Inference
```bash
python main.py
```

---

## 💡 Why This Project Deserves Your Attention?

* **See Through PagedAttention**: The complex C++ memory paging logic is simplified into Python `list` operations within the `BlockManager` class, allowing you to grasp the core secret of vLLM at a glance.
* **Engineering Reference**: Completely demonstrates how to handle multi-process `spawn`, multi-GPU NCCL communication, and production-ready asynchronous result queues.
* **Zero-Refactoring Migration**: Uses a weight loading method fully compatible with HuggingFace, making it easy for developers to quickly test their private models.

---

## 📊 Inference Workflow Analysis

vLLM-Lite implements a streamlined **iterative scheduling strategy**:

1.  **Prefill Phase**: Aggregates new requests, uses `flash_attn_varlen_func` for efficient prompt preprocessing, and allocates physical blocks for the KV Cache.
2.  **Decode Phase**: For ongoing requests, retrieves physical block indices from the `BlockManager` and performs efficient single-token iteration using the paged cache.
3.  **Resource Reclamation**: Once a sequence triggers a stop token (EOS) or reaches its maximum length, the corresponding physical blocks are immediately returned to the `free_blocks` queue, enabling dynamic memory reuse.

---

## 🤝 Contribution & Discussion

Looking forward to your participation! You can contribute by:
* Optimizing the `BlockManager` allocation algorithm to reduce fragmentation.
* Adapting more model architectures like `DeepSeek`, `Mixtral`, etc.
* Improving multi-GPU communication efficiency or introducing kernel fusion.

Feel free to submit a **Pull Request** or open an **Issue** for discussion!

---

## 📄 License

This project is licensed under the **[MIT](./LICENSE)** License.
