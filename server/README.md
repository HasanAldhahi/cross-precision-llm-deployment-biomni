# Server Module

> **Model Inference Infrastructure**: vLLM-based serving with multi-GPU load balancing for biomedical LLM evaluation.

This module provides the infrastructure for deploying and evaluating the quantized models using vLLM with multi-GPU support.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Module Structure](#-module-structure)
- [Server Architecture](#-server-architecture)
- [Installation](#-installation)
- [Usage](#-usage)
- [Logs](#-logs)

---

## 🎯 Overview

The server module enables high-throughput inference for evaluating different model configurations:

| Feature | Description |
|---------|-------------|
| **Backend** | vLLM with FP8/LoRA support |
| **Multi-GPU** | 4x GPU load balancing |
| **Endpoints** | OpenAI-compatible API |
| **Evaluation** | Eval1 biomedical benchmark |

---

## 📁 Module Structure

```
server/
├── requirements.txt              # Python dependencies
├── README.md                     # This file
│
├── 📂 Scripts
│   ├── run_server.sh             # Main server launch script
│   ├── run_client.sh             # Evaluation client launcher
│   ├── run_client_Eval1_benchmark.py  # Eval1 benchmark runner
│   ├── load_balancer.py          # Multi-GPU request distributor
│   └── diagnose_servers.sh       # Health check utility
│
├── 📂 Documentation
│   └── LOAD_BALANCER_FIX.md      # Load balancer troubleshooting
│
└── 📂 Logs
    ├── vllm_gpu0.log             # GPU 0 server log
    ├── vllm_gpu1.log             # GPU 1 server log
    ├── vllm_gpu2.log             # GPU 2 server log
    ├── vllm_gpu3.log             # GPU 3 server log
    └── load_balancer.log         # Load balancer log
```

---

## 🏗️ Server Architecture

### Multi-GPU Setup

```
┌─────────────────────────────────────────────────┐
│                  Load Balancer                  │
│              (Round-Robin Distribution)         │
│                  Port: 8000                     │
└────────────┬───────────┬───────────┬───────────┘
             │           │           │           │
        ┌────▼───┐  ┌────▼───┐  ┌────▼───┐  ┌────▼───┐
        │ vLLM   │  │ vLLM   │  │ vLLM   │  │ vLLM   │
        │ GPU 0  │  │ GPU 1  │  │ GPU 2  │  │ GPU 3  │
        │ :8001  │  │ :8002  │  │ :8003  │  │ :8004  │
        └────────┘  └────────┘  └────────┘  └────────┘
```

### Supported Model Configurations

| Configuration | Command Flag | Memory/GPU |
|---------------|--------------|------------|
| R0-32B-BF16 | `--model r0-bf16` | ~17GB |
| R0-32B-FP8 (Method C) | `--model r0-fp8` | ~9GB |
| Qwen-FP8 + LoRA (Method A) | `--model qwen-fp8-lora` | ~10GB |

---

## 🔧 Installation

```bash
pip install -r requirements.txt
```

### Key Dependencies

- `vllm>=0.4.0` - High-performance LLM serving
- `openai` - API client for evaluation
- `aiohttp` - Async HTTP for load balancer
- `python-dotenv` - Environment variable management

---

## 💻 Usage

### Start Server

```bash
# Full precision baseline
bash run_server.sh --model r0-bf16 --gpus 4

# Method C: Direct quantization
bash run_server.sh --model r0-fp8 --gpus 4

# Method A: FP8 + LoRA
bash run_server.sh --model qwen-fp8-lora --gpus 4 \
    --lora-path ../brain_surgery/lora_extraction_results/Method_A_lora_basic_original_base_rank_256
```

### Run Evaluation

```bash
bash run_client.sh --benchmark eval1
# OR
python run_client_Eval1_benchmark.py --server http://localhost:8000
```

### Diagnose Issues

```bash
bash diagnose_servers.sh
```

---

## 📋 Logs

| Log | Description |
|-----|-------------|
| `vllm_gpu{0-3}.log` | Per-GPU vLLM server output |
| `load_balancer.log` | Request distribution logs |

Logs contain:
- Model loading progress
- Request/response latencies
- Error traces
- Memory usage statistics

---

## ⚠️ Troubleshooting

See [LOAD_BALANCER_FIX.md](LOAD_BALANCER_FIX.md) for common issues and solutions.
