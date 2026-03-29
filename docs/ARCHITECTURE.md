# Technical Architecture: Local LLM Deployment

## Overview
This document describes the architecture and deployment options for running Large Language Models locally on standard hardware.

## 1. Local LLM Deployment Architecture

```
┌─────────────────────────────────────────────────┐
│          User Interface / Application           │
│  (Streamlit, Web UI, CLI, or Python Scripts)   │
└──────────────────┬──────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────┐
│      LLM Inference Engine / Runtime             │
│   (Ollama / llama.cpp / vLLM / Transformers)   │
└──────────────────┬──────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────┐
│        Quantized LLM Model (Local)              │
│  (GGUF, AWQ, GPTQ formats - 4-8GB RAM)         │
└──────────────────┬──────────────────────────────┘
                   │
        ┌──────────┴──────────┐
        ▼                     ▼
   ┌─────────┐         ┌──────────┐
   │   CPU   │         │   GPU    │
   │         │         │ (Optional│
   │ (8+ GB) │         │  NVIDIA) │
   └─────────┘         └──────────┘
```

## 2. Deployment Options

### Option A: CPU-Based Deployment (Recommended for Start)

**Tools:** Ollama or llama.cpp

**Characteristics:**
- No GPU required
- Works on any machine with 16GB+ RAM
- Moderate latency (1-5 seconds per token)
- Stable and predictable performance
- Lower setup complexity

**Workflow:**
```
1. Download GGUF quantized model
2. Run with Ollama or llama.cpp
3. Access via REST API or CLI
4. Integrate with applications
```

**Example Command:**
```bash
ollama run mistral:7b-instruct-q4
```

**Recommended Models:**
- Mistral 7B (Q4_K_M)
- LLaMA 2 7B (Q4_K_M)
- Phi 2.7B (Q4_K_M)

### Option B: GPU-Based Deployment (Advanced)

**Tools:** vLLM, Transformers, or Ollama with GPU

**Characteristics:**
- Requires NVIDIA GPU (RTX 3060 or better recommended)
- Fast inference (10-50 tokens/second)
- Higher memory overhead
- More setup complexity
- Better for production workloads

**Quantization Methods:**
- **AWQ** (Activation-Aware Quantization) - Preferred
- **GPTQ** (General Post-Training Quantization) - Alternative
- **INT8** - Post-training quantization

**Workflow:**
```
1. Download AWQ/GPTQ quantized model
2. Load with vLLM or Transformers
3. Run inference on GPU
4. Scale with batch processing
```

**Example Setup:**
```python
from vllm import LLM, SamplingParams

llm = LLM(model="mistralai/Mistral-7B-Instruct-v0.1")
sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=100)
outputs = llm.generate(prompts, sampling_params)
```

## 3. Model Quantization Overview

### Why Quantization?
Local machines have limited RAM. Quantization compresses models without significant quality loss.

### Quantization Levels

| Format | Size | Quality | Speed | Use Case |
|--------|------|---------|-------|----------|
| FP16   | 100% | Highest | Slowest | GPU with large VRAM |
| INT8   | 50%  | Good    | Medium | GPU standard |
| INT4   | 25%  | Good    | Fast   | CPU/small GPU |
| GGUF Q4| 25%  | Good    | Fast   | CPU optimal |

### Quantization Type: GGUF (CPU)
- **Format:** GGUF (GPTQ-like format)
- **Q4_K_M:** 4-bit quantization with medium K values
- **Tools:** llama.cpp, Ollama
- **Advantage:** Optimized for CPU inference

### Quantization Type: AWQ (GPU)
- **Activation-Aware Quantization**
- **Preserves** activations that matter most
- **Tools:** vLLM, AutoGPTQ
- **Advantage:** Better quality at 4-bit

### Quantization Type: GPTQ (GPU)
- **General Post-Training Quantization**
- **Iterative method** for quantization
- **Tools:** AutoGPTQ, vLLM
- **Advantage:** Wide model support

## 4. Hybrid Setup (Local + Cloud)

For applications needing both speed and capability:

```
┌──────────────────────────────────────────┐
│         Application / User Request       │
└────────────────┬─────────────────────────┘
                 │
         ┌───────▼─────────┐
         │   Router/Logic  │
         └───────┬─────────┘
                 │
      ┌──────────┴──────────┐
      ▼                     ▼
┌──────────────┐   ┌──────────────────┐
│  Local LLM   │   │    Cloud LLM     │
│ (Fast, Free) │   │ (Capable, Slow)  │
│              │   │                  │
│ Simple tasks │   │ Complex tasks    │
│ Quick QA     │   │ Advanced analysis│
└──────────────┘   └──────────────────┘
```

**Routing Logic:**
- **Local:** Document Q&A, simple summarization, fast responses
- **Cloud:** Complex reasoning, generation, specialized tasks

## 5. Vector Database Integration (for RAG)

For Document Q&A use case:

```
┌──────────────────────────────────┐
│      Document Input (PDF/TXT)    │
└────────────────┬─────────────────┘
                 │
         ┌───────▼──────────┐
         │ Text Chunking    │
         └───────┬──────────┘
                 │
         ┌───────▼──────────────┐
         │ Embedding Generation │
         │ (Sentence Transformers│
         └───────┬──────────────┘
                 │
         ┌───────▼──────────────────┐
         │ Vector Store (FAISS/etc) │
         └───────┬──────────────────┘
                 │
         ┌───────▼──────────┐
         │ User Query       │
         └───────┬──────────┘
                 │
         ┌───────▼──────────────┐
         │ Similarity Search    │
         └───────┬──────────────┘
                 │
         ┌───────▼──────────────┐
         │ LLM with Context     │
         │ + Retrieved Docs     │
         └───────┬──────────────┘
                 │
         ┌───────▼──────────────┐
         │ Generated Answer     │
         └──────────────────────┘
```

**Tools:**
- FAISS (Facebook AI Similarity Search)
- Chroma (Open-source vector DB)
- Milvus (Scalable vector DB)

## 6. Hardware Reference

### CPU-Only Setup
```
CPU:         Intel i7/AMD Ryzen 7 (8+ cores)
RAM:         32 GB DDR4+
Storage:     NVMe SSD (500GB+)
GPU:         None
Performance: ~2-3 tokens/second
Cost:        $400-800
```

### CPU + GPU Setup
```
CPU:         Intel i7/AMD Ryzen 7 (8+ cores)
RAM:         32 GB DDR4+
GPU:         NVIDIA RTX 4060 Ti (8GB) or better
Storage:     NVMe SSD (500GB+)
Performance: ~10-30 tokens/second
Cost:        $800-2000
```

## 7. Model Landscape

### Recommended Models (7B-8B range)

**LLaMA 2 7B**
- Source: Meta
- Quantization: GGUF (CPU), AWQ (GPU)
- Use: General purpose, RAG tasks
- Quality: High

**Mistral 7B**
- Source: Mistral AI
- Quantization: GGUF (CPU), AWQ (GPU)
- Use: Fast, instruction-following
- Quality: Very High

**Phi 2.7B**
- Source: Microsoft
- Quantization: GGUF
- Use: Lightweight, coding tasks
- Quality: Good for size

**Gemma 7B**
- Source: Google
- Quantization: GGUF, GPTQ
- Use: General purpose, coding
- Quality: High

### Model Size Recommendations

| Model Size | RAM (CPU) | RAM (GPU) | Speed | Quality |
|------------|-----------|-----------|-------|---------|
| 3B         | 8 GB      | 4 GB      | Fast  | Fair    |
| 7B         | 16 GB     | 6 GB      | Good  | Good    |
| 13B        | 32 GB     | 10 GB     | Good  | Very Good |
| 34B+       | 64+ GB    | 24 GB+    | Slow  | Excellent |

## 8. Performance Benchmarks

### Token Generation Speed

**CPU (Intel i9, 16GB RAM):**
- LLaMA 7B Q4: ~1-2 tokens/second
- Mistral 7B Q4: ~2-3 tokens/second
- Phi 2.7B Q4: ~3-5 tokens/second

**GPU (RTX 4060 Ti, 8GB):**
- LLaMA 7B AWQ: ~15-20 tokens/second
- Mistral 7B AWQ: ~20-30 tokens/second
- Phi 2.7B FP16: ~30-40 tokens/second

## 9. Summary Table

| Aspect | CPU | GPU |
|--------|-----|-----|
| Setup  | Easy | Medium |
| Cost   | $400 | $1000+ |
| Speed  | Slow | Fast |
| Models | 7B optimal | 13B+ possible |
| Quant  | GGUF Q4 | AWQ/GPTQ |
| Use    | Dev/Test | Production |

## Next Steps
- See [SETUP.md](SETUP.md) for detailed installation
- See [BEST_PRACTICES.md](BEST_PRACTICES.md) for optimization tips
- See [usecases/](usecases/) for example implementations