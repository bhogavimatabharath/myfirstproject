# Best Practices for Local LLM Deployment

## Overview
This guide provides practical recommendations for optimizing performance, quality, and cost when running LLMs locally.

## 1. Model Selection

### Choosing the Right Model Size

**7B Models (Recommended Start)**
- ✅ Fit on most hardware (16GB RAM)
- ✅ Fast enough for interactive use
- ✅ Good balance of quality and speed
- ✅ Examples: Mistral 7B, LLaMA 2 7B, Phi

**3B-5B Models (Lightweight)**
- ✅ Run on 8GB RAM or less
- ✅ Very fast inference
- ❌ Limited reasoning capability
- ✅ Use for: Simple Q&A, classification

**13B-34B Models (Advanced)**
- ❌ Require 32GB+ RAM (quantized)
- ❌ Slow on CPU, acceptable on GPU
- ✅ Better reasoning and understanding
- ✅ Use for: Complex tasks, production

### Model Quality Indicators

**Check these factors when selecting a model:**

1. **License**: Use open models (Apache 2.0, MIT, etc.)
2. **Benchmarks**: Look at HELM, LLaMA-bench scores
3. **Instruction Tuning**: Prefer chat/instruct versions
4. **Community**: Popularity on Hugging Face is a good sign
5. **Use Case Match**: Domain-specific models perform better

### Recommended Models by Use Case

| Use Case | Model | Size | Reason |
|----------|-------|------|--------|
| General Q&A | Mistral 7B | 7B | Fast, capable, good instruction following |
| Code Tasks | Phi 2.7B | 2.7B | Trained on code, lightweight |
| Creative Writing | LLaMA 2 7B | 7B | Good at generation, balanced |
| Document Analysis | Mistral 7B | 7B | Good long-context, understanding |

---

## 2. Quantization Strategy

### When to Use Each Quantization Level

| Scenario | Quantization | Reason |
|----------|--------------|--------|
| Limited RAM (8GB) | INT4/Q4 | Only option that fits |
| Standard RAM (16GB) | INT4/Q4 | Recommended, best speed-quality |
| Plenty of RAM (32GB+) | INT8 or FP16 | Better quality, still reasonable speed |
| Quality critical | INT8 or FP16 | Trade speed for quality |
| Speed critical | INT4 | Maximum speed trade |

### Quantization Best Practices

**For CPU (GGUF Format):**
```bash
# Q4_K_M: Optimal balance (RECOMMENDED)
ollama pull mistral:7b-q4-k-m

# Q6_K: Better quality, slower
ollama pull mistral:7b-q6-k

# Q3_K: Smallest, significant quality loss
ollama pull mistral:7b-q3-k-m
```

**For GPU (AWQ/GPTQ):**
```bash
# AWQ: Better quality (RECOMMENDED for GPU)
# Download from TheBloke/Mistral-7B-Instruct-v0.1-AWQ

# GPTQ: Good alternative, more models available
# Download from TheBloke/Mistral-7B-Instruct-v0.1-GPTQ
```

### Memory vs Quality Trade-off

```
Quality        │ FP16 ──► INT8 ──► INT4
               │  ▲        ▲        ▲
               │  High   Good      Fair
               │
Speed          │ INT4 ──► INT8 ──► FP16
               │  ▲        ▲        ▲
               │  Fast   Medium    Slow
               │
RAM Required   │ FP16 ──► INT8 ──► INT4
               │  ▲        ▲        ▲
               │ 16GB     8GB      4GB
```

**Rule of Thumb:** Start with Q4/INT4, move up if quality is insufficient.

---

## 3. Context Length Optimization

### What is Context Length?
Maximum number of tokens the model can consider at once. Longer context = more memory.

### Managing Context Length

```python
# Too long context = OOM errors, slow processing
# Too short context = Information loss

# Recommended: 2K tokens (default)
# Maximum: 4K tokens (CPU), 8K tokens (GPU)

from vllm import LLM, SamplingParams

llm = LLM(
    model="mistral-7b-awq",
    quantization="awq",
    gpu_memory_utilization=0.8
)

# Limit context to prevent OOM
sampling_params = SamplingParams(
    max_tokens=512,  # Keep individual responses short
)
```

### Best Practices for Context

1. **Summarize old conversation** after 5-10 turns
2. **Use RAG** for document context instead of long prompts
3. **Batch similar tasks** together
4. **Limit prompt length** to 500-1000 tokens for interactive use

---

## 4. Performance Tuning

### CPU Performance Tips

```bash
# 1. Use GGUF with optimizations
# Download GGUF Q4_K_M format (best CPU performance)

# 2. Set thread count to CPU cores
export OLLAMA_NUM_THREAD=8  # If you have 8 cores

# 3. Use faster models
ollama pull mistral:7b  # Faster than llama2

# 4. Enable GPU acceleration if available
export OLLAMA_CUDA=1    # For NVIDIA GPUs
export OLLAMA_METAL=1   # For Apple Silicon
```

### GPU Performance Tips

```python
from vllm import LLM

llm = LLM(
    model="mistral-7b-awq",
    quantization="awq",
    
    # Key performance settings:
    gpu_memory_utilization=0.9,  # Use up to 90% GPU memory
    max_model_len=4096,           # Limit context for speed
    dtype="float16",              # Use FP16 instead of FP32
    tensor_parallel_size=2        # If you have multiple GPUs
)
```

### Batch Processing for Throughput

```python
# Process multiple prompts at once (more efficient)
from vllm import LLM, SamplingParams

llm = LLM(model="mistral-7b-awq", quantization="awq")

prompts = [
    "What is AI?",
    "Explain machine learning",
    "What is deep learning?"
]

sampling_params = SamplingParams(max_tokens=100)

# Process all at once
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(f"{output.prompt} → {output.outputs[0].text}\n")
```

---

## 5. Retrieval-Augmented Generation (RAG)

### When to Use RAG

Use RAG when:
- ✅ You have specific documents to search
- ✅ You need factual accuracy
- ✅ You want to reduce hallucinations
- ✅ You have limited context window

### RAG Architecture

```
Document → Chunk → Embed → Vector DB
                               │
                        ┌──────┘
                        │
Query → Embed → Search → Retrieve
                            │
                        ┌───┴─────┐
                        │         │
                    Context   Original Prompt
                        │         │
                        └─────┬───┘
                              │
                          LLM Response
```

### RAG Best Practices

1. **Chunk Size:** 256-512 tokens (balance coverage vs specificity)
2. **Overlap:** 50-100 tokens (capture context across chunks)
3. **Embedding Model:** Use lightweight (sentence-transformers)
4. **Vector DB:** FAISS for local, Chroma for in-memory
5. **Retrieval Count:** Use top 3-5 most relevant chunks

### RAG Implementation Example

```python
from langchain import OpenAI, VectorStore, FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.llms import Ollama

# Setup local LLM
llm = Ollama(model="mistral:7b-instruct-q4")

# Setup embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load and chunk documents
from langchain.document_loaders import PyPDFLoader
loader = PyPDFLoader("document.pdf")
documents = loader.load()

splitter = CharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)
docs = splitter.split_documents(documents)

# Create vector store
vector_store = FAISS.from_documents(docs, embeddings)

# Create RAG chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever(k=3)
)

# Query
result = qa.run("What is the main topic of this document?")
print(result)
```

---

## 6. Response Caching

### Cache Similar Responses

```python
from functools import lru_cache
from vllm import LLM

llm = LLM(model="mistral-7b-awq", quantization="awq")

@lru_cache(maxsize=128)
def get_llm_response(prompt):
    """Cache LLM responses for identical prompts"""
    output = llm.generate(prompt)
    return output[0].outputs[0].text

# Repeated prompt uses cached result
response1 = get_llm_response("What is AI?")
response2 = get_llm_response("What is AI?")  # Returns cached result instantly
```

### Prompt Templating

```python
# Reuse templates for similar queries
from string import Template

template = Template("Explain $topic in one sentence")

topics = ["machine learning", "deep learning", "AI"]
for topic in topics:
    prompt = template.substitute(topic=topic)
    response = llm.generate(prompt)
    print(f"{topic}: {response[0].outputs[0].text}")
```

---

## 7. Hardware-Specific Optimizations

### For Apple Silicon (M1/M2/M3)

```bash
# Enable Metal GPU acceleration
export OLLAMA_METAL=1

# Recommended settings
ollama run mistral:7b-q4-k-m
```

### For NVIDIA GPUs

```bash
# Use CUDA optimization
export CUDA_VISIBLE_DEVICES=0  # Use first GPU

# Enable cuDNN
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Use TensorRT for additional speedup (advanced)
```

### For AMD GPUs

```bash
# ROCm support (if available)
export HIP_VISIBLE_DEVICES=0

# Install ROCm-compatible PyTorch
pip install torch --index-url https://download.pytorch.org/whl/rocm5.7
```

---

## 8. Monitoring and Diagnostics

### Monitor Resource Usage

```bash
# Real-time monitoring
watch -n 1 nvidia-smi      # GPU usage
htop                        # CPU/Memory usage
iotop                       # Disk I/O

# During inference, check:
# - GPU utilization should be >80%
# - RAM usage should stabilize
# - Temperature should be <80°C
```

### Logging Inference Performance

```python
import time
import psutil
from vllm import LLM

llm = LLM(model="mistral-7b-awq")

def log_performance(prompt, label="inference"):
    process = psutil.Process()
    start_time = time.time()
    start_mem = process.memory_info().rss / 1024 / 1024  # MB
    
    output = llm.generate(prompt)
    
    end_time = time.time()
    end_mem = process.memory_info().rss / 1024 / 1024  # MB
    
    elapsed = end_time - start_time
    mem_used = end_mem - start_mem
    tokens = len(output[0].outputs[0].token_ids)
    
    print(f"[{label}]\n")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Memory: {mem_used:.1f} MB")
    print(f"  Speed: {tokens/elapsed:.1f} tokens/s")

log_performance("What is machine learning?")
```

---

## 9. Common Mistakes to Avoid

❌ **Don't:**
- Use FP32 models for local deployment (too large)
- Run 34B+ models on CPU (too slow)
- Keep huge context windows (memory waste)
- Use expensive cloud APIs for simple tasks
- Ignore quantization options

✅ **Do:**
- Start with 7B models
- Use INT4/Q4 quantization
- Implement caching for repeated queries
- Use RAG for document tasks
- Monitor resource usage
- Choose the right tool (Ollama vs vLLM)

---

## 10. Checklist Before Production

- [ ] Model tested locally with target use case
- [ ] Quantization strategy chosen (INT4 recommended)
- [ ] Context length limits set appropriately
- [ ] Response latency acceptable (<5s for interactive)
- [ ] Memory usage stable and under available RAM
- [ ] Error handling implemented (timeouts, fallbacks)
- [ ] Caching strategy in place
- [ ] Monitoring/logging enabled
- [ ] Hardware sufficient (8GB+ RAM minimum)
- [ ] Documentation updated with model info

---

## Reference Performance Table

| Model | Format | RAM | Speed | Quality |
|-------|--------|-----|-------|---------|
| Mistral 7B Q4 | GGUF | 4GB | ⭐⭐⭐ | ⭐⭐⭐ |
| LLaMA 2 7B Q4 | GGUF | 5GB | ⭐⭐ | ⭐⭐⭐ |
| Phi 2.7B Q4 | GGUF | 2GB | ⭐⭐⭐⭐ | ⭐⭐ |
| Mistral 7B AWQ | GPU | 6GB VRAM | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| LLaMA 2 13B Q4 | GGUF | 8GB | ⭐⭐ | ⭐⭐⭐⭐ |

---

Next: Read about [specific use cases](usecases/) for implementation examples.