# Model Selection

## Objective
Select 3 open-source LLM models for comprehensive testing across all evaluation approaches.

## Selection Criteria
1. Open-source and freely available
2. Small enough to run locally (2-3B parameters)
3. Good balance of accuracy and speed
4. Well-documented and widely used
5. Compatible with Ollama and Groq API

## Selected Models

### 1. Qwen/Qwen2.5-3B-Instruct
- Parameters: 3 billion
- Developer: Alibaba Cloud
- Strengths: Fast inference, efficient architecture, good reasoning capabilities, strong multilingual support
- Use Cases: General-purpose, instruction-following
- Availability: Ollama, Hugging Face

### 2. meta-llama/Llama-3.2-3B-Instruct
- Parameters: 3 billion
- Developer: Meta AI
- Strengths: Strong performance, widely adopted, excellent community support, good for fine-tuning
- Use Cases: Instruction-following, chat, classification
- Availability: Ollama, Hugging Face

### 3. google/gemma-2-2b-it
- Parameters: 2 billion
- Developer: Google
- Strengths: Lightweight, resource-efficient, good for constrained environments, fast inference
- Use Cases: Edge deployment, mobile, resource-constrained
- Availability: Ollama, Hugging Face

## Rationale

These models were chosen because they:
- Can run locally on consumer hardware
- Are fast enough for real-time inference
- Have proven performance on NLP tasks
- Are well-documented with active communities
- Support both local (Ollama) and cloud (Groq) inference

## Alternative Models Considered

Not Selected:
- Llama-3.1-70B: Too large for local inference
- GPT-4: Not open-source, expensive API
- Claude: Not open-source, API-only
- Mistral-7B: Larger than needed, slower

## Implementation

### Local Inference (Ollama)
```bash
ollama pull qwen2.5:3b-instruct
ollama pull llama3.2:3b
ollama pull gemma:2b
```

### Cloud Inference (Groq)
- Llama models available via Groq API
- Faster inference (1-2s per email vs 30-50s local)
- Used for single LLM and debate system evaluations

## Expected Performance

Based on model benchmarks:
- Accuracy: 85-95% on classification tasks
- Speed (local): 30-50 seconds per email
- Speed (cloud): 1-2 seconds per email
- Memory: 4-6 GB RAM required
