# Phase 7: Fine-Tuning LLM for Phishing Detection

## Objective

Fine-tune a smaller LLM on phishing detection data to improve accuracy beyond zero-shot performance while maintaining the simplicity of a single-model approach.

## Hypothesis

Fine-tuning a single LLM on task-specific data can:
- Improve accuracy from 91-97% toward traditional ML levels (98-99%)
- Maintain simplicity (single model, no multi-agent complexity)
- Provide better performance than complex multi-agent systems
- Learn phishing-specific patterns and indicators

## Methodology

### Model Selection

**Chosen**: Qwen/Qwen2.5-1.5B-Instruct

**Rationale**:
- Smaller model (1.5B) trains faster than 3B
- Still capable for classification tasks
- Efficient with LoRA fine-tuning
- Good base performance

**Alternatives Considered**:
- Qwen2.5-3B: Larger, slower training
- Llama-3.2-3B: Good but slower
- Gemma-2-2B: Less efficient for fine-tuning

### Fine-Tuning Approach

**Method**: LoRA (Low-Rank Adaptation)
- Efficient: Only trains small adapter layers
- Fast: 10-15 minutes vs hours for full fine-tuning
- Memory-efficient: Fits on single GPU
- Preserves base model knowledge

**Framework**: Unsloth
- 2x faster than standard Hugging Face
- Optimized for LoRA training
- Better memory efficiency
- Supports latest models

### Training Configuration

```python
# LoRA Configuration
lora_config = {
    "r": 16,                    # LoRA rank
    "lora_alpha": 16,           # LoRA alpha
    "lora_dropout": 0.05,       # Dropout for regularization
    "target_modules": [         # Layers to adapt
        "q_proj", "k_proj", 
        "v_proj", "o_proj",
        "gate_proj", "up_proj", 
        "down_proj"
    ]
}

# Training Configuration
training_config = {
    "num_train_epochs": 1,
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 4,
    "learning_rate": 2e-4,
    "max_steps": 100,           # Reduced for faster training
    "warmup_steps": 5,
    "logging_steps": 10,
    "save_steps": 50,
    "fp16": True,               # Mixed precision
    "optim": "adamw_8bit"       # Memory-efficient optimizer
}
```

### Training Data

**Source**: Enron dataset (preprocessed)
- Total: 3,000 emails (1,500 phishing + 1,500 legitimate)
- Training: 2,400 emails (80%)
- Validation: 600 emails (20%)

**Reduced for Debugging**:
- Training: 400-500 emails (to avoid training hangs)
- Steps: 100 (instead of full epoch)

**Format**:
```json
{
    "instruction": "Classify this email as phishing or legitimate.",
    "input": "[email text]",
    "output": "phishing" or "legitimate"
}
```

### Test Data

1. **Enron Test Set**: 600 emails (from 20% split)
2. **Combined Test Set**: 2,000 emails (separate dataset)

## Infrastructure

### Platform: Kaggle

**Why Kaggle**:
- Free GPU access (Tesla T4)
- 30 hours/week GPU quota
- Pre-installed ML libraries
- Easy dataset upload

**GPU Specifications**:
- Model: Tesla T4
- Memory: 15.64 GB
- CUDA: 12.8
- Compute Capability: 7.5

### Local Issues (Why Not Local)

**GPU Error**:
```
Unable to determine the device handle for GPU 0000:01:00.0: GPU is lost.
Reboot the system to recover this GPU
```

**Problems**:
- GPU driver issues
- System reboot required
- Unstable for long training runs
- Limited VRAM

## Current Status

### Training Progress

**Initialization**: ✅ Complete
- Model loaded successfully
- LoRA adapters added
- Dataset prepared (2,400 examples)
- Training started

**Training**: ⏳ Stuck
- Hangs after initialization
- No progress for 2+ hours
- No error messages
- Process appears frozen

### Attempts Made

#### Attempt 1: Full Training
- Model: Qwen2.5-3B
- Data: 2,400 samples
- Steps: 150 (full epoch)
- Result: ❌ Hung indefinitely

#### Attempt 2: Simplified
- Model: Qwen2.5-1.5B (smaller)
- Data: 500 samples (reduced)
- Steps: 100 (reduced)
- Config: `packing=False` (avoid bugs)
- Result: ⏳ Still hanging

### Debugging Steps Taken

1. ✅ Reduced model size (3B → 1.5B)
2. ✅ Reduced training data (2,400 → 500)
3. ✅ Reduced steps (150 → 100)
4. ✅ Disabled packing (`packing=False`)
5. ✅ Simplified configuration
6. ⏳ Still investigating...

## Known Issues

### 1. Training Hangs

**Symptom**: Training starts but never progresses
**Possible Causes**:
- Unsloth version compatibility
- Kaggle environment issues
- Dataset formatting problems
- Memory issues (silent failure)

### 2. Unsloth Warnings

```
Unsloth: Dropout = 0 is supported for fast patching. 
You are using dropout = 0.05. 
Unsloth will patch all other layers, except LoRA matrices, 
causing a performance hit.
```

**Impact**: Slower training, but should still work

### 3. CUDA Warnings

```
Unable to register cuFFT factory
Unable to register cuDNN factory
Unable to register cuBLAS factory
```

**Impact**: Warnings only, shouldn't prevent training

## Expected Results (When Complete)

### Performance Targets

**Conservative Estimate**:
- Enron: 93-95% accuracy (up from 91%)
- Combined: 98-99% accuracy (up from 97%)

**Optimistic Estimate**:
- Enron: 95-97% accuracy
- Combined: 99%+ accuracy (matching traditional ML)

### Comparison with Other Approaches

| Approach | Enron Accuracy | Combined Accuracy | Speed | Status |
|----------|----------------|-------------------|-------|--------|
| Traditional ML | 98.00% | 99.50% | 601,765 emails/s | ✅ Best |
| Single LLM (Zero-Shot) | 91.00% | 97.00% | 0.625 emails/s | ✅ Good |
| **Fine-Tuned LLM** | **93-97%** | **98-99%** | **~0.5 emails/s** | ⏳ Pending |
| Debate System | 76.00% | 54.00% | 0.133 emails/s | ❌ Poor |
| LangGraph | 55.00% | 53.00% | 0.165 emails/s | ❌ Worst |

**Expected Ranking**: Traditional ML ≈ Fine-Tuned LLM > Zero-Shot LLM > Debate > LangGraph

## Advantages of Fine-Tuning

### 1. Task-Specific Learning
- Learns phishing patterns from data
- Adapts to specific email formats
- Improves on edge cases

### 2. Maintains Simplicity
- Still single model (no multi-agent complexity)
- Same inference speed as zero-shot
- Easy to deploy

### 3. Better Than Multi-Agent
- Higher accuracy expected (93-97% vs 76%)
- Faster (0.5 emails/s vs 0.13 emails/s)
- More reliable (no multi-agent failures)

### 4. Potential to Match Traditional ML
- Could reach 98-99% accuracy
- With better interpretability
- And zero-shot generalization

## Disadvantages of Fine-Tuning

### 1. Requires Training
- 10-30 minutes training time
- Needs GPU access
- Requires training data

### 2. Still Slower Than Traditional ML
- 0.5 emails/s vs 600k emails/s
- 1 million times slower
- Not suitable for high-volume

### 3. Infrastructure Requirements
- GPU needed for training
- Cloud platform or local GPU
- More complex deployment

### 4. Training Instability
- Current issues with hanging
- Requires debugging
- Platform-dependent

## Next Steps

### Option 1: Debug Kaggle Training (Recommended)
1. Try different Unsloth version
2. Test with even smaller dataset (100 samples)
3. Try different model (Llama instead of Qwen)
4. Check Kaggle forums for similar issues
5. Contact Unsloth support

### Option 2: Alternative Platform
1. Google Colab (free GPU)
2. AWS SageMaker (paid)
3. Azure ML (paid)
4. Local GPU (after reboot)

### Option 3: Alternative Fine-Tuning Method
1. Use standard Hugging Face Transformers
2. Slower but more stable
3. Better documentation
4. More control over process

### Option 4: Skip Fine-Tuning
1. Document current results (Phases 1-6)
2. Create comprehensive final report
3. Note fine-tuning as future work
4. Focus on conclusions from existing data

## Lessons Learned (So Far)

### 1. Platform Matters
- Local GPU issues forced cloud migration
- Kaggle has its own quirks
- Platform stability is critical

### 2. Debugging is Time-Consuming
- Training hangs are hard to debug
- No clear error messages
- Trial and error required

### 3. Simplification Helps
- Smaller model trains faster
- Less data reduces issues
- Simpler config easier to debug

### 4. Fine-Tuning is Promising
- If we can get it working
- Should outperform multi-agent
- Simpler than complex systems

## Recommendations

### For Completing Phase 7

**Priority 1**: Debug Kaggle training
- Most likely to succeed
- Free GPU access
- Already set up

**Priority 2**: Try Google Colab
- Similar to Kaggle
- Different environment might work
- Also free

**Priority 3**: Use alternative method
- Standard Transformers
- More stable
- Slower but reliable

### For Production Use (Current State)

**Best Option**: Traditional ML
- 98-99% accuracy
- Extremely fast
- Proven and reliable

**Alternative**: Zero-Shot LLM
- 91-97% accuracy
- No training needed
- Good for novel patterns

**Avoid**: Multi-agent systems
- Lower accuracy (53-76%)
- High failure rates
- Added complexity

## Code Implementation (When Working)

```python
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments

# Load model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-1.5B-Instruct",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True
)

# Add LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    use_gradient_checkpointing="unsloth"
)

# Training
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    dataset_text_field="text",
    max_seq_length=2048,
    args=TrainingArguments(
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=100,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        optim="adamw_8bit",
        output_dir="outputs"
    )
)

trainer.train()
```

## Files Generated

- `phase7_finetune_llm.py` - Training script (original)
- `phase7_finetune_simplified.py` - Simplified version
- `PHASE7_STATUS.md` - Current status
- `PHASE7_SETUP.md` - Kaggle setup instructions

## Current Conclusion

Fine-tuning shows promise but faces technical challenges:
- ✅ Approach is sound (LoRA + Unsloth)
- ✅ Infrastructure available (Kaggle GPU)
- ✅ Data prepared correctly
- ⏳ Training hangs (debugging in progress)

**If successful**, fine-tuning could:
- Match traditional ML accuracy (98-99%)
- Maintain single-model simplicity
- Outperform all multi-agent approaches
- Provide best LLM-based solution

**Current recommendation**: Continue debugging while documenting existing results from Phases 1-6.

## Status: ⚠️ BLOCKED - Training Stuck

**Current Situation**: Training remains stuck on Kaggle after multiple attempts and simplifications. The process hangs indefinitely after initialization with no progress or error messages.

**Attempts Made**:
1. Reduced model size (3B → 1.5B)
2. Reduced training data (2,400 → 500 → 100 samples)
3. Reduced steps (150 → 100 → 50)
4. Disabled packing (`packing=False`)
5. Simplified configuration
6. Multiple Kaggle sessions

**All attempts**: Still hanging at training initialization

**Recommendation**: Given the persistent issues and time constraints, it's recommended to:
1. Document Phase 7 as "attempted but blocked by technical issues"
2. Create final report with Phases 1-6 results
3. Note fine-tuning as future work when stable environment is available
4. Focus on conclusions from existing comprehensive data (6 phases completed)
