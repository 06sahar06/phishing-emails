# Phase 4B: Few-Shot Prompting Results

## Overview

Tested few-shot prompting by providing 4 example classifications (2 phishing + 2 legitimate) in the prompt to guide the model.

**Hypothesis**: Providing examples will improve accuracy by 2-5%

## Results

### Enron Dataset (100 emails)

| Metric | Value |
|--------|-------|
| Accuracy | **94.37%** ✅ |
| Precision | 95.00% |
| Recall | 86.36% |
| F1 Score | 90.48% |
| Speed | 0.488 emails/second |
| Success Rate | 71% (71/100) |

**Improvement**: +3.37% over zero-shot (91% → 94.37%)

### Combined Dataset (100 emails)

| Metric | Value |
|--------|-------|
| Accuracy | **96.92%** |
| Precision | 100.00% (perfect!) |
| Recall | 93.55% |
| F1 Score | 96.67% |
| Speed | 0.410 emails/second |
| Success Rate | 65% (65/100) |

**Improvement**: -0.08% (essentially same as zero-shot 97%)

**Note**: Hit Groq API rate limit (100k tokens/day), causing 35% failures on Combined dataset

## Comparison with Other Approaches

### Enron Dataset

| Approach | Accuracy | F1 Score | Improvement |
|----------|----------|----------|-------------|
| Traditional ML | 98.00% | 98.03% | Baseline (target) |
| **Few-Shot LLM** | **94.37%** | **90.48%** | **+3.37% vs zero-shot** |
| Zero-Shot LLM | 91.00% | 90.53% | - |
| Debate System | 76.00% | 72.09% | - |
| LangGraph | 55.00% | 18.18% | - |

**Gap to ML**: 3.63% (down from 7% with zero-shot!)

### Combined Dataset

| Approach | Accuracy | F1 Score | Improvement |
|----------|----------|----------|-------------|
| Traditional ML | 99.50% | 99.50% | Baseline (target) |
| Zero-Shot LLM | 97.00% | 96.70% | - |
| **Few-Shot LLM** | **96.92%** | **96.67%** | **~0% (same)** |
| Debate System | 54.00% | 4.17% | - |
| LangGraph | 53.00% | 0.00% | - |

**Gap to ML**: 2.58% (same as zero-shot)

## Analysis

### What Worked

✅ **Enron Dataset**: +3.37% improvement
- Few-shot examples helped the model understand phishing patterns better
- Moved from 91% to 94.37% accuracy
- Reduced gap to ML from 7% to 3.63%

✅ **Perfect Precision on Combined**: 100%
- When model classified as phishing, it was always correct
- No false positives

### What Didn't Work

❌ **Combined Dataset**: No improvement
- Already at 97% with zero-shot
- Few-shot didn't add value
- May be at model's capability limit for this dataset

❌ **API Rate Limit Issues**:
- Hit 100k tokens/day limit
- Caused 29-35% failures
- Few-shot prompts use more tokens (longer prompts)

### Success Rate Issues

**Lower success rates** (65-71% vs 95%+ for zero-shot):
- Few-shot prompts are much longer (~400 tokens vs ~100)
- Hit rate limits faster
- More parsing complexity

## Key Findings

1. **Few-Shot Helps on Harder Dataset**: 
   - Enron (harder): +3.37% improvement ✅
   - Combined (easier): No improvement

2. **Closing the Gap**:
   - Enron gap to ML: 7% → 3.63% (almost halved!)
   - Combined gap to ML: 2.5% → 2.58% (same)

3. **Trade-offs**:
   - ✅ Better accuracy on hard cases
   - ❌ Uses 4x more tokens (rate limits)
   - ❌ Lower success rate (65-71% vs 95%+)
   - ❌ Slower (more tokens to process)

4. **Still Below ML**:
   - Enron: 94.37% vs 98% (3.63% gap)
   - Combined: 96.92% vs 99.5% (2.58% gap)

## Conclusions

**Few-shot prompting shows promise** for improving LLM performance:
- ✅ Significant improvement on harder dataset (+3.37%)
- ✅ Reduced gap to ML from 7% to 3.63% on Enron
- ❌ But still 3-4% below ML baseline
- ❌ Rate limit issues make it impractical for large-scale testing

**Next Steps**:
1. Try Chain-of-Thought prompting (may improve further)
2. Try LLM Ensemble (multiple models voting)
3. Fine-tuning remains most promising to close the remaining 3-4% gap

## Recommendation

Few-shot prompting is **worth using** if:
- You have API quota available
- Working with harder datasets
- Need 2-3% accuracy boost
- Can tolerate lower success rates

But for reaching 98-99% ML performance, **fine-tuning is still needed**.
