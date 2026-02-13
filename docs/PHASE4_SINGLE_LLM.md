# Phase 4: Single LLM Evaluation

## Objective

Evaluate individual Large Language Models (LLMs) for phishing detection using zero-shot classification to compare against traditional ML baseline.

## Methodology

### Models Tested
1. **Llama-3.1-8B-Instant** - Fast, efficient 8B parameter model
2. **Llama-3.3-70B-Versatile** - Larger, more capable 70B parameter model
3. **Mixtral-8x7B** - Mixture of experts model (failed due to API issues)

### Infrastructure
- **API**: Groq Cloud API (fast inference)
- **Alternative**: Ollama (local inference, much slower)
- **Reason for Groq**: 30-50x faster than local Ollama

### Prompt Design
```
You are a phishing email detection expert. Analyze this email and determine if it's phishing or legitimate.

Email:
{email_text}

Respond with ONLY one word: "phishing" or "legitimate"
```

### Configuration
- **Temperature**: 0.1 (deterministic, consistent)
- **Max Tokens**: 10 (short response)
- **Sample Size**: 100 emails per dataset (50 phishing + 50 legitimate)

### Evaluation Metrics
- Accuracy: Overall correctness
- Precision: Phishing detection accuracy
- Recall: Percentage of phishing caught
- F1 Score: Balance of precision and recall
- Speed: Emails processed per second

## Results

### Enron Dataset (100 emails)

| Model | Accuracy | Precision | Recall | F1 Score | Speed (emails/s) | Avg Time |
|-------|----------|-----------|--------|----------|------------------|----------|
| Llama-3.1-8B-Instant | 72.00% | 64.47% | 98.00% | 77.78% | 0.523 | 1.9s |
| **Llama-3.3-70B** | **91.00%** | **95.56%** | **86.00%** | **90.53%** | **0.625** | **1.6s** |

**Winner**: Llama-3.3-70B
- 91% accuracy (19% better than 8B model)
- 95.56% precision (very few false positives)
- 86% recall (catches most phishing)
- Faster than smaller model (better optimization)

### Combined Dataset (100 emails)

| Model | Accuracy | Precision | Recall | F1 Score | Speed (emails/s) | Avg Time |
|-------|----------|-----------|--------|----------|------------------|----------|
| Llama-3.1-8B-Instant | 91.00% | 85.19% | 97.87% | 91.09% | 0.372 | 2.7s |
| **Llama-3.3-70B** | **97.00%** | **100.00%** | **93.62%** | **96.70%** | **0.453** | **2.2s** |

**Winner**: Llama-3.3-70B
- 97% accuracy (6% better than 8B model)
- 100% precision (zero false positives!)
- 93.62% recall (excellent detection rate)

## Analysis

### Model Size Impact

**Llama-3.1-8B-Instant**:
- ✓ Faster inference
- ✓ Lower cost
- ✗ Lower accuracy (72-91%)
- ✗ High recall but low precision (aggressive detection)
- ✗ More false positives

**Llama-3.3-70B-Versatile**:
- ✓ Much better accuracy (91-97%)
- ✓ Excellent precision (95-100%)
- ✓ Balanced recall (86-94%)
- ✓ Actually faster on Groq (better optimization)
- ✗ Higher cost per API call

**Conclusion**: Larger model (70B) significantly outperforms smaller model (8B) on this task.

### Dataset Differences

**Enron Dataset** (harder):
- 91% accuracy with Llama-3.3-70B
- More diverse email types
- Internal corporate emails mixed with phishing
- Some ambiguous cases

**Combined Dataset** (easier):
- 97% accuracy with Llama-3.3-70B
- Clearer phishing patterns
- More distinct vocabulary
- Better class separation

**Pattern**: Both LLMs and traditional ML find Combined dataset easier.

### Comparison with Traditional ML (Phase 3)

#### Enron Dataset
| Approach | Accuracy | F1 Score | Speed (emails/s) |
|----------|----------|----------|------------------|
| **Traditional ML (Logistic Regression)** | **98.00%** | **98.03%** | **601,765** |
| **Single LLM (Llama-3.3-70B)** | **91.00%** | **90.53%** | **0.625** |
| Single LLM (Llama-3.1-8B) | 72.00% | 77.78% | 0.523 |

**Gap**: Traditional ML is 7% more accurate and 1 million times faster.

#### Combined Dataset
| Approach | Accuracy | F1 Score | Speed (emails/s) |
|----------|----------|----------|------------------|
| **Traditional ML (Naive Bayes)** | **99.50%** | **99.50%** | **125,178** |
| **Single LLM (Llama-3.3-70B)** | **97.00%** | **96.70%** | **0.453** |
| Single LLM (Llama-3.1-8B) | 91.00% | 91.09% | 0.372 |

**Gap**: Traditional ML is 2.5% more accurate and 275,000 times faster.

## Key Findings

### What LLMs Do Well
1. **Zero-Shot Performance**: No training required, works immediately
2. **Semantic Understanding**: Can understand context and meaning
3. **Reasonable Accuracy**: 91-97% is competitive for many use cases
4. **Explainability Potential**: Could explain reasoning (not tested here)
5. **Adaptability**: Can handle novel phishing patterns

### What LLMs Struggle With
1. **Speed**: 0.5 emails/second vs 100k+ for traditional ML
2. **Accuracy Gap**: 2-7% lower than traditional ML
3. **Cost**: API calls cost money (traditional ML is free after training)
4. **Consistency**: Small variations in prompt can change results
5. **Resource Usage**: Requires powerful GPUs or API access

### Error Analysis

**Llama-3.3-70B on Enron** (9 errors out of 100):
- False Positives: 2 (legitimate marked as phishing)
  - Emails with urgent language
  - Emails requesting action
- False Negatives: 7 (phishing marked as legitimate)
  - Sophisticated phishing mimicking legitimate emails
  - Well-written phishing without obvious red flags

**Llama-3.3-70B on Combined** (3 errors out of 100):
- False Positives: 0 (perfect precision!)
- False Negatives: 3 (phishing marked as legitimate)
  - Very subtle phishing attempts
  - Professional-looking emails

## Advantages of LLMs

### 1. Zero-Shot Capability
- No training data required
- Works on day one
- Can adapt to new patterns

### 2. Semantic Understanding
- Understands context, not just keywords
- Can detect sophisticated social engineering
- Recognizes intent and tone

### 3. Flexibility
- Easy to update prompts
- Can add new criteria
- Can explain decisions

### 4. Generalization
- May handle novel attacks better
- Not limited to training data patterns
- Can reason about new scenarios

## Disadvantages of LLMs

### 1. Speed
- 0.5 emails/second (too slow for high-volume)
- 1-2 seconds per email
- Cannot handle real-time filtering at scale

### 2. Cost
- API calls cost money
- Groq: ~$0.10 per 1M tokens
- Traditional ML: free after training

### 3. Accuracy
- 91-97% vs 98-99% for traditional ML
- 2-7% gap may matter for production

### 4. Infrastructure
- Requires API access or powerful GPU
- Network dependency
- Potential downtime

## Use Cases

### When to Use LLMs
- ✓ Low-volume email filtering (<1000 emails/day)
- ✓ Need zero-shot capability (no training data)
- ✓ Want explainable decisions
- ✓ Handling novel/evolving phishing patterns
- ✓ Research and experimentation

### When to Use Traditional ML
- ✓ High-volume email filtering (>10k emails/day)
- ✓ Have training data available
- ✓ Need maximum speed
- ✓ Want lowest cost
- ✓ Production deployment

## Groq API Performance

**Why Groq is Fast**:
- Custom LPU (Language Processing Unit) hardware
- Optimized for inference
- 30-50x faster than local Ollama
- Competitive pricing

**Groq vs Ollama** (Llama-3.1-8B):
- Groq: 1.9 seconds per email
- Ollama: 30-50 seconds per email
- **Speedup**: 15-25x faster

## Prompt Engineering Insights

**What Worked**:
- Simple, direct instructions
- Requesting single-word response
- Clear role definition ("phishing detection expert")
- Low temperature (0.1) for consistency

**What Didn't Work** (tested but not shown):
- Complex multi-step reasoning (slower, no better)
- Chain-of-thought prompting (inconsistent parsing)
- JSON output (parsing errors)

## Next Steps

Phase 5 will test **multi-agent debate systems** to see if multiple LLMs collaborating can:
- Improve accuracy beyond 91-97%
- Reduce false positives/negatives through consensus
- Provide better reasoning through debate
- Approach or exceed traditional ML performance (98-99%)

**Hypothesis**: Multiple perspectives (attacker vs defender) may catch errors that single LLM misses.

## Code Implementation

```python
from groq import Groq

client = Groq(api_key="your-api-key")

def classify_email(email_text):
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{
            "role": "user",
            "content": f"You are a phishing email detection expert. "
                      f"Analyze this email and determine if it's phishing or legitimate.\n\n"
                      f"Email:\n{email_text}\n\n"
                      f"Respond with ONLY one word: 'phishing' or 'legitimate'"
        }],
        temperature=0.1,
        max_tokens=10
    )
    return response.choices[0].message.content.strip().lower()
```

## Files Generated

- `phase4_single_llm_groq.py` - Evaluation script
- `phase4_llm_groq_results.json` - Detailed metrics
- `PHASE4_RESULTS.md` - Summary report

## Conclusion

Single LLMs achieve **competitive but not superior** performance compared to traditional ML:
- **Accuracy**: 91-97% (good, but 2-7% below traditional ML)
- **Speed**: 0.5 emails/second (1 million times slower)
- **Cost**: Ongoing API costs vs one-time training

**Best Use Case**: Zero-shot scenarios where training data is unavailable or when handling novel phishing patterns that traditional ML might miss.

**For Production**: Traditional ML remains superior for high-volume, cost-effective phishing detection.
