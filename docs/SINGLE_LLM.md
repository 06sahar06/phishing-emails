# Single LLM Evaluation

## Objective

Evaluate individual Large Language Models for phishing detection using zero-shot classification to compare against traditional ML baseline.

## Methodology

### Models Tested
1. Llama-3.1-8B-Instant - Fast, efficient 8B parameter model
2. Llama-3.3-70B-Versatile - Larger, more capable 70B parameter model

### Infrastructure
- API: Groq Cloud API (fast inference)
- Alternative: Ollama (local inference, 30-50x slower)

### Prompt Design
```
You are a phishing email detection expert. Analyze this email and determine if it's phishing or legitimate.

Email:
{email_text}

Respond with ONLY one word: "phishing" or "legitimate"
```

### Configuration
- Temperature: 0.1 (deterministic, consistent)
- Max Tokens: 10 (short response)
- Sample Size: 100 emails per dataset (50 phishing + 50 legitimate)

## Results

### Enron Dataset (100 emails)

| Model | Accuracy | Precision | Recall | F1 Score | Speed (emails/s) |
|-------|----------|-----------|--------|----------|------------------|
| Llama-3.1-8B | 72.00% | 64.47% | 98.00% | 77.78% | 0.523 |
| Llama-3.3-70B | 91.00% | 95.56% | 86.00% | 90.53% | 0.625 |

Best: Llama-3.3-70B (91% accuracy, 19% better than 8B model)

### Combined Dataset (100 emails)

| Model | Accuracy | Precision | Recall | F1 Score | Speed (emails/s) |
|-------|----------|-----------|--------|----------|------------------|
| Llama-3.1-8B | 91.00% | 85.19% | 97.87% | 91.09% | 0.372 |
| Llama-3.3-70B | 97.00% | 96.00% | 93.62% | 96.70% | 0.453 |

Best: Llama-3.3-70B (97% accuracy, 96% precision)

## Analysis

### Model Size Impact

Llama-3.1-8B:
- Faster inference, lower cost
- Lower accuracy (72-91%)
- High recall but low precision
- More false positives

Llama-3.3-70B:
- Much better accuracy (91-97%)
- Excellent precision (95-96%)
- Balanced recall (86-94%)
- Actually faster on Groq (better optimization)

Conclusion: Larger model (70B) significantly outperforms smaller model (8B).

### Comparison with Traditional ML

#### Enron Dataset
| Approach | Accuracy | F1 Score | Speed (emails/s) |
|----------|----------|----------|------------------|
| Traditional ML | 98.00% | 98.03% | 601,765 |
| Single LLM | 91.00% | 90.53% | 0.625 |

Gap: Traditional ML is 7% more accurate and 1 million times faster.

#### Combined Dataset
| Approach | Accuracy | F1 Score | Speed (emails/s) |
|----------|----------|----------|------------------|
| Traditional ML | 99.50% | 99.50% | 125,178 |
| Single LLM | 97.00% | 96.70% | 0.453 |

Gap: Traditional ML is 2.5% more accurate and 275,000 times faster.

## Key Findings

### LLM Advantages
1. Zero-Shot Performance: No training required
2. Semantic Understanding: Can understand context and meaning
3. Reasonable Accuracy: 91-97% is competitive
4. Adaptability: Can handle novel phishing patterns

### LLM Limitations
1. Speed: 0.5 emails/second vs 100k+ for traditional ML
2. Accuracy Gap: 2-7% lower than traditional ML
3. Cost: API calls cost money
4. Infrastructure: Requires powerful GPUs or API access

## Use Cases

### When to Use LLMs
- Low-volume email filtering (<1000 emails/day)
- Need zero-shot capability (no training data)
- Want explainable decisions
- Handling novel/evolving phishing patterns

### When to Use Traditional ML
- High-volume email filtering (>10k emails/day)
- Have training data available
- Need maximum speed
- Want lowest cost
- Production deployment

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

## Conclusion

Single LLMs achieve competitive but not superior performance compared to traditional ML:
- Accuracy: 91-97% (good, but 2-7% below traditional ML)
- Speed: 0.5 emails/second (1 million times slower)
- Cost: Ongoing API costs vs one-time training

Best Use Case: Zero-shot scenarios where training data is unavailable or when handling novel phishing patterns.

For Production: Traditional ML remains superior for high-volume, cost-effective phishing detection.
