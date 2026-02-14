# Multi-Agent Debate System

## Objective

Test if multiple LLMs debating can improve accuracy beyond single LLM performance through collaborative reasoning.

## Hypothesis

Multiple agents with different perspectives (attacker vs defender) can catch errors that single LLM misses and improve accuracy toward traditional ML levels (98-99%).

## Architecture

### Three-Agent System

1. Attacker Agent (Llama-3.1-8B)
   - Identifies phishing threats
   - Assumes email is malicious
   - Temperature: 0.7 (creative)

2. Defender Agent (Llama-3.1-8B)
   - Argues for legitimacy
   - Assumes email is legitimate
   - Temperature: 0.3 (conservative)

3. Judge Agent (Llama-3.3-70B)
   - Makes final decision
   - Neutral arbiter
   - Temperature: 0.1 (deterministic)

### Workflow
```
Email → Attacker Agent → Phishing Analysis
     → Defender Agent → Legitimacy Analysis
     → Judge Agent → Final Decision
```

## Results

### Enron Dataset (100 emails)

| Version | Accuracy | Precision | Recall | F1 Score | Success Rate | Speed (emails/s) |
|---------|----------|-----------|--------|----------|--------------|------------------|
| Original | 69.00% | 75.68% | 56.00% | 64.37% | 58% | 0.096 |
| Improved | 76.00% | 86.11% | 62.00% | 72.09% | 62% | 0.133 |

Improvement: +7% accuracy, +8% F1, +38% faster

### Combined Dataset (100 emails)

| Version | Accuracy | Precision | Recall | F1 Score | Success Rate | Speed (emails/s) |
|---------|----------|-----------|--------|----------|--------------|------------------|
| Original | 55.00% | 85.00% | 4.26% | 8.16% | 2% | 0.091 |
| Improved | 54.00% | 85.00% | 2.13% | 4.17% | 1% | 0.120 |

Issue: Both versions failed catastrophically (98-99% failure rate)

## Analysis

### What Worked (Enron)
- Improved prompts increased accuracy by 7%
- Better parsing reduced failures
- 38% speed improvement
- Higher precision (86%)

### What Failed (Combined)
- 98-99% failure rate
- Likely causes: Longer emails, API timeouts, encoding issues
- System defaulted to "legitimate" for almost all emails

### Comparison with Other Approaches

#### Enron Dataset
| Approach | Accuracy | F1 Score | Speed (emails/s) |
|----------|----------|----------|------------------|
| Traditional ML | 98.00% | 98.03% | 601,765 |
| Single LLM | 91.00% | 90.53% | 0.625 |
| Debate System | 76.00% | 72.09% | 0.133 |

Gap: Debate is 15% worse than single LLM

#### Combined Dataset
| Approach | Accuracy | F1 Score | Speed (emails/s) |
|----------|----------|----------|------------------|
| Traditional ML | 99.50% | 99.50% | 125,178 |
| Single LLM | 97.00% | 96.70% | 0.453 |
| Debate System | 54.00% | 4.17% | 0.120 |

Gap: Debate failed completely (43% worse than single LLM)

## Key Findings

### 1. Complexity Hurts Performance
- 3 sequential API calls increase failure risk
- Each call is a potential failure point
- Parsing errors compound across agents
- More opportunities for inconsistency

### 2. Speed Penalty
- Single LLM: 1.6-2.2 seconds per email
- Debate System: 7.5-11 seconds per email
- Slowdown: 3-5x slower

### 3. Dataset Sensitivity
- Enron (shorter emails): 62% success rate, 76% accuracy
- Combined (longer emails): 1-2% success rate, complete failure
- Conclusion: Debate systems are fragile

### 4. No Accuracy Benefit
- Expected: Multiple perspectives would catch errors
- Reality: More complexity introduced more errors
- Agents don't truly "debate" (no back-and-forth)
- Judge often ignores one agent's input

## Why Multi-Agent Debate Fails

### Theoretical Benefits (Expected)
- Error correction through consensus
- Multiple perspectives catch edge cases
- Balanced analysis reduces bias

### Actual Problems (Reality)
- More failure points (3x API calls)
- Slower processing (3-5x slower)
- Parsing complexity loses information
- No true collaboration (sequential, not interactive)
- Judge bias (tends to favor one agent)

## Recommendations

### For Phishing Detection

Don't Use Debate Systems:
- 15-43% worse accuracy than single LLM
- 3-5x slower
- High failure rate (38-99%)
- Added complexity without benefits

Use Traditional ML Instead:
- 98-99% accuracy
- 1 million times faster
- Reliable and proven

Use Single LLM If Needed:
- 91-97% accuracy
- 3-5x faster than debate
- Simpler and more reliable

## Lessons Learned

1. Simpler is Better: More agents does not equal better performance
2. Speed Matters: 3-5x slowdown is significant
3. Failure Compounds: Multiple API calls multiply failure risk
4. Dataset Sensitivity: Works on simple inputs, fails on complex
5. No Magic Bullet: Debate doesn't automatically improve accuracy

## Code Implementation

```python
def debate_classify(email_text):
    # Attacker analysis
    attacker_response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{
            "role": "user",
            "content": f"You are a security analyst. List 3-5 phishing indicators:\n\n{email_text}"
        }],
        temperature=0.7
    )
    
    # Defender analysis
    defender_response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{
            "role": "user",
            "content": f"List 3-5 reasons this email might be legitimate:\n\n{email_text}"
        }],
        temperature=0.3
    )
    
    # Judge decision
    judge_response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{
            "role": "user",
            "content": f"Email: {email_text}\n\n"
                      f"Phishing indicators: {attacker_response.choices[0].message.content}\n\n"
                      f"Legitimacy indicators: {defender_response.choices[0].message.content}\n\n"
                      f"Final decision (one word): phishing or legitimate?"
        }],
        temperature=0.1
    )
    
    return judge_response.choices[0].message.content.strip().lower()
```

## Conclusion

Multi-agent debate systems do not improve phishing detection and should be avoided:
- Lower accuracy: 76% vs 91% for single LLM
- Much slower: 7.5s vs 1.6s per email
- Higher failure rate: 38-99% vs <5% for single LLM
- Added complexity without benefits

Recommendation: Stick with traditional ML (98-99%) or single LLM (91-97%) for phishing detection.
