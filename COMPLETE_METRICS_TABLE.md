# Complete Metrics Table: All Approaches on Both Datasets

## Enron Dataset (3,000 emails: 1,500 phishing + 1,500 legitimate)

| Approach | Accuracy | Precision | Recall | F1 Score | Speed (emails/s) | Success Rate | Status |
|----------|----------|-----------|--------|----------|------------------|--------------|--------|
| **Logistic Regression** | **98.00%** | **96.45%** | **99.67%** | **98.03%** | **601,765** | **100%** | ✅ Best |
| Naive Bayes | 97.83% | 97.99% | 97.67% | 97.83% | 125,178 | 100% | ✅ Excellent |
| Random Forest | 97.17% | 96.39% | 98.00% | 97.19% | 13,104 | 100% | ✅ Excellent |
| Llama-3.3-70B (Single LLM) | 91.00% | 95.56% | 86.00% | 90.53% | 0.625 | ~95% | ✅ Good |
| Llama-3.1-8B (Single LLM) | 72.00% | 64.47% | 98.00% | 77.78% | 0.523 | ~95% | ⚠️ Mediocre |
| Debate System (Improved) | 76.00% | 86.11% | 62.00% | 72.09% | 0.133 | 62% | ⚠️ Poor |
| Debate System (Original) | 69.00% | 75.68% | 56.00% | 64.37% | 0.096 | 58% | 