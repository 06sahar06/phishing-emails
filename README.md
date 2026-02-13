# Phishing Email Detection: Comprehensive ML & LLM Analysis

A systematic comparison of traditional ML models, single LLM agents, multi-agent debate systems, and graph-based approaches for phishing email detection.

## Project Overview

This project explores how far we can push Large Language Models (LLMs) for phishing email detection, with the goal of matching or exceeding traditional ML performance (98-99% accuracy).

**Research Question**: Can LLMs, through various techniques (zero-shot, multi-agent, fine-tuning), achieve performance comparable to traditional ML for phishing detection?

**Datasets**:
- **Enron Dataset**: 3,000 emails (1,500 phishing + 1,500 legitimate)
- **Combined Dataset**: 2,000 emails (1,000 phishing + 1,000 legitimate)



## Quick Results Summary

### Enron Dataset (3,000 emails)

| Approach | Accuracy | Precision | Recall | F1 Score | Speed (emails/s) | Status |
|----------|----------|-----------|--------|----------|------------------|--------|
| **Traditional ML** | **98.00%** | **96.45%** | **99.67%** | **98.03%** | **601,765** | Best |
| **Single LLM** | **91.00%** | **95.56%** | **86.00%** | **90.53%** | **0.625** | Good |
| Debate System | 76.00% | 86.11% | 62.00% | 72.09% | 0.133 | Poor |
| LangGraph | 55.00% | 100.00%* | 10.00%* | 18.18% | 0.165 | Worst |
| Fine-Tuned LLM | N/A | N/A | N/A | N/A | N/A | Blocked |

*LangGraph: 100% precision but only 10% recall = rarely detected phishing (86% failure rate), but when it did, it was correct

### Combined Dataset (2,000 emails)

| Approach | Accuracy | Precision | Recall | F1 Score | Speed (emails/s) | Status |
|----------|----------|-----------|--------|----------|------------------|--------|
| **Traditional ML** | **99.50%** | **99.50%** | **99.50%** | **99.50%** | **125,178** | Best |
| **Single LLM** | **97.00%** | **100.00%** | **93.62%** | **96.70%** | **0.453** | Good |
| Debate System | 54.00% | 100.00%* | 2.13%* | 4.17% | 0.120 | Failed |
| LangGraph | 53.00% | 0.00%** | 0.00%** | 0.00% | 0.145 | Failed |
| Fine-Tuned LLM | N/A | N/A | N/A | N/A | N/A | Blocked |

*Debate: 100% precision but only 2.13% recall = rarely detected phishing, but when it did, it was correct  
**LangGraph: 0% metrics = classified everything as legitimate, never detected any phishing (98% failure rate)

**Progress Toward Goal**: 
- **Baseline (ML)**: 98-99% accuracy - the target to match
- **Zero-Shot LLM**: 91-97% accuracy - promising start, 2-7% gap
- **Multi-Agent Attempts**: 53-76% accuracy - complexity hurt performance
- **Fine-Tuning (Next)**: Expected to close the gap to 93-99%, potentially matching ML

## Evaluation Phases

### Phase 1: Model Selection (Complete)
Selected 3 open-source LLMs for testing:
- **Qwen/Qwen2.5-3B-Instruct**: Fast, efficient, good reasoning
- **meta-llama/Llama-3.2-3B-Instruct**: Strong performance, widely adopted
- **google/gemma-2-2b-it**: Lightweight, resource-efficient

[Detailed Documentation](docs/PHASE1_MODEL_SELECTION.md)

### Phase 2: Data Preprocessing (Complete)
- Cleaned and standardized three raw datasets
- Created balanced samples (50/50 phishing/legitimate)
- Generated two test datasets: Enron (3k) and Combined (2k)
- Separate preprocessing for each dataset due to different formats

[Detailed Documentation](docs/PHASE2_DATA_PREPROCESSING.md)

### Phase 3: Traditional ML Baseline (Complete)
Tested classical ML models with TF-IDF features on both datasets:

**Enron Dataset**:
- **Logistic Regression**: 98.00% accuracy, 98.03% F1
- **Naive Bayes**: 97.83% accuracy, 97.83% F1
- **Random Forest**: 97.17% accuracy, 97.19% F1

**Combined Dataset**:
- **Naive Bayes**: 99.50% accuracy, 99.50% F1
- **Random Forest**: 99.50% accuracy, 99.50% F1
- **Logistic Regression**: 99.25% accuracy, 99.24% F1

**Result**: Excellent baseline (97-99% accuracy), extremely fast (100k+ emails/second)

[Detailed Documentation](docs/PHASE3_TRADITIONAL_ML.md)

### Phase 4: Single LLM Evaluation (Complete)
Tested LLMs via Groq API (zero-shot classification) on both datasets:

**Enron Dataset**:
- **Llama-3.3-70B**: 91% accuracy, 90.53% F1
- **Llama-3.1-8B**: 72% accuracy, 77.78% F1

**Combined Dataset**:
- **Llama-3.3-70B**: 97% accuracy, 96.70% F1
- **Llama-3.1-8B**: 91% accuracy, 91.09% F1

**Result**: Competitive performance (91-97%), but slower and below traditional ML

[Detailed Documentation](docs/PHASE4_SINGLE_LLM.md)

### Phase 5: Multi-Agent Debate System (Complete)
Three-agent system (Attacker, Defender, Judge) tested on both datasets:

**Enron Dataset**:
- **Improved Version**: 76% accuracy, 72.09% F1
- **Original Version**: 69% accuracy, 64.37% F1

**Combined Dataset**:
- **Both Versions Failed**: 54-55% accuracy, 2-8% F1, 98-99% failure rate

**Result**: Underperformed single LLM, high failure rates, 3-5x slower

[Detailed Documentation](docs/PHASE5_DEBATE_SYSTEM.md)

### Phase 6: Graph-Based Systems (LangGraph) (Complete)
Structured workflow using LangGraph tested on both datasets:

**Enron Dataset**:
- 55% accuracy, 18.18% F1, 14% success rate (86% failures)

**Combined Dataset**:
- 53% accuracy, 0.00% F1, 2% success rate (98% failures)

**Result**: Worst performance of all approaches, added complexity without benefits

[Detailed Documentation](docs/PHASE6_LANGGRAPH.md)

### Phase 7: Fine-Tuning (Blocked)
Fine-tuning Qwen2.5-1.5B with LoRA on Kaggle GPU:
- **Status**: Blocked - Training stuck after multiple attempts
- **Issue**: Hangs indefinitely at initialization, no progress after 2+ hours
- **Attempts**: Reduced model size, data, steps - still hanging
- **Conclusion**: Technical issues prevent completion

[Detailed Documentation](docs/PHASE7_FINETUNING.md)

## Evaluation Metrics

All approaches evaluated using:
- **Accuracy**: Overall correctness
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1 Score**: Harmonic mean of precision and recall
- **Speed**: Emails processed per second

## Complete Documentation

- [Complete Project Summary](COMPLETE_PROJECT_SUMMARY.md) - Comprehensive overview of all phases
- [Phase 1: Model Selection](docs/PHASE1_MODEL_SELECTION.md)
- [Phase 2: Data Preprocessing](docs/PHASE2_DATA_PREPROCESSING.md)
- [Phase 3: Traditional ML](docs/PHASE3_TRADITIONAL_ML.md)
- [Phase 4: Single LLM](docs/PHASE4_SINGLE_LLM.md)
- [Phase 5: Debate System](docs/PHASE5_DEBATE_SYSTEM.md)
- [Phase 6: LangGraph](docs/PHASE6_LANGGRAPH.md)
- [Phase 7: Fine-Tuning](docs/PHASE7_FINETUNING.md)

## Project Structure

```
phishing-detection-project/
├── data/                          # Raw datasets
│   ├── enron.csv
│   ├── legit.csv
│   └── phishing.csv
├── docs/                          # Detailed phase documentation
│   ├── PHASE1_MODEL_SELECTION.md
│   ├── PHASE2_DATA_PREPROCESSING.md
│   ├── PHASE3_TRADITIONAL_ML.md
│   ├── PHASE4_SINGLE_LLM.md
│   ├── PHASE5_DEBATE_SYSTEM.md
│   ├── PHASE6_LANGGRAPH.md
│   └── PHASE7_FINETUNING.md
├── notebooks/                     # Evaluation scripts
│   ├── preprocess_*.py           # Data preprocessing
│   ├── traditional_ml_baseline.py # ML baseline
│   ├── single_llm_groq.py        # Zero-shot LLM
│   ├── few_shot_prompting.py     # Few-shot LLM (best)
│   ├── chain_of_thought_prompting.py
│   ├── llm_ensemble.py
│   ├── debate_system.py
│   ├── langgraph_system.py
│   ├── finetune_colab.ipynb      # Fine-tuning notebook
│   └── README.md                 # Notebooks guide
├── results/                       # Evaluation results
│   ├── enron_preprocessed_3k.csv
│   ├── combined_preprocessed_2k.csv
│   ├── PHASE*_RESULTS.md
│   └── PHASE4B_FEW_SHOT_RESULTS.md
├── COMPLETE_PROJECT_SUMMARY.md   # Comprehensive overview
├── ADDITIONAL_TECHNIQUES.md      # Guide to additional methods
├── COLAB_INSTRUCTIONS.md         # Fine-tuning instructions
├── README.md                      # This file
└── requirements.txt               # Dependencies
```

## Key Findings

### LLM Performance Journey

**Goal**: Push LLMs to match traditional ML (98-99% accuracy)

**Attempts**:
1. **Zero-Shot LLM (Phase 4)**: 91-97% accuracy
   - Good starting point, but 2-7% gap from ML
   - No training required
   - Shows LLMs understand phishing concepts

2. **Multi-Agent Debate (Phase 5)**: 54-76% accuracy
   - Hypothesis: Multiple perspectives would improve accuracy
   - Reality: Complexity introduced more errors
   - Lesson: More agents does not equal better performance

3. **Graph-Based Systems (Phase 6)**: 53-55% accuracy
   - Hypothesis: Structured workflows would help coordination
   - Reality: Added overhead without benefits
   - Lesson: Framework complexity doesn't solve fundamental issues

4. **Fine-Tuning (Phase 7)**: Expected 93-99% accuracy
   - Hypothesis: Task-specific training will close the gap
   - Status: Blocked by technical issues, attempting on Colab
   - Expected: Should get within 1-2% of ML performance

### What We Learned

**What Helps LLMs**:
- Larger models (70B > 8B)
- Task-specific fine-tuning (expected)
- Simple, direct prompts
- Lower temperature for consistency

**What Hurts LLMs**:
- Multi-agent complexity
- Sequential API calls
- Complex orchestration
- Graph-based overhead

**The Path Forward**:
- Fine-tuning single LLM is most promising
- Keep it simple - avoid multi-agent complexity
- Focus on model quality over system architecture

## Recommendations

### Research Goal: Pushing LLMs to Match ML

**Current Status**:
- ML Baseline: 98-99% accuracy ← Target
- Best LLM (Zero-Shot): 91-97% accuracy ← 2-7% gap
- Fine-Tuned LLM: TBD (expected 93-99%) ← Should close the gap

**For Achieving ML-Level Performance with LLMs**:

**Recommended Approach**:
1. Fine-tune single LLM on task-specific data (Phase 7)
2. Use larger models (70B parameters)
3. Keep architecture simple (avoid multi-agent)
4. Optimize prompts for consistency

**Avoid**:
- Multi-agent systems (76% accuracy, worse than single LLM)
- Graph-based orchestration (55% accuracy, added complexity)
- Complex workflows (more failure points)

### For Production Use (Current State)

**If you need 98-99% accuracy NOW**:
- Use Traditional ML (Logistic Regression or Naive Bayes)
- Extremely fast (600k emails/second)
- Proven and reliable

**If you want LLM flexibility**:
- Use Single LLM (Llama-3.3-70B) at 91-97% accuracy
- Zero-shot capability for novel attacks
- Wait for fine-tuning results to potentially reach 93-99%

### Future Work to Close the Gap

1. **Complete Phase 7 Fine-Tuning** (Most Promising)
   - Expected to reach 93-99% accuracy
   - Should match or nearly match ML performance
   
2. **Ensemble Approach**
   - Combine fine-tuned LLM + Traditional ML
   - Could achieve best of both worlds

3. **Prompt Optimization**
   - Few-shot learning with examples
   - Chain-of-thought for complex cases

4. **Model Selection**
   - Test even larger models (405B Llama)
   - Try specialized models

## Technologies Used

- **Languages**: Python 3.12
- **ML Libraries**: scikit-learn, pandas, numpy
- **LLM Frameworks**: Transformers, Unsloth, LangChain, LangGraph
- **APIs**: Groq (fast LLM inference), Ollama (local inference)
- **Fine-tuning**: Unsloth with LoRA
- **Platforms**: Local (Windows), Kaggle (GPU training)

## Requirements

```bash
# Traditional ML
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0

# LLM Inference
transformers>=4.35.0
torch>=2.1.0
groq>=0.4.0

# Agent Frameworks
langchain>=0.1.0
langgraph>=0.0.20

# Fine-tuning
unsloth>=2024.1.0
trl>=0.7.0
```

Install all dependencies:
```bash
pip install -r requirements.txt
```

## Getting Started

### 1. Clone Repository
```bash
git clone https://github.com/06sahar06/phishing-emails.git
cd phishing-emails
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Traditional ML (Best Approach)
```bash
python notebooks/traditional_ml_baseline.py
```

### 4. Test LLM Approaches (Requires Groq API Key)
```bash
export GROQ_API_KEY="your-api-key-here"

# Zero-shot
python notebooks/single_llm_groq.py

# Few-shot (best LLM result: 94.37%)
python notebooks/few_shot_prompting.py

# Other approaches
python notebooks/chain_of_thought_prompting.py
python notebooks/llm_ensemble.py
```

### 5. Fine-Tuning (Google Colab)
See `COLAB_INSTRUCTIONS.md` for detailed steps.

## Detailed Results

### Enron Dataset (3,000 emails)

| Approach | Accuracy | Precision | Recall | F1 Score | Speed (emails/s) |
|----------|----------|-----------|--------|----------|------------------|
| Logistic Regression | 98.00% | 96.45% | 99.67% | 98.03% | 601,765 |
| Naive Bayes | 97.83% | 97.99% | 97.67% | 97.83% | 125,178 |
| Random Forest | 97.17% | 96.39% | 98.00% | 97.19% | 13,104 |
| Llama-3.3-70B (Single) | 91.00% | 95.56% | 86.00% | 90.53% | 0.625 |
| Debate System | 76.00% | 86.11% | 62.00% | 72.09% | 0.133 |
| LangGraph | 55.00% | 100.00% | 10.00% | 18.18% | 0.165 |

### Combined Dataset (2,000 emails)

| Approach | Accuracy | Precision | Recall | F1 Score | Speed (emails/s) |
|----------|----------|-----------|--------|----------|------------------|
| Naive Bayes | 99.50% | 99.50% | 99.50% | 99.50% | 125,178 |
| Random Forest | 99.50% | 100.00% | 99.00% | 99.50% | 12,318 |
| Logistic Regression | 99.25% | 100.00% | 98.50% | 99.24% | Very Fast |
| Llama-3.3-70B (Single) | 97.00% | 100.00% | 93.62% | 96.70% | 0.453 |
| Debate System | 54.00% | 100.00% | 2.13% | 4.17% | 0.120 |
| LangGraph | 53.00% | 0.00% | 0.00% | 0.00% | 0.145 |

## Future Work

1. **Complete Phase 7 Fine-Tuning**: Resolve Kaggle training issues or try alternative platforms
   - Try Google Colab with different GPU
   - Use standard Hugging Face Transformers instead of Unsloth
   - Test on local GPU after system reboot
   - Consider cloud platforms (AWS, Azure) with more stable environments
   
2. **Ensemble Methods**: Combine traditional ML + LLM predictions for best of both worlds

3. **Real-time API**: Deploy best model (Traditional ML) as REST API for production use

4. **Explainability**: Add SHAP/LIME for model interpretability and feature importance

5. **Novel Attacks**: Test on zero-day phishing patterns and adversarial examples

6. **Production Deployment**: Containerize and deploy best model with monitoring

7. **Cross-Dataset Validation**: Test models trained on one dataset against others

## Contributing

This is a research project. For questions or suggestions, please open an issue on GitHub.

## License

MIT License - See LICENSE file for details

## Citation

If you use this work, please cite:
```
Phishing Email Detection: Comprehensive ML & LLM Analysis
GitHub: https://github.com/06sahar06/phishing-emails
Year: 2026
```

## Acknowledgments

- Enron email dataset
- Groq for fast LLM inference API
- Unsloth for efficient fine-tuning
- Open-source LLM community

## Contact

GitHub: [@06sahar06](https://github.com/06sahar06)
Repository: [phishing-emails](https://github.com/06sahar06/phishing-emails)
