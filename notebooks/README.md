# Notebooks

This folder contains all scripts for the phishing detection project.

## Data Preprocessing

- `preprocess_enron.py` - Clean and prepare Enron dataset
- `preprocess_legit.py` - Clean and prepare legitimate emails dataset
- `preprocess_phishing.py` - Clean and prepare phishing emails dataset
- `create_combined_dataset.py` - Merge legit + phishing into combined dataset

## Evaluation Scripts

### Traditional ML
- `traditional_ml_baseline.py` - Test Logistic Regression, Naive Bayes, Random Forest
  - **Results**: 98-99% accuracy (baseline to beat)

### Single LLM Approaches
- `single_llm_groq.py` - Zero-shot classification with Groq API
  - **Results**: 91-97% accuracy

- `few_shot_prompting.py` - Provide 4 examples in prompt
  - **Results**: 94.37% accuracy on Enron (+3.37% improvement)

- `chain_of_thought_prompting.py` - Ask for step-by-step reasoning
  - **Status**: Ready to test (expected 92-98%)

- `llm_ensemble.py` - Multiple models voting
  - **Status**: Ready to test (expected 93-98%)

### Multi-Agent Systems
- `debate_system.py` - Three agents (Attacker, Defender, Judge)
  - **Results**: 76% accuracy (underperformed single LLM)

- `langgraph_system.py` - Graph-based workflow
  - **Results**: 55% accuracy (worst performance)

### Fine-Tuning
- `finetune_colab.ipynb` - Google Colab notebook for fine-tuning
  - **Status**: Ready to run (expected 95-99%)
  - **Platform**: Google Colab with T4 GPU
  - **Time**: 15-30 minutes

## Usage

### Run Preprocessing
```bash
python preprocess_enron.py
python preprocess_legit.py
python preprocess_phishing.py
python create_combined_dataset.py
```

### Run Evaluations
```bash
# Traditional ML baseline
python traditional_ml_baseline.py

# Single LLM (requires Groq API key)
python single_llm_groq.py

# Few-shot prompting (best LLM result so far)
python few_shot_prompting.py

# Other approaches
python chain_of_thought_prompting.py
python llm_ensemble.py
python debate_system.py
python langgraph_system.py
```

### Fine-Tuning
1. Open `finetune_colab.ipynb` in Google Colab
2. Enable GPU (Runtime > Change runtime type > T4 GPU)
3. Upload datasets when prompted
4. Run all cells

## Results Summary

| Approach | Enron | Combined | Status |
|----------|-------|----------|--------|
| Traditional ML | 98.00% | 99.50% | ✅ Baseline |
| Few-Shot LLM | 94.37% | 96.92% | ✅ Best LLM |
| Zero-Shot LLM | 91.00% | 97.00% | ✅ Good |
| Debate System | 76.00% | 54.00% | ⚠️ Poor |
| LangGraph | 55.00% | 53.00% | ❌ Failed |
| Fine-Tuned LLM | TBD | TBD | ⏳ Pending |

## Goal

Push LLMs to match traditional ML performance (98-99% accuracy).

**Current Progress**: 94.37% on Enron (3.63% gap remaining)

**Next Step**: Fine-tuning expected to close the gap to 95-99%
