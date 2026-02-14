# Data Preprocessing

## Objective
Clean and prepare three phishing email datasets for all subsequent testing.

## Input Datasets

### 1. Enron Dataset (enron.csv)
- Source: Enron email corpus
- Size: 33,716 emails
- Format: Message ID, Subject, Message, Spam/Ham, Date
- Labels: "spam" (phishing) / "ham" (legitimate)
- Distribution: 17,171 spam, 16,545 ham

### 2. Legit Dataset (legit.csv)
- Source: Legitimate emails from various sources
- Size: 1,000 emails
- Format: sender, receiver, date, subject, body, urls, label
- Labels: All label=0 (legitimate)
- Content: Technical discussions, mailing lists

### 3. Phishing Dataset (phishing.csv)
- Source: Known phishing emails
- Size: 1,000 emails
- Format: sender, receiver, date, subject, body, urls, label
- Labels: All label=1 (phishing)
- Content: Delivery scams, account verification, crypto scams

## Preprocessing Steps

### 1. Load and Analyze
```python
df = pd.read_csv("dataset.csv")
print(df.columns)
print(df['label'].value_counts())
```

### 2. Clean Text
- Remove extra whitespace
- Handle missing values
- Normalize line breaks
- Truncate very long emails

### 3. Standardize Labels
- Convert to binary: 0 (legitimate) / 1 (phishing)
- Ensure consistency across datasets

### 4. Create Balanced Samples
- Sample equal numbers of phishing and legitimate
- Use random_state=42 for reproducibility

### 5. Combine Fields
- Merge subject + body into single text field
- Keep original fields for analysis

## Output Datasets

### 1. Enron Preprocessed (enron_preprocessed_3k.csv)
- Size: 3,000 emails
- Distribution: 1,500 phishing + 1,500 legitimate (50/50)
- Columns: text, subject, message, date, label
- Avg Length: 1,448 characters
- Range: 11 - 90,646 characters

### 2. Legit Preprocessed (legit_preprocessed_1.5k.csv)
- Size: 1,000 emails (all available)
- Distribution: 100% legitimate
- Columns: text, subject, message, sender, date, label
- Avg Length: 3,929 characters
- Range: 82 - 70,035 characters

### 3. Phishing Preprocessed (phishing_preprocessed_1.5k.csv)
- Size: 1,000 emails (all available)
- Distribution: 100% phishing
- Columns: text, subject, message, sender, date, label
- Avg Length: 1,141 characters
- Range: 95 - 23,715 characters

### 4. Combined Preprocessed (combined_preprocessed_2k.csv)
- Size: 2,000 emails
- Distribution: 1,000 phishing + 1,000 legitimate (50/50)
- Source: Legit + Phishing datasets merged
- Columns: text, subject, message, sender, date, label
- Avg Length: 2,535 characters

## Data Quality

### Enron Dataset
- Balanced distribution (50/50)
- No missing values in critical fields
- Reasonable text lengths
- Diverse email types

### Combined Dataset
- Balanced distribution (50/50)
- More technical content than Enron
- Longer average email length
- Mix of modern phishing tactics

## Scripts

1. preprocess_enron.py - Process Enron dataset
2. preprocess_legit.py - Process legitimate emails
3. preprocess_phishing.py - Process phishing emails
4. create_combined_dataset.py - Merge legit + phishing

## Usage

```bash
python notebooks/preprocess_enron.py
python notebooks/preprocess_legit.py
python notebooks/preprocess_phishing.py
python notebooks/create_combined_dataset.py
```

## Key Decisions

### Why 3k for Enron?
- Large enough for statistical significance
- Balanced 50/50 split possible
- Manageable for LLM testing

### Why Combined Dataset?
- Legit + Phishing datasets have different format than Enron
- Provides second test set with different characteristics
- More modern phishing examples

### Why Balanced Samples?
- Prevents model bias toward majority class
- Fair comparison of precision and recall
- Realistic for production scenarios

## Challenges

1. Different Formats: Each dataset had unique structure
   - Solution: Separate preprocessing script for each

2. Limited Samples: Legit and Phishing only had 1k each
   - Solution: Use all available, create 2k combined dataset

3. Text Length Variation: Some emails very long (90k+ chars)
   - Solution: Keep full text, truncate during model inference

## Statistics Summary

| Dataset | Total | Phishing | Legitimate | Avg Length | Balance |
|---------|-------|----------|------------|------------|---------|
| Enron | 3,000 | 1,500 | 1,500 | 1,448 | 50/50 |
| Combined | 2,000 | 1,000 | 1,000 | 2,535 | 50/50 |
