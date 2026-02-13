# Google Colab Fine-Tuning Instructions

## Quick Start

1. **Open the notebook in Google Colab**
   - Go to: https://colab.research.google.com/
   - Click "File" > "Upload notebook"
   - Upload: `notebooks/phase7_colab_finetuning.ipynb`

2. **Enable GPU**
   - Click "Runtime" > "Change runtime type"
   - Select "T4 GPU" from the dropdown
   - Click "Save"

3. **Upload your datasets**
   - When prompted, upload these files:
     - `results/enron_preprocessed_3k.csv`
     - `results/combined_preprocessed_2k.csv`

4. **Run all cells**
   - Click "Runtime" > "Run all"
   - Wait 15-30 minutes for training to complete

5. **Download results**
   - The notebook will automatically download `phase7_finetuned_results.json`
   - Save this file to your project

## What the Notebook Does

1. ✅ Installs all required libraries
2. ✅ Checks GPU availability
3. ✅ Loads and prepares your datasets
4. ✅ Loads Qwen2.5-1.5B model with 4-bit quantization
5. ✅ Adds LoRA adapters for efficient fine-tuning
6. ✅ Trains for 200 steps (~15-30 minutes)
7. ✅ Evaluates on both Enron and Combined datasets
8. ✅ Compares with previous approaches
9. ✅ Saves results as JSON

## Expected Results

**Training Time**: 15-30 minutes with T4 GPU

**Expected Performance**:
- Enron: 93-97% accuracy (up from 91% zero-shot)
- Combined: 98-99% accuracy (up from 97% zero-shot)

**Comparison**:
- Traditional ML: 98-99% (still best)
- Fine-Tuned LLM: 93-99% (NEW - should be close to traditional ML)
- Zero-Shot LLM: 91-97%
- Debate System: 54-76%
- LangGraph: 53-55%

## Advantages of This Approach

✅ **More Stable**: Uses standard Hugging Face Transformers (not Unsloth)
✅ **Well-Tested**: Proven approach used by thousands
✅ **Better Documentation**: Extensive community support
✅ **Memory Efficient**: 4-bit quantization + LoRA
✅ **Fast**: 15-30 minutes on free T4 GPU

## Troubleshooting

### "No GPU available"
- Make sure you selected GPU in Runtime settings
- Try disconnecting and reconnecting

### "Out of memory"
- Reduce `per_device_train_batch_size` from 4 to 2
- Reduce `max_seq_length` from 512 to 256

### "Training too slow"
- Reduce `max_steps` from 200 to 100
- This will be faster but slightly less accurate

### "Upload failed"
- Try mounting Google Drive instead (see Option 2 in notebook)
- Upload files to Drive first, then mount

## After Training

1. **Download the results JSON**
2. **Copy it to**: `phishing-detection-project/results/`
3. **Update documentation** with Phase 7 results
4. **Create final report** comparing all approaches

## Alternative: Use Smaller Sample

If training is still too slow, you can modify the notebook to use fewer samples:

```python
# In Step 4, add this line after loading data:
train_df = train_df.sample(n=500, random_state=42)  # Use only 500 samples
```

This will train faster (5-10 minutes) but may be slightly less accurate.

## Files You'll Need

From your project, upload these to Colab:
1. `results/enron_preprocessed_3k.csv` (required)
2. `results/combined_preprocessed_2k.csv` (required)

## Files You'll Get

After training, download:
1. `phase7_finetuned_results.json` - Evaluation metrics
2. `phishing-finetuned-final/` - Fine-tuned model (optional, large)

## Next Steps

Once you have the results:
1. Update `docs/PHASE7_FINETUNING.md` with actual results
2. Update `README.md` with Phase 7 metrics
3. Update `COMPLETE_PROJECT_SUMMARY.md`
4. Create final comparison table
5. Push to GitHub

## Support

If you encounter issues:
1. Check the error message in Colab
2. Try the troubleshooting steps above
3. Reduce batch size or max_steps
4. Try with smaller sample (500 emails)

Good luck! 🚀
