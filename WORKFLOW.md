# Complete Workflow: LLM-Based Image Classification to Model Training

This document describes the end-to-end workflow for classifying skin images using the LLM-based approach (v2) and training models.

## Overview

```
Raw Images
    ↓
[Step 1: LLM Analysis - skin-quality-qa]
    ↓
metadata_enriched.csv (with LLM labels)
    ↓
[Step 2: Data Preparation - Optional model training setup]
    ↓
[Step 3: Model Training - Train ML models with reliable labels]
    ↓
Trained Models + Evaluation Metrics
```

---

## Step 1: LLM-Based Image Classification (skin-quality-qa)

### Purpose
Use Google's Gemini API to automatically analyze images and classify them based on quality artifacts. This replaces the manual classification process from v1.

### Setup

1. **Navigate to the project:**
   ```bash
   cd skin-quality-qa
   ```

2. **Create environment file:**
   ```bash
   cp .env.example .env
   ```

3. **Add API key:**
   Edit `.env` and add your Google Gemini API key:
   ```
   GOOGLE_API_KEY=your_actual_api_key_here
   ```

4. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Prepare Images

Place your raw images in the expected directory structure:
```
skin-quality-qa/
└── data/
    └── train/
        ├── image1.jpg
        ├── image2.png
        └── ...
```

Also ensure `data/metadata.csv` exists with at least a filename column:
```csv
filename
image1.jpg
image2.png
```

### Run LLM Analysis

```bash
# Process images (40/day by default)
python -m src.enrich_labels_with_llm

# Or process custom number of images
python -m src.enrich_labels_with_llm 50
```

### Output

The process generates `data/metadata_enriched.csv`:
```csv
filename,blurry,hairy,bubble,ruler,vignette,gel_border
image1.jpg,0,0,1,0,0,0
image2.png,1,0,0,0,0,1
```

### Important Notes

- **Rate Limiting**: Free tier allows ~50 requests/day. Script processes 40/day with 6-second delays.
- **Resume**: If interrupted, the script resumes from the last processed image.
- **Cost**: Free tier sufficient for most projects; premium tier available if needed.

---

## Step 2: Data Preparation (Optional - for model training)

If you want to train ML models with the LLM-classified images, follow this step.

### Organize Images by Label

Create a directory structure matching the classification results:

```
datasetv2/
├── blurry/
│   ├── image1.jpg
│   └── ...
├── bubble/
│   ├── image2.jpg
│   └── ...
├── clean/
│   └── ...
├── hairy/
│   └── ...
└── marked/
    └── ...
```

### Map LLM Output to Labels

The LLM output has one of these patterns:
- **clean**: All artifacts = 0 (no defects)
- **blurry**: blurry = 1
- **hairy**: hairy = 1
- **bubble**: bubble = 1
- **marked/ruler/vignette**: ruler = 1, vignette = 1, gel_border = 1

Use the metadata_enriched.csv to organize images accordingly.

### (Optional) Run Data Preparation Script

If using the legacy v1 scripts:
```bash
cd ../skin-noise-detection
python src/prepare.py
```

This will:
- Detect all class folders
- Split images into train/validation/test (70%/15%/15%)
- Organize into data/ directory

---

## Step 3: Model Training (Optional)

If you want to train classification models with the labeled dataset.

### Using v1 Reference (Legacy)

Navigate to the skin-noise-detection project:

```bash
cd ../skin-noise-detection
pip install -r requirements.txt
```

### Train Single Model

```bash
python src/model.py
```

This trains a MobileNetV2 model on your dataset.

### Compare Multiple Models

```bash
python src/compare.py
```

Compares MobileNetV2, ResNet50, and NASNetMobile architectures.
Outputs: `final_comparison_results.csv`

### Analyze Errors

```bash
python src/analyzeerrors.py
```

Generates detailed error analysis and confusion matrices.

---

## Configuration Reference

### skin-quality-qa (LLM Analysis)

Edit `src/config.py`:
```python
DEFAULT_DAILY_LIMIT = 40          # Images to process per day
REQUEST_DELAY = 6                 # Seconds between API requests
RETRY_ATTEMPTS = 5                # Retry attempts for failed requests
RETRY_DELAY_BASE = 30             # Base delay for exponential backoff (seconds)
MODEL_NAME = "gemini-2.0-flash"   # Gemini model to use
```

### skin-noise-detection (Model Training)

Edit `src/model.py`:
```python
IMAGE_SIZE = (450, 600)           # Input image size
BATCH_SIZE = 32                   # Training batch size
EPOCHS = 30                        # Number of training epochs
LEARNING_RATE = 0.0001            # Learning rate
```

---

## Expected Results

### LLM Classification (Step 1)
- ✅ `metadata_enriched.csv` with 6 artifact classifications
- ✅ Automatic progress saving (resume capability)
- ✅ Processing time: ~10 seconds per image
- ✅ Cost: Free tier sustainable

### Model Training (Step 3)
- ✅ Trained model files: `best_model_*.h5`
- ✅ Metrics: Accuracy, F1-score, Precision, Recall
- ✅ Confusion matrices for error analysis
- ✅ Better accuracy with consistent LLM labels (vs manual classification)

---

## Troubleshooting

### Issue: GOOGLE_API_KEY not found
**Solution**: Ensure `.env` file exists and contains `GOOGLE_API_KEY=your_key`

### Issue: API quota exceeded (429 error)
**Solution**: 
- Default 40 images/day respects free tier limits
- Wait until next day to continue
- Or upgrade to Google Cloud paid plan

### Issue: Image format not supported
**Solution**: Ensure images are JPG, PNG, or supported formats by Gemini API

### Issue: Model training runs out of memory
**Solution**: 
- Reduce `BATCH_SIZE` in config (try 16 or 8)
- Enable GPU acceleration
- Use smaller `IMAGE_SIZE`

---

## Best Practices

1. **Start small**: Test with 10-20 images first before processing entire dataset
2. **Verify output**: Check a few entries in `metadata_enriched.csv` to ensure quality
3. **Monitor API usage**: Track your Google API quota
4. **Save checkpoints**: The LLM script auto-saves progress; model training saves best weights
5. **Use GPU**: GPU training is 5-10x faster than CPU for model training

---

## Next Steps

After completing the workflow:

1. **Evaluate Models**: Use confusion matrices to identify misclassified categories
2. **Improve Data**: Consider manually reviewing edge cases from `metadata_enriched.csv`
3. **Deploy Models**: Package best model for production use
4. **Iterate**: Retrain with additional or refined images to improve accuracy

---

## Reference Links

- [skin-quality-qa](./skin-quality-qa/README.md) - LLM-based classification details
- [skin-noise-detection](./skin-noise-detection/README.md) - Model training details
- [Google Gemini API Docs](https://ai.google.dev/)
- [TensorFlow/Keras Docs](https://www.tensorflow.org/)

---

**Last Updated**: January 2026
