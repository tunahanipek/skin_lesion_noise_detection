# Skin Quality QA (v2)

**Automated skin image quality assessment using Google's Gemini API**

This is the **second generation** of the skin image classification project. Unlike the legacy [skin-noise-detection](../skin-noise-detection/) which relies on manual classification, this version uses AI/LLM to automatically analyze and classify images based on quality artifacts.

## 🎯 Purpose

This project evolves the classification workflow by:
1. **Automating image analysis** using Google's Gemini API instead of manual classification
2. **Providing more reliable labels** for training ML models
3. **Improving model accuracy** with consistent, AI-based artifact detection

## Features

- ✅ **Automated LLM Analysis**: Uses Google Gemini API to analyze images for quality artifacts
- ✅ **Artifact Detection**: Identifies 6 artifact types:
  - blurry, hairy, bubble, ruler, vignette, gel_border
- ✅ **Rate Limiting**: Respects Google's free tier limits (~50 requests/day)
- ✅ **Resume Capability**: Continues from last processed image if interrupted
- ✅ **Structured Output**: Generates `metadata_enriched.csv` with LLM classifications

## Project Structure

```
skin-quality-qa/
├── .github/workflows/     # CI/CD automation
├── data/                  # Dataset (excluded from git)
│   ├── train/
│   └── metadata.csv
├── models/                # Trained models
├── src/                   # Source code
│   ├── config.py         # Configuration
│   ├── model.py          # Model architecture
│   ├── data_loader.py    # Data loading
│   └── enrich_labels_with_llm.py  # Main script
├── tests/                 # Unit tests
├── requirements.txt       # Python dependencies
├── .env.example          # Environment template
└── Dockerfile            # Docker configuration
```

## Workflow

```
Raw Images (data/train/)
    ↓
[Gemini API Analysis via enrich_labels_with_llm.py]
    ↓
metadata_enriched.csv (with LLM-based labels)
    ↓
[Can be used to train ML models - see notes below]
    ↓
Trained Models (with better accuracy)
```

## Installation

1. Clone the repository
2. Navigate to the `skin-quality-qa` directory
3. Create `.env` file from `.env.example`:
   ```bash
   cp .env.example .env
   ```
4. Add your Google Gemini API key to `.env`:
   ```
   GOOGLE_API_KEY=your_api_key_here
   ```
5. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Process images with default limit (40/day):
```bash
python -m src.enrich_labels_with_llm
```

### Process custom number of images:
```bash
python -m src.enrich_labels_with_llm 50
```

### Run tests:
```bash
python -m pytest tests/
```

## Configuration

Edit `src/config.py` to adjust:
- Daily request limit
- Request delay (for RPM rate limiting)
- Retry attempts and backoff strategy

## Rate Limits

- **Free Tier**: ~50 requests/day, ~15 RPM
- Current settings: 40 images/day with 6-second delays
- Automatically resumes from last processed image

## Timeline

- 646 total images
- 40 images/day processing
- Estimated: ~16 days to complete

## Integration with Model Training

The output of this LLM enrichment (`metadata_enriched.csv`) can be used to train ML models with more reliable labels. See the [Legacy Project](#legacy-project---v1) section for model training details.

### Next Steps After LLM Analysis:
1. Use `metadata_enriched.csv` output as training data
2. Follow the model training pipeline from [skin-noise-detection](../skin-noise-detection/)
3. Train ML models (MobileNetV2, ResNet50, etc.) with improved, LLM-based labels

## Docker

Build and run in Docker:
```bash
docker build -t skin-quality-qa .
docker run -e GOOGLE_API_KEY=your_key_here skin-quality-qa
```

## Legacy Project - v1

For the **manual classification approach**, see [skin-noise-detection](../skin-noise-detection/). This older version demonstrates:
- Manual image classification workflow
- Model training and comparison
- Error analysis

### Comparison: v1 vs v2

| Aspect | v1 (Legacy) | v2 (Current) |
|--------|------------|------------|
| Classification | Manual | LLM-based (Gemini API) |
| Artifacts | 5 classes | 6 artifact types |
| Configuration | Hardcoded | config.py |
| Reliability | User-dependent | Consistent AI-based |
| Effort | High (manual) | Low (automated) |
| Scalability | Limited | Free tier friendly |

## License

MIT
