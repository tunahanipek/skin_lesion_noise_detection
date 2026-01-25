# Skin Image Quality Analysis - Project Collection

This repository contains two complementary projects for automated skin image quality assessment and classification. They represent two generations of approaches to the same problem.

## 📚 Projects Overview

### 1. **skin-quality-qa** (v2 - Current/Recommended) ⭐
**LLM-Based Automated Classification**

- Uses Google's Gemini API for automated image analysis
- Detects 6 artifact types: blurry, hairy, bubble, ruler, vignette, gel_border
- Generates structured labels automatically
- Rate-limiting friendly (free tier compatible)
- High scalability and consistency

**When to use**: For production workflows needing automated, scalable classification

[→ See skin-quality-qa README](./skin-quality-qa/README.md)

---

### 2. **skin-noise-detection** (v1 - Legacy)
**Manual Classification with Model Training**

- Demonstrates manual image classification approach
- Supports multiple model architectures (MobileNetV2, ResNet50, NASNetMobile)
- Includes comprehensive error analysis
- Works with 5 image classes: blurry, bubble, clean, hairy, marked

**When to use**: For reference, legacy workflows, or understanding the original approach

[→ See skin-noise-detection README](./skin-noise-detection/README.md)

---

## 🔄 Evolution: From v1 to v2

```
┌─────────────────────────────────────────────────────────────┐
│                    v1: Manual Process                        │
├─────────────────────────────────────────────────────────────┤
│  Raw Images → Manual Classification → Labeled Dataset        │
│  (Slow, Error-prone, High effort)                            │
└─────────────────────────────────────────────────────────────┘
                            ↓ (Evolved to)
┌─────────────────────────────────────────────────────────────┐
│                    v2: LLM Automation                         │
├─────────────────────────────────────────────────────────────┤
│  Raw Images → Gemini API Analysis → metadata_enriched.csv    │
│  (Fast, Consistent, Low effort, Scalable)                    │
└─────────────────────────────────────────────────────────────┘
```

## 🎯 Comparison Table

| Aspect | v1 (Manual) | v2 (LLM) |
|--------|------------|---------|
| **Classification Method** | Human manual review | Google Gemini API |
| **Speed** | Slow (~5 min/image) | Fast (~10 sec/image) |
| **Consistency** | Subjective/Variable | Consistent AI-based |
| **Artifacts Detected** | 5 classes | 6 artifact types |
| **Scalability** | Limited | Excellent |
| **Cost** | Free (time) | Free tier available |
| **Maintenance** | High (human effort) | Low (automated) |
| **Recommended** | Legacy reference | ✅ Current |

## 💡 Recommended Workflow

1. **Start with v2** (skin-quality-qa)
   - Run Gemini API analysis on your images
   - Get `metadata_enriched.csv` with LLM-based labels

2. **Prepare data** 
   - Use the enriched metadata to organize images

3. **Train models** (optional)
   - Use the labeled dataset to train ML models
   - Reference v1's model training scripts if needed

## 🛠️ Quick Start

### For LLM-Based Classification (v2):
```bash
cd skin-quality-qa
cp .env.example .env
# Add your Google API key to .env
pip install -r requirements.txt
python -m src.enrich_labels_with_llm
```

### For Manual Classification (v1):
```bash
cd skin-noise-detection
pip install -r requirements.txt
# Place your images in datasetv2/ folder
python src/prepare.py
python src/model.py
```

## 📝 Documentation

- **[skin-quality-qa](./skin-quality-qa/README.md)** - LLM-based image analysis using Gemini API
- **[skin-noise-detection](./skin-noise-detection/README.md)** - Manual classification and model training reference

## 🔑 Key Concepts

### Artifacts Detected (v2)
- **blurry**: Image is out of focus
- **hairy**: Contains distracting hair/artifact hair
- **bubble**: Air bubbles or gaps present
- **ruler**: Ruler, scale, or frame markings visible
- **vignette**: Darkening at image corners
- **gel_border**: Gel boundaries or liquid bubble edges

### Model Architectures (v1 Reference)
- MobileNetV2 (lightweight, mobile-friendly)
- ResNet50 (deep residual network)
- NASNetMobile (neural architecture search)

## 📊 Project Structure

```
.
├── skin-quality-qa/          (v2: LLM Classification)
│   ├── src/
│   │   ├── enrich_labels_with_llm.py    (Main LLM analysis)
│   │   ├── config.py                     (Configuration)
│   │   └── data_loader.py               (Data utilities)
│   └── README.md
│
├── skin-noise-detection/     (v1: Manual + Model Training)
│   ├── src/
│   │   ├── prepare.py                    (Data prep)
│   │   ├── model.py                      (Model training)
│   │   ├── compare.py                    (Compare models)
│   │   └── analyzeerrors.py             (Error analysis)
│   └── README.md
│
└── README.md                 (This file)
```

## 🤝 Contributing

Contributions are welcome! Please refer to the individual project READMEs for contribution guidelines.

## 📄 License

Both projects are licensed under the MIT License.

---

**Last Updated**: January 2026

**Questions?** Check the individual README files in each project folder.
