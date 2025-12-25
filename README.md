# Skin Noise Detection Project

## Overview
This project aims to develop a robust model for detecting noise and artifacts in skin lesion images. The model is trained on a comprehensive dataset and evaluated on its ability to accurately classify various types of skin image defects and quality issues.

## Dataset Classes
The dataset includes five classes representing different types of skin image artifacts and conditions:
- **blurry**: Blurry or out-of-focus images
- **bubble**: Bubble or air gap artifacts
- **clean**: Clean, high-quality images without artifacts
- **hairy**: Images with visible hair or hair-like artifacts
- **marked**: Images with markings, labels, or annotations

## Project Structure
```
skin-noise-detection/
├── best_model_*.h5          # Pre-trained model files for different architectures
├── final_comparison_results.csv  # Comparison results between models
├── README.md               # This file
├── requirements.txt        # Python dependencies
├── data/                   # Dataset directory (created by prepare script)
│   ├── train/             # Training dataset
│   │   ├── blurry/
│   │   ├── bubble/
│   │   ├── clean/
│   │   ├── hairy/
│   │   └── marked/
│   ├── validation/        # Validation dataset
│   │   ├── blurry/
│   │   ├── bubble/
│   │   ├── clean/
│   │   ├── hairy/
│   │   └── marked/
│   └── test/              # Test dataset
│       ├── blurry/
│       ├── bubble/
│       ├── clean/
│       ├── hairy/
│       └── marked/
└── src/                    # Source code
    ├── prepare.py         # Data preparation and splitting script
    ├── model.py          # Model architecture and training script
    └── compare.py        # Model comparison and evaluation script
```

## Getting Started

### Prerequisites
- Python 3.8 or higher
- TensorFlow/Keras
- Required Python packages (see requirements.txt)

### Installation and Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare the dataset:**
   
   Place your raw dataset in the source directory following this structure:
   ```
   C:\proje\datasetv2\
   ├── blurry/
   ├── bubble/
   ├── clean/
   ├── hairy/
   └── marked/
   ```

   Then run the preparation script:
   ```bash
   python src/prepare.py
   ```

   This script will:
   - Automatically detect all class folders
   - Split images into train/validation/test sets (70%/15%/15%)
   - Organize them into the `data/` directory

3. **Train the model:**

   Using MobileNetV2 transfer learning:
   ```bash
   python src/model.py
   ```

4. **Compare multiple models:**

   Compare different architectures (MobileNetV2, ResNet50, NASNetMobile):
   ```bash
   python src/compare.py
   ```

   This will generate `final_comparison_results.csv` with performance metrics.

## Model Architectures

The project supports multiple pre-trained architectures for transfer learning:

- **MobileNetV2**: Lightweight model optimized for mobile and edge devices
- **ResNet50**: Deep residual network with 50 layers
- **NASNetMobile**: Mobile-optimized neural architecture search model

## Training Configuration

Key hyperparameters can be adjusted in the scripts:
- **Image Size**: 224x224 pixels (or 450x600 for model.py)
- **Batch Size**: 32
- **Epochs**: 50 (compare.py) or 30 (model.py)
- **Learning Rate**: 0.0001
- **Data Augmentation**: Rotation, zoom, horizontal flip, and shift

## Output

After running the comparison script, results are saved to:
- **Models**: `best_model_mobilenetv2.h5`, `best_model_resnet50.h5`, `best_model_nasnetmobile.h5`
- **Results**: `final_comparison_results.csv` containing accuracy, F1-score, precision, and recall

## Notes

- GPU acceleration is recommended for faster training
- The scripts automatically set up GPU memory growth to prevent OOM errors
- Pre-trained ImageNet weights are used as the base for transfer learning
- All models use categorical cross-entropy loss for multi-class classification

## Contributing

Contributions are welcome. Please feel free to submit pull requests with improvements, bug fixes, or new features.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
