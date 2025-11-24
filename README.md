# Skin and Noise Detection Project

## Overview
This project aims to develop a robust model for noise detection in skin lesion images. The model will be trained using a dataset that includes various images, and it will be evaluated on its ability to accurately identify skin regions and detect noise.

## Classes
The dataset is divided into the following classes:
- bubble
- hairy
- marked
- light
- clean

## Project Structure
The project is organized into the following directories and files:

- **data/**: Contains the datasets used for training, testing, and validation.
  - **raw/**: Place uploaded dataset here. Each class should be a subfolder (e.g. data/raw/bubble, data/raw/hairy, ...).
  - **train/**: Training dataset (created by the splitter script).
  - **test/**: Testing dataset (created by the splitter script).
  - **validation/**: Validation dataset (created by the splitter script).

- **notebooks/**: Contains Jupyter notebooks for exploratory data analysis, model training, and evaluation.
  - **skin_noise_detection.ipynb**: Main notebook for analysis and model development.

- **src/**: Contains source code for the project.
  - **preprocessing.py**: Functions for data preprocessing, including loading datasets, data augmentation, and normalization.
  - **model.py**: Defines the model architecture for skin and noise detection, including training and evaluation logic.
  - **utils.py**: Utility functions for logging, metrics calculation, and visualization.
  - **split_dataset.py**: Script to split data/raw into train/test/validation.

- **requirements.txt**: Lists the Python packages required for the project.

## How to upload and split the dataset
1. Place your dataset into `data/raw/` with one subfolder per class:
   - data/raw/bubble/
   - data/raw/hairy/
   - data/raw/marked/
   - data/raw/light/
   - data/raw/clean/

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the splitter script from project root (Windows):
   ```
   python src\split_dataset.py --source data\raw --dest data --train 0.7 --val 0.15 --test 0.15 --seed 42
   ```

4. Verify resulting folders:
   - data/train/<class>/
   - data/validation/<class>/
   - data/test/<class>/

## Usage Guidelines
- Use `notebooks/skin_noise_detection.ipynb` for exploration and model experiments.
- Modify `src/preprocessing.py`, `src/model.py`, and `src/utils.py` as needed.
- Keep large raw data out of Git (add `data/raw/` to `.gitignore`).

## Contributing
Contributions are welcome. Submit a pull request with changes and a brief description.

## License
This project is licensed under the MIT License.