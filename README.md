# Skin and Noise Detection Project

## Overview
This project aims to develop a robust model for noise detection in skin lesion images. The model will be trained using a dataset that includes various images, and it will be evaluated on its ability to accurately identify skin regions and detect noise.

## Classes
The dataset is divided into the following classes:
- band
- bubble
- clean
- hairy
- marked

## Project Structure
The project is organized into the following directories and files:

- **data/**: Contains the datasets used for training, testing, and validation.
  - **train/**: Training dataset (created by the prepare script).
  - **test/**: Testing dataset (created by the prepare script).
  - **validation/**: Validation dataset (created by the prepare script).

- **notebooks/**: Contains Jupyter notebooks for exploratory data analysis, model training, and evaluation.
  - **skin_noise_detection.ipynb**: Main notebook for analysis and model development.

- **src/**: Contains source code for the project.
  - **preprocessing.py**: Functions for data preprocessing, including loading datasets, data augmentation, and normalization.
  - **model.py**: Defines the model architecture for skin and noise detection, including training and evaluation logic.
  - **utils.py**: Utility functions for logging, metrics calculation, and visualization.
  - **train.py**: Script to train the model using MobileNetV2 transfer learning.
  - **prepare.py**: Script to organize data from raw dataset into train/test/validation splits.

- **requirements.txt**: Lists the Python packages required for the project.

## How to use the project
1. Place your raw dataset in `C:\proje\dataset\` with one subfolder per class:
   - C:\proje\dataset\band\
   - C:\proje\dataset\bubble\
   - C:\proje\dataset\clean\
   - C:\proje\dataset\hairy\
   - C:\proje\dataset\marked\

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the preparation script from project root:
   ```
   python src\prepare.py
   ```

4. Run the training script:
   ```
   python src\train.py
   ```

5. Verify resulting folders in `data/`:
   - data/train/<class>/
   - data/validation/<class>/
   - data/test/<class>/

## Usage Guidelines
- Use `notebooks/skin_noise_detection.ipynb` for exploration and model experiments.
- Modify `src/preprocessing.py`, `src/model.py`, and `src/utils.py` as needed.
- Keep large raw data out of Git (add `data/` to `.gitignore`).

## Contributing
Contributions are welcome. Submit a pull request with changes and a brief description.

## License
This project is licensed under the MIT License.