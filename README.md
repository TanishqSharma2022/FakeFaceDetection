# Real vs Fake Face Classification

## Description
This project uses machine learning to classify images of faces as real or fake. The model employs convolutional neural networks (CNNs) to differentiate between real and fake faces based on images. The project includes preprocessing steps for facial recognition and training the model on a dataset of real and fake face images.

## Setup

### Prerequisites
- Python 3.10+
- Virtual environment (recommended)


### Create and activate a virtual environment

```bash
python -m venv myenv
source myenv/bin/activate  # For Windows use `myenv\Scripts\activate`
```

### Install all the dependencies
```bash
pip install -r requirements.txt
```

`To install the dlib library, use the custom wheel file in utils folder. This file runs on python 3.10. You would need a different version of this file to install on some other Python version. `

`Use the shape predictor file for face landmarks detection in the utils folder.`


### Download the dataset

*-* The dataset is not included in the repository due to its large size. You can download it from the following link:




### Project Structure
```bash

- project_root/
  - data/            # Folder with scripts to download and preprocess the dataset
  - model/           # Folder to store trained model weights
  - notebooks/       # Jupyter notebooks for analysis and experimentation
  - src/             # Source code for training, evaluation, and prediction
  - requirements.txt # Python dependencies for the project
  - README.md        # Project documentation

```
