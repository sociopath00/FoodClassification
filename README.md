# FoodClassification

This is a toy project which uses Pretrained model (EfficientNet) to predict if the image contains Pizza, Sushi or Stake. 

## Pre-requisite

- Python 
- UV package manager

## Setup

1. Clone the repository

    `git clone https://github.com/sociopath00/FoodClassification.git`

2. Change the directory and Create Virtual Env using UV

    `uv venv`

3. Install packages

    `uv pip install -r pyproject.toml`

4. Optional: Download the data
    
    `python download_data.py`

5. Train the model

    `python -m src.model_training`

6. Run the inference code

    `python -m src.inference`

4. Run the python script to launch Stream-lit UI

    `python app.py`




