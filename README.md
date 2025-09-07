# FoodClassification

This is a toy project which uses Pretrained model (EfficientNet) to predict if the image contains Pizza, Sushi or Stake. 

## Pre-requisite

- Python 
- UV package manager

## Setup

1. Clone the repository

    `git clone https://github.com/sociopath00/FoodClassification.git`

2. Install the requirements and dependecies

    `uv sync`

4. Download the data for training
    
    `python download_data.py`

5. Train the model

    `python -m src.model_training`

6. Run the inference code

    `python -m src.inference`

4. Run the python script to launch Stream-lit UI

    `streamlit run app.py`




