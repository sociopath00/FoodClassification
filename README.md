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

3. Download the data for training
    
    `uv run download_data.py` 
    
    or 
    
    `python download_data.py`

4. Train the model

    `uv run -m src.model_training`

    or

    `python -m src.model_training`

5. Run the inference code

    `uv run -m src.inference`

    or

    `python -m src.inference`

4. Run the python script to launch Stream-lit UI

    `uv run streamlit run app.py`

    or

    `streamlit run app.py`




