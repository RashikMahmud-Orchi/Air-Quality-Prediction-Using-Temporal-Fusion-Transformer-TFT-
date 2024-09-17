# Air Quality Prediction Using Temporal Fusion Transformer (TFT)  
This project implements a deep learning model based on the Temporal Fusion Transformer (TFT) architecture to predict future air quality levels, specifically the concentration of carbon monoxide (CO) in the atmosphere. The model was trained on the Air Quality UCI dataset, containing time-series data of various atmospheric measurements, including CO, benzene, and nitrogen oxides. The main objective is to forecast CO levels 24 hours ahead, using past observations and temporal features such as day, month, and hour.  

## Key Features:  
- Data Preprocessing: The dataset is cleaned by handling missing values and removing invalid readings. Continuous features are normalized and engineered, and the target variable (CO) is shifted by 24 hours for prediction.  
- Exploratory Data Analysis (EDA): Statistical summaries and visualizations are used to understand the distribution and relationships between atmospheric components.  
- Temporal Fusion Transformer (TFT): The TFT model captures temporal patterns using attention mechanisms and is well-suited for time-series forecasting. The model predicts future air quality while considering dynamic and static features.  
- Baseline Model: A baseline is established using simple models for comparison against TFT's performance.  
- Model Tuning & Optimization: Hyperparameters like learning rate, hidden size, and dropout rate are tuned using techniques like random search and the PyTorch Lightning learning rate finder.  
- Evaluation: The model's performance is evaluated using metrics such as Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and visual comparisons of predicted vs actual CO levels.

## Results:  
The final model achieved an MAE of approximately 1.89 and RMSE of 2.43, indicating a strong ability to predict CO concentrations 24 hours into the future. Predictions were visualized to compare against actual recorded data.  

## Requirements  
```
pytorch-forecasting
lightning
torch
pandas
numpy
matplotlib
scikit-learn
optuna
```
Run the following to install all required packages.     
```
pip install -r requirements.txt
```

However to set up the Pytorch to run with CUDA use following:  


## Technologies Used:  

- Python: Data processing and model development  
- PyTorch Lightning: Model training and evaluation  
- TFT Model: Time-series forecasting  
- Pandas, NumPy: Data manipulation  
- Matplotlib: Data visualization  
- Optuna/PyTorch Lightning Tuner: Hyperparameter tuning  
