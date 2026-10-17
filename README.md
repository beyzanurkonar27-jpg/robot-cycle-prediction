# robot-cycle-prediction
Predicting robot operating cycle from joint sensor data (current, temperature, speed) using linear regression

## What it does
- Fills missing values using a rolling median/mode
- Explores correlations between sensors and cycle
- Trains a linear regression model with and without a `Robot_state` feature
- Compares models using MAE, MSE, RMSE and R²
