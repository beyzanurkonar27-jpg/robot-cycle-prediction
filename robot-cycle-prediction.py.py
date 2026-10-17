# 1.Import Libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from pathlib import Path

# 2. Create Output Directories
Path("images").mkdir(exist_ok=True)
Path("results").mkdir(exist_ok=True)

# 3. Define Sensor Columns

sensor_columns = [
    "Current_J0",
    "Temperature_T0",
    "Current_J1",
    "Temperature_J1",
    "Current_J2",
    "Temperature_J2",
    "Current_J3",
    "Temperature_J3",
    "Current_J4",
    "Temperature_J4",
    "Current_J5",
    "Temperature_J5",
    "Speed_J0",
    "Speed_J1",
    "Speed_J2",
    "Speed_J3",
    "Speed_J4",
    "Speed_J5",
    "Tool_current"
]

temperature_columns = ["Temperature_T0","Temperature_J1","Temperature_J2","Temperature_J3","Temperature_J4","Temperature_J5"]

speeds = [ "Speed_J0", "Speed_J1", "Speed_J2", "Speed_J3", "Speed_J4", "Speed_J5"]

# 4. Model Evaluation Function
def evaluate_model(df,data,target):
    
    x = df[data]
    y = df[target]
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    model = LinearRegression()
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    
    MAE = mean_absolute_error(y_test,y_pred)
    MSE = mean_squared_error(y_test, y_pred)
    RMSE = np.sqrt(MSE)
    R2  = r2_score(y_test, y_pred)
    
    metrics = {
        "MAE": MAE,
        "MSE": MSE,
        "RMSE": RMSE,
        "R2": R2
    }
    
    return metrics,y_test,y_pred

# 5. Load Dataset
df = pd.read_excel("dataset_02052023.xlsx")
df.columns = df.columns.str.strip()
print(df.shape)
print(df.columns)
df.info()

# 6. Handle Missing Values
print(df.isnull().sum())
missing_rows = df[df.isnull().any(axis=1)]
print(missing_rows.shape)
print(missing_rows.index)

for i in missing_rows.index:
    for j in sensor_columns:
        if pd.isnull(df.loc[i,j]):
            df.loc[i,j] = df.loc[i-5:i+5, j].median()
            
print(df.isnull().sum())       

robot_stop_missing = df[df["Robot_ProtectiveStop"].isnull()]
for i in robot_stop_missing.index:
        df.loc[i,"Robot_ProtectiveStop"] = df.loc[i-5:i+5, "Robot_ProtectiveStop"].mode()[0]  
        
print(df.isnull().sum())
print(df[sensor_columns].isnull().sum())
print(df["Robot_ProtectiveStop"].isnull().sum())
print(df["cycle"].isnull().sum())

# 7. Exploratory Data Analysis
correlation_columns = sensor_columns + ["cycle", "Robot_ProtectiveStop"]
correlation_matrix = df[correlation_columns].corr()

plt.figure(figsize=(14, 10))
sns.heatmap(correlation_matrix,annot=False)
plt.title("Correlation Matrix")
plt.tight_layout()
plt.savefig("images/correlation_heatmap.png",dpi = 300, bbox_inches = "tight")
plt.show()


# 8. Temperature and Cycle Relationship
fig, axes = plt.subplots(2,3,figsize=(18, 10))
for ax , column in zip(axes.flat, temperature_columns):

    
    x = df[column]
    y = df["cycle"]

    ax.scatter(x,y, alpha=0.5)
    
    coefficient = np.polyfit(x, y, 1)
    trend_line  = np.poly1d(coefficient)
    
    x_sorted = np.sort(x)
    ax.plot(x_sorted, trend_line(x_sorted))
   
    ax.set_xlabel(column)
    ax.set_ylabel("cycle")
    ax.set_title(f"{column} vs Cycle")
    
    
plt.tight_layout()
plt.savefig("images/cycle_temperature_association.png", dpi=300, bbox_inches ="tight")
plt.show()
 
# 9. Sensor Correlations with Cycle   
cycle_correlation = correlation_matrix.loc[sensor_columns,"cycle"]

cycle_correlation = cycle_correlation.sort_values(
    key=abs,
    ascending=False
)
print("\nSensor correlations with cycle:")
print(cycle_correlation)



cycle_correlation.sort_values().plot(kind="barh",figsize=(10, 8))

plt.xlabel("Correlation with Cycle")
plt.ylabel("Sensor Variable")
plt.title("Robot Sensor Variables Associated with Cycle")
plt.tight_layout()
plt.savefig("images/sensor_variables_associated_cycle.png", dpi=300, bbox_inches="tight")
plt.show()

# 10. Model Without Robot State
metrics_without, y_test, y_pred = evaluate_model(df, sensor_columns, "cycle")
print("\nModel Without Robot State:")
print(metrics_without)

result_without_robot_state = pd.DataFrame({
    "Actual": y_test,
    "Predicted": y_pred,
    }, index=y_test.index)

result_without_robot_state["Error"] = (result_without_robot_state["Actual"]- result_without_robot_state["Predicted"]) 
result_without_robot_state["Absolute_Error"] = result_without_robot_state["Error"].abs()
result_without_robot_state.to_csv("results/predictions_without_robot_state.csv")

# 11. Actual vs Predicted Plot
plt.figure(figsize=(10,9))
plt.scatter(y_test, y_pred, alpha=0.5)

plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()], linestyle= "--")

plt.xlabel("Actual cycle")
plt.ylabel("Predicted cycle")
plt.title("Actual vs Predicted Cycle")
plt.savefig("images/actual_vs_predicted_cycle.png", dpi=300, bbox_inches="tight")
plt.show()

# 12. Prediction Error Distribution
plt.figure(figsize=(10,9))
error_without = y_test-y_pred
plt.hist(error_without, bins=20)
plt.xlabel("Prediction error")
plt.ylabel("Frequency")
plt.title("Prediction Error Distribution")
plt.tight_layout()
plt.savefig("images/prediction_error_distribution.png", dpi=300, bbox_inches="tight")
plt.show()

print("Mean prediction error:", error_without.mean())


# 13. Create Robot State Feature

df["Robot_state"] = (df[speeds] != 0).any(axis=1).astype(int)
print("\nRobot state distribution:")
print(df["Robot_state"].value_counts())  

# 14. Model With Robot State
features_with_robot_state = sensor_columns + ["Robot_state"]  
metrics_with, y_test_with, y_pred_with  =evaluate_model(df, features_with_robot_state, "cycle")

print("\nModel With Robot State:")
print(metrics_with)

result_with_robot_state = pd.DataFrame({
    "Actual": y_test_with,
    "Predicted": y_pred_with,
    }, index=y_test.index)
result_with_robot_state["Error"] = (result_with_robot_state["Actual"] - result_with_robot_state["Predicted"])
result_with_robot_state["Absolute_Error"] = (result_with_robot_state["Error"].abs())
result_with_robot_state.to_csv("results/predictions_with_robot_state.csv")

# 15. Compare Model Performance

comparison = pd.DataFrame({
    "Without Robot State": metrics_without,
    "With Robot State": metrics_with
})

print("\nModel Comparison:")
print(comparison)

comparison.to_csv("results/comparison.csv")
