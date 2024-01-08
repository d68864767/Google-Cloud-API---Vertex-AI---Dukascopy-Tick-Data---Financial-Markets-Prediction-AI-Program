```python
import json
import pandas as pd
from google.cloud import aiplatform
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib

# Load configuration
with open('config.json') as config_file:
    config = json.load(config_file)

# Initialize Vertex AI client
aiplatform.init(project=config['project_id'], location=config['location'])

# Data Preprocessing
def preprocess_data(data_path):
    # Load the data
    data = pd.read_csv(data_path)
    
    # Here you would add any data cleaning or feature engineering steps
    # For simplicity, we assume the data is already preprocessed and ready for training
    
    return data

# Model Training
def train_model(data, hyperparameters):
    # Split the data into features and target
    X = data.drop('target_column', axis=1)  # Replace 'target_column' with the actual name of your target column
    y = data['target_column']
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Initialize the model with the provided hyperparameters
    model = RandomForestRegressor(
        n_estimators=hyperparameters['num_epochs'],
        max_depth=len(hyperparameters['hidden_units']),
        random_state=42
    )
    
    # Train the model
    model.fit(X_train_scaled, y_train)
    
    # Evaluate the model
    predictions = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, predictions)
    print(f"Model Mean Squared Error: {mse}")
    
    return model, scaler

# Prediction
def make_predictions(model, scaler, data, prediction_horizon):
    # Here you would add any steps necessary to prepare your data for prediction
    # For simplicity, we assume the data is already in the correct format
    
    # Standardize the features
    data_scaled = scaler.transform(data)
    
    # Make predictions
    predictions = model.predict(data_scaled)
    
    # Create a DataFrame with the predictions
    prediction_df = pd.DataFrame(predictions, columns=['Predictions'])
    
    # Save the predictions to a .csv file
    prediction_df.to_csv(f"{config['output_directory']}/predictions.csv", index=False)
    
    return prediction_df

def main():
    # Preprocess the data
    data = preprocess_data(config['data_file_path'])
    
    # Train the model
    model, scaler = train_model(data, config['model_hyperparameters'])
    
    # Save the model and scaler
    joblib.dump(model, f"{config['output_directory']}/model.joblib")
    joblib.dump(scaler, f"{config['output_directory']}/scaler.joblib")
    
    # Make predictions for the next 12 months
    # Here you would add any steps necessary to generate the data for the next 12 months
    # For simplicity, we assume we have a DataFrame `future_data` ready for prediction
    future_data = pd.DataFrame()  # Placeholder for actual future data preparation
    predictions = make_predictions(model, scaler, future_data, config['prediction_horizon'])
    
    print("Predictions saved to .csv file.")

if __name__ == "__main__":
    main()
```
