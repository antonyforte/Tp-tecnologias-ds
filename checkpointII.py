import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def calculate_errors(actual, predicted):
    """Calculate various error metrics."""
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual, predicted)
    r2 = r2_score(actual, predicted)
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'MAPE': mape
    }

def plot_indicators(df_mean, predictions_df, actual_values):
    for _, row in predictions_df.iterrows():
        conjunto_id = row['IdeConjUndConsumidoras']
        indicador = row['SigIndicador']
        desc_conjunto = row['DscConjUndConsumidoras']
        
        df_conj = df_mean[(df_mean['IdeConjUndConsumidoras'] == conjunto_id) & 
                          (df_mean['SigIndicador'] == indicador)]
        
        df_conj = df_conj.sort_values('AnoIndice')
        
        plt.figure(figsize=(12, 6))
        
        # Plot all historical data up to 2021 as training data
        historical_data = df_conj[df_conj['AnoIndice'] <= 2021]
        sns.lineplot(x=historical_data['AnoIndice'], y=historical_data['VlrIndiceEnviado'], 
                    marker='o', label="Training Data", color='blue')
        
        # Create continuous line from 2021 to 2022-2023
        last_training_point = historical_data.iloc[-1]
        
        # Plot predicted values with connection to last training point
        future_years = [2021, 2022, 2023]  # Include 2021 for connection
        predicted_values = [last_training_point['VlrIndiceEnviado'], row['2022'], row['2023']]
        sns.lineplot(x=future_years, y=predicted_values, 
                    marker='o', linestyle='dashed', label="Predicted Data", color='orange')
        
        # Plot actual values with connection to last training point
        actual_vals = [last_training_point['VlrIndiceEnviado'],
                      actual_values[(conjunto_id, indicador)]['2022'],
                      actual_values[(conjunto_id, indicador)]['2023']]
        sns.lineplot(x=future_years, y=actual_vals, 
                    marker='s', label="Actual Data", color='green')
        
        plt.xlabel("Year")
        plt.ylabel(indicador)
        plt.title(f"{desc_conjunto} - {indicador}\nPrediction vs Actual Values")
        plt.legend()
        plt.grid(True)
        
        # Add error metrics
        actual_comparison = [actual_values[(conjunto_id, indicador)]['2022'],
                           actual_values[(conjunto_id, indicador)]['2023']]
        predicted_comparison = [row['2022'], row['2023']]
        errors = calculate_errors(np.array(actual_comparison), np.array(predicted_comparison))
        error_text = f"RMSE: {errors['RMSE']:.2f}\nMAPE: {errors['MAPE']:.2f}%"
        plt.text(0.02, 0.98, error_text, transform=plt.gca().transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.savefig(f"plots/validation_{conjunto_id}_{indicador}.png")
        plt.close()

def prepare_data(df):
    df['DatGeracaoConjuntoDados'] = pd.to_datetime(df['DatGeracaoConjuntoDados'], format='%d-%m-%Y')
    df_mean = df.groupby(['IdeConjUndConsumidoras', 'DscConjUndConsumidoras', 
                         'SigIndicador', 'AnoIndice'])['VlrIndiceEnviado'].mean().reset_index()
    return df_mean

def create_features(df_mean, conjunto_id, indicador, train_years):
    df_conj = df_mean[(df_mean['IdeConjUndConsumidoras'] == conjunto_id) & 
                      (df_mean['SigIndicador'] == indicador) &
                      (df_mean['AnoIndice'].isin(train_years))]
    
    df_conj = df_conj.sort_values('AnoIndice')
    
    X = []
    y = []
    
    for i in range(3, len(df_conj)):
        features = df_conj['VlrIndiceEnviado'].iloc[i-3:i].values
        target = df_conj['VlrIndiceEnviado'].iloc[i]
        X.append(features)
        y.append(target)
    
    return np.array(X), np.array(y), df_conj['DscConjUndConsumidoras'].iloc[0]

def predict_future_values(X, y, last_values, years_to_predict=2):
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X, y)
    
    future_predictions = []
    current_values = last_values.copy()
    
    for _ in range(years_to_predict):
        prediction = rf_model.predict([current_values])[0]
        future_predictions.append(prediction)
        current_values = np.append(current_values[1:], prediction)
    
    return future_predictions

def main(data_path):
    df = pd.read_csv(data_path, sep=';', decimal=',', encoding='latin1')
    
    # Filter out unwanted indicators
    df = df[~df['SigIndicador'].isin(['NDIACRI', 'VLCLACRI'])]
    
    df_mean = prepare_data(df)
    train_years = list(range(2011, 2022))
    combinations = df_mean[['IdeConjUndConsumidoras', 'SigIndicador']].drop_duplicates()
    
    predictions = []
    actual_values = {}
    error_metrics = []
    
    for _, row in combinations.iterrows():
        conjunto_id = row['IdeConjUndConsumidoras']
        indicador = row['SigIndicador']
        
        try:
            X, y, desc_conjunto = create_features(df_mean, conjunto_id, indicador, train_years)
            
            if len(X) > 0:
                # Get last three years from training data for prediction
                last_values = df_mean[
                    (df_mean['IdeConjUndConsumidoras'] == conjunto_id) & 
                    (df_mean['SigIndicador'] == indicador) &
                    (df_mean['AnoIndice'].isin([2019, 2020, 2021]))
                ].sort_values('AnoIndice')['VlrIndiceEnviado'].values
                
                # Get actual values for 2022 and 2023
                actual_vals = df_mean[
                    (df_mean['IdeConjUndConsumidoras'] == conjunto_id) & 
                    (df_mean['SigIndicador'] == indicador) &
                    (df_mean['AnoIndice'].isin([2022, 2023]))
                ].sort_values('AnoIndice')['VlrIndiceEnviado'].values
                
                if len(actual_vals) == 2:  # Only process if we have both 2022 and 2023 values
                    future_vals = predict_future_values(X, y, last_values)
                    
                    predictions.append({
                        'IdeConjUndConsumidoras': conjunto_id,
                        'DscConjUndConsumidoras': desc_conjunto,
                        'SigIndicador': indicador,
                        '2022': future_vals[0],
                        '2023': future_vals[1]
                    })
                    
                    actual_values[(conjunto_id, indicador)] = {
                        '2022': actual_vals[0],
                        '2023': actual_vals[1]
                    }
                    
                    # Calculate errors
                    errors = calculate_errors(actual_vals, future_vals)
                    errors.update({
                        'IdeConjUndConsumidoras': conjunto_id,
                        'DscConjUndConsumidoras': desc_conjunto,
                        'SigIndicador': indicador
                    })
                    error_metrics.append(errors)
                    
        except Exception as e:
            print(f"Error processing conjunto {conjunto_id} with indicator {indicador}: {str(e)}")
    
    predictions_df = pd.DataFrame(predictions)
    error_metrics_df = pd.DataFrame(error_metrics)
    
    return df_mean, predictions_df, error_metrics_df, actual_values

if __name__ == "__main__":
    data_path = "atendimento-ocorrencias/indicador_atendimento_filtrado.csv"
    df_mean, predictions_df, error_metrics_df, actual_values = main(data_path)
    
    print("\nPredictions for 2022 and 2023:")
    print(predictions_df)
    
    print("\nError Metrics:")
    print(error_metrics_df)
    
    # Save results
    predictions_df.to_csv("validation_predictions.csv", index=False, sep=';', decimal=',', encoding='latin1')
    error_metrics_df.to_csv("error_metrics.csv", index=False, sep=';', decimal=',', encoding='latin1')
    
    # Generate plots
    plot_indicators(df_mean, predictions_df, actual_values)