import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.data.loader import load_data, load_config
from src.models.lstm_model import LSTMModel
from src.models.lstm_dataloader import get_lstm_dataloader
from src.tracking.mlflow_utils import set_experiment, start_run, log_params, log_metrics, log_model
import os
import mlflow.models
import mlflow.pyfunc
import numpy as np
from mlflow.models.signature import infer_signature

config = load_config()
set_experiment(config['mlflow']['experiment_name'])

def main():
    df = load_data(processed=True)
    params = config['lstm']
    dataloader = get_lstm_dataloader(
        df,
        batch_size=params['batch_size'],
        sequence_length=params['sequence_length'],
        target_col='sales',
        id_col='product_id',
        shuffle=True,
        scaler_path='models/lstm/feature_scaler.joblib',
        feature_cols_path='models/lstm/feature_cols.joblib'
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_features = len(dataloader.dataset.feature_cols)
    model = LSTMModel(
        input_size=num_features,
        hidden_size=params['hidden_size'],
        num_layers=params['num_layers'],
        dropout=params['dropout']
    ).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])
    model.train()
    for epoch in range(params['epochs']):
        epoch_loss = 0
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * x.size(0)
        avg_loss = epoch_loss / len(dataloader.dataset)
        print(f"Epoch {epoch+1}/{params['epochs']}, Loss: {avg_loss:.4f}")
    with start_run(run_name='lstm_train'):
        # Add model name tag
        mlflow.set_tag("model_name", "lstm")
        
        log_params(params)
        
        # Calculate and log performance metrics
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        import numpy as np
        
        # Get predictions on training data
        model.eval()
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for x, y in dataloader:
                x = x.to(device)
                y = y.to(device)
                output = model(x)
                predictions.extend(output.cpu().numpy().flatten())
                actuals.extend(y.cpu().numpy().flatten())
        
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        mae = mean_absolute_error(actuals, predictions)
        rmse = np.sqrt(mean_squared_error(actuals, predictions))
        mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100
        
        log_metrics({
            'final_loss': avg_loss,
            'mae': mae,
            'rmse': rmse,
            'mape': mape
        })
        
        # Save model locally
        os.makedirs('models/lstm', exist_ok=True)
        torch.save(model.state_dict(), 'models/lstm/lstm_model.pt')
        
        # MLflow model signature and input example
        feature_cols = dataloader.dataset.feature_cols
        sample_x, _ = next(iter(dataloader))
        input_example = sample_x[:1].cpu().numpy()
        signature = infer_signature(input_example, model(sample_x[:1].to(device)).detach().cpu().numpy())
        mlflow.pytorch.log_model(model, 'lstm_model', signature=signature, input_example=input_example)

if __name__ == '__main__':
    main()
