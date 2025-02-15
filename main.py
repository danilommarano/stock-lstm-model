import numpy as np
import tensorflow as tf
import yfinance as yf
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta

app = FastAPI()

model = tf.keras.models.load_model("lstm_stock_model.h5")

symbol = 'AAPL'

class StockRequest(BaseModel):
    date: str

def get_last_60_days(symbol):
    """Baixa os últimos 60 dias de preços de uma ação"""
    df = yf.download(symbol, period="70d")
    df = df[['Close']].dropna()
    return df

@app.post("/predict/")
def predict_stock(data: StockRequest):
    """API para prever o preço de fechamento de uma ação em uma data futura"""
    
    df = get_last_60_days(symbol)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df)

    last_60_days = scaled_data[-60:].reshape(1, 60, 1)
    today = df.index[-1]  # Última data disponível no dataset
    target_date = datetime.strptime(data.date, "%Y-%m-%d")

    while today < target_date:
        predicted_price = model.predict(last_60_days)
        predicted_price = np.reshape(predicted_price, (1, 1, 1))
        last_60_days = np.concatenate((last_60_days[:, 1:, :], predicted_price), axis=1)
        today += timedelta(days=1)

    predicted_price = scaler.inverse_transform(predicted_price.reshape(-1, 1))

    return {
        "symbol": symbol,
        "predicted_date": data.date,
        "predicted_price": float(predicted_price[0][0])
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
