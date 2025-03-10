from flask import Flask, jsonify, request
from flask_cors import CORS
import yfinance as yf
import pandas as pd
from prophet import Prophet
import requests
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import datetime
import os

# Descargar recursos necesarios para NLTK
nltk.download('vader_lexicon')

# Configuración
NEWS_API_KEY = os.environ.get('NEWS_API_KEY', 'tu_clave_aquí')  # Obtener de variables de entorno o usar valor predeterminado

app = Flask(__name__)
CORS(app)  # Habilitar CORS para todas las rutas

def get_stock_data(symbol):
    """Obtener datos históricos de acciones"""
    stock = yf.Ticker(symbol)
    data = stock.history(period="6mo")
    return data

def predict_stock(symbol):
    """Predecir precios de acciones"""
    data = get_stock_data(symbol)
    # Convertir el índice a datetime y eliminar timezone
    df = pd.DataFrame({'ds': data.index.tz_localize(None), 'y': data['Close']})
    df.reset_index(inplace=True, drop=True)  # Eliminar la columna 'index' duplicada
    
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)
    
    return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(30).to_dict(orient="records")

def get_news(symbol):
    """Obtener noticias relacionadas con un símbolo de acción"""
    try:
        company_name = yf.Ticker(symbol).info.get('shortName', symbol)
        
        # Calcular fecha de hace una semana
        today = datetime.datetime.now()
        week_ago = today - datetime.timedelta(days=7)
        
        # Consultar API de noticias
        url = f"https://newsapi.org/v2/everything?q={company_name}&from={week_ago.strftime('%Y-%m-%d')}&sortBy=popularity&apiKey={NEWS_API_KEY}"
        
        response = requests.get(url)
        if response.status_code != 200:
            print(f"Error al obtener noticias: {response.status_code}")
            return []
        
        news_data = response.json()
        
        # Procesar y devolver noticias
        articles = news_data.get('articles', [])
        return articles
    except Exception as e:
        print(f"Error en get_news: {str(e)}")
        return []

def analyze_news_sentiment(news_articles):
    """Analizar el sentimiento de las noticias"""
    if not news_articles:
        return {"compound": 0, "positive": 0, "negative": 0, "neutral": 0}
    
    sia = SentimentIntensityAnalyzer()
    
    all_sentiments = []
    for article in news_articles:
        title = article.get('title', '')
        description = article.get('description', '')
        content = title + " " + description
        
        sentiment = sia.polarity_scores(content)
        all_sentiments.append(sentiment)
    
    # Calcular promedio de sentimientos
    compound = sum(s['compound'] for s in all_sentiments) / len(all_sentiments)
    positive = sum(s['pos'] for s in all_sentiments) / len(all_sentiments)
    negative = sum(s['neg'] for s in all_sentiments) / len(all_sentiments)
    neutral = sum(s['neu'] for s in all_sentiments) / len(all_sentiments)
    
    return {
        "compound": compound,
        "positive": positive,
        "negative": negative,
        "neutral": neutral
    }

@app.route('/')
def home():
    return jsonify({"status": "API is running"})

@app.route('/predict', methods=['GET'])
def predict():
    """Endpoint para predecir precios de acciones"""
    symbol = request.args.get('symbol', 'AAPL')
    predictions = predict_stock(symbol)
    return jsonify(predictions)

@app.route('/news', methods=['GET'])
def news():
    """Endpoint para obtener noticias relacionadas con una acción"""
    symbol = request.args.get('symbol', 'AAPL')
    news_articles = get_news(symbol)
    return jsonify(news_articles)

@app.route('/sentiment', methods=['GET'])
def sentiment():
    """Endpoint para obtener análisis de sentimiento de noticias"""
    symbol = request.args.get('symbol', 'AAPL')
    news_articles = get_news(symbol)
    sentiment_analysis = analyze_news_sentiment(news_articles)
    return jsonify(sentiment_analysis)

@app.route('/market-prediction', methods=['GET'])
def market_prediction():
    """Endpoint para obtener predicción combinada (precio + noticias)"""
    try:
        symbol = request.args.get('symbol', 'AAPL')
        
        # Obtener predicciones de precio
        price_predictions = predict_stock(symbol)
        
        # Obtener análisis de sentimiento
        news_articles = get_news(symbol)
        sentiment_analysis = analyze_news_sentiment(news_articles)
        
        # Combinar resultados
        result = {
            "price_predictions": price_predictions,
            "sentiment_analysis": sentiment_analysis,
            "news": news_articles[:5]  # Incluir las 5 noticias más relevantes
        }
        
        return jsonify(result)
    except Exception as e:
        print(f"Error en market_prediction: {str(e)}")
        # Devolver un mensaje de error y código de estado 500
        return jsonify({"error": str(e), "message": "Error al procesar la solicitud"}), 500

if __name__ == '__main__':
    # Modo desarrollo local
    app.run(debug=True, host='0.0.0.0')
    
# Esto es necesario para Render/Gunicorn
# No eliminar esta variable
application = app 
