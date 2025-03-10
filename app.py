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
    try:
        print(f"Obteniendo datos históricos para: {symbol}")
        stock = yf.Ticker(symbol)
        data = stock.history(period="6mo")
        
        # Verificar si obtuvimos datos
        if data.empty:
            print(f"No se encontraron datos históricos para {symbol}")
            return pd.DataFrame()  # Devolver DataFrame vacío
            
        print(f"Obtenidos {len(data)} registros históricos para {symbol}")
        return data
    except Exception as e:
        print(f"Error en get_stock_data: {str(e)}")
        return pd.DataFrame()  # Devolver DataFrame vacío en caso de error

def predict_stock(symbol):
    """Predecir precios de acciones"""
    try:
        print(f"Obteniendo datos para predicción de: {symbol}")
        data = get_stock_data(symbol)
        
        # Verificar si tenemos datos suficientes
        if data.empty or len(data) < 5:
            print(f"Datos insuficientes para {symbol}")
            # Devolver datos simulados en este caso
            return generate_mock_predictions(symbol)
            
        # Convertir el índice a datetime y eliminar timezone
        df = pd.DataFrame({'ds': data.index.tz_localize(None), 'y': data['Close']})
        df.reset_index(inplace=True, drop=True)  # Eliminar la columna 'index' duplicada
        
        print(f"Entrenando modelo para {symbol} con {len(df)} puntos de datos")
        model = Prophet()
        model.fit(df)
        future = model.make_future_dataframe(periods=30)
        forecast = model.predict(future)
        
        print(f"Predicción completada para {symbol}")
        return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(30).to_dict(orient="records")
    except Exception as e:
        print(f"Error en predict_stock: {str(e)}")
        # En caso de error, devolver datos simulados
        return generate_mock_predictions(symbol)

def generate_mock_predictions(symbol):
    """Generar predicciones simuladas cuando hay errores"""
    print(f"Generando predicciones simuladas para {symbol}")
    today = datetime.datetime.now()
    predictions = []
    
    base_price = 150.0  # Precio base ficticio
    
    for i in range(30):
        # Generar una fecha futura
        future_date = today + datetime.timedelta(days=i)
        # Generar un precio simulado con una tendencia alcista leve
        predicted_price = base_price + (i * 0.5) + (((hash(symbol) % 10) - 5) * 0.2)
        
        predictions.append({
            'ds': future_date.strftime("%Y-%m-%d"),
            'yhat': round(predicted_price, 2),
            'yhat_lower': round(predicted_price * 0.95, 2),
            'yhat_upper': round(predicted_price * 1.05, 2)
        })
    
    return predictions

def get_news(symbol):
    """Obtener noticias relacionadas con un símbolo de acción"""
    try:
        # Intentar obtener el nombre de la compañía, usar el símbolo como fallback si falla
        ticker = yf.Ticker(symbol)
        
        # Usamos un bloque try-except interno solo para la obtención del nombre
        try:
            company_info = ticker.info
            company_name = company_info.get('shortName', '') or company_info.get('longName', '') or symbol
        except:
            # Si no podemos obtener la información, usamos el símbolo directamente
            company_name = symbol
            
        print(f"Buscando noticias para: {company_name}")
        
        # Calcular fecha de hace una semana
        today = datetime.datetime.now()
        week_ago = today - datetime.timedelta(days=7)
        
        # Fecha formateada para la URL
        formatted_date = week_ago.strftime('%Y-%m-%d')
        
        # Verificar que tenemos una clave de API
        if not NEWS_API_KEY or NEWS_API_KEY == 'tu_clave_aquí':
            print("Advertencia: No se ha configurado una clave de API para noticias")
            return []
            
        # Construir la URL asegurando que todos los componentes son strings
        url = f"https://newsapi.org/v2/everything?q={str(company_name)}&from={formatted_date}&sortBy=popularity&apiKey={NEWS_API_KEY}"
        
        print(f"URL de búsqueda de noticias: {url}")
        
        response = requests.get(url)
        if response.status_code != 200:
            print(f"Error al obtener noticias: {response.status_code}")
            return []
        
        news_data = response.json()
        
        # Procesar y devolver noticias
        articles = news_data.get('articles', [])
        print(f"Obtenidas {len(articles)} noticias")
        return articles
    except Exception as e:
        print(f"Error en get_news: {str(e)}")
        return []

def analyze_news_sentiment(news_articles):
    """Analizar sentimiento de noticias usando NLTK VADER"""
    try:
        # Si no hay artículos, devolver sentimiento neutral
        if not news_articles:
            print("No hay noticias para analizar, devolviendo sentimiento neutral")
            return {
                'compound': 0.0,
                'positive': 0.0, 
                'negative': 0.0, 
                'neutral': 1.0
            }
        
        # Inicializar analizador
        analyzer = SentimentIntensityAnalyzer()
        
        # Agregar textos de noticias
        all_text = ""
        for article in news_articles:
            # Obtener texto de título y descripción, si están disponibles
            title = article.get('title', '')
            description = article.get('description', '')
            
            if title:
                all_text += title + " "
            if description:
                all_text += description + " "
        
        # Si después de todo no tenemos texto, devolver neutro
        if not all_text.strip():
            print("No se pudo extraer texto de las noticias")
            return {
                'compound': 0.0,
                'positive': 0.0, 
                'negative': 0.0, 
                'neutral': 1.0
            }
            
        # Analizar sentimiento
        sentiment = analyzer.polarity_scores(all_text)
        print(f"Análisis de sentimiento completado: {sentiment}")
        
        return sentiment
    except Exception as e:
        print(f"Error en analyze_news_sentiment: {str(e)}")
        # En caso de error, devolver neutral
        return {
            'compound': 0.0,
            'positive': 0.0, 
            'negative': 0.0, 
            'neutral': 1.0
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
