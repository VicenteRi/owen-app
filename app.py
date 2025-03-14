from flask import Flask, jsonify, request
from flask_cors import CORS
import yfinance as yf
import pandas as pd
import numpy as np
from prophet import Prophet
import requests
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import datetime
import os
from finance_api import register_finance_routes
import multiprocessing
# Importar módulo de predicción personalizado
from prediction_models import (
    get_model_cache_path, 
    check_model_cache, 
    save_model_cache,
    add_technical_indicators,
    test_stationarity,
    train_prophet_model,
    train_arima_model,
    combine_forecasts,
    generate_mock_predictions
)

# Descargar recursos necesarios para NLTK
nltk.download('vader_lexicon', quiet=True)

# Configuración
NEWS_API_KEY = os.environ.get('NEWS_API_KEY', 'tu_clave_aquí')  # Obtener de variables de entorno o usar valor predeterminado

app = Flask(__name__)
CORS(app)  # Habilitar CORS para todas las rutas

# Register financial routes
register_finance_routes(app)

def get_stock_data(symbol, period="6mo"):
    """Obtener datos históricos de acciones"""
    try:
        print(f"Obteniendo datos históricos para: {symbol} con periodo {period}")
        stock = yf.Ticker(symbol)
        
        # Intentar con periodos más largos para entrenar mejor el modelo
        data = stock.history(period=period)
        
        # Si los datos son insuficientes, intentar con más datos
        if len(data) < 60 and period == "6mo":
            print(f"Datos insuficientes, intentando con periodo más largo")
            data = stock.history(period="1y")
            
        if len(data) < 90 and period == "1y":
            print(f"Datos insuficientes, intentando con periodo más largo")
            data = stock.history(period="2y")
        
        # Verificar si obtuvimos datos
        if data.empty:
            print(f"No se encontraron datos históricos para {symbol}")
            return pd.DataFrame()  # Devolver DataFrame vacío
            
        print(f"Obtenidos {len(data)} registros históricos para {symbol}")
        
        # Añadir indicadores técnicos usando nuestra función
        data = add_technical_indicators(data)
        
        return data
    except Exception as e:
        print(f"Error en get_stock_data: {str(e)}")
        return pd.DataFrame()  # Devolver DataFrame vacío en caso de error

def predict_stock(symbol):
    """Predecir precios de acciones con múltiples modelos avanzados"""
    try:
        print(f"Obteniendo datos para predicción de: {symbol}")
        # Obtener más datos históricos para mejorar la predicción
        data = get_stock_data(symbol, period="1y")
        
        # Verificar si tenemos datos suficientes
        if data.empty or len(data) < 30:
            print(f"Datos insuficientes para {symbol} ({len(data) if not data.empty else 0} puntos)")
            # Intentar con más datos
            data = get_stock_data(symbol, period="2y")
            
            if data.empty or len(data) < 30:
                print("Aún insuficiente, usando datos simulados")
                return generate_mock_predictions(symbol)
        
        # Preparar datos para Prophet
        df = pd.DataFrame({
            'ds': data.index.tz_localize(None),
            'y': data['Close']
        })
        df.reset_index(inplace=True, drop=True)
        
        # Ejecutar ambos modelos en paralelo usando multiprocessing
        with multiprocessing.Pool(processes=2) as pool:
            # Iniciar ambos procesos de entrenamiento
            prophet_result = pool.apply_async(train_prophet_model, (df, symbol))
            arima_result = pool.apply_async(train_arima_model, (df, symbol))
            
            # Esperar resultados
            prophet_forecast = None
            arima_forecast = None
            
            try:
                prophet_forecast = prophet_result.get(timeout=60)  # 60 segundos timeout
                print(f"Prophet completado para {symbol}")
            except Exception as e:
                print(f"Error en Prophet: {e}")
            
            try:
                arima_forecast = arima_result.get(timeout=60)  # 60 segundos timeout
                print(f"ARIMA completado para {symbol}")
            except Exception as e:
                print(f"Error en ARIMA: {e}")
        
        # Combinar predicciones
        ensemble_result = combine_forecasts(prophet_forecast, arima_forecast)
        
        # Si todo falla, usar datos simulados
        if ensemble_result is None:
            print("Error en todos los modelos, usando datos simulados")
            return generate_mock_predictions(symbol)
        
        print(f"Predicción de ensamble completada para {symbol}")
        return ensemble_result
    except Exception as e:
        print(f"Error en predict_stock: {str(e)}")
        # En caso de error, devolver datos simulados
        return generate_mock_predictions(symbol)

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
                'positive': 0.33, 
                'negative': 0.33, 
                'neutral': 0.34
            }
        
        # Inicializar analizador
        analyzer = SentimentIntensityAnalyzer()
        
        # Recopilar sentimientos de cada artículo individualmente
        all_sentiments = []
        for article in news_articles:
            # Obtener texto de título y descripción, si están disponibles
            title = article.get('title', '')
            description = article.get('description', '')
            
            text = ""
            if title:
                text += title + ". "
            if description:
                text += description
                
            # Solo analizar si hay texto
            if text.strip():
                sentiment = analyzer.polarity_scores(text)
                all_sentiments.append(sentiment)
        
        # Si no se pudo analizar ningún artículo
        if not all_sentiments:
            print("No se pudo analizar ningún artículo")
            return {
                'compound': 0.0,
                'positive': 0.33, 
                'negative': 0.33, 
                'neutral': 0.34
            }
            
        # Calcular promedio de sentimientos
        compound = sum(s['compound'] for s in all_sentiments) / len(all_sentiments)
        positive = sum(s['pos'] for s in all_sentiments) / len(all_sentiments)
        negative = sum(s['neg'] for s in all_sentiments) / len(all_sentiments)
        neutral = sum(s['neu'] for s in all_sentiments) / len(all_sentiments)
        
        # Balancear el sentimiento compound (no dejarlo tan extremo)
        # Escalar el valor compound para que no sea tan extremo (1.0)
        if compound > 0.3:
            # Reducir valores muy positivos
            compound = 0.1 + (compound * 0.5)
        elif compound < -0.3:
            # Reducir valores muy negativos
            compound = -0.1 + (compound * 0.5)
        
        # Asegurar que compound está en el rango [-1, 1]
        compound = max(-1.0, min(1.0, compound))
        
        print(f"Análisis de sentimiento ajustado: compound={compound}, pos={positive}, neg={negative}, neu={neutral}")
        
        # Devolver con los nombres que espera el frontend
        return {
            'compound': compound,
            'positive': positive, 
            'negative': negative, 
            'neutral': neutral
        }
    except Exception as e:
        print(f"Error en analyze_news_sentiment: {str(e)}")
        # En caso de error, devolver neutral balanceado
        return {
            'compound': 0.0,
            'positive': 0.33, 
            'negative': 0.33, 
            'neutral': 0.34
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
        
        # Calcular porcentajes de cambio para evitar NaN%
        if price_predictions and len(price_predictions) > 1:
            # Obtener precios actuales para referencia
            try:
                current_price = float(yf.Ticker(symbol).history(period="1d")['Close'].iloc[-1])
                print(f"Precio actual de {symbol}: {current_price}")
            except Exception as e:
                print(f"Error al obtener precio actual: {str(e)}")
                # Usar el primer precio de predicción como referencia si no podemos obtener el actual
                current_price = float(price_predictions[0]['yhat'])
            
            # Añadir campos de porcentaje de cambio a cada predicción
            for i, pred in enumerate(price_predictions):
                # Agregar porcentaje de cambio con respecto al precio actual
                try:
                    pred_price = float(pred['yhat'])
                    pct_change = ((pred_price - current_price) / current_price) * 100
                    pred['pct_change'] = round(pct_change, 2)
                    
                    # También calcular el cambio respecto al día anterior en la predicción
                    if i > 0:
                        prev_price = float(price_predictions[i-1]['yhat'])
                        daily_pct = ((pred_price - prev_price) / prev_price) * 100
                        pred['daily_pct'] = round(daily_pct, 2)
                    else:
                        # Para el primer día, usar el cambio respecto al precio actual
                        pred['daily_pct'] = round(pct_change, 2)
                except Exception as e:
                    print(f"Error al calcular porcentajes para predicción {i}: {str(e)}")
                    pred['pct_change'] = 0.0
                    pred['daily_pct'] = 0.0
        
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

@app.route('/current-price', methods=['GET'])
def current_price():
    """Endpoint para obtener el precio actual de una acción"""
    try:
        symbol = request.args.get('symbol', 'AAPL')
        
        # Obtener datos del mercado
        stock = yf.Ticker(symbol)
        
        # Obtener el precio de cierre más reciente
        hist = stock.history(period="1d")
        
        if hist.empty:
            return jsonify({"error": "No se pudieron obtener datos para este símbolo"}), 404
            
        current_price = float(hist['Close'].iloc[-1])
        
        # Obtener datos adicionales si están disponibles
        try:
            info = stock.info
            company_name = info.get('shortName', symbol)
            change = info.get('regularMarketChangePercent', 0.0)
            previous_close = info.get('regularMarketPreviousClose', 0.0)
            market_cap = info.get('marketCap', 0)
            volume = info.get('volume', 0)
        except:
            company_name = symbol
            change = 0.0
            previous_close = 0.0
            market_cap = 0
            volume = 0
        
        return jsonify({
            "symbol": symbol,
            "price": current_price,
            "change_percent": change,
            "previous_close": previous_close,
            "company_name": company_name,
            "market_cap": market_cap,
            "volume": volume
        })
        
    except Exception as e:
        print(f"Error obteniendo precio actual: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/search', methods=['GET'])
def search_symbols():
    """Endpoint para buscar símbolos de acciones"""
    try:
        # Obtener el término de búsqueda
        query = request.args.get('query', '')
        print(f"Buscando símbolos con consulta: {query}")
        
        if not query:
            # Si no hay consulta, devolver algunas acciones populares
            popular_stocks = [
                {"symbol": "AAPL", "name": "Apple Inc."},
                {"symbol": "MSFT", "name": "Microsoft Corporation"},
                {"symbol": "GOOGL", "name": "Alphabet Inc."},
                {"symbol": "AMZN", "name": "Amazon.com, Inc."},
                {"symbol": "TSLA", "name": "Tesla, Inc."},
                {"symbol": "META", "name": "Meta Platforms, Inc."},
                {"symbol": "NVDA", "name": "NVIDIA Corporation"},
                {"symbol": "NFLX", "name": "Netflix, Inc."},
                {"symbol": "JPM", "name": "JPMorgan Chase & Co."},
                {"symbol": "DIS", "name": "The Walt Disney Company"}
            ]
            return jsonify(popular_stocks)
        
        # Realizar búsqueda usando yfinance
        import yfinance as yf
        
        # Usar tickers.search() si está disponible, de lo contrario usar un enfoque personalizado
        try:
            # Intento usar método moderno de búsqueda (si está disponible)
            tickers = yf.Tickers('')
            results = tickers.tickers
            search_results = []
            
            # Filtrar resultados que coincidan con la consulta
            for symbol, ticker in results.items():
                try:
                    info = ticker.info
                    if 'shortName' in info and (
                        query.lower() in symbol.lower() or 
                        query.lower() in info['shortName'].lower()
                    ):
                        search_results.append({
                            "symbol": symbol,
                            "name": info.get('shortName', symbol)
                        })
                except:
                    continue
                
                # Limitar a 20 resultados
                if len(search_results) >= 20:
                    break
            
            return jsonify(search_results)
        
        except Exception as inner_e:
            print(f"Método de búsqueda avanzado falló: {str(inner_e)}")
            # Método alternativo: buscar símbolos comunes que coincidan con la consulta
            common_symbols = {
                "AAPL": "Apple Inc.",
                "MSFT": "Microsoft Corporation",
                "GOOGL": "Alphabet Inc.",
                "AMZN": "Amazon.com, Inc.",
                "TSLA": "Tesla, Inc.",
                "META": "Meta Platforms, Inc.",
                "NVDA": "NVIDIA Corporation",
                "NFLX": "Netflix, Inc.",
                "JPM": "JPMorgan Chase & Co.",
                "DIS": "The Walt Disney Company",
                "V": "Visa Inc.",
                "PG": "Procter & Gamble Co.",
                "JNJ": "Johnson & Johnson",
                "KO": "The Coca-Cola Company",
                "CSCO": "Cisco Systems, Inc.",
                "MCD": "McDonald's Corporation",
                "ADBE": "Adobe Inc.",
                "INTC": "Intel Corporation",
                "IBM": "International Business Machines",
                "PYPL": "PayPal Holdings, Inc.",
                "BA": "Boeing Company",
                "GE": "General Electric Company",
                "VZ": "Verizon Communications Inc.",
                "T": "AT&T Inc.",
                "WMT": "Walmart Inc.",
                "TGT": "Target Corporation",
                "F": "Ford Motor Company",
                "GM": "General Motors Company",
                "NKE": "Nike, Inc.",
                "HD": "Home Depot, Inc."
            }
            
            results = []
            for symbol, name in common_symbols.items():
                if query.lower() in symbol.lower() or query.lower() in name.lower():
                    results.append({"symbol": symbol, "name": name})
            
            return jsonify(results)
            
    except Exception as e:
        print(f"Error en búsqueda de símbolos: {str(e)}")
        return jsonify([]), 500

if __name__ == '__main__':
    # Modo desarrollo local
    app.run(debug=True, host='0.0.0.0')
    
# Esto es necesario para Render/Gunicorn
# No eliminar esta variable
application = app 
