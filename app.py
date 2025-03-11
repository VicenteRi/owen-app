from flask import Flask, jsonify, request
from flask_cors import CORS
import yfinance as yf
import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
import requests
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import datetime
from datetime import timedelta
import os
from finance_api import register_finance_routes
# Modelos de predicción adicionales
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
# Para paralelización
import multiprocessing
import warnings
# Para caché
import joblib
import hashlib
from pathlib import Path

# Ignorar advertencias para evitar salida excesiva
warnings.filterwarnings("ignore")

# Crear directorio cache si no existe
CACHE_DIR = Path("model_cache")
CACHE_DIR.mkdir(exist_ok=True)

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
        
        # Añadir características técnicas
        if len(data) > 5:
            # Media móvil de 5 y 20 días
            data['MA5'] = data['Close'].rolling(window=5).mean()
            data['MA20'] = data['Close'].rolling(window=20).mean()
            
            # Volatilidad (desviación estándar en una ventana)
            data['Volatility'] = data['Close'].rolling(window=20).std()
            
            # Momentum (cambio porcentual de n días)
            data['Momentum'] = data['Close'].pct_change(periods=5)
            
            # Volumen relativo (ratio respecto a la media de volumen)
            data['RelVolume'] = data['Volume'] / data['Volume'].rolling(window=20).mean()
            
            # RSI (Relative Strength Index) simplificado
            delta = data['Close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            
            # Evitar división por cero
            avg_loss = avg_loss.replace(0, 0.001)
            
            rs = avg_gain / avg_loss
            data['RSI'] = 100 - (100 / (1 + rs))
            
            # Limpiar NaNs resultantes de las ventanas móviles
            data = data.fillna(method='bfill').fillna(method='ffill')
        
        return data
    except Exception as e:
        print(f"Error en get_stock_data: {str(e)}")
        return pd.DataFrame()  # Devolver DataFrame vacío en caso de error

def get_model_cache_path(symbol, model_type):
    """Genera una ruta única para guardar el modelo en caché"""
    # Crear un hash para el símbolo y tipo de modelo
    hash_key = hashlib.md5(f"{symbol}_{model_type}".encode()).hexdigest()
    return CACHE_DIR / f"{symbol}_{model_type}_{hash_key}.joblib"

def check_model_cache(symbol, model_type, max_age_days=1):
    """Verifica si existe un modelo en caché y si aún es válido"""
    cache_path = get_model_cache_path(symbol, model_type)
    
    if cache_path.exists():
        # Verificar la fecha de última modificación
        mtime = cache_path.stat().st_mtime
        cache_date = datetime.datetime.fromtimestamp(mtime)
        max_age = datetime.datetime.now() - timedelta(days=max_age_days)
        
        # Si el caché es más reciente que max_age_days
        if cache_date > max_age:
            try:
                # Cargar el caché
                cached_data = joblib.load(cache_path)
                print(f"Usando modelo {model_type} en caché para {symbol}")
                return cached_data
            except Exception as e:
                print(f"Error cargando caché: {e}")
    
    return None

def save_model_cache(data, symbol, model_type):
    """Guarda el modelo en caché"""
    try:
        cache_path = get_model_cache_path(symbol, model_type)
        joblib.dump(data, cache_path)
        print(f"Modelo {model_type} guardado en caché para {symbol}")
    except Exception as e:
        print(f"Error guardando caché: {e}")

def test_stationarity(timeseries):
    """Prueba si una serie de tiempo es estacionaria usando el test Augmented Dickey-Fuller"""
    try:
        # Test ADF
        result = adfuller(timeseries.values)
        p_value = result[1]
        
        # Si p-value es menor que 0.05, la serie es estacionaria
        return p_value < 0.05
    except Exception as e:
        print(f"Error en test_stationarity: {e}")
        return False

def train_prophet_model(df, symbol, seasonality_mode='multiplicative'):
    """Entrenar modelo Prophet con optimizaciones"""
    # Verificar caché
    cached_forecast = check_model_cache(symbol, 'prophet')
    if cached_forecast is not None:
        return cached_forecast
    
    try:
        print(f"Entrenando modelo Prophet para {symbol}")
        
        # Configurar modelo con parámetros mejorados
        model = Prophet(
            seasonality_mode=seasonality_mode,  # Modo de estacionalidad
            changepoint_prior_scale=0.05,       # Control de flexibilidad para cambios de tendencia
            seasonality_prior_scale=10.0,       # Peso para la estacionalidad 
            daily_seasonality=False,            # No usar estacionalidad diaria para datos financieros
            weekly_seasonality=True,            # Usar estacionalidad semanal (patrones comerciales)
            yearly_seasonality=True             # Usar estacionalidad anual
        )
        
        # Añadir estacionalidades específicas del mercado si tenemos suficientes datos
        if len(df) > 100:
            # Estacionalidad mensual (efecto fin de mes, anuncios, etc.)
            model.add_seasonality(
                name='monthly',
                period=30.5,
                fourier_order=5
            )
            
            # Estacionalidad trimestral (reportes trimestrales)
            model.add_seasonality(
                name='quarterly',
                period=91.25,
                fourier_order=5
            )
        
        # Ajustar modelo
        model.fit(df)
        
        # Generar predicciones futuras
        future = model.make_future_dataframe(periods=30)
        forecast = model.predict(future)
        
        # Guardar en caché
        forecast_result = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(30).to_dict(orient="records")
        save_model_cache(forecast_result, symbol, 'prophet')
        
        return forecast_result
    except Exception as e:
        print(f"Error en train_prophet_model: {e}")
        return None

def train_arima_model(data, symbol):
    """Entrena un modelo ARIMA para predicción"""
    # Verificar caché
    cached_forecast = check_model_cache(symbol, 'arima')
    if cached_forecast is not None:
        return cached_forecast
    
    try:
        print(f"Entrenando modelo ARIMA para {symbol}")
        
        # Preparar datos
        y = data['y'].values
        
        # Verificar estacionaridad
        is_stationary = test_stationarity(data['y'])
        d_param = 0 if is_stationary else 1
        
        # Parámetros ARIMA (p,d,q)
        # p: orden autoregresivo
        # d: grado de diferenciación 
        # q: orden de media móvil
        p, q = 5, 1
        
        # Ajustar modelo ARIMA
        model = ARIMA(y, order=(p, d_param, q))
        model_fit = model.fit()
        
        # Predecir 30 días hacia el futuro
        forecast_steps = 30
        forecast = model_fit.forecast(steps=forecast_steps)
        
        # Generar dataframe con formato similar a Prophet
        last_date = data['ds'].iloc[-1]
        forecast_dates = [last_date + datetime.timedelta(days=i+1) for i in range(forecast_steps)]
        
        # Calcular intervalos de confianza
        # Usamos la desviación estándar de los datos históricos para estimar la incertidumbre
        std_dev = np.std(y)
        
        # Crear objeto de resultado en formato compatible
        result = []
        for i, date in enumerate(forecast_dates):
            pred = float(forecast[i])
            # Asegurar que las predicciones no sean negativas
            pred = max(pred, 0.01)
            
            # Intervalos de confianza
            confidence_interval = std_dev * (1 + i * 0.05)  # Aumentar con el tiempo
            
            result.append({
                'ds': date.strftime('%Y-%m-%d'),
                'yhat': round(pred, 2),
                'yhat_lower': round(max(pred - confidence_interval, 0.01), 2),
                'yhat_upper': round(pred + confidence_interval, 2)
            })
        
        # Guardar en caché
        save_model_cache(result, symbol, 'arima')
        
        return result
    except Exception as e:
        print(f"Error en train_arima_model: {e}")
        return None

def combine_forecasts(prophet_forecast, arima_forecast, ensemble_weights=None):
    """Combina las predicciones de diferentes modelos usando un enfoque de ponderación"""
    if prophet_forecast is None and arima_forecast is None:
        return None
    
    # Usar pesos por defecto si no se proporcionan
    if ensemble_weights is None:
        # Dar más peso a Prophet en general
        ensemble_weights = {'prophet': 0.7, 'arima': 0.3}
    
    try:
        # Si solo tenemos una predicción, devolver esa
        if prophet_forecast is None:
            return arima_forecast
        if arima_forecast is None:
            return prophet_forecast
        
        # Convertir a dataframes para facilitar la combinación
        df_prophet = pd.DataFrame(prophet_forecast)
        df_arima = pd.DataFrame(arima_forecast)
        
        # Asegurar que las fechas coincidan
        df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])
        df_arima['ds'] = pd.to_datetime(df_arima['ds'])
        
        # Combinar por fecha
        df_result = df_prophet.merge(
            df_arima, 
            on='ds', 
            suffixes=('_prophet', '_arima')
        )
        
        # Combinar predicciones con ponderación
        df_result['yhat'] = (
            df_result['yhat_prophet'] * ensemble_weights['prophet'] + 
            df_result['yhat_arima'] * ensemble_weights['arima']
        )
        
        # Ajustar los intervalos de predicción
        df_result['yhat_lower'] = (
            df_result['yhat_lower_prophet'] * ensemble_weights['prophet'] + 
            df_result['yhat_lower_arima'] * ensemble_weights['arima']
        )
        
        df_result['yhat_upper'] = (
            df_result['yhat_upper_prophet'] * ensemble_weights['prophet'] + 
            df_result['yhat_upper_arima'] * ensemble_weights['arima']
        )
        
        # Reformatear para devolver
        result = df_result[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_dict(orient="records")
        
        # Convertir fechas a string para JSON
        for item in result:
            item['ds'] = item['ds'].strftime('%Y-%m-%d')
            item['yhat'] = round(item['yhat'], 2)
            item['yhat_lower'] = round(item['yhat_lower'], 2)
            item['yhat_upper'] = round(item['yhat_upper'], 2)
        
        return result
    except Exception as e:
        print(f"Error combinando predicciones: {e}")
        # En caso de error, devolver la predicción de Prophet si está disponible
        if prophet_forecast is not None:
            return prophet_forecast
        return arima_forecast

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

def generate_mock_predictions(symbol):
    """Generar predicciones simuladas cuando hay errores - Versión mejorada"""
    print(f"Generando predicciones simuladas para {symbol}")
    today = datetime.datetime.now()
    predictions = []
    
    # Intentar obtener precio actual real
    try:
        stock = yf.Ticker(symbol)
        hist = stock.history(period="1d")
        if not hist.empty:
            base_price = float(hist['Close'].iloc[-1])
            print(f"Usando precio actual real: {base_price} para simulación")
        else:
            raise ValueError("Sin datos actuales")
    except Exception as e:
        print(f"No se pudo obtener precio actual: {e}")
        # Generar un precio base usando el hash del símbolo para que sea consistente
        seed = sum(ord(c) for c in symbol)
        base_price = 100.0 + (seed % 200)  # Precio base entre 100 y 300
    
    # Analizar tendencias recientes para hacer simulación más realista
    try:
        # Obtener datos de los últimos 30 días
        recent_data = stock.history(period="1mo")
        if len(recent_data) > 5:
            # Calcular tendencia reciente (cambio porcentual promedio)
            changes = recent_data['Close'].pct_change().dropna()
            avg_change = changes.mean() * 100  # En porcentaje
            
            # Calcular volatilidad
            volatility = changes.std() * 100  # En porcentaje
            
            # Ajustar simulación con datos reales
            print(f"Tendencia reciente: {avg_change:.2f}%, volatilidad: {volatility:.2f}%")
            
            # Usar tendencia real con una suavización
            trend = avg_change * 0.7  # Suavizar para no exagerar
            
            # Volatilidad diaria basada en datos reales (con un mínimo razonable)
            daily_vol = max(0.5, volatility * 0.8)  # Al menos 0.5% de volatilidad
        else:
            raise ValueError("Datos históricos insuficientes")
    except Exception as e:
        print(f"Usando tendencia y volatilidad simuladas: {e}")
        # Tendencia diaria (entre -0.5% y +1.5%)
        seed = sum(ord(c) for c in symbol)
        trend = 0.5 + (seed % 10) / 10.0  # Entre 0.5% y 1.5%
        if seed % 3 == 0:  # Un tercio de las veces, tendencia negativa
            trend = -trend / 2  # Tendencia negativa más suave
        
        # Volatilidad diaria
        daily_vol = 0.8  # 0.8% por defecto
    
    # Crear ciclos en los datos simulados para mayor realismo
    # Frecuencias de ciclos
    cycles = []
    cycle_count = 2 + (sum(ord(c) for c in symbol) % 3)  # 2-4 ciclos
    
    for i in range(cycle_count):
        # Periodo del ciclo (entre 5 y 15 días)
        period = 5 + (ord(symbol[i % len(symbol)]) % 10)
        # Amplitud (entre 0.5% y 2% del precio)
        amplitude = base_price * (0.005 + (ord(symbol[i % len(symbol)]) % 15) / 1000)
        # Fase inicial
        phase = (ord(symbol[i % len(symbol)]) % 100) / 100 * 2 * np.pi
        
        cycles.append({
            'period': period,
            'amplitude': amplitude,
            'phase': phase
        })
    
    current_price = base_price
    
    for i in range(30):
        # Generar una fecha futura
        future_date = today + datetime.timedelta(days=i)
        
        # Efecto tendencia
        trend_effect = trend * i / 100  # Efecto acumulativo
        
        # Efecto ciclos
        cycle_effect = 0
        for cycle in cycles:
            # Efecto sinusoidal
            t = i / cycle['period']
            cycle_effect += cycle['amplitude'] * np.sin(2 * np.pi * t + cycle['phase'])
        
        # Efecto memoria (autocorrelación)
        memory_effect = 0
        if i > 0:
            # El precio reciente influye en el siguiente (efecto memoria)
            price_diff = predictions[i-1]['yhat'] - base_price
            memory_effect = price_diff * 0.2  # 20% de la diferencia anterior
        
        # Calcular variación diaria (ruido aleatorio basado en volatilidad)
        noise = current_price * (daily_vol / 100) * ((hash(f"{symbol}_{i}") % 100) / 50 - 1)
        
        # Calcular nuevo precio combinando todos los efectos
        if i > 0:
            # Usar precio anterior como base para el siguiente
            base_for_calc = predictions[i-1]['yhat']
            # Añadir efectos
            current_price = base_for_calc * (1 + trend_effect) + cycle_effect + memory_effect + noise
        else:
            # Para el primer día, usar el precio base
            current_price = base_price * (1 + trend_effect) + cycle_effect + noise
        
        # Asegurar que el precio no sea negativo
        current_price = max(current_price, 0.01)
        
        # Calcular límites inferior y superior (más amplios con el tiempo)
        confidence_interval = (daily_vol / 100) * current_price * (1 + i * 0.05)
        lower_bound = current_price - confidence_interval
        upper_bound = current_price + confidence_interval
        
        # Crear predicción
        prediction = {
            'ds': future_date.strftime("%Y-%m-%d"),
            'yhat': round(current_price, 2),
            'yhat_lower': round(max(lower_bound, 0.01), 2),
            'yhat_upper': round(upper_bound, 2)
        }
        
        # Añadir porcentajes de cambio
        if i == 0:
            prediction['pct_change'] = 0.0
            prediction['daily_pct'] = 0.0
        else:
            # Cambio desde el precio base
            pct_change = ((current_price - base_price) / base_price) * 100
            prediction['pct_change'] = round(pct_change, 2)
            
            # Cambio diario
            daily_pct = ((current_price - predictions[i-1]['yhat']) / predictions[i-1]['yhat']) * 100
            prediction['daily_pct'] = round(daily_pct, 2)
        
        predictions.append(prediction)
    
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
