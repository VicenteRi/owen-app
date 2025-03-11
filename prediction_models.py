"""
Módulo de modelos de predicción avanzados para el backend.
Contiene implementaciones de varios algoritmos de predicción financiera.
"""

import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import yfinance as yf
import datetime
from datetime import timedelta
import joblib
import hashlib
from pathlib import Path
import os
import warnings
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('prediction_models')

# Ignorar advertencias para evitar salida excesiva
warnings.filterwarnings("ignore")

# Crear directorio cache si no existe
CACHE_DIR = Path("model_cache")
CACHE_DIR.mkdir(exist_ok=True)

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
                logger.info(f"Usando modelo {model_type} en caché para {symbol}")
                return cached_data
            except Exception as e:
                logger.error(f"Error cargando caché: {e}")
    
    return None

def save_model_cache(data, symbol, model_type):
    """Guarda el modelo en caché"""
    try:
        cache_path = get_model_cache_path(symbol, model_type)
        joblib.dump(data, cache_path)
        logger.info(f"Modelo {model_type} guardado en caché para {symbol}")
    except Exception as e:
        logger.error(f"Error guardando caché: {e}")

def add_technical_indicators(data):
    """Añade indicadores técnicos al DataFrame de precios"""
    if len(data) <= 5:
        return data
        
    # Crear una copia para evitar SettingWithCopyWarning
    df = data.copy()
    
    # Media móvil de 5 y 20 días
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    
    # Volatilidad (desviación estándar en una ventana)
    df['Volatility'] = df['Close'].rolling(window=20).std()
    
    # Momentum (cambio porcentual de n días)
    df['Momentum'] = df['Close'].pct_change(periods=5)
    
    # Volumen relativo (ratio respecto a la media de volumen)
    if 'Volume' in df.columns:
        df['RelVolume'] = df['Volume'] / df['Volume'].rolling(window=20).mean()
    
    # RSI (Relative Strength Index) simplificado
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    
    # Evitar división por cero
    avg_loss = avg_loss.replace(0, 0.001)
    
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD (Moving Average Convergence Divergence)
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # Limpiar NaNs resultantes de las ventanas móviles
    df = df.fillna(method='bfill').fillna(method='ffill')
    
    return df

def test_stationarity(timeseries):
    """Prueba si una serie de tiempo es estacionaria usando el test Augmented Dickey-Fuller"""
    try:
        # Test ADF
        result = adfuller(timeseries.values)
        p_value = result[1]
        
        # Si p-value es menor que 0.05, la serie es estacionaria
        return p_value < 0.05
    except Exception as e:
        logger.error(f"Error en test_stationarity: {e}")
        return False

def train_prophet_model(df, symbol, seasonality_mode='multiplicative'):
    """Entrena un modelo Prophet con optimizaciones"""
    # Verificar caché
    cached_forecast = check_model_cache(symbol, 'prophet')
    if cached_forecast is not None:
        return cached_forecast
    
    try:
        logger.info(f"Entrenando modelo Prophet para {symbol}")
        
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
        logger.error(f"Error en train_prophet_model: {e}")
        return None

def train_arima_model(data, symbol):
    """Entrena un modelo ARIMA para predicción"""
    # Verificar caché
    cached_forecast = check_model_cache(symbol, 'arima')
    if cached_forecast is not None:
        return cached_forecast
    
    try:
        logger.info(f"Entrenando modelo ARIMA para {symbol}")
        
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
        logger.error(f"Error en train_arima_model: {e}")
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
        logger.error(f"Error combinando predicciones: {e}")
        # En caso de error, devolver la predicción de Prophet si está disponible
        if prophet_forecast is not None:
            return prophet_forecast
        return arima_forecast

def generate_mock_predictions(symbol):
    """Genera predicciones simuladas cuando hay errores - Versión mejorada"""
    logger.info(f"Generando predicciones simuladas para {symbol}")
    today = datetime.datetime.now()
    predictions = []
    
    # Intentar obtener precio actual real
    try:
        stock = yf.Ticker(symbol)
        hist = stock.history(period="1d")
        if not hist.empty:
            base_price = float(hist['Close'].iloc[-1])
            logger.info(f"Usando precio actual real: {base_price} para simulación")
        else:
            raise ValueError("Sin datos actuales")
    except Exception as e:
        logger.warning(f"No se pudo obtener precio actual: {e}")
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
            logger.info(f"Tendencia reciente: {avg_change:.2f}%, volatilidad: {volatility:.2f}%")
            
            # Usar tendencia real con una suavización
            trend = avg_change * 0.7  # Suavizar para no exagerar
            
            # Volatilidad diaria basada en datos reales (con un mínimo razonable)
            daily_vol = max(0.5, volatility * 0.8)  # Al menos 0.5% de volatilidad
        else:
            raise ValueError("Datos históricos insuficientes")
    except Exception as e:
        logger.warning(f"Usando tendencia y volatilidad simuladas: {e}")
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
