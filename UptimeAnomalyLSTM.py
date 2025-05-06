import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
from reportlab.lib.utils import ImageReader
import logging
import time
import traceback
from datetime import datetime
import os
import pickle
import json
import pyodbc
from sklearn.preprocessing import RobustScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Definisi konstanta global
SEQUENCE_LENGTH = 10
THRESHOLD_PERCENTILE = 95
MODEL_PATH = "LSTM_Model/uptime_lstm/model.keras"
SCALER_PATH = "LSTM_Model/uptime_lstm/scaler.pkl"
THRESHOLD_PATH = "LSTM_Model/uptime_lstm/threshold.pkl"

# Variabel global
g_model = None
g_scaler = RobustScaler()
g_reconstruction_error_threshold = None

def create_model(input_dim, sequence_length=SEQUENCE_LENGTH):
    """
    Membuat arsitektur model LSTM Autoencoder
    
    Args:
        input_dim: Dimensi input (jumlah fitur)
        sequence_length: Panjang sequence untuk input LSTM
        
    Returns:
        Model: Model LSTM Autoencoder
    """
    # Encoder
    encoder_inputs = Input(shape=(sequence_length, input_dim))
    encoder = LSTM(64, activation='relu', return_sequences=True)(encoder_inputs)
    encoder = Dropout(0.2)(encoder)
    encoder = LSTM(32, activation='relu', return_sequences=False)(encoder)
    encoder = Dropout(0.2)(encoder)
    
    # Representasi laten
    latent_dim = 16
    latent_representation = Dense(latent_dim)(encoder)
    
    # Decoder
    decoder = RepeatVector(sequence_length)(latent_representation)
    decoder = LSTM(32, activation='relu', return_sequences=True)(decoder)
    decoder = Dropout(0.2)(decoder)
    decoder = LSTM(64, activation='relu', return_sequences=True)(decoder)
    decoder = Dropout(0.2)(decoder)
    decoder_outputs = TimeDistributed(Dense(input_dim))(decoder)
    
    # Membuat model autoencoder
    model = Model(encoder_inputs, decoder_outputs)
    
    # Compile model
    model.compile(optimizer='adam', loss='mse')
    
    return model

def preprocess_data(df, features, sequence_length=SEQUENCE_LENGTH):
    """
    Preprocess data dan buat sequence untuk LSTM
    
    Args:
        df: DataFrame dengan data uptime
        features: List kolom fitur yang akan digunakan
        sequence_length: Panjang sequence untuk input LSTM
        
    Returns:
        X: Data sequence untuk input LSTM
        df_preprocessed: DataFrame setelah preprocessing
    """
    global g_scaler
    
    # Ambil fitur yang diperlukan
    df_features = df[features].copy()
    
    # Normalisasi fitur
    scaled_data = g_scaler.fit_transform(df_features)
    df_scaled = pd.DataFrame(scaled_data, columns=features)
    
    # Buat sequences untuk LSTM
    X = []
    for i in range(len(df_scaled) - sequence_length + 1):
        X.append(scaled_data[i:(i + sequence_length)])
    
    X = np.array(X)
    
    return X, df_scaled

def train_model(X, epochs=50, batch_size=32, sequence_length=SEQUENCE_LENGTH, threshold_percentile=THRESHOLD_PERCENTILE):
    """
    Melatih model LSTM Autoencoder
    
    Args:
        X: Data sequence untuk training
        epochs: Jumlah epoch untuk training
        batch_size: Ukuran batch
        sequence_length: Panjang sequence untuk input LSTM
        threshold_percentile: Persentil yang digunakan untuk menentukan threshold anomali
        
    Returns:
        history: Hasil training model
    """
    global g_model, g_reconstruction_error_threshold
    
    # Cek dimensi input
    _, timesteps, features = X.shape
    
    # Buat model jika belum ada
    if g_model is None:
        g_model = create_model(features, sequence_length)
    
    # Callback untuk early stopping dan model checkpoint
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    # Pastikan direktori model ada
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    
    model_checkpoint = ModelCheckpoint(
        filepath=MODEL_PATH,
        save_best_only=True,
        monitor='val_loss'
    )
    
    # Train model
    history = g_model.fit(
        X, X,  # Autoencoder: input = output
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        callbacks=[early_stopping, model_checkpoint],
        verbose=1
    )
    
    # Calculate reconstruction error on training data
    reconstructions = g_model.predict(X)
    mse = np.mean(np.square(X - reconstructions), axis=(1, 2))
    
    # Set threshold based on percentile of reconstruction errors
    g_reconstruction_error_threshold = np.percentile(mse, threshold_percentile)
    
    # Save scaler and threshold
    os.makedirs(os.path.dirname(SCALER_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(THRESHOLD_PATH), exist_ok=True)
    
    with open(SCALER_PATH, 'wb') as f:
        pickle.dump(g_scaler, f)
    
    with open(THRESHOLD_PATH, 'wb') as f:
        pickle.dump(g_reconstruction_error_threshold, f)
    
    logger.info(f"Model saved to {MODEL_PATH}")
    logger.info(f"Reconstruction error threshold: {g_reconstruction_error_threshold}")
    
    return history

def load_saved_model():
    """
    Load model, scaler, dan threshold yang sudah disimpan
    """
    global g_model, g_scaler, g_reconstruction_error_threshold
    
    g_model = load_model(MODEL_PATH)
    
    with open(SCALER_PATH, 'rb') as f:
        g_scaler = pickle.load(f)
    
    with open(THRESHOLD_PATH, 'rb') as f:
        g_reconstruction_error_threshold = pickle.load(f)
    
    logger.info("Model, scaler, dan threshold berhasil dimuat")

def detect_anomalies(df, features, sequence_length=SEQUENCE_LENGTH):
    """
    Deteksi anomali pada data baru
    
    Args:
        df: DataFrame dengan data uptime
        features: List kolom fitur yang akan digunakan
        sequence_length: Panjang sequence untuk input LSTM
        
    Returns:
        df: DataFrame dengan hasil deteksi anomali
    """
    global g_model, g_reconstruction_error_threshold
    
    # Preprocess data
    X, df_scaled = preprocess_data(df, features, sequence_length)
    
    # Get reconstructions
    reconstructions = g_model.predict(X)
    
    # Calculate reconstruction error (MSE)
    mse = np.mean(np.square(X - reconstructions), axis=(1, 2))
    
    # Create DataFrame with results
    # Pad with NaN values for the first (sequence_length-1) entries
    pad_length = sequence_length - 1
    reconstruction_errors = np.concatenate([np.array([np.nan] * pad_length), mse])
    
    # Add reconstruction error to the original DataFrame
    df['reconstruction_error'] = reconstruction_errors
    
    # Add anomaly flag based on threshold
    df['anomaly'] = df['reconstruction_error'] > g_reconstruction_error_threshold
    
    # Add anomaly score (normalized reconstruction error)
    max_error = df['reconstruction_error'].max()
    min_error = df['reconstruction_error'].min()
    if max_error > min_error:
        df['anomaly_score'] = (df['reconstruction_error'] - min_error) / (max_error - min_error)
    else:
        df['anomaly_score'] = 0
    
    # Remove NaN values (from padding)
    df = df.dropna()
    
    return df

def GlobalHandler(uptime_data):
    """
    Handler utama untuk deteksi anomali menggunakan LSTM Autoencoder
    
    Args:
        uptime_data: Data uptime dari database
        
    Returns:
        Dictionary dengan hasil deteksi anomali
    """
    try:
        start_time = time.time()
        
        logger.info("Memproses data uptime...")
        column_names = ['fw_days_uptime', 'fw_number_of_users', 
                   'fw_load_avg_1_min', 'fw_load_avg_5_min', 'fw_load_avg_15_min', 'created_at']
        
        # Convert pyodbc.Row objects to regular Python lists or tuples
        data_list = [tuple(row) for row in uptime_data]
        
        # Create DataFrame from the converted data
        df = pd.DataFrame(data_list, columns=column_names)
        
        print(df.shape)

        if df.empty:
            logger.warning("Tidak ada data uptime yang ditemukan")
            return {
                "status": "error",
                "message": "Tidak ada data uptime yang ditemukan",
                "normal_data": None,
                "anomalies": None,
                "visualizations": None
            }
        
        logger.info(f"Memproses {len(df)} data uptime")
        
        # Ensure created_at is datetime type
        if not pd.api.types.is_datetime64_any_dtype(df['created_at']):
            df['created_at'] = pd.to_datetime(df['created_at'])
        
        # Sort data by timestamp
        df = df.sort_values('created_at')
        
        # Define features for anomaly detection
        features = ['fw_days_uptime', 'fw_number_of_users', 
                   'fw_load_avg_1_min', 'fw_load_avg_5_min', 'fw_load_avg_15_min']
        
        # Handle missing values if any
        df[features] = df[features].fillna(df[features].mean())
        
        # Check if model exists, if not return error
        model_path = MODEL_PATH
        if os.path.exists(model_path):
            logger.info("Memuat model yang sudah ada...")
            load_saved_model()
        else:
            logger.error(f"Model tidak ditemukan di path: {model_path}")
            return {
                "status": "error",
                "message": f"Model LSTM tidak ditemukan di path: {model_path}. Pastikan model sudah dilatih terlebih dahulu.",
                "normal_data": None,
                "anomalies": None,
                "visualizations": None
            }
        
        # Detect anomalies
        df_with_anomalies = detect_anomalies(df, features)
        
        # Separate normal and anomaly data
        normal_data = df_with_anomalies[~df_with_anomalies['anomaly']].copy()
        anomalies = df_with_anomalies[df_with_anomalies['anomaly']].copy()
        
        # Sort anomalies by score (highest first)
        anomalies = anomalies.sort_values('anomaly_score', ascending=False)
        
        # Generate visualizations
        visualizations = create_visualizations(df_with_anomalies, normal_data, anomalies)
        
        # Get feature importance
        feature_importance = get_feature_importance(df_with_anomalies, anomalies, features)
        
        # Get anomaly insights
        insights = get_anomaly_insights(df_with_anomalies, anomalies, features)
        
        # Log execution time
        execution_time = time.time() - start_time
        logger.info(f"Deteksi anomali selesai dalam {execution_time:.2f} detik")
        logger.info(f"Ditemukan {len(anomalies)} anomali dari {len(df_with_anomalies)} data")
        
        # Prepare result
        result = {
            "status": "success",
            "execution_time": execution_time,
            "total_records": len(df_with_anomalies),
            "normal_count": len(normal_data),
            "anomaly_count": len(anomalies),
            "normal_data": normal_data.to_dict('records') if not normal_data.empty else [],
            "anomalies": anomalies.to_dict('records') if not anomalies.empty else [],
            "feature_importance": feature_importance.to_dict('records') if len(feature_importance) > 0 else [],
            "insights": insights,
            "visualizations": visualizations
        }
        
        return result
        
    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"Error dalam deteksi anomali: {str(e)}\n{tb}")
        return {
            "status": "error",
            "message": str(e),
            "traceback": tb,
            "normal_data": None,
            "anomalies": None,
            "visualizations": None
        }

# Fungsi get_feature_importance dan get_anomaly_insights tetap sama seperti sebelumnya
def get_feature_importance(df, anomalies, features):
    """
    Calculate importance of each feature in identifying anomalies
    
    Args:
        df: Full DataFrame with anomaly column
        anomalies: DataFrame containing only anomalies
        features: List of feature column names
    
    Returns:
        DataFrame with feature importance scores
    """
    if anomalies.empty:
        return pd.DataFrame()
        
    # Calculate the mean and std for each feature in normal data
    normal_df = df[~df['anomaly']]
    
    result = {}
    for feature in features:
        normal_mean = normal_df[feature].mean()
        normal_std = normal_df[feature].std() if len(normal_df) > 1 else 1.0
        
        # Calculate z-scores for anomalies
        if normal_std == 0:  # Avoid division by zero
            normal_std = 1.0
            
        anomalies_z = abs((anomalies[feature] - normal_mean) / normal_std)
        result[feature] = anomalies_z.mean() if not anomalies_z.empty else 0
    
    # Create and return a DataFrame sorted by importance
    importance_df = pd.DataFrame({
        'feature': list(result.keys()),
        'importance_score': list(result.values())
    })
    
    return importance_df.sort_values('importance_score', ascending=False)

def get_anomaly_insights(df, anomalies, features):
    """
    Generate insights about detected anomalies
    
    Args:
        df: Full DataFrame with anomaly column
        anomalies: DataFrame containing only anomalies
        features: List of feature column names
    
    Returns:
        Dictionary with insights about anomalies
    """
    if anomalies.empty:
        return {"summary": "Tidak ditemukan anomali"}
    
    # Calculate summary statistics
    normal_df = df[~df['anomaly']]
    
    insights = {
        "summary": f"Ditemukan {len(anomalies)} anomali dari total {len(df)} data",
        "metrics": {}
    }
    
    # Add insights for each feature
    for feature in features:
        normal_mean = normal_df[feature].mean()
        anomaly_mean = anomalies[feature].mean()
        
        # Determine if anomalies are higher or lower than normal
        if anomaly_mean > normal_mean:
            direction = "lebih tinggi"
            pct_diff = ((anomaly_mean - normal_mean) / normal_mean) * 100 if normal_mean != 0 else 0
        else:
            direction = "lebih rendah"
            pct_diff = ((normal_mean - anomaly_mean) / normal_mean) * 100 if normal_mean != 0 else 0
        
        insights["metrics"][feature] = {
            "normal_mean": normal_mean,
            "anomaly_mean": anomaly_mean,
            "direction": direction,
            "percent_difference": pct_diff,
            "description": f"Rata-rata {feature} pada data anomali {direction} {pct_diff:.1f}% dibanding normal"
        }
    
    # Find the most severe anomaly time periods
    if not anomalies.empty:
        top_anomalies = anomalies.sort_values('anomaly_score', ascending=False).head(3)
        insights["top_anomalies"] = []
        
        for _, row in top_anomalies.iterrows():
            anomaly_info = {
                "time": row['created_at'].strftime("%Y-%m-%d %H:%M:%S") if hasattr(row['created_at'], 'strftime') else str(row['created_at']),
                "score": row['anomaly_score'],
                "key_metrics": {}
            }
            
            # Identify the most unusual metrics for this anomaly
            for feature in features:
                normal_mean = normal_df[feature].mean()
                normal_std = normal_df[feature].std() if len(normal_df) > 1 else 1.0
                
                if normal_std == 0:
                    normal_std = 1.0
                
                z_score = abs((row[feature] - normal_mean) / normal_std)
                
                if z_score > 2:  # Only include significantly unusual metrics
                    anomaly_info["key_metrics"][feature] = {
                        "value": row[feature],
                        "z_score": z_score,
                        "normal_range": f"{normal_mean - 2*normal_std:.2f} - {normal_mean + 2*normal_std:.2f}"
                    }
            
            insights["top_anomalies"].append(anomaly_info)
    
    return insights

def create_visualizations(df, normal_data, anomalies):
    """
    Create visualization plots for uptime data analysis
    
    Args:
        df: Full DataFrame with anomaly column
        normal_data: DataFrame containing only normal points
        anomalies: DataFrame containing only anomalies
    
    Returns:
        Dictionary with visualization objects
    """
    visualizations = {}
    
    try:
        # Plot reconstruction errors
        plt.figure(figsize=(10, 6))
        plt.plot(df['created_at'], df['reconstruction_error'], 'b-', alpha=0.5)
        if not anomalies.empty:
            plt.scatter(anomalies['created_at'], anomalies['reconstruction_error'], c='red', label='Anomali')
        plt.axhline(y=df['reconstruction_error'].median(), color='green', linestyle='--', label='Median Error')
        plt.axhline(y=g_reconstruction_error_threshold, color='red', linestyle='--', label='Threshold')
        plt.title('Reconstruction Error dari LSTM Autoencoder')
        plt.ylabel('Reconstruction Error')
        plt.xlabel('Waktu')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save figure to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        img = ImageReader(buf)
        visualizations['reconstruction_error_plot'] = img
        plt.close()
        
        # Time series plot of load averages with anomalies
        plt.figure(figsize=(10, 6))
        plt.plot(normal_data['created_at'], normal_data['fw_load_avg_1_min'], 'b.', label='Normal 1 min', alpha=0.5)
        plt.plot(normal_data['created_at'], normal_data['fw_load_avg_5_min'], 'g.', label='Normal 5 min', alpha=0.5)
        plt.plot(normal_data['created_at'], normal_data['fw_load_avg_15_min'], 'c.', label='Normal 15 min', alpha=0.5)
        
        if not anomalies.empty:
            plt.plot(anomalies['created_at'], anomalies['fw_load_avg_1_min'], 'ro', label='Anomali 1 min')
            plt.plot(anomalies['created_at'], anomalies['fw_load_avg_5_min'], 'mo', label='Anomali 5 min')
            plt.plot(anomalies['created_at'], anomalies['fw_load_avg_15_min'], 'yo', label='Anomali 15 min')
        
        plt.title('Load Average dengan Deteksi Anomali')
        plt.ylabel('Load Average')
        plt.xlabel('Waktu')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save figure to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        img = ImageReader(buf)
        visualizations['load_avg_plot'] = img
        plt.close()
        
        # Number of users plot
        plt.figure(figsize=(10, 6))
        plt.plot(normal_data['created_at'], normal_data['fw_number_of_users'], 'b.', label='Normal', alpha=0.5)
        
        if not anomalies.empty:
            plt.plot(anomalies['created_at'], anomalies['fw_number_of_users'], 'ro', label='Anomali')
        
        plt.title('Jumlah Pengguna dengan Deteksi Anomali')
        plt.ylabel('Jumlah Pengguna')
        plt.xlabel('Waktu')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save figure to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        img = ImageReader(buf)
        visualizations['users_plot'] = img
        plt.close()
        
        # Uptime plot
        plt.figure(figsize=(10, 6))
        plt.plot(normal_data['created_at'], normal_data['fw_days_uptime'], 'b.', label='Normal', alpha=0.5)
        
        if not anomalies.empty:
            plt.plot(anomalies['created_at'], anomalies['fw_days_uptime'], 'ro', label='Anomali')
        
        plt.title('Uptime dengan Deteksi Anomali')
        plt.ylabel('Uptime (hari)')
        plt.xlabel('Waktu')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save figure to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        img = ImageReader(buf)
        visualizations['uptime_plot'] = img
        plt.close()
        
        # Create scatter plot of load_avg vs number_of_users
        plt.figure(figsize=(10, 6))
        plt.scatter(normal_data['fw_load_avg_5_min'], normal_data['fw_number_of_users'], 
                   c='blue', label='Normal', alpha=0.5)
        
        if not anomalies.empty:
            plt.scatter(anomalies['fw_load_avg_5_min'], anomalies['fw_number_of_users'], 
                        c='red', label='Anomali')
        
        plt.title('Load Average vs Jumlah Pengguna')
        plt.xlabel('Load Average (5 min)')
        plt.ylabel('Jumlah Pengguna')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save figure to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        img = ImageReader(buf)
        visualizations['scatter_plot'] = img
        plt.close()
        
        # Add 3D visualization
        if not anomalies.empty and len(df) > 10:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            # Plot normal data
            ax.scatter(normal_data['fw_load_avg_1_min'], 
                       normal_data['fw_load_avg_5_min'], 
                       normal_data['fw_number_of_users'], 
                       c='blue', marker='o', label='Normal', alpha=0.6)
            
            # Plot anomalies
            ax.scatter(anomalies['fw_load_avg_1_min'], 
                       anomalies['fw_load_avg_5_min'], 
                       anomalies['fw_number_of_users'], 
                       c='red', marker='*', s=100, label='Anomali')
            
            ax.set_xlabel('Load Avg (1 min)')
            ax.set_ylabel('Load Avg (5 min)')
            ax.set_zlabel('Jumlah Pengguna')
            ax.set_title('Visualisasi 3D Anomali')
            ax.legend()
            
            # Save figure to buffer
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100)
            buf.seek(0)
            img = ImageReader(buf)
            visualizations['3d_plot'] = img
            plt.close()
        
    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"Error dalam pembuatan visualisasi: {str(e)}\n{tb}")
        
    return visualizations