import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import io
from reportlab.lib.utils import ImageReader
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def GlobalHandler(uptime_data, contamination=0.05):
    """
    Deteksi anomali pada data uptime firewall menggunakan Isolation Forest.
    
    Parameters:
    -----------
    uptime_data : list of tuples
        Data uptime dari query SQL yang berisi:
        (current_time, days_uptime, uptime, number_of_user, load_avg_1, load_avg_5, load_avg_15)
    contamination : float, default=0.05
        Proporsi estimasi outlier dalam dataset
    
    Returns:
    --------
    dict
        Hasil deteksi anomali dan visualisasi
    """
    try:
        start_time = time.time()
        
        # Convert data to DataFrame
        logger.info("Memproses data uptime...")
        column_names = ['fw_days_uptime', 'fw_number_of_users', 
                   'fw_load_avg_1','fw_load_avg_5', 'fw_load_avg_15', 'created_at']
        
        df = pd.DataFrame(uptime_data, columns=column_names)
        
        # Check if we have data
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
        
        # Preprocess data
        # Convert current_time to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(df['created_at']):
            df['created_at'] = pd.to_datetime(df['created_at'])
        
        # Select only numerical features for anomaly detection
        features = ['fw_days_uptime', 'fw_number_of_users', 
                   'fw_load_avg_1_min','fw_load_avg_5_min', 'fw_load_avg_15_min']
        X = df[features]
        
        # Handle missing values if any
        X = X.fillna(X.mean())
        
        # Apply Isolation Forest
        logger.info("Menjalankan algoritma Isolation Forest...")
        model = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_jobs=-1  # Use all available cores
        )
        
        # Train and predict
        df['anomaly'] = model.fit_predict(X)
        
        # Convert predictions: -1 for anomalies, 1 for normal data points
        # Convert to boolean: True for anomalies, False for normal
        df['anomaly'] = df['anomaly'] == -1
        
        # Calculate anomaly score (higher score means more anomalous)
        df['anomaly_score'] = -model.score_samples(X)
        
        # Separate normal and anomaly data
        normal_data = df[~df['anomaly']].copy()
        anomalies = df[df['anomaly']].copy()
        
        # Sort anomalies by score (highest first)
        anomalies = anomalies.sort_values('anomaly_score', ascending=False)
        
        # Create visualizations
        # visualizations = create_visualizations(df, normal_data, anomalies)
        
        # Get feature importance
        feature_importance = get_feature_importance(df, anomalies, features)
        
        # Get anomaly insights
        insights = get_anomaly_insights(df, anomalies, features)
        
        # Log execution time
        execution_time = time.time() - start_time
        logger.info(f"Deteksi anomali selesai dalam {execution_time:.2f} detik")
        logger.info(f"Ditemukan {len(anomalies)} anomali dari {len(df)} data")
        
        # Prepare result
        result = {
            "status": "success",
            "execution_time": execution_time,
            "total_records": len(df),
            "normal_count": len(normal_data),
            "anomaly_count": len(anomalies),
            "normal_data": normal_data.to_dict('records') if not normal_data.empty else [],
            "anomalies": anomalies.to_dict('records') if not anomalies.empty else [],
            "feature_importance": feature_importance.to_dict('records') if len(feature_importance) > 0 else [],
            "insights": insights,
            # "visualizations": visualizations
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Error dalam deteksi anomali: {str(e)}")
        return {
            "status": "error",
            "message": str(e),
            "normal_data": None,
            "anomalies": None,
            "visualizations": None
        }

def get_feature_importance(df, anomalies, features):
    """
    Analisis fitur yang paling berkontribusi pada anomali.
    
    Parameters:
    -----------
    df : DataFrame
        DataFrame asli dengan semua data
    anomalies : DataFrame
        DataFrame yang hanya berisi anomali
    features : list
        Daftar nama fitur yang digunakan untuk deteksi anomali
    
    Returns:
    --------
    DataFrame
        Skor kepentingan fitur
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
    Menghasilkan insight dari anomali yang terdeteksi.
    
    Parameters:
    -----------
    df : DataFrame
        DataFrame asli dengan semua data
    anomalies : DataFrame
        DataFrame yang hanya berisi anomali
    features : list
        Daftar nama fitur yang digunakan untuk deteksi anomali
    
    Returns:
    --------
    dict
        Wawasan tentang anomali
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
    if not anomalies.empty and 'current_time' in anomalies.columns:
        top_anomalies = anomalies.sort_values('anomaly_score', ascending=False).head(3)
        insights["top_anomalies"] = []
        
        for _, row in top_anomalies.iterrows():
            anomaly_info = {
                "time": row['current_time'].strftime("%Y-%m-%d %H:%M:%S") if hasattr(row['current_time'], 'strftime') else str(row['current_time']),
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
    Membuat visualisasi untuk data normal dan anomali.
    
    Parameters:
    -----------
    df : DataFrame
        DataFrame lengkap dengan kolom anomaly
    normal_data : DataFrame
        Data yang diklasifikasikan sebagai normal
    anomalies : DataFrame
        Data yang diklasifikasikan sebagai anomali
    
    Returns:
    --------
    dict
        Dictionary berisi objek ImageReader dari visualisasi
    """
    visualizations = {}
    
    try:
        # Time series plot of load averages with anomalies
        plt.figure(figsize=(10, 6))
        plt.plot(normal_data['current_time'], normal_data['load_avg_1'], 'b.', label='Normal 1 min', alpha=0.5)
        plt.plot(normal_data['current_time'], normal_data['load_avg_5'], 'g.', label='Normal 5 min', alpha=0.5)
        plt.plot(normal_data['current_time'], normal_data['load_avg_15'], 'c.', label='Normal 15 min', alpha=0.5)
        
        if not anomalies.empty:
            plt.plot(anomalies['current_time'], anomalies['load_avg_1'], 'ro', label='Anomali 1 min')
            plt.plot(anomalies['current_time'], anomalies['load_avg_5'], 'mo', label='Anomali 5 min')
            plt.plot(anomalies['current_time'], anomalies['load_avg_15'], 'yo', label='Anomali 15 min')
        
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
        plt.plot(normal_data['current_time'], normal_data['number_of_user'], 'b.', label='Normal', alpha=0.5)
        
        if not anomalies.empty:
            plt.plot(anomalies['current_time'], anomalies['number_of_user'], 'ro', label='Anomali')
        
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
        plt.plot(normal_data['current_time'], normal_data['uptime'], 'b.', label='Normal', alpha=0.5)
        
        if not anomalies.empty:
            plt.plot(anomalies['current_time'], anomalies['uptime'], 'ro', label='Anomali')
        
        plt.title('Uptime dengan Deteksi Anomali')
        plt.ylabel('Uptime (detik)')
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
        plt.scatter(normal_data['load_avg_5'], normal_data['number_of_user'], 
                   c='blue', label='Normal', alpha=0.5)
        
        if not anomalies.empty:
            plt.scatter(anomalies['load_avg_5'], anomalies['number_of_user'], 
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
        
    except Exception as e:
        logger.error(f"Error dalam pembuatan visualisasi: {str(e)}")
        
    return visualizations