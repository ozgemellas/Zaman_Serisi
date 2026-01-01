import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from math import radians, sin, cos, sqrt, atan2
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, BatchNormalization, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import visualize_results # Görselleştirme modülü eklendi

# ------------------------------------------------------------------------------
# 1. Veri Yükleme
# ------------------------------------------------------------------------------
def load_traffic_data(data_path="data"):
    """
    data/ klasöründeki tüm CSV dosyalarını okur, birleştirir ve temizler.
    """
    print(f"Veriler {data_path} klasöründen yükleniyor...")
    all_files = glob.glob(os.path.join(data_path, "*.csv"))
    
    if not all_files:
        raise FileNotFoundError(f"{data_path} klasöründe hiç CSV dosyası bulunamadı!")

    df_list = []
    for filename in all_files:
        try:
            df = pd.read_csv(filename)
            # Sütun isimlerini standartlaştır (büyük harf, boşluksuz)
            df.columns = [c.strip().upper() for c in df.columns]
            df_list.append(df)
        except Exception as e:
            print(f"Hata: {filename} okunamadı. {e}")

    if not df_list:
        raise ValueError("Hiçbir dosya okunamadı.")

    full_df = pd.concat(df_list, ignore_index=True)
    
    # Tarih formatını düzelt
    full_df['DATE_TIME'] = pd.to_datetime(full_df['DATE_TIME'])
    
    # Eksik değerleri doldur (basit doldurma, detaylısı pivot aşamasında)
    full_df.ffill(inplace=True)
    full_df.bfill(inplace=True)
    
    # Tarihe göre sırala
    full_df.sort_values('DATE_TIME', inplace=True)
    full_df.reset_index(drop=True, inplace=True)
    
    print(f"Toplam {len(full_df)} satır veri yüklendi.")
    return full_df

# ------------------------------------------------------------------------------
# 2. Zaman Öznitelikleri Üret
# ------------------------------------------------------------------------------
def add_sensor_features(df):
    """
    Sensör bazlı özellikler: Rolling Mean, Speed Diff.
    Pivot öncesi hesaplanmalı.
    """
    print("Sensör türev özellikleri ekleniyor...")
    if 'GEOHASH' in df.columns and 'AVERAGE_SPEED' in df.columns:
        df['ROLLING_MEAN_3H'] = df.groupby('GEOHASH')['AVERAGE_SPEED'].transform(lambda x: x.rolling(window=3, min_periods=1).mean())
        df['SPEED_DIFF'] = df.groupby('GEOHASH')['AVERAGE_SPEED'].transform(lambda x: x.diff().fillna(0))
    return df

def add_temporal_features(df):
    """
    Zaman bazlı öznitelikler ekler: HOUR, DAY, MONTH, WEEKDAY, SEASON, SIN/COS TIME.
    """
    print("Zaman öznitelikleri ekleniyor...")
    # df = df.copy() # Bellek tasarrufu için kopyalamayı kaldırdık
    
    df['HOUR'] = df['DATE_TIME'].dt.hour
    df['DAY'] = df['DATE_TIME'].dt.day
    df['MONTH'] = df['DATE_TIME'].dt.month
    df['DAYOFWEEK'] = df['DATE_TIME'].dt.dayofweek
    df['IS_WEEKEND'] = (df['DAYOFWEEK'] >= 5).astype(int)
    
    # Mevsim Etiketi (Vektörize)
    # 0: KIŞ, 1: İLKBAHAR, 2: YAZ, 3: SONBAHAR
    conditions = [
        df['MONTH'].isin([12, 1, 2]),
        df['MONTH'].isin([3, 4, 5]),
        df['MONTH'].isin([6, 7, 8]),
        df['MONTH'].isin([9, 10, 11])
    ]
    choices = [0, 1, 2, 3]
    df['SEASON'] = np.select(conditions, choices, default=0)
    
    # Sin/Cos Dönüşümü
    df['HOUR_SIN'] = np.sin(2 * np.pi * df['HOUR'] / 24)
    df['HOUR_COS'] = np.cos(2 * np.pi * df['HOUR'] / 24)
    df['DAY_SIN'] = np.sin(2 * np.pi * df['DAYOFWEEK'] / 7)
    df['DAY_COS'] = np.cos(2 * np.pi * df['DAYOFWEEK'] / 7)
    
    return df

# ------------------------------------------------------------------------------
# 3. Sensör Seçimi (Spatial Filtering)
# ------------------------------------------------------------------------------
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371.0 # Dünya yarıçapı (km)
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

def select_nearby_sensors(df, K=40):
    """
    En çok veriye sahip sensörü merkez seçip, ona en yakın K sensörü döndürür.
    """
    print(f"En yakın {K} sensör seçiliyor...")
    
    # Her sensörün (GEOHASH) ortalama konumu ve veri sayısı
    sensor_stats = df.groupby('GEOHASH').agg({
        'LATITUDE': 'mean',
        'LONGITUDE': 'mean',
        'AVERAGE_SPEED': 'count'
    }).rename(columns={'AVERAGE_SPEED': 'COUNT'})
    
    # En çok verisi olan sensörü merkez seç
    center_sensor = sensor_stats.sort_values('COUNT', ascending=False).iloc[0]
    center_lat = center_sensor['LATITUDE']
    center_lon = center_sensor['LONGITUDE']
    print(f"Merkez Sensör: {sensor_stats.sort_values('COUNT', ascending=False).index[0]} (Veri Sayısı: {center_sensor['COUNT']})")
    
    # Mesafeleri hesapla
    sensor_stats['DISTANCE'] = sensor_stats.apply(
        lambda row: haversine_distance(center_lat, center_lon, row['LATITUDE'], row['LONGITUDE']), axis=1
    )
    
    # En yakın K sensörü seç
    selected_sensors = sensor_stats.sort_values('DISTANCE').head(K).index.tolist()
    
    return selected_sensors, sensor_stats.loc[selected_sensors]

# ------------------------------------------------------------------------------
# 4. Çok Sensörlü Pivot Veri Üret
# ------------------------------------------------------------------------------
def prepare_multisensor_pivot(df, selected_sensors):
    """
    Seçilen sensörler için veriyi pivotlar:
    Satırlar: DATE_TIME
    Sütunlar: S1_SPEED, S1_VEH, S2_SPEED, S2_VEH ...
    """
    print("Pivot tablo oluşturuluyor...")
    
    # Sadece seçilen sensörleri filtrele
    df_filtered = df[df['GEOHASH'].isin(selected_sensors)].copy()
    
    # Pivot işlemi
    # Her sensör için SPEED, VEHICLES, ROLLING_MEAN ve SPEED_DIFF sütunlarını yan yana getireceğiz
    pivot_df = df_filtered.pivot_table(
        index='DATE_TIME',
        columns='GEOHASH',
        values=['AVERAGE_SPEED', 'NUMBER_OF_VEHICLES', 'ROLLING_MEAN_3H', 'SPEED_DIFF']
    )
    
    # Sütun isimlerini düzleştir: SPEED_GEOHASH1, VEH_GEOHASH1 gibi
    pivot_df.columns = [f"{col[0]}_{col[1]}" for col in pivot_df.columns]
    
    # Zaman serisi olduğu için eksik saatleri doldurmak önemli
    # Tam zaman aralığını oluştur
    full_idx = pd.date_range(start=pivot_df.index.min(), end=pivot_df.index.max(), freq='H')
    pivot_df = pivot_df.reindex(full_idx)
    
    # Eksik değerleri doldur
    pivot_df.interpolate(method='linear', inplace=True)
    pivot_df.ffill(inplace=True)
    pivot_df.bfill(inplace=True)
    
    # Index ismini geri getir
    pivot_df.index.name = 'DATE_TIME'
    pivot_df.reset_index(inplace=True)
    
    # Ek zaman özelliklerini tekrar ekle (Pivotta kaybolmuş olabilir veya tekilleştirmek için)
    # Sadece ilk sensörün zaman özelliklerini veya yeniden hesaplayarak ekleyebiliriz.
    # Burada yeniden hesaplamak en temizi.
    pivot_df = add_temporal_features(pivot_df)
    
    print(f"Pivot tablo hazır. Boyut: {pivot_df.shape}")
    return pivot_df

# ------------------------------------------------------------------------------
# 5. Veri Ölçeklendirme
# ------------------------------------------------------------------------------
def scale_features(data_pivot, selected_sensors):
    """
    X için RobustScaler kullanır.
    Y (Target) ölçeklemesi burada YAPILMAZ (Main içinde y_train üzerinde fit edilecek).
    """
    print("Veriler ölçeklendiriliyor (Sadece X)...")
    
    # Hedef sütunlar (Sadece SPEED olanlar)
    target_cols = [c for c in data_pivot.columns if 'AVERAGE_SPEED' in c]
    
    # Feature sütunları (DATE_TIME hariç hepsi)
    feature_cols = [c for c in data_pivot.columns if c != 'DATE_TIME']
    
    # Scaler'ları tanımla
    scaler_X = RobustScaler()
    # scaler_y main fonksiyonunda tanımlanacak
    
    # Veriyi ayır
    data_X = data_pivot[feature_cols].values
    data_y = data_pivot[target_cols].values
    
    # Fit ve Transform
    data_X_scaled = scaler_X.fit_transform(data_X)
    # data_y_scaled = scaler_y.fit_transform(data_y) # İPTAL
    
    return data_X_scaled, data_y, scaler_X, None, feature_cols, target_cols

# ------------------------------------------------------------------------------
# 6. Sequence Oluşturma
# ------------------------------------------------------------------------------
def create_sequences(data_X, data_y, lookback=24, horizons=[1, 3, 6]):
    """
    LSTM için (samples, lookback, features) formatında veri hazırlar.
    Çıktı Y: (samples, num_sensors * len(horizons))
    """
    print(f"Sequence oluşturuluyor (Lookback: {lookback}, Horizons: {horizons})...")
    
    X, y = [], []
    
    # En büyük horizon kadar sondan kırpmamız lazım
    max_horizon = max(horizons)
    
    for i in range(lookback, len(data_X) - max_horizon):
        # Girdi: i-lookback'ten i'ye kadar olan veriler
        X.append(data_X[i-lookback:i])
        
        # Çıktı: i+h anındaki hız değerleri (tüm sensörler için)
        targets = []
        for h in horizons:
            # i anı şu an, tahmin etmek istediğimiz i+h (fakat index 0-based olduğu için i+h-1 gibi düşünmeliyiz? 
            # Hayır, data_y[i] şu anki değer. Gelecek değer data_y[i+h-1] değil, data_y[i+h-1] eğer i bir sonraki adımsa...
            # Basitçe: i anındayız (son veri data_X[i-1]). Hedef data_y[i + h - 1] (pandas shift mantığı gibi)
            # data_X[i-1] t zamanı ise, data_y[i] t+1 zamanıdır (eğer data_y data_X ile aynı hizadaysa).
            # Burada data_X ve data_y aynı hizalı (satır satır).
            # X penceresi: [t-23, ..., t] (index i-1 son eleman)
            # Tahmin: t+1, t+3, t+6.
            # Index olarak: i (1 saat sonra), i+2 (3 saat sonra), i+5 (6 saat sonra)
            targets.append(data_y[i - 1 + h]) # DÜZELTME: Index kayması düzeltildi data_y[i - 1 + h] 
            
        # targets şekli: (3, num_sensors). Bunu düzleştirelim (3 * num_sensors)
        y.append(np.concatenate(targets))
        
    return np.array(X), np.array(y)

# ------------------------------------------------------------------------------
# 7. Model Mimarisi
# ------------------------------------------------------------------------------
def build_lstm_model(input_shape, output_dim):
    """
    Bidirectional LSTM Modeli
    """
    print("Model inşa ediliyor...")
    
    inputs = Input(shape=input_shape)
    
    # 1. LSTM Katmanı
    x = Bidirectional(LSTM(128, return_sequences=True))(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    # 2. LSTM Katmanı
    x = Bidirectional(LSTM(64, return_sequences=False))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    # Dense Çıkış
    outputs = Dense(output_dim)(x) # Aktivasyon linear (regresyon)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    optimizer = Adam(learning_rate=0.0005)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    
    model.summary()
    return model



# ------------------------------------------------------------------------------
# 8. Model Eğitme
# ------------------------------------------------------------------------------
def train_model(model, X_train, y_train, X_val, y_val, epochs=200, batch_size=32):
    """
    Modeli eğitir.
    """
    print("Model eğitimi başlıyor...")
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6, verbose=1)
    ]
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    return history

# ------------------------------------------------------------------------------
# 9. Performans Değerlendirme
# ------------------------------------------------------------------------------
def evaluate_model(model, X_test, y_test, scaler_y, num_sensors, horizons=[1, 3, 6]):
    """
    RMSE, MAE hesaplar ve grafikleri çizer.
    """
    print("Model değerlendiriliyor...")
    
    # Tahmin yap
    y_pred_scaled = model.predict(X_test)
    
    # Ölçeği geri al (Inverse Transform)
    # Scaler artık flattened (samples, horizons*sensors) üzerinde fit edildiği için direkt inverse edilebilir
    y_pred_inv_flat = scaler_y.inverse_transform(y_pred_scaled)
    y_test_inv_flat = scaler_y.inverse_transform(y_test)
    
    num_horizons = len(horizons)
    
    y_test_reshaped = y_test_inv_flat.reshape(-1, num_horizons, num_sensors)
    y_pred_reshaped = y_pred_inv_flat.reshape(-1, num_horizons, num_sensors)
    
    # Sonuçları saklamak için
    results = {}
    
    for i, h in enumerate(horizons):
        print(f"\n--- Horizon: {h} Saat ---")
        
        # O anki horizon için veriler
        y_true_h = y_test_reshaped[:, i, :]
        y_pred_h = y_pred_reshaped[:, i, :]
        
        # Inverse transform zaten yapıldı
        y_true_inv = y_true_h
        y_pred_inv = y_pred_h
        
        # Metrikler
        rmse = np.sqrt(mean_squared_error(y_true_inv, y_pred_inv))
        mae = mean_absolute_error(y_true_inv, y_pred_inv)
        
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        
        results[f'RMSE_{h}h'] = rmse
        results[f'MAE_{h}h'] = mae
        
        # Örnek bir sensör için grafik (İlk sensör)
        plt.figure(figsize=(12, 4))
        plt.plot(y_true_inv[:200, 0], label='Gerçek', alpha=0.7)
        plt.plot(y_pred_inv[:200, 0], label='Tahmin', alpha=0.7)
        plt.title(f"{h} Saat Sonrası Tahmin (Sensör 0)")
        plt.legend()
        plt.show()

    return results

# ------------------------------------------------------------------------------
# 10. Görselleştirmeler
# ------------------------------------------------------------------------------
def plot_traffic_map(sensor_info, output_path="traffic_map.html"):
    """
    Sensörleri haritada gösterir.
    """
    print("Trafik haritası oluşturuluyor...")
    center_lat = sensor_info['LATITUDE'].mean()
    center_lon = sensor_info['LONGITUDE'].mean()
    
    m = folium.Map(location=[center_lat, center_lon], zoom_start=11)
    
    for idx, row in sensor_info.iterrows():
        folium.CircleMarker(
            location=[row['LATITUDE'], row['LONGITUDE']],
            radius=5,
            popup=f"Geohash: {idx}\nVeri: {row['COUNT']}",
            color='blue',
            fill=True
        ).add_to(m)
        
    m.save(output_path)
    print(f"Harita kaydedildi: {output_path}")

def plot_predictions_bar(results):
    """
    Horizon bazlı RMSE değerlerini çizer.
    """
    horizons = ['1h', '3h', '6h']
    rmses = [results.get(f'RMSE_{h}', 0) for h in horizons]
    
    plt.figure(figsize=(8, 5))
    plt.bar(horizons, rmses, color=['green', 'orange', 'red'])
    plt.title("Tahmin Hataları (RMSE)")
    plt.ylabel("RMSE (km/h)")
    plt.show()

def plot_traffic_stats(df):
    """
    Genel trafik istatistiklerini çizer.
    """
    plt.figure(figsize=(10, 5))
    # sns.histplot(df['AVERAGE_SPEED'], bins=50, kde=True) # Seaborn/Pandas sürüm uyumsuzluğu ve performans nedeniyle iptal
    plt.hist(df['AVERAGE_SPEED'], bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel("Ortalama Hız")
    plt.ylabel("Frekans")
    plt.title("Hız Dağılımı")
    plt.show()



# ------------------------------------------------------------------------------
# 11. Model Kaydetme
# ------------------------------------------------------------------------------
def save_model_artifacts(model, path="models"):
    if not os.path.exists(path):
        os.makedirs(path)
    model.save(os.path.join(path, "traffic_model.h5"))
    print(f"Model kaydedildi: {path}/traffic_model.h5")

# ------------------------------------------------------------------------------
# 12. Tahmin Scripti (Inference)
# ------------------------------------------------------------------------------
def predict_future_hours(model, data_pivot, scaler_X, scaler_y, horizons=[1, 3, 6], lookback=24):
    """
    Son 24 saati alıp gelecek 1, 3, 6 saati tahmin eder.
    """
    print("\nGelecek tahminleri yapılıyor...")
    
    # Son 24 saati al (Scale edilmiş veri üzerinden gitmek lazım ama burada raw'dan alıp scale edelim)
    # data_pivot zaten raw ama feature'ları tam.
    
    # Feature sütunlarını bul (scale_features fonksiyonundaki mantıkla aynı olmalı)
    feature_cols = [c for c in data_pivot.columns if c != 'DATE_TIME']
    last_sequence = data_pivot[feature_cols].iloc[-lookback:].values
    
    # Scale et
    last_sequence_scaled = scaler_X.transform(last_sequence)
    
    # Model girdisi (1, 24, features)
    input_seq = np.expand_dims(last_sequence_scaled, axis=0)
    
    # Tahmin
    pred_scaled = model.predict(input_seq)
    
    # Inverse Transform
    # Scaler flattened targets üzerinde fit edildi
    pred_inv_flat = scaler_y.inverse_transform(pred_scaled)
    
    # Reshape ve Inverse Transform
    num_sensors = len([c for c in data_pivot.columns if 'AVERAGE_SPEED' in c])
    # DÜZELTME: Reshape işlemi dinamikleştirildi
    pred_reshaped = pred_inv_flat.reshape(1, len(horizons), num_sensors)
    
    print("\n--- TAHMİN SONUÇLARI (Ortalama Hızlar) ---")
    
    for i, h in enumerate(horizons):
        pred_h = pred_reshaped[:, i, :]
        # Inverse zaten yapıldı
        avg_speed = np.mean(pred_h)
        print(f"{h} Saat Sonra İstanbul Geneli Tahmini Ortalama Hız: {avg_speed:.2f} km/s")

# ------------------------------------------------------------------------------
# 13. Ana Çalıştırma Fonksiyonu
# ------------------------------------------------------------------------------
def main():
    # Klasör kontrolü
    if not os.path.exists("models"): os.makedirs("models")
    
    # 1. Veri Yükle
    try:
        df = load_traffic_data("data")
    except Exception as e:
        print(f"Veri yükleme hatası: {e}")
        return

    # 2. Sensör Seçimi (PERFORMANS İÇİN ÖNE ALINDI)
    # 40 milyon satırda işlem yapmadan önce veriyi azaltıyoruz.
    selected_sensors, sensor_info = select_nearby_sensors(df, K=40)
    plot_traffic_map(sensor_info)
    
    print(f"Veri {len(selected_sensors)} sensör için filtreleniyor...")
    df = df[df['GEOHASH'].isin(selected_sensors)].copy()
    print(f"Filtrelenmiş veri boyutu: {len(df)}")

    # 3. Zaman Özellikleri (Artık filtrelenmiş veride çalışıyor, çok daha hızlı)
    df = add_temporal_features(df)
    df = add_sensor_features(df) # DÜZELTME: Sensor özellikleri pivot öncesi eklendi
    
    # İstatistikleri çiz
    plot_traffic_stats(df)

    # 4. Pivot
    pivot_df = prepare_multisensor_pivot(df, selected_sensors)

    # 5. Ölçekleme (DÜZELTME: scaler_y artık burada fit edilmiyor, ham Y dönüyor)
    X_scaled, y_raw, scaler_X, _, feature_cols, target_cols = scale_features(pivot_df, selected_sensors)
    
    # 6. Sequence
    X, y = create_sequences(X_scaled, y_raw, lookback=24, horizons=[1, 3, 6])
    print(f"Eğitim Verisi Şekli: X={X.shape}, y={y.shape}")
    
    # Eğitim/Test Ayrımı
    train_size = int(len(X) * 0.9)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train_raw, y_test_raw = y[:train_size], y[train_size:]
    
    # Doğrulama ayrımı (Eğitim setinden %10 ayıralım veya fit içinde validation_split kullanalım, 
    # ama burada manuel ayırmak daha kontrollü)
    val_size = int(len(X_train) * 0.1)
    X_val = X_train[-val_size:]
    y_val_raw = y_train_raw[-val_size:]
    X_train = X_train[:-val_size]
    y_train_raw = y_train_raw[:-val_size]

    # DÜZELTME: scaler_y, flattened y_train üzerinde fit ediliyor
    print("Scaler_y flattened target verisi üzerinde fit ediliyor...")
    scaler_y = RobustScaler()
    y_train = scaler_y.fit_transform(y_train_raw)
    y_val = scaler_y.transform(y_val_raw)
    y_test = scaler_y.transform(y_test_raw)
    
    # 7. Model Kur
    input_shape = (X_train.shape[1], X_train.shape[2])
    output_dim = y_train.shape[1]
    model = build_lstm_model(input_shape, output_dim)
    
    # 8. Eğit
    history = train_model(model, X_train, y_train, X_val, y_val, epochs=200)
    
    # Eğitim grafiği
    plt.figure(figsize=(10, 4))
    plt.plot(history.history['loss'], label='Eğitim Kaybı')
    plt.plot(history.history['val_loss'], label='Doğrulama Kaybı')
    plt.title("Model Eğitim Kaybı")
    plt.legend()
    plt.show()

    # 9. Değerlendir
    results = evaluate_model(model, X_test, y_test, scaler_y, len(selected_sensors))
    plot_predictions_bar(results)

    # 10. Kaydet
    save_model_artifacts(model)
    
    # 11. Gelişmiş Görselleştirme (visualize_results.py kullanımı)
    print("\n--- Gelişmiş Grafikler Oluşturuluyor ---")
    visualize_results.generate_all_visualizations(
        history=history,
        model=model,
        X_test=X_test,
        y_test=y_test,
        scaler_y=scaler_y,
        num_sensors=len(selected_sensors),
        horizons=[1, 3, 6]
    )
    
    # 12. Inference (Son durum için)
    predict_future_hours(model, pivot_df, scaler_X, scaler_y, horizons=[1, 3, 6])

if __name__ == "__main__":
    main()
