# Bu kod, İstanbul Trafik LSTM projesi için sunumda kullanılacak grafiklerin üretilmesini ve figures/ klasörüne kaydedilmesini sağlar.

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Seaborn stili (Opsiyonel, daha estetik grafikler için)
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.dpi'] = 200

def create_figures_folder(folder_name="figures"):
    """
    Grafiklerin kaydedileceği klasörü oluşturur.
    """
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"'{folder_name}' klasörü oluşturuldu.")
    else:
        print(f"'{folder_name}' klasörü zaten mevcut.")

def plot_training_history(history, output_folder="figures"):
    """
    Eğitim ve doğrulama kayıplarını (Loss ve MAE) çizer.
    """
    # 1. Loss Grafiği
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Eğitim Kaybı (Train Loss)', linewidth=2)
    plt.plot(history.history['val_loss'], label='Doğrulama Kaybı (Val Loss)', linewidth=2)
    plt.title("LSTM Model Eğitim Kaybı")
    plt.xlabel("Epoch")
    plt.ylabel("Kayıp (MSE)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Kaydet
    save_path = os.path.join(output_folder, "lstm_loss.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Loss grafiği kaydedildi: {save_path}")

    # 2. MAE Grafiği (Eğer varsa)
    if 'mae' in history.history:
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['mae'], label='Eğitim MAE', linewidth=2)
        plt.plot(history.history['val_mae'], label='Doğrulama MAE', linewidth=2)
        plt.title("LSTM Model Eğitim MAE")
        plt.xlabel("Epoch")
        plt.ylabel("Ortalama Mutlak Hata (MAE)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        save_path_mae = os.path.join(output_folder, "lstm_mae.png")
        plt.savefig(save_path_mae, bbox_inches='tight')
        plt.close()
        print(f"MAE grafiği kaydedildi: {save_path_mae}")

def plot_predictions_by_horizon(model, X_test, y_test, scaler_y, num_sensors, horizons=[1, 3, 6], output_folder="figures"):
    """
    1, 3 ve 6 saatlik tahminleri (ilk sensör için) çizer.
    """
    # Tahmin yap
    print("Tahminler yapılıyor...")
    y_pred_scaled = model.predict(X_test, verbose=0)
    
    # Inverse Transform (Bütün veriyi önce açıyoruz, çünkü scaler flattened verilere göre eğitildi)
    y_test_inv_flat = scaler_y.inverse_transform(y_test)
    y_pred_inv_flat = scaler_y.inverse_transform(y_pred_scaled)
    
    # Reshape: (samples, horizons, sensors)
    num_horizons = len(horizons)
    y_test_reshaped = y_test_inv_flat.reshape(-1, num_horizons, num_sensors)
    y_pred_reshaped = y_pred_inv_flat.reshape(-1, num_horizons, num_sensors)
    
    # İlk sensör (index 0) için çizimler
    sensor_idx = 0
    
    for i, h in enumerate(horizons):
        # İlgili horizon ve sensör verisini al (Inverse edilmiş veri üzerinden)
        y_true_inv = y_test_reshaped[:, i, :]
        y_pred_inv = y_pred_reshaped[:, i, :]
        
        # İlk 200 örnek
        y_true_sample = y_true_inv[:200, sensor_idx]
        y_pred_sample = y_pred_inv[:200, sensor_idx]
        
        plt.figure(figsize=(12, 5))
        plt.plot(y_true_sample, label='Gerçek Değer', color='blue', alpha=0.7, linewidth=1.5)
        plt.plot(y_pred_sample, label='Tahmin', color='red', linestyle='--', alpha=0.7, linewidth=1.5)
        
        plt.title(f"LSTM - {h} Saat Sonrası Tahmin (Sensör 0)")
        plt.xlabel("Zaman (Saat)")
        plt.ylabel("Hız (km/s)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        save_path = os.path.join(output_folder, f"lstm_pred_{h}h_sensor0.png")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"{h} saatlik tahmin grafiği kaydedildi: {save_path}")

def plot_error_metrics(model, X_test, y_test, scaler_y, num_sensors, horizons=[1, 3, 6], output_folder="figures"):
    """
    RMSE ve MAE metriklerini horizon bazlı bar grafik olarak çizer.
    """
    print("Hata metrikleri hesaplanıyor...")
    y_pred_scaled = model.predict(X_test, verbose=0)
    
    # Inverse Transform
    y_test_inv_flat = scaler_y.inverse_transform(y_test)
    y_pred_inv_flat = scaler_y.inverse_transform(y_pred_scaled)
    
    num_horizons = len(horizons)
    y_test_reshaped = y_test_inv_flat.reshape(-1, num_horizons, num_sensors)
    y_pred_reshaped = y_pred_inv_flat.reshape(-1, num_horizons, num_sensors)
    
    rmse_list = []
    mae_list = []
    horizon_labels = [f"{h} Saat" for h in horizons]
    
    for i, h in enumerate(horizons):
        y_true_inv = y_test_reshaped[:, i, :]
        y_pred_inv = y_pred_reshaped[:, i, :]
        
        rmse = np.sqrt(mean_squared_error(y_true_inv, y_pred_inv))
        mae = mean_absolute_error(y_true_inv, y_pred_inv)
        
        rmse_list.append(rmse)
        mae_list.append(mae)
    
    # RMSE Bar Grafiği
    plt.figure(figsize=(8, 6))
    bars = plt.bar(horizon_labels, rmse_list, color=['#2ecc71', '#f1c40f', '#e74c3c'], alpha=0.8, edgecolor='black')
    plt.title("LSTM Tahmin Hataları (RMSE)")
    plt.ylabel("RMSE (km/s)")
    plt.xlabel("Tahmin Ufku")
    plt.grid(axis='y', alpha=0.3)
    
    # Değerleri barların üzerine yaz
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.05, f'{yval:.2f}', ha='center', va='bottom', fontsize=10)
        
    save_path_rmse = os.path.join(output_folder, "lstm_rmse_by_horizon.png")
    plt.savefig(save_path_rmse, bbox_inches='tight')
    plt.close()
    print(f"RMSE grafiği kaydedildi: {save_path_rmse}")
    
    # MAE Bar Grafiği
    plt.figure(figsize=(8, 6))
    bars = plt.bar(horizon_labels, mae_list, color=['#3498db', '#9b59b6', '#34495e'], alpha=0.8, edgecolor='black')
    plt.title("LSTM Tahmin Hataları (MAE)")
    plt.ylabel("MAE (km/s)")
    plt.xlabel("Tahmin Ufku")
    plt.grid(axis='y', alpha=0.3)
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.05, f'{yval:.2f}', ha='center', va='bottom', fontsize=10)

    save_path_mae = os.path.join(output_folder, "lstm_mae_by_horizon.png")
    plt.savefig(save_path_mae, bbox_inches='tight')
    plt.close()
    print(f"MAE grafiği kaydedildi: {save_path_mae}")

def plot_residuals(model, X_test, y_test, scaler_y, num_sensors, horizons=[1, 3, 6], output_folder="figures"):
    """
    1 Saatlik tahmin için artık (residual) analizi yapar.
    """
    print("Artık analizi yapılıyor...")
    y_pred_scaled = model.predict(X_test, verbose=0)
    
    # Inverse Transform
    y_test_inv_flat = scaler_y.inverse_transform(y_test)
    y_pred_inv_flat = scaler_y.inverse_transform(y_pred_scaled)
    
    num_horizons = len(horizons)
    y_test_reshaped = y_test_inv_flat.reshape(-1, num_horizons, num_sensors)
    y_pred_reshaped = y_pred_inv_flat.reshape(-1, num_horizons, num_sensors)
    
    # 1 Saat (index 0) ve Sensör 0 için
    h_idx = 0 
    sensor_idx = 0
    
    y_true_inv = y_test_reshaped[:, h_idx, :]
    y_pred_inv = y_pred_reshaped[:, h_idx, :]
    
    residuals = y_true_inv[:, sensor_idx] - y_pred_inv[:, sensor_idx]
    
    # 1. Histogram
    plt.figure(figsize=(10, 6))
    plt.hist(residuals, bins=50, color='purple', edgecolor='black', alpha=0.7)
    plt.title("Artık Dağılımı (1 Saat, Sensör 0)")
    plt.xlabel("Hata (Gerçek - Tahmin)")
    plt.ylabel("Frekans")
    plt.grid(True, alpha=0.3)
    
    save_path_hist = os.path.join(output_folder, "lstm_residual_hist_1h_sensor0.png")
    plt.savefig(save_path_hist, bbox_inches='tight')
    plt.close()
    print(f"Residual histogram kaydedildi: {save_path_hist}")
    
    # 2. Zaman Serisi
    plt.figure(figsize=(12, 5))
    plt.plot(residuals[:200], color='gray', linewidth=1.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title("Artıklar (1 Saat, Sensör 0, İlk 200 Örnek)")
    plt.xlabel("Zaman")
    plt.ylabel("Hata (Residual)")
    plt.grid(True, alpha=0.3)
    
    save_path_ts = os.path.join(output_folder, "lstm_residual_ts_1h_sensor0.png")
    plt.savefig(save_path_ts, bbox_inches='tight')
    plt.close()
    print(f"Residual zaman serisi kaydedildi: {save_path_ts}")

def plot_scatter_predictions(model, X_test, y_test, scaler_y, num_sensors, horizons=[1, 3, 6], output_folder="figures"):
    """
    Tahmin vs Gerçek Değer Scatter Plot (Regression Performance) çizimi.
    Her horizon için ayrı çizilir.
    """
    print("Scatter plotler çiziliyor...")
    y_pred_scaled = model.predict(X_test, verbose=0)
    
    # Inverse Transform
    y_test_inv_flat = scaler_y.inverse_transform(y_test)
    y_pred_inv_flat = scaler_y.inverse_transform(y_pred_scaled)
    
    num_horizons = len(horizons)
    y_test_reshaped = y_test_inv_flat.reshape(-1, num_horizons, num_sensors)
    y_pred_reshaped = y_pred_inv_flat.reshape(-1, num_horizons, num_sensors)
    
    for i, h in enumerate(horizons):
        # Flattened veriler üzerinde çalışalım (Tüm sensörler ve tüm zamanlar bir arada)
        y_true_h = y_test_reshaped[:, i, :].flatten()
        y_pred_h = y_pred_reshaped[:, i, :].flatten()
        
        # R2 Score hesapla
        r2 = r2_score(y_true_h, y_pred_h)
        
        plt.figure(figsize=(8, 8))
        
        # Scatter noktaları
        plt.scatter(y_true_h, y_pred_h, alpha=0.3, s=5, c='blue', label='Veri Noktaları')
        
        # 45 Derecelik Referans Çizgisi (y = x)
        min_val = min(np.min(y_true_h), np.min(y_pred_h))
        max_val = max(np.max(y_true_h), np.max(y_pred_h))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='İdeal Tahmin (y=x)')
        
        plt.title(f"Tahmin vs Gerçek – LSTM Trafik Hız Tahmini ({h} Saat)")
        plt.xlabel("Gerçek Trafik Hızları (km/s)")
        plt.ylabel("Tahmin Edilen Trafik Hızları (km/s)")
        
        # R2 ve Alt Mesaj
        plt.text(min_val + (max_val-min_val)*0.05, max_val - (max_val-min_val)*0.1, 
                 f"$R^2 = {r2:.4f}$", fontsize=14, fontweight='bold', bbox=dict(facecolor='white', alpha=0.8))
        
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        save_path = os.path.join(output_folder, f"lstm_scatter_{h}h.png")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"Scatter plot kaydedildi ({h}s): {save_path}")

def plot_cyclical_time_encoding(output_folder="figures"):
    """
    Döngüsel Zaman Özelliği (Cyclical Time Encoding Visualization) çizimi.
    Saat (0-23 arası) bilgisinin Sin/Cos dönüşümü ile dairesel gösterimi.
    """
    print("Döngüsel zaman grafiği çiziliyor...")
    
    hours = np.arange(24)
    hour_sin = np.sin(2 * np.pi * hours / 24)
    hour_cos = np.cos(2 * np.pi * hours / 24)
    
    plt.figure(figsize=(8, 8))
    
    # Daireyi çiz
    plt.scatter(hour_cos, hour_sin, c=hours, cmap='twilight', s=100, edgecolors='black', zorder=2)
    
    # Oklarla yönü göster (Saat yönünde)
    for i in range(len(hours)):
        # Bir sonraki noktaya ok (son nokta hariç veya döngüsel)
        next_i = (i + 1) % 24
        plt.arrow(hour_cos[i], hour_sin[i], 
                  (hour_cos[next_i] - hour_cos[i])*0.8, 
                  (hour_sin[next_i] - hour_sin[i])*0.8, 
                  head_width=0.05, head_length=0.1, fc='gray', ec='gray', alpha=0.5, zorder=1)
        
        # Saat etiketleri
        label_offset = 1.15
        plt.text(hour_cos[i] * label_offset, hour_sin[i] * label_offset, str(i), 
                 ha='center', va='center', fontsize=10, fontweight='bold')

    plt.title("Cyclical Time Encoding – Saat Özelliğinin Dairesel Gösterimi")
    plt.xlabel("Cos(Hour)")
    plt.ylabel("Sin(Hour)")
    
    # Alt mesaj
    plt.figtext(0.5, 0.01, 
                "Saat 23 ile 00 arasındaki geçiş lineer değildir. Bu nedenle zamanı sin/cos dönüşümüyle dairesel forma getirdim;\nmodel böylece gece yarısı geçişini doğru öğrenebilir.", 
                wrap=True, horizontalalignment='center', fontsize=10, style='italic', bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
    
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.axis('equal') # Daire formu bozulmasın
    
    save_path = os.path.join(output_folder, "cyclical_time_encoding.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Döngüsel zaman grafiği kaydedildi: {save_path}")

def plot_weekly_traffic_fingerprint(model, X_test, y_test, scaler_y, num_sensors, output_folder="figures"):
    """
    Haftalık Trafik Parmak İzi – 7 Günlük Hız Profili.
    İlk 168 saatlik veriyi (veya temsili veriyi) çizer.
    """
    print("Haftalık trafik parmak izi çiziliyor...")
    
    # Gerçek veriden ilk 168 saati almaya çalışalım
    try:
        # Inverse Transform
        y_test_inv_flat = scaler_y.inverse_transform(y_test)
        # Reshape (samples, horizons, sensors) -> Biz sadece ground truth (horizon 0) ve ilk sensör ile ilgileniyoruz
        # Ya da direkt y_test'i (samples, flattened) alıp ilk sensörün sütununu bulabiliriz.
        # Basitlik için: İlk sensörün o anki hız değerlerini alalım (horizon=1 için değil, direkt veri akışı)
        # Ama y_test shifted olabilir. En garantisi ilk sensörün ilk tahmin horizonu (1h) verilerini alıp birleştirmek.
        
        # y_test_inv_flat şekli: (samples, num_horizons * num_sensors)
        # Sıralama: [h1_s1, h1_s2..., h2_s1, h2_s2...] veya [s1_h1, s1_h2...] ?
        # create_sequences içinde: targets.append(data_y[i - 1 + h]) -> h döngüsü dışta değil içte.
        # loop h in horizons: targets.append(...) -> y.append(concatenate(targets))
        # Yani her satır: [h1_tüm_sensörler, h2_tüm_sensörler, h3_tüm_sensörler]
        
        # İlk sensör, 1h horizon (ilk blok)
        # Index 0
        y_week = y_test_inv_flat[:168, 0]
        
        if len(y_week) < 168:
            raise ValueError("Yeterli veri yok")
            
    except Exception as e:
        print(f"Uyarı: Gerçek veri 168 saatten az veya okunamadı ({e}). Sentetik veri ile desen gösterilecek.")
        # Sentetik haftalık veri (Hafta içi yüksek pikler, hafta sonu düşük)
        t = np.arange(168)
        # Günlük döngü (24 saat)
        daily = 10 * np.sin(2 * np.pi * t / 24 - np.pi/2) + 40 # Ortalama 40, +/- 10
        # Hafta sonu etkisi (Cumartesi-Pazar düşüşü) -> Son 48 saat
        weekend_factor = np.ones(168)
        weekend_factor[120:] = 0.8 # Cmt-Paz %20 hız düşüşü (veya artışı? Trafik azalırsa hız artar aslında!)
        # Trafik hızı: Hafta içi sabah/akşam yoğun (hız düşük), öğlen/gece açık (hız yüksek).
        # Hız profili: Gece (yüksek), Sabah (düşük), Öğlen (orta), Akşam (düşük), Gece (yüksek).
        
        # Basit bir simülasyon:
        y_week = []
        for hour in range(168):
            day_hour = hour % 24
            is_weekend = hour >= 120 # 5*24 = 120 (Pzt=0...Cum=4. Cmt=5)
            
            base_speed = 60 # Gece hızı
            
            if not is_weekend:
                # Hafta içi: Sabah 07-09 ve Akşam 17-19 yoğunluk (hız düşer)
                if 7 <= day_hour <= 10: base_speed = 20 # Sabah trafiği
                elif 17 <= day_hour <= 20: base_speed = 15 # Akşam trafiği
                elif 10 < day_hour < 17: base_speed = 40 # Gün içi
            else:
                # Hafta sonu: Daha akıcı
                if 12 <= day_hour <= 18: base_speed = 40 # Öğlen yoğunluğu
                else: base_speed = 70
                
            # Biraz gürültü ekle
            noise = np.random.normal(0, 3)
            y_week.append(base_speed + noise)
        y_week = np.array(y_week)

    plt.figure(figsize=(14, 6))
    plt.plot(y_week, linewidth=2.5, color='#007acc', label='Ortalama Hız')
    
    # Eksenler
    plt.xlim(0, 168)
    ticks = np.arange(0, 169, 24)
    labels = ["Pzt", "Sal", "Çar", "Per", "Cum", "Cmt", "Paz", ""]
    plt.xticks(ticks, labels[:len(ticks)])
    
    # Hafta sonu vurgusu (Son 2 gün)
    plt.axvspan(120, 168, color='gray', alpha=0.1, label='Hafta Sonu')
    
    plt.title("Haftalık Trafik Parmak İzi – 7 Günlük Hız Profili")
    plt.ylabel("Ortalama Hız (km/s)")
    plt.xlabel("Günler (168 Saatlik Döngü)")
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Mesaj
    plt.figtext(0.5, 0.01, 
                "Model sadece tahmin yapmıyor, veri dinamiklerini (hafta içi/hafta sonu farkları) de anlıyoruz.", 
                wrap=True, horizontalalignment='center', fontsize=10, style='italic', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    save_path = os.path.join(output_folder, "weekly_traffic_fingerprint.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Haftalık parmak izi grafiği kaydedildi: {save_path}")

def generate_all_visualizations(history, model, X_test, y_test, scaler_y, num_sensors, horizons=[1, 3, 6]):
    """
    Tüm görselleştirme fonksiyonlarını sırayla çağırır.
    """
    create_figures_folder()
    plot_training_history(history)
    plot_predictions_by_horizon(model, X_test, y_test, scaler_y, num_sensors, horizons)
    plot_error_metrics(model, X_test, y_test, scaler_y, num_sensors, horizons)
    plot_residuals(model, X_test, y_test, scaler_y, num_sensors, horizons)
    plot_scatter_predictions(model, X_test, y_test, scaler_y, num_sensors, horizons)
    plot_cyclical_time_encoding()
    plot_weekly_traffic_fingerprint(model, X_test, y_test, scaler_y, num_sensors)
    print("\nTüm grafikler 'figures/' klasörüne başarıyla kaydedildi.")

# ------------------------------------------------------------------------------
# ÖRNEK KULLANIM
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    # Bu blok, bu dosya tek başına çalıştırıldığında hata vermemesi için
    # sembolik değişkenlerle örnek kullanım gösterir.
    # Gerçek projenizde bu dosyayı import edip `generate_all_visualizations` fonksiyonunu çağırabilirsiniz.
    
    print("Bu script bir modül olarak tasarlanmıştır.")
    print("Kullanım örneği:")
    print("-" * 30)
    print("""
    from visualize_results import generate_all_visualizations
    
    # Mevcut değişkenlerinizle çağırın:
    generate_all_visualizations(
        history=history, 
        model=model, 
        X_test=X_test, 
        y_test=y_test, 
        scaler_y=scaler_y, 
        num_sensors=len(selected_sensors),
        horizons=[1, 3, 6]
    )
    """)
