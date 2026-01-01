import pandas as pd
import folium
import glob
import os
import numpy as np # Numpy eklendi

def haversine_vectorized(lat1, lon1, lat2, lon2):
    """
    Vektörize edilmiş Haversine formülü (NumPy ile çok daha hızlı).
    Döngü kullanmadan tüm mesafeleri tek seferde hesaplar.
    """
    R = 6371  # Dünya yarıçapı (km)
    
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    
    a = np.sin(dphi/2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    return R * c

def get_color(speed):
    if speed < 30: return '#8B0000'   # Koyu Kırmızı (Yoğun Trafik)
    elif 30 <= speed < 50: return '#FF8C00' # Turuncu
    elif 50 <= speed < 80: return '#32CD32' # Yeşil
    else: return '#1E90FF'            # Mavi (Akıcı Trafik)

def main():
    print("🚀 Veriler optimize edilerek okunuyor...")
    path = "data"
    all_files = glob.glob(os.path.join(path, "*.csv"))
    
    if not all_files:
        print("HATA: Dosya bulunamadı!")
        return

    # Veri okuma kısmı aynı
    df_list = []
    for filename in all_files:
        try:
            df = pd.read_csv(filename, usecols=['GEOHASH', 'LATITUDE', 'LONGITUDE', 'AVERAGE_SPEED'])
            df_list.append(df)
        except Exception as e:
            print(f"Atlanan dosya: {filename}")
            
    if not df_list: return

    full_df = pd.concat(df_list, ignore_index=True)
    
    # 1. Merkez Belirleme
    print("📍 Merkez istasyon istatistiksel olarak belirleniyor...")
    center_geohash = full_df['GEOHASH'].value_counts().idxmax()
    
    # Sensör özetleri
    sensor_stats = full_df.groupby('GEOHASH').agg({
        'LATITUDE': 'mean',
        'LONGITUDE': 'mean',
        'AVERAGE_SPEED': 'mean'
    }).reset_index()
    
    # Merkez verilerini çek
    center_row = sensor_stats[sensor_stats['GEOHASH'] == center_geohash].iloc[0]
    center_lat, center_lon = center_row['LATITUDE'], center_row['LONGITUDE']
    
    # 2. HIZLI MESAFE HESABI (Vektörizasyon)
    print("⚡ Mesafeler NumPy ile hesaplanıyor...")
    # Merkez hariç diğerlerini al
    others = sensor_stats[sensor_stats['GEOHASH'] != center_geohash].copy()
    
    # Tek satırda tüm mesafeleri hesapla (For döngüsü yok!)
    others['DISTANCE'] = haversine_vectorized(
        center_lat, center_lon, 
        others['LATITUDE'].values, others['LONGITUDE'].values
    )
    
    # En yakın 40 komşuyu seç
    neighbors = others.sort_values('DISTANCE').head(40)
    print(f"✅ Ağ topolojisi için en yakın {len(neighbors)} sensör seçildi.")
    
    # 3. HARİTA VE AĞ GÖRSELLEŞTİRME
    print("🎨 Harita ve Ağ çizgileri çiziliyor (Normal Renkli Mod)...")
    
    # --- DEĞİŞİKLİK BURADA YAPILDI ---
    # tiles="CartoDB positron" parametresi kaldırıldı. 
    # Artık varsayılan (renkli) OpenStreetMap kullanılacak.
    m = folium.Map(location=[center_lat, center_lon], zoom_start=13) 
    # ---------------------------------

    # Bağlantı Çizgileri (Topology Lines)
    # Harita renkli olduğu için çizgileri biraz daha belirgin (koyu gri) yapalım
    for _, row in neighbors.iterrows():
        folium.PolyLine(
            locations=[[center_lat, center_lon], [row['LATITUDE'], row['LONGITUDE']]],
            color="#444444", # Daha koyu gri
            weight=1.5,      # Biraz daha kalın
            opacity=0.5,
            dash_array='5, 5' 
        ).add_to(m)

    # Merkez Sensör (Star Icon veya Büyük Daire)
    folium.CircleMarker(
        location=[center_lat, center_lon],
        radius=15,
        color='black',
        weight=3,
        fill=True,
        fill_color='red',
        fill_opacity=1.0,
        popup=f"<b>MERKEZ (TARGET)</b><br>Hız: {center_row['AVERAGE_SPEED']:.1f} km/s",
        tooltip="MERKEZ SENSÖR"
    ).add_to(m)

    # Komşu Sensörler
    for _, row in neighbors.iterrows():
        folium.CircleMarker(
            location=[row['LATITUDE'], row['LONGITUDE']],
            radius=8,
            color='white', 
            weight=1,
            fill=True,
            fill_color=get_color(row['AVERAGE_SPEED']),
            fill_opacity=0.9, # Renkler daha canlı görünsün diye opaklık arttı
            popup=f"Mesafe: {row['DISTANCE']:.2f} km<br>Hız: {row['AVERAGE_SPEED']:.1f}",
            tooltip=f"{row['AVERAGE_SPEED']:.0f} km/s"
        ).add_to(m)

    # 4. LEJANT EKLEME (HTML)
    legend_html = '''
     <div style="position: fixed; 
     bottom: 50px; left: 50px; width: 150px; height: 130px; 
     border:2px solid grey; z-index:9999; font-size:12px;
     background-color:white; opacity: 0.9; padding: 10px;">
     <b>Hız Lejantı</b><br>
     <i style="background:#8B0000;width:10px;height:10px;display:inline-block;border-radius:50%"></i> < 30 km/s (Yoğun)<br>
     <i style="background:#FF8C00;width:10px;height:10px;display:inline-block;border-radius:50%"></i> 30-50 km/s<br>
     <i style="background:#32CD32;width:10px;height:10px;display:inline-block;border-radius:50%"></i> 50-80 km/s<br>
     <i style="background:#1E90FF;width:10px;height:10px;display:inline-block;border-radius:50%"></i> > 80 km/s (Akıcı)<br>
     </div>
     '''
    m.get_root().html.add_child(folium.Element(legend_html))

    output_file = "traffic_network_map_renkli.html"
    m.save(output_file)
    print(f"🎉 İşlem tamam! Normal renkli harita '{output_file}' dosyasına kaydedildi.")

if __name__ == "__main__":
    main()