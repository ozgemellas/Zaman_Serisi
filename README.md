
# İstanbul Trafik Tahmini (Time Series Traffic Prediction)

Bu proje, İstanbul'daki trafik sensör verilerini kullanarak gelecekteki trafik yoğunluğunu (hızını) tahmin eden bir Derin Öğrenme (Deep Learning) modelidir. **Bi-Directional LSTM (Uzun Kısa Süreli Bellek)** mimarisi kullanılarak, trafik akışının zamansal ve mekansal özellikleri modellenmiştir.

## Proje Hakkında

Proje, İBB Açık Veri Portalı'ndan alınan trafik verilerini işler, belirli sensör bölgeleri için zaman serisi tahmini yapar ve sonuçları görselleştirir.

**Temel Özellikler:**
*   **Mekansal Filtreleme:** Veri setindeki en yoğun sensörü merkez kabul ederek, ona en yakın 40 sensörü (komşuyu) analize dahil eder (Haversine formülü ile).
*   **Zaman Öznitelikleri:** Saat, gün, ay, hafta sonu bilgisi ve döngüsel (Cyclical) zaman kodlaması (Sin/Cos dönüşümü).
*   **Derin Öğrenme Modeli:** Çift Yönlü (Bidirectional) LSTM katmanları ile zaman serisi tahmini.
*   **Çoklu Ufuk Tahmini:** Gelecek 1, 3 ve 6 saat için hız tahmini.
*   **Görselleştirme:** Folium ile interaktif haritalar ve Matplotlib/Seaborn ile detaylı performans grafikleri.

## Dosya Yapısı

*   `istanbul_traffic_prediction.py`: Ana model eğitim dosyası. Veriyi yükler, işler, modeli eğitir ve sonuçları kaydeder.
*   `create_map.py`: Trafik ağını harita üzerinde görselleştiren ve HTML çıktısı üreten script.
*   `visualize_results.py`: Model sonuçlarını (Loss, RMSE, MAE, Scatter Plot vb.) görselleştirmek için kullanılan yardımcı modül.
*   `mimarioluşturma.py`: Projenin sistem mimarisini blok diyagram olarak çizen script.
*   `traffic_network_map_renkli.html`: Proje tarafından üretilen, sensör ağını ve trafik durumunu gösteren interaktif harita.

## Kurulum ve Kullanım

Gerekli kütüphaneleri yüklemek için:

```bash
pip install pandas numpy matplotlib seaborn folium tensorflow scikit-learn
```

Modeli eğitmek için:

```bash
python istanbul_traffic_prediction.py
```

Harita oluşturmak için:

```bash
python create_map.py
```

## Veri Seti

Bu proje "data" klasörü altındaki CSV dosyaları ile çalışır.
*Not: Veri dosyaları boyutları nedeniyle (her biri >100MB) bu depoya eklenmemiştir.*

## Yazar
**Özge Mellaş**  
[GitHub Profilim](https://github.com/ozgemellas)
