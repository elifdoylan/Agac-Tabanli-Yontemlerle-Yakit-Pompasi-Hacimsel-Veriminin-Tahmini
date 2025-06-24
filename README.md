# Bitirme Projesi - Makine Öğrenmesi ile Veri Analizi

**Öğrenci:** Elif Doylan  
**Öğrenci Numarası:** 20360859003  
**Proje Konusu:** Regresyon Analizi için Kapsamlı Makine Öğrenmesi Pipeline'ı


## Proje Özeti

Bu bitirme projesi, endüstriyel veri setleri üzerinde kapsamlı makine öğrenmesi analizi yapmak için geliştirilmiştir. Proje, veri ön işleme, çoklu model eğitimi, performans değerlendirme ve sonuç raporlama süreçlerini otomatikleştirir.

## Proje Hedefleri

- Gerçek dünya veri setleri üzerinde makine öğrenmesi uygulaması
- Çoklu regresyon modeli karşılaştırması
- Kapsamlı veri görselleştirme ve analizi
- Profesyonel raporlama ve sonuç sunumu
- Python ve bilimsel kütüphaneler ile pratik deneyim

## Teknik Özellikler

### Veri İşleme
- Otomatik eksik değer doldurma
- Kategorik değişken encoding
- Özellik ölçeklendirme (StandardScaler)
- Akıllı train/test ayrımı

### Makine Öğrenmesi Modelleri
- **Lineer Regresyon**: Temel doğrusal model
- **Ridge Regresyon**: L2 regularizasyonlu model
- **Polinom Regresyon**: 3. derece polinom özellikleri
- **Karar Ağacı**: Ağaç tabanlı regresyon
- **Rastgele Orman**: Topluluk ağaç modeli
- **Gradyan Artırma**: Gradient Boosting
- **XGBoost**: Gelişmiş gradient boosting
- **Topluluk Modeli**: Voting regressor kombinasyonu

### Değerlendirme Metrikleri
- **MAE**: Ortalama Mutlak Hata
- **RMSE**: Kök Ortalama Kare Hata
- **R²**: Açıklanan Varyans Oranı
- **Cross-Validation**: 5-fold çapraz doğrulama

## Kurulum ve Çalıştırma

### Kurulum Adımları

1. **Gerekli paketleri yükleyin:**
```bash
pip install -r requirements.txt
```

2. **Veri dosyasını yerleştirin:**
   - `latest_data.xlsx` dosyasını proje dizinine koyun

3. **Analizi çalıştırın:**
```bash
python main.py
```

## Çıktılar ve Sonuçlar

### Grafik Çıktıları
1. **Korelasyon Matrisi**: Değişkenler arası ilişkileri gösterir
2. **Model Karşılaştırması**: CV vs Test performansı
3. **Tahmin Grafikleri**: Her model için scatter plot'lar

### Veri Çıktıları
1. **model_results.csv**: Tüm modellerin performans metrikleri
2. **predictions.csv**: Test seti tahminleri
3. **analysis.log**: Detaylı işlem kayıtları

## Metodoloji

### Veri Ön İşleme Süreci
1. **Veri Yükleme**: Excel dosyasından otomatik yükleme
2. **Eksik Değer Analizi**: Numerik ve kategorik değişkenler için farklı stratejiler
3. **Encoding**: Label encoding ile kategorik değişken dönüşümü
4. **Normalizasyon**: StandardScaler ile özellik ölçeklendirme

### Model Eğitimi ve Değerlendirme
1. **Veri Bölme**: Hub değerlerine göre stratejik train/test ayrımı
2. **Cross-Validation**: 5-fold çapraz doğrulama ile güvenilir performans ölçümü
3. **Model Karşılaştırması**: Çoklu metrik ile objektif değerlendirme
4. **Ensemble Learning**: Voting regressor ile model kombinasyonu
