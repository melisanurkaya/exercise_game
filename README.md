# exercise_game

# Yapay Zeka Projesi: Egzersiz Tanıma

Bu proje, gerçek zamanlı kamera görüntüsünden kişinin yaptığı egzersizleri tanıyan bir model geliştirmek için tasarlanmıştır. Ayrıca, kullanıcıya rastgele seçilen egzersizleri yapması için bir web arayüzü sunar ve her bir egzersizi doğru bir şekilde yapması durumunda puan verir.

## Veri Kümesi 

Bu projede kullanılan egzersiz görüntüleri Kaggle'dan indirilmiştir. Kaggle'daki veri kümesine [bu bağlantıdan](https://www.kaggle.com/datasets/hasyimabdillah/workoutexercises-images) erişebilirsiniz. Kaggle veri kümesini indirdikten sonra, `workout_data` adında bir klasör oluşturun ve indirdiğiniz dosyayı bu klasöre çıkarın.

## Kurulum

Proje klasörünü bilgisayarınıza klonlayın veya indirin. Ardından, aşağıdaki komutu kullanarak gerekli Python paketlerini yükleyin:

pip install -r requirements.txt


## Kullanım

Projeyi çalıştırmak için aşağıdaki adımları izleyin:

1. Gerçek zamanlı egzersiz tanıma modelini çalıştırın:

```bash
python realtime_detection.py

2. Flask sunucusunu başlatın:

python app.py

3. Tarayıcınızda http://localhost:5000/ adresine gidin ve egzersiz tanıma uygulamasını kullanmaya başlayın.
Gereksinimler
Bu projeyi çalıştırmak için aşağıdaki Python paketlerine ihtiyacınız vardır:
NumPy
Pandas
OpenCV (opencv-python)
Scikit-learn
TensorFlow
Keras
Flask
Flask-CORS


Bu paketleri tek seferde yüklemek için aşağıdaki komutu kullanabilirsiniz:

pip install -r requirements.txt

Katkıda Bulunma
Bu proje, katkıda bulunanları her zaman memnuniyetle karşılar. Lütfen herhangi bir hata rapor edin veya önerilerde bulunun.

