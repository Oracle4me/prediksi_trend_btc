# Bitcoin Trend Predictor

Aplikasi ini digunakan untuk memprediksi arah trend Bitcoin menggunakan pendekatan **klasifikasi trend**, bukan prediksi harga regresi murni.

Model utama yang digunakan adalah **LSTM**, dengan model pembanding:

- MLP(Multi Layer Preceptron)
- XGBoost
- Random Forest

Dataset yang digunakan adalah data historis Bitcoin dari **Yahoo Finance / CSV lokal** mulai dari tahun **2021 sampai data terbaru**.

Target prediksi menggunakan horizon **14 hari ke depan** dengan 3 kelas trend:

```text
0 = Turun
1 = Naik
```

---

## 1. Fitur Aplikasi

- Mengambil data historis Bitcoin dari file CSV atau Yahoo Finance.
- Membuat fitur teknikal dari data OHLCV.
- Melatih empat model klasifikasi trend:
  - LSTM
  - MLP
  - XGBoost
  - Random Forest

- Menampilkan hasil evaluasi model:
  - Balanced Accuracy
  - Precision
  - Recall
  - F1 Score

- Menampilkan grafik harga aktual dan estimasi harga berbasis hasil prediksi trend.
- Menampilkan prediksi trend Bitcoin untuk horizon 14 hari ke depan menggunakan LSTM.
- Dashboard berbasis Flask dengan tampilan web interaktif.

---

## 2. Struktur Folder

```text
prediksi_trend_btc/
│
├── data/
│   └── btc_data.csv
│
├── models/
│   ├── saved/
│   ├── lstm_model.py
│   └── ml_models.py
│
├── static/
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── app.js
│
├── templates/
│   └── index.html
│
├── utils/
│   └── data_loader.py
│
├── app.py
├── train_all.py
├── requirements.txt
├── README.md
└── AlurPrediksiTrendBTC.txt
```

---

## 3. Persyaratan Sistem

Sebelum menjalankan project, pastikan sudah menginstall:

- Python v3.12
- pip
- Virtual environment Python
- Koneksi internet untuk mengambil data dari Yahoo Finance jika data belum tersedia

---

## 4. Instalasi Project

### 4.1. Clone atau buka folder project

Masuk ke folder project:

```bash
cd prediksi_trend_btc
```

### 4.2. Buat virtual environment

Windows:

```bash
python -m venv venv
```

Aktifkan virtual environment:

```bash
venv\Scripts\activate
```

Linux / macOS:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 4.3. Install dependency

```bash
pip install -r requirements.txt
```

---

## 5. Menjalankan Training Model

Sebelum membuka dashboard, jalankan proses training terlebih dahulu:

```bash
python train_all.py
```

Proses ini akan melatih empat model:

1. LSTM
2. XGBoost
3. MLP
4. Random Forest

Setelah training selesai, hasil model dan file evaluasi akan disimpan ke folder:

```text
models/saved/
```

File yang dihasilkan antara lain:

```text
lstm_model.keras
mlp_model.keras
rf_model.pkl
xgb_model.pkl
lstm_predictions.csv
mlp_predictions.csv
rf_predictions.csv
xgb_predictions.csv
lstm_metrics.json
mlp_metrics.json
rf_metrics.json
xgb_metrics.json
training_summary.json
training_summary.csv
```

---

## 6. Menjalankan Aplikasi Web

Setelah proses training selesai, jalankan aplikasi Flask:

```bash
python app.py
```

Kemudian buka browser dan akses:

```text
http://127.0.0.1:5000
```

Dashboard akan menampilkan:

- Harga Bitcoin terkini
- Metrik evaluasi model
- Grafik harga aktual
- Grafik prediksi LSTM, MLP, Random Forest, dan XGBoost
- Prediksi trend Bitcoin 14 hari ke depan
- Kesimpulan hasil evaluasi model

---

## 7. Alur Penggunaan Singkat

Urutan penggunaan project:

```text
1. Install Python
2. Buat dan aktifkan virtual environment
3. Install requirements
4. Jalankan python train_all.py
5. Tunggu proses training selesai
6. Jalankan python app.py
7. Buka http://127.0.0.1:5000
```

---

## 8. Penjelasan Model

### 8.1. LSTM

LSTM digunakan sebagai model utama karena mampu mempelajari pola data deret waktu atau time-series. Model membaca sequence data historis Bitcoin untuk memprediksi arah trend dalam horizon 14 hari ke depan.

Pada penelitian ini, LSTM difokuskan untuk klasifikasi trend:

```text
0 = Turun
1 = Naik
```

### 8.2. MLP (Multi Layer Preceptron)

MLP digunakan sebagai model deep learning pembanding. Model ini memanfaatkan fitur-fitur teknikal yang telah diekstraksi dari data historis Bitcoin, seperti Moving Average, RSI, MACD, Momentum, Volatility, dan indikator lainnya untuk melakukan klasifikasi arah trend.

### 8.3. XGBoost

XGBoost digunakan sebagai model machine learning pembanding untuk data tabular dengan fitur teknikal seperti Open, High, Low, Close, Volume, RSI, MACD, Moving Average, Momentum, dan Volatility.

### 8.4. Random Forest

Random Forest digunakan sebagai model pembanding berbasis ensemble tree. Model ini membangun banyak decision tree dan menentukan kelas trend berdasarkan hasil voting dari banyak pohon keputusan.

---

## 9. Metrik Evaluasi

Evaluasi dilakukan menggunakan empat metrik utama, yaitu Balanced Accuracy, Precision, Recall, dan F1 Score. Keempat metrik ini digunakan untuk menilai kemampuan model dalam memprediksi arah trend Bitcoin (Naik atau Turun). Model terbaik dipilih berdasarkan **F1 Score**, dengan **Balanced Accuracy** sebagai pembanding apabila terdapat model yang memiliki nilai F1 Score yang sama.

### 9.1. Balanced Accuracy

Balanced Accuracy mengukur rata-rata akurasi pada masing-masing kelas (Naik dan Turun), sehingga memberikan gambaran yang lebih adil terhadap performa model ketika distribusi data antar kelas tidak seimbang.

Metrik ini digunakan untuk memastikan bahwa model memiliki kemampuan yang baik dalam mengenali kedua kelas trend, bukan hanya kelas yang jumlah datanya lebih banyak.

### 9.2. Precision

Precision mengukur ketepatan prediksi model terhadap suatu kelas.

Dalam konteks prediksi trend Bitcoin, precision menunjukkan seberapa dapat dipercaya prediksi model ketika model memprediksi suatu arah trend, khususnya trend naik.

Semakin tinggi nilai precision, semakin sedikit prediksi positif yang salah.

### 9.3. Recall

Recall mengukur kemampuan model dalam menemukan seluruh data aktual dari suatu kelas.

Dalam penelitian ini, recall menunjukkan seberapa baik model mampu mendeteksi kondisi trend naik yang benar-benar terjadi.

Semakin tinggi nilai recall, semakin sedikit kondisi trend yang terlewat oleh model.

### 9.4. F1 Score

F1 Score merupakan rata-rata harmonik antara Precision dan Recall.

Metrik ini digunakan sebagai acuan utama dalam menentukan model terbaik karena mampu menilai keseimbangan antara ketepatan prediksi (Precision) dan kemampuan mendeteksi seluruh data aktual (Recall).

Semakin tinggi nilai F1 Score, semakin baik keseimbangan performa model dalam melakukan klasifikasi arah trend Bitcoin.

---

## 10. Interpretasi Hasil

Model terbaik ditentukan berdasarkan **F1 Score** pada data pengujian. Apabila terdapat model yang memiliki nilai F1 Score yang sama, maka **Balanced Accuracy** digunakan sebagai metrik pembanding.

Selain F1 Score, Balanced Accuracy, Precision, dan Recall juga dianalisis untuk memberikan gambaran yang lebih lengkap mengenai karakteristik masing-masing model dalam memprediksi arah trend Bitcoin.

Contoh interpretasi hasil:

```text
Precision tinggi         = Prediksi arah trend yang dihasilkan model lebih tepat dan dapat dipercaya.
Recall tinggi            = Model mampu mendeteksi lebih banyak kondisi trend aktual.
F1 Score tinggi          = Model memiliki keseimbangan yang baik antara Precision dan Recall.
Balanced Accuracy tinggi = Model memiliki performa yang seimbang pada kelas trend Naik maupun Turun.
```

Dalam penelitian ini, model terbaik dipilih berdasarkan **F1 Score** karena metrik tersebut mampu mengevaluasi keseimbangan antara ketepatan prediksi (Precision) dan kemampuan model dalam mendeteksi seluruh kondisi trend aktual (Recall). Balanced Accuracy digunakan sebagai metrik pendukung untuk memastikan bahwa model tetap memiliki performa yang baik pada kedua kelas, terutama apabila distribusi data tidak sepenuhnya seimbang.

Karena fokus penelitian adalah klasifikasi arah trend Bitcoin (Naik atau Turun), maka metrik regresi seperti MAE, RMSE, R², dan MAPE tidak digunakan sebagai metrik evaluasi utama.

---

## 11. Catatan Penting

- Model memprediksi arah trend Bitcoin, bukan harga pasti.
- Estimasi harga pada dashboard hanya turunan dari hasil prediksi trend.
- Dashboard mengonversi harga ke IDR menggunakan kurs USD/IDR terbaru.
- Prediksi trend menggunakan horizon 14 hari ke depan.
- Hasil prediksi bukan merupakan saran investasi.
- Cryptocurrency memiliki volatilitas tinggi sehingga hasil prediksi dapat berubah mengikuti kondisi pasar.

---

## 12. Troubleshooting

### 12.1. Module tidak ditemukan

Jika muncul error seperti:

```text
ModuleNotFoundError
```

Pastikan dependency sudah diinstall:

```bash
pip install -r requirements.txt
```

### 12.2. Model belum tersedia

Jika dashboard belum menampilkan hasil prediksi, jalankan training terlebih dahulu:

```bash
python train_all.py
```

### 12.3. Aplikasi tidak bisa dibuka

Pastikan Flask sudah berjalan:

```bash
python app.py
```

Lalu buka:

```text
http://127.0.0.1:5000
```

### 12.4. Data atau model tidak sesuai

Jika terjadi error karena jumlah fitur model tidak cocok, hapus isi folder berikut:

```text
models/saved/
```

Kemudian jalankan ulang:

```bash
python train_all.py
```

---

## 13. Perintah Utama

Training model:

```bash
python train_all.py
```

Menjalankan dashboard:

```bash
python app.py
```

Membuka aplikasi:

```text
http://127.0.0.1:5000
```

---

## 14. Disclaimer

Hasil prediksi yang ditampilkan oleh model tidak menjamin pergerakan harga Bitcoin di masa depan dan bukan merupakan saran investasi.
