# 🧠 Analisis Sentimen Bahasa Indonesia

Aplikasi analisis sentimen teks berbahasa Indonesia menggunakan **Flask** sebagai backend API dan **Streamlit** sebagai frontend UI.

## 📁 Struktur Proyek

```
sentiment_app/
├── app.py               # Flask Backend API
├── streamlit_app.py     # Streamlit Frontend UI
├── requirements.txt     # Dependensi Python
├── data/
│   └── dataset.csv      # Dataset bahasa Indonesia (200+ contoh)
└── README.md
```

## 🧑‍💻 Cara Menjalankan (Lokal)

### 1. Install dependensi
```bash
pip install -r requirements.txt
```

### 2. Jalankan Flask API
```bash
python app.py
# API berjalan di http://localhost:5000
```

### 3. Jalankan Streamlit (di terminal terpisah)
```bash
streamlit run streamlit_app.py
# UI berjalan di http://localhost:8501
```

> **Catatan:** Saat deploy di Streamlit Cloud, Flask otomatis berjalan di background thread.

---

## 🌐 Deploy ke Streamlit Cloud

1. Push semua file ke GitHub repository
2. Buka [share.streamlit.io](https://share.streamlit.io)
3. Klik **New App**
4. Pilih repository dan set **Main file path** ke `streamlit_app.py`
5. Klik **Deploy**

---

## 🔌 Endpoint Flask API

| Method | Endpoint | Deskripsi |
|--------|----------|-----------|
| GET | `/` | Info API |
| GET | `/health` | Status kesehatan |
| GET | `/models` | Daftar model tersedia |
| GET | `/stats` | Statistik model & dataset |
| POST | `/predict` | Prediksi satu teks |
| POST | `/predict/batch` | Prediksi banyak teks |
| POST | `/train` | Latih ulang model |

### Contoh Request `/predict`
```json
POST /predict
{
    "text": "Produk ini sangat bagus dan berkualitas tinggi!",
    "model": "naive_bayes"
}
```

### Contoh Response
```json
{
    "success": true,
    "result": {
        "text_original": "Produk ini sangat bagus dan berkualitas tinggi!",
        "text_preprocessed": "produk bagus berkualitas tinggi",
        "prediction": "positif",
        "prediction_label": "Positif 😊",
        "confidence": {
            "positif": 87.3,
            "negatif": 5.2,
            "netral": 7.5
        },
        "model_used": "naive_bayes"
    }
}
```

### Contoh Request `/predict/batch`
```json
POST /predict/batch
{
    "texts": [
        "Produk ini sangat bagus!",
        "Kualitas sangat buruk dan mengecewakan.",
        "Barang sudah sampai kondisi normal."
    ],
    "model": "logistic_regression"
}
```

---

## 🤖 Model yang Tersedia

| Model | Deskripsi |
|-------|-----------|
| `naive_bayes` | Naive Bayes (MultinomialNB) - Cepat & ringan |
| `logistic_regression` | Logistic Regression - Akurasi tinggi |
| `svm` | Support Vector Machine (LinearSVC) - Terbaik untuk teks |

---

## ⚙️ Pipeline NLP

1. **Lowercase** → Semua teks diubah ke huruf kecil
2. **Hapus URL & mention** → Bersihkan link dan @username
3. **Hapus karakter non-alfabet**
4. **Hapus stopwords** bahasa Indonesia
5. **TF-IDF Vectorization** (n-gram 1-2, max 5000 fitur)
6. **Klasifikasi** dengan model pilihan

---

## 📊 Dataset

Dataset terdiri dari **200+ contoh** teks berbahasa Indonesia dengan 3 kelas:
- **Positif** 😊 - Review produk positif, pujian, kepuasan
- **Negatif** 😞 - Review produk negatif, keluhan, kekecewaan  
- **Netral** 😐 - Pernyataan faktual, netral tanpa emosi kuat

Sumber: Dataset buatan sendiri berdasarkan pola review umum Indonesia.

---

## 🛠️ Tech Stack

- **Backend**: Flask + scikit-learn
- **Frontend**: Streamlit + Plotly
- **NLP**: TF-IDF + Naive Bayes / Logistic Regression / SVM
- **Bahasa**: Python 3.9+
