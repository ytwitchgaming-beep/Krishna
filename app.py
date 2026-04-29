from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import re
import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

# ─── Stopwords Bahasa Indonesia ───────────────────────────────────────────────
STOPWORDS_ID = {
    'yang', 'dan', 'di', 'ke', 'dari', 'ini', 'itu', 'dengan', 'untuk',
    'adalah', 'ada', 'atau', 'juga', 'pada', 'dalam', 'saya', 'kami',
    'kita', 'anda', 'mereka', 'dia', 'ia', 'nya', 'akan', 'sudah',
    'sudah', 'telah', 'tidak', 'bukan', 'jangan', 'tak', 'lebih',
    'sangat', 'sangat', 'sekali', 'paling', 'sudah', 'bisa', 'dapat',
    'seperti', 'saat', 'serta', 'oleh', 'agar', 'tetapi', 'namun',
    'namun', 'karena', 'jika', 'kalau', 'maka', 'lalu', 'kemudian',
    'pun', 'lah', 'kah', 'deh', 'dong', 'sih', 'kok', 'ya', 'yah',
    'oh', 'eh', 'ah', 'ih', 'uh', 'wah', 'hah', 'hem', 'hmm'
}

# ─── Text Preprocessing ───────────────────────────────────────────────────────
def preprocess_text(text):
    """Preprocessing teks bahasa Indonesia"""
    if not isinstance(text, str):
        return ""
    
    # Lowercase
    text = text.lower()
    
    # Hapus URL
    text = re.sub(r'http\S+|www\S+', '', text)
    
    # Hapus mention & hashtag
    text = re.sub(r'@\w+|#\w+', '', text)
    
    # Hapus karakter non-alfabet (kecuali spasi)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    
    # Hapus spasi berlebih
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Hapus stopwords
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS_ID and len(t) > 2]
    
    return ' '.join(tokens)

# ─── Model Manager ────────────────────────────────────────────────────────────
# Lokasi file cache model (disimpan di dalam repo agar persisten di Streamlit Cloud)
MODEL_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model_cache')
MODEL_CACHE_FILE = os.path.join(MODEL_CACHE_DIR, 'sentiment_model.pkl')

class SentimentModel:
    def __init__(self):
        self.vectorizer = None
        self.models = {}
        self.active_model = 'naive_bayes'
        self.is_trained = False
        self.training_stats = {}
        self.label_map = {
            'positif': 'Positif 😊',
            'negatif': 'Negatif 😞',
            'netral': 'Netral 😐'
        }

    # ── Persistensi ───────────────────────────────────────────────────────────

    def save(self, path=MODEL_CACHE_FILE):
        """Simpan model yang sudah dilatih ke file .pkl"""
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            payload = {
                'vectorizer': self.vectorizer,
                'models': self.models,
                'training_stats': self.training_stats,
            }
            with open(path, 'wb') as f:
                pickle.dump(payload, f)
            print(f"[INFO] Model disimpan ke {path}")
            return True
        except Exception as e:
            print(f"[WARNING] Gagal menyimpan model: {e}")
            return False

    def load(self, path=MODEL_CACHE_FILE):
        """Muat model dari file .pkl jika tersedia. Return True jika berhasil."""
        if not os.path.exists(path):
            return False
        try:
            with open(path, 'rb') as f:
                payload = pickle.load(f)
            self.vectorizer = payload['vectorizer']
            self.models = payload['models']
            self.training_stats = payload['training_stats']
            self.is_trained = True
            print(f"[INFO] Model dimuat dari cache: {path}")
            return True
        except Exception as e:
            print(f"[WARNING] Gagal memuat model cache: {e}")
            return False

    # ── Training ──────────────────────────────────────────────────────────────

    def train(self, data_path='data/dataset.csv', force=False):
        """
        Train semua model dari dataset.

        Args:
            data_path: Path ke file CSV dataset.
            force: Jika True, paksa training ulang meski cache sudah ada.
        """
        # Coba muat dari cache terlebih dahulu (kecuali force=True)
        if not force and self.load():
            return True, "Model dimuat dari cache (tidak perlu training ulang)."

        try:
            df = pd.read_csv(data_path)
            df = df.dropna()
            df['text_clean'] = df['text'].apply(preprocess_text)
            df = df[df['text_clean'].str.len() > 0]

            X = df['text_clean']
            y = df['label']

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            # TF-IDF Vectorizer
            self.vectorizer = TfidfVectorizer(
                ngram_range=(1, 2),
                max_features=5000,
                min_df=1,
                max_df=0.95
            )
            X_train_vec = self.vectorizer.fit_transform(X_train)
            X_test_vec = self.vectorizer.transform(X_test)

            # Train 3 model
            classifiers = {
                'naive_bayes': MultinomialNB(alpha=0.5),
                'logistic_regression': LogisticRegression(max_iter=1000, C=1.0),
                'svm': LinearSVC(max_iter=2000, C=1.0)
            }

            stats = {}
            for name, clf in classifiers.items():
                clf.fit(X_train_vec, y_train)
                y_pred = clf.predict(X_test_vec)
                acc = accuracy_score(y_test, y_pred)
                report = classification_report(y_test, y_pred, output_dict=True)
                cm = confusion_matrix(y_test, y_pred, labels=['positif', 'negatif', 'netral'])

                self.models[name] = clf
                stats[name] = {
                    'accuracy': round(acc * 100, 2),
                    'report': report,
                    'confusion_matrix': cm.tolist(),
                    'labels': ['positif', 'negatif', 'netral']
                }

            # Dataset stats
            self.training_stats = {
                'models': stats,
                'dataset': {
                    'total': len(df),
                    'train_size': len(X_train),
                    'test_size': len(X_test),
                    'distribution': df['label'].value_counts().to_dict()
                }
            }

            self.is_trained = True

            # Simpan hasil training ke cache
            self.save()

            return True, "Model berhasil dilatih dan disimpan ke cache!"

        except Exception as e:
            return False, str(e)
    
    def predict(self, text, model_name=None):
        """Prediksi sentimen satu teks"""
        if not self.is_trained:
            return None, "Model belum dilatih"
        
        model_name = model_name or self.active_model
        if model_name not in self.models:
            return None, f"Model '{model_name}' tidak ditemukan"
        
        text_clean = preprocess_text(text)
        if not text_clean:
            return None, "Teks tidak valid setelah preprocessing"
        
        text_vec = self.vectorizer.transform([text_clean])
        clf = self.models[model_name]
        
        prediction = clf.predict(text_vec)[0]
        
        # Confidence score (hanya untuk model yang support)
        confidence = None
        if hasattr(clf, 'predict_proba'):
            proba = clf.predict_proba(text_vec)[0]
            classes = clf.classes_
            confidence = {c: round(float(p) * 100, 2) for c, p in zip(classes, proba)}
        elif hasattr(clf, 'decision_function'):
            decision = clf.decision_function(text_vec)[0]
            classes = clf.classes_
            # Softmax approximation untuk SVM
            exp_d = np.exp(decision - np.max(decision))
            proba = exp_d / exp_d.sum()
            confidence = {c: round(float(p) * 100, 2) for c, p in zip(classes, proba)}
        
        return {
            'text_original': text,
            'text_preprocessed': text_clean,
            'prediction': prediction,
            'prediction_label': self.label_map.get(prediction, prediction),
            'confidence': confidence,
            'model_used': model_name
        }, None
    
    def predict_batch(self, texts, model_name=None):
        """Prediksi sentimen batch teks"""
        results = []
        for text in texts:
            result, err = self.predict(text, model_name)
            if err:
                results.append({'text': text, 'error': err})
            else:
                results.append(result)
        return results

# ─── Inisialisasi Model ───────────────────────────────────────────────────────
sentiment_model = SentimentModel()

# Auto-train saat startup
@app.before_request
def initialize():
    global sentiment_model
    if not sentiment_model.is_trained:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(base_dir, 'data', 'dataset.csv')
        success, msg = sentiment_model.train(data_path)
        if not success:
            print(f"[WARNING] Auto-train gagal: {msg}")

# ─── Routes ───────────────────────────────────────────────────────────────────
@app.route('/')
def home():
    return jsonify({
        'status': 'running',
        'message': 'Sentiment Analysis API - Bahasa Indonesia',
        'version': '1.0.0',
        'endpoints': {
            'POST /predict': 'Prediksi sentimen satu teks',
            'POST /predict/batch': 'Prediksi sentimen banyak teks',
            'GET /stats': 'Statistik model dan dataset',
            'GET /models': 'Daftar model tersedia',
            'POST /train': 'Latih ulang model',
            'GET /health': 'Health check'
        }
    })

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'model_trained': sentiment_model.is_trained
    })

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'Field "text" dibutuhkan'}), 400
    
    text = data.get('text', '').strip()
    model_name = data.get('model', 'naive_bayes')
    
    if not text:
        return jsonify({'error': 'Teks tidak boleh kosong'}), 400
    
    result, error = sentiment_model.predict(text, model_name)
    if error:
        return jsonify({'error': error}), 500
    
    return jsonify({'success': True, 'result': result})

@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    data = request.get_json()
    if not data or 'texts' not in data:
        return jsonify({'error': 'Field "texts" (array) dibutuhkan'}), 400
    
    texts = data.get('texts', [])
    model_name = data.get('model', 'naive_bayes')
    
    if not isinstance(texts, list) or len(texts) == 0:
        return jsonify({'error': 'texts harus array yang tidak kosong'}), 400
    
    if len(texts) > 100:
        return jsonify({'error': 'Maksimal 100 teks per request'}), 400
    
    results = sentiment_model.predict_batch(texts, model_name)
    
    # Summary
    sentiments = [r.get('prediction') for r in results if 'prediction' in r]
    summary = {
        'total': len(results),
        'positif': sentiments.count('positif'),
        'negatif': sentiments.count('negatif'),
        'netral': sentiments.count('netral')
    }
    
    return jsonify({'success': True, 'results': results, 'summary': summary})

@app.route('/stats')
def stats():
    if not sentiment_model.is_trained:
        return jsonify({'error': 'Model belum dilatih'}), 503
    
    return jsonify({'success': True, 'stats': sentiment_model.training_stats})

@app.route('/models')
def get_models():
    model_info = {
        'naive_bayes': {
            'name': 'Naive Bayes',
            'description': 'Model probabilistik berbasis teorema Bayes, cepat dan efisien',
            'pros': 'Cepat, interpretable, baik untuk teks',
            'cons': 'Asumsi independensi fitur'
        },
        'logistic_regression': {
            'name': 'Logistic Regression',
            'description': 'Model linear untuk klasifikasi dengan estimasi probabilitas',
            'pros': 'Output probabilitas, robust',
            'cons': 'Lebih lambat dari Naive Bayes'
        },
        'svm': {
            'name': 'Support Vector Machine',
            'description': 'Model margin maksimum untuk pemisahan kelas',
            'pros': 'Akurasi tinggi, efektif di ruang dimensi tinggi',
            'cons': 'Tidak langsung memberikan probabilitas'
        }
    }
    
    # Tambahkan akurasi jika model sudah dilatih
    if sentiment_model.is_trained:
        for key in model_info:
            if key in sentiment_model.training_stats.get('models', {}):
                model_info[key]['accuracy'] = sentiment_model.training_stats['models'][key]['accuracy']
    
    return jsonify({'models': model_info, 'active': sentiment_model.active_model})

@app.route('/train', methods=['POST'])
def train():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, 'data', 'dataset.csv')
    # force=True → selalu latih ulang dari data, tidak pakai cache lama
    success, msg = sentiment_model.train(data_path, force=True)
    if success:
        return jsonify({'success': True, 'message': msg})
    else:
        return jsonify({'success': False, 'error': msg}), 500

if __name__ == '__main__':
    print("🚀 Starting Sentiment Analysis API...")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, 'data', 'dataset.csv')
    success, msg = sentiment_model.train(data_path)
    print(f"✅ {msg}" if success else f"❌ {msg}")
    app.run(debug=True, host='0.0.0.0', port=5000)
