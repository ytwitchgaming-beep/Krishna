import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import time
import os
import sys

# ─── Konfigurasi Halaman ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="Analisis Sentimen Bahasa Indonesia",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Inisialisasi Flask API (mode embedded) ───────────────────────────────────
@st.cache_resource
def start_flask_in_background():
    """Jalankan Flask di background thread saat di Streamlit Cloud"""
    import threading
    
    # Tambahkan direktori app ke path
    app_dir = os.path.dirname(os.path.abspath(__file__))
    if app_dir not in sys.path:
        sys.path.insert(0, app_dir)
    
    from app import app, sentiment_model
    
    # Pre-train model
    data_path = os.path.join(app_dir, 'data', 'dataset.csv')
    if not sentiment_model.is_trained:
        sentiment_model.train(data_path)
    
    # Jalankan Flask di thread terpisah
    def run_flask():
        app.run(host='127.0.0.1', port=5000, debug=False, use_reloader=False)
    
    thread = threading.Thread(target=run_flask, daemon=True)
    thread.start()
    time.sleep(2)  # Tunggu Flask siap
    return "http://127.0.0.1:5000"

# Start Flask
API_BASE = start_flask_in_background()

# ─── Helper Functions ─────────────────────────────────────────────────────────
def call_api(endpoint, method='GET', data=None):
    """Wrapper untuk memanggil Flask API"""
    url = f"{API_BASE}{endpoint}"
    try:
        if method == 'GET':
            response = requests.get(url, timeout=30)
        else:
            response = requests.post(url, json=data, timeout=30)
        return response.json(), None
    except Exception as e:
        return None, str(e)

def get_sentiment_color(sentiment):
    colors = {
        'positif': '#10b981',
        'negatif': '#ef4444',
        'netral': '#6b7280'
    }
    return colors.get(sentiment, '#6b7280')

def get_sentiment_emoji(sentiment):
    emojis = {
        'positif': '😊',
        'negatif': '😞',
        'netral': '😐'
    }
    return emojis.get(sentiment, '❓')

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
    
    * { font-family: 'Plus Jakarta Sans', sans-serif; }
    
    .main { background: #0f0f14; }
    
    .stApp {
        background: linear-gradient(135deg, #0f0f14 0%, #13131a 50%, #0f0f14 100%);
    }
    
    /* Header */
    .hero-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 20px;
        padding: 40px;
        margin-bottom: 32px;
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    .hero-header::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(ellipse at center, rgba(99,102,241,0.1) 0%, transparent 60%);
        pointer-events: none;
    }
    .hero-title {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #a78bfa, #60a5fa, #34d399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        line-height: 1.2;
    }
    .hero-subtitle {
        color: rgba(255,255,255,0.6);
        font-size: 1rem;
        margin-top: 10px;
        font-weight: 400;
    }
    
    /* Card */
    .card {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 16px;
        backdrop-filter: blur(10px);
    }
    
    /* Result Card */
    .result-positive {
        background: linear-gradient(135deg, rgba(16,185,129,0.1), rgba(16,185,129,0.05));
        border: 1px solid rgba(16,185,129,0.3);
        border-radius: 16px;
        padding: 24px;
    }
    .result-negative {
        background: linear-gradient(135deg, rgba(239,68,68,0.1), rgba(239,68,68,0.05));
        border: 1px solid rgba(239,68,68,0.3);
        border-radius: 16px;
        padding: 24px;
    }
    .result-neutral {
        background: linear-gradient(135deg, rgba(107,114,128,0.1), rgba(107,114,128,0.05));
        border: 1px solid rgba(107,114,128,0.3);
        border-radius: 16px;
        padding: 24px;
    }
    
    .prediction-badge {
        display: inline-block;
        font-size: 2rem;
        font-weight: 800;
        padding: 8px 24px;
        border-radius: 50px;
        margin-bottom: 12px;
    }
    .badge-positive { 
        background: rgba(16,185,129,0.2); 
        color: #10b981;
        border: 1px solid rgba(16,185,129,0.4);
    }
    .badge-negative { 
        background: rgba(239,68,68,0.2); 
        color: #ef4444;
        border: 1px solid rgba(239,68,68,0.4);
    }
    .badge-neutral { 
        background: rgba(107,114,128,0.2); 
        color: #9ca3af;
        border: 1px solid rgba(107,114,128,0.4);
    }
    
    /* Metric Card */
    .metric-box {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #a78bfa;
    }
    .metric-label {
        font-size: 0.8rem;
        color: rgba(255,255,255,0.5);
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 4px;
    }
    
    /* Section title */
    .section-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: rgba(255,255,255,0.9);
        margin-bottom: 16px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    /* Tag */
    .model-tag {
        display: inline-block;
        background: rgba(99,102,241,0.2);
        color: #a78bfa;
        border: 1px solid rgba(99,102,241,0.3);
        border-radius: 6px;
        padding: 2px 10px;
        font-size: 0.75rem;
        font-weight: 600;
        font-family: 'JetBrains Mono', monospace;
    }
    
    /* Override Streamlit */
    .stTextArea textarea {
        background: rgba(255,255,255,0.05) !important;
        border: 1px solid rgba(255,255,255,0.15) !important;
        border-radius: 12px !important;
        color: white !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
    }
    .stTextArea textarea:focus {
        border-color: rgba(99,102,241,0.6) !important;
        box-shadow: 0 0 0 2px rgba(99,102,241,0.2) !important;
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        padding: 10px 24px !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        transition: all 0.2s !important;
    }
    .stButton>button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 20px rgba(99,102,241,0.4) !important;
    }
    
    .stSelectbox select {
        background: rgba(255,255,255,0.05) !important;
        border-radius: 10px !important;
        color: white !important;
    }
    
    [data-testid="stSidebar"] {
        background: rgba(15,15,20,0.95) !important;
        border-right: 1px solid rgba(255,255,255,0.08) !important;
    }
    
    h1, h2, h3 { color: white !important; }
    p, label, span { color: rgba(255,255,255,0.8) !important; }
    
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(255,255,255,0.03) !important;
        border-radius: 12px !important;
        padding: 4px !important;
        gap: 4px !important;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent !important;
        color: rgba(255,255,255,0.5) !important;
        border-radius: 8px !important;
        font-weight: 500 !important;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(99,102,241,0.3) !important;
        color: white !important;
    }
    
    .preprocessed-text {
        background: rgba(255,255,255,0.05);
        border: 1px dashed rgba(255,255,255,0.15);
        border-radius: 8px;
        padding: 12px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
        color: rgba(255,255,255,0.6);
        margin-top: 8px;
    }
    
    .stAlert { border-radius: 10px !important; }
    
    /* Table styling */
    .stDataFrame { border-radius: 12px !important; }
</style>
""", unsafe_allow_html=True)

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Konfigurasi Model")
    st.markdown("---")
    
    # Ambil info model
    models_data, err = call_api('/models')
    
    model_options = {
        'naive_bayes': '🔵 Naive Bayes',
        'logistic_regression': '🟣 Logistic Regression',
        'svm': '🟠 Support Vector Machine'
    }
    
    selected_model = st.selectbox(
        "Pilih Model Klasifikasi",
        options=list(model_options.keys()),
        format_func=lambda x: model_options[x],
        help="Pilih algoritma machine learning yang ingin digunakan"
    )
    
    # Tampilkan info model
    if models_data and not err:
        model_info = models_data['models'].get(selected_model, {})
        st.markdown(f"""
        <div class="card" style="margin-top:16px">
            <div style="font-size:0.8rem; color:rgba(255,255,255,0.5); margin-bottom:8px">📖 Info Model</div>
            <div style="font-size:0.85rem; color:rgba(255,255,255,0.8); margin-bottom:12px">{model_info.get('description', '-')}</div>
            <div style="font-size:0.75rem; color:#10b981; margin-bottom:4px">✅ {model_info.get('pros', '-')}</div>
            <div style="font-size:0.75rem; color:#f59e0b; margin-bottom:12px">⚠️ {model_info.get('cons', '-')}</div>
            <div style="font-size:1.1rem; font-weight:700; color:#a78bfa">
                🎯 Akurasi: {model_info.get('accuracy', '?')}%
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Retrain button
    if st.button("🔄 Latih Ulang Model", use_container_width=True):
        with st.spinner("Melatih model..."):
            result, err = call_api('/train', method='POST')
            if result and result.get('success'):
                st.success("✅ Model berhasil dilatih ulang!")
                st.rerun()
            else:
                st.error(f"❌ Gagal: {err}")
    
    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.75rem; color:rgba(255,255,255,0.3); text-align:center">
        🇮🇩 Sentiment Analysis ID<br>
        Flask API + Streamlit UI<br>
        <span style="color:rgba(99,102,241,0.6)">v1.0.0</span>
    </div>
    """, unsafe_allow_html=True)

# ─── Main Content ─────────────────────────────────────────────────────────────

# Hero Header
st.markdown("""
<div class="hero-header">
    <div class="hero-title">🧠 Analisis Sentimen Bahasa Indonesia</div>
    <div class="hero-subtitle">
        Deteksi sentimen teks berbahasa Indonesia menggunakan Machine Learning<br>
        <span style="color:rgba(167,139,250,0.7)">Naive Bayes · Logistic Regression · Support Vector Machine</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Cek status API
health, err = call_api('/health')
if err or not health:
    st.error("⚠️ Flask API tidak dapat dijangkau. Silakan refresh halaman.")
    st.stop()

# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "✍️ Analisis Teks", 
    "📦 Analisis Batch",
    "📊 Statistik Model"
])

# ═══════════════════════════════════════════════════════
# TAB 1: ANALISIS TEKS TUNGGAL
# ═══════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-title">✍️ Masukkan Teks untuk Dianalisis</div>', unsafe_allow_html=True)
    
    # Contoh teks
    col_ex1, col_ex2, col_ex3 = st.columns(3)
    with col_ex1:
        if st.button("😊 Contoh Positif", use_container_width=True):
            st.session_state['input_text'] = "Produk ini sangat bagus dan berkualitas tinggi, pelayanan seller juga sangat ramah dan pengiriman cepat!"
    with col_ex2:
        if st.button("😞 Contoh Negatif", use_container_width=True):
            st.session_state['input_text'] = "Sangat kecewa dengan produk ini, kualitas sangat buruk dan tidak sesuai deskripsi sama sekali."
    with col_ex3:
        if st.button("😐 Contoh Netral", use_container_width=True):
            st.session_state['input_text'] = "Produk sudah sampai, kondisi normal dan sesuai dengan foto yang ada di deskripsi."
    
    # Input teks
    default_text = st.session_state.get('input_text', '')
    input_text = st.text_area(
        "Teks Bahasa Indonesia",
        value=default_text,
        height=120,
        placeholder="Ketik atau tempel teks berbahasa Indonesia di sini...\n\nContoh: 'Produk ini sangat bagus dan berkualitas tinggi!'",
        label_visibility="collapsed"
    )
    
    col_btn1, col_btn2, col_spacer = st.columns([1, 1, 4])
    with col_btn1:
        analyze_btn = st.button("🔍 Analisis Sentimen", use_container_width=True, type="primary")
    with col_btn2:
        if st.button("🗑️ Hapus", use_container_width=True):
            st.session_state['input_text'] = ''
            st.rerun()
    
    # Hasil Analisis
    if analyze_btn and input_text.strip():
        with st.spinner("Menganalisis sentimen..."):
            result_data, err = call_api('/predict', method='POST', data={
                'text': input_text,
                'model': selected_model
            })
        
        if err or not result_data:
            st.error(f"❌ Gagal menganalisis: {err}")
        else:
            result = result_data['result']
            prediction = result['prediction']
            confidence = result.get('confidence', {})
            
            st.markdown("---")
            st.markdown('<div class="section-title">📊 Hasil Analisis</div>', unsafe_allow_html=True)
            
            col_result, col_detail = st.columns([1, 2])
            
            with col_result:
                class_map = {
                    'positif': ('result-positive', 'badge-positive'),
                    'negatif': ('result-negative', 'badge-negative'),
                    'netral': ('result-neutral', 'badge-neutral')
                }
                card_class, badge_class = class_map.get(prediction, ('result-neutral', 'badge-neutral'))
                emoji = get_sentiment_emoji(prediction)
                
                st.markdown(f"""
                <div class="{card_class}">
                    <div style="text-align:center">
                        <div style="font-size:4rem">{emoji}</div>
                        <div class="prediction-badge {badge_class}">
                            {prediction.upper()}
                        </div>
                        <div style="color:rgba(255,255,255,0.5); font-size:0.8rem; margin-top:8px">
                            Model: <span class="model-tag">{selected_model}</span>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col_detail:
                # Confidence bar chart
                if confidence:
                    conf_df = pd.DataFrame({
                        'Sentimen': list(confidence.keys()),
                        'Probabilitas (%)': list(confidence.values())
                    })
                    
                    color_map = {
                        'positif': '#10b981',
                        'negatif': '#ef4444',
                        'netral': '#6b7280'
                    }
                    colors = [color_map.get(s, '#6366f1') for s in conf_df['Sentimen']]
                    
                    fig = go.Figure(go.Bar(
                        x=conf_df['Probabilitas (%)'],
                        y=conf_df['Sentimen'],
                        orientation='h',
                        marker_color=colors,
                        text=[f"{v:.1f}%" for v in conf_df['Probabilitas (%)']],
                        textposition='auto',
                    ))
                    fig.update_layout(
                        title="Distribusi Kepercayaan Model",
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font_color='rgba(255,255,255,0.8)',
                        height=200,
                        margin=dict(l=0, r=0, t=40, b=0),
                        xaxis=dict(range=[0, 100], gridcolor='rgba(255,255,255,0.1)'),
                        yaxis=dict(gridcolor='rgba(255,255,255,0.1)')
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Teks yang diproses
                st.markdown(f"""
                <div style="margin-top:8px">
                    <div style="font-size:0.75rem; color:rgba(255,255,255,0.4); margin-bottom:4px">
                        📝 Teks setelah preprocessing:
                    </div>
                    <div class="preprocessed-text">{result['text_preprocessed']}</div>
                </div>
                """, unsafe_allow_html=True)
    
    elif analyze_btn:
        st.warning("⚠️ Silakan masukkan teks terlebih dahulu.")

# ═══════════════════════════════════════════════════════
# TAB 2: ANALISIS BATCH
# ═══════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-title">📦 Analisis Teks Massal</div>', unsafe_allow_html=True)
    
    input_method = st.radio(
        "Metode Input",
        ["✏️ Ketik Manual", "📁 Upload CSV"],
        horizontal=True
    )
    
    batch_texts = []
    
    if input_method == "✏️ Ketik Manual":
        batch_input = st.text_area(
            "Teks (satu teks per baris)",
            height=200,
            placeholder="Masukkan teks satu per baris:\n\nProduk ini sangat bagus!\nKualitas jelek tidak worth it.\nBarang sampai sesuai deskripsi.",
            label_visibility="visible"
        )
        if batch_input.strip():
            batch_texts = [t.strip() for t in batch_input.strip().split('\n') if t.strip()]
    
    else:
        uploaded_file = st.file_uploader(
            "Upload file CSV (kolom 'text' wajib ada)",
            type=['csv'],
            help="CSV harus memiliki kolom bernama 'text'"
        )
        if uploaded_file:
            try:
                df_upload = pd.read_csv(uploaded_file)
                if 'text' in df_upload.columns:
                    batch_texts = df_upload['text'].dropna().tolist()
                    st.success(f"✅ {len(batch_texts)} baris teks berhasil dimuat")
                    st.dataframe(df_upload.head(5), use_container_width=True)
                else:
                    st.error("❌ CSV harus memiliki kolom 'text'")
            except Exception as e:
                st.error(f"❌ Gagal membaca file: {e}")
    
    if batch_texts:
        st.markdown(f"📌 **{len(batch_texts)} teks** siap dianalisis")
    
    if st.button("🚀 Analisis Semua Teks", use_container_width=False, type="primary") and batch_texts:
        with st.spinner(f"Menganalisis {len(batch_texts)} teks..."):
            result_data, err = call_api('/predict/batch', method='POST', data={
                'texts': batch_texts[:100],
                'model': selected_model
            })
        
        if err or not result_data:
            st.error(f"❌ Gagal: {err}")
        else:
            results = result_data['results']
            summary = result_data['summary']
            
            st.markdown("---")
            
            # Summary metrics
            col_t, col_p, col_n, col_neu = st.columns(4)
            with col_t:
                st.markdown(f'<div class="metric-box"><div class="metric-value">{summary["total"]}</div><div class="metric-label">Total Teks</div></div>', unsafe_allow_html=True)
            with col_p:
                st.markdown(f'<div class="metric-box"><div class="metric-value" style="color:#10b981">{summary["positif"]}</div><div class="metric-label">Positif 😊</div></div>', unsafe_allow_html=True)
            with col_n:
                st.markdown(f'<div class="metric-box"><div class="metric-value" style="color:#ef4444">{summary["negatif"]}</div><div class="metric-label">Negatif 😞</div></div>', unsafe_allow_html=True)
            with col_neu:
                st.markdown(f'<div class="metric-box"><div class="metric-value" style="color:#9ca3af">{summary["netral"]}</div><div class="metric-label">Netral 😐</div></div>', unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            col_pie, col_table = st.columns([1, 2])
            
            with col_pie:
                # Pie chart
                pie_data = {
                    'Sentimen': ['Positif', 'Negatif', 'Netral'],
                    'Jumlah': [summary['positif'], summary['negatif'], summary['netral']]
                }
                fig_pie = px.pie(
                    pie_data,
                    values='Jumlah',
                    names='Sentimen',
                    color='Sentimen',
                    color_discrete_map={
                        'Positif': '#10b981',
                        'Negatif': '#ef4444',
                        'Netral': '#6b7280'
                    },
                    hole=0.5
                )
                fig_pie.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font_color='rgba(255,255,255,0.8)',
                    showlegend=True,
                    legend=dict(bgcolor='rgba(0,0,0,0)'),
                    height=280,
                    margin=dict(l=0, r=0, t=20, b=0)
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col_table:
                # Tabel hasil
                table_data = []
                for r in results:
                    if 'prediction' in r:
                        table_data.append({
                            'Teks': r['text_original'][:80] + ('...' if len(r['text_original']) > 80 else ''),
                            'Sentimen': f"{get_sentiment_emoji(r['prediction'])} {r['prediction'].capitalize()}",
                        })
                
                if table_data:
                    df_results = pd.DataFrame(table_data)
                    st.dataframe(df_results, use_container_width=True, height=260)
            
            # Download hasil
            full_results = []
            for r in results:
                if 'prediction' in r:
                    full_results.append({
                        'teks_asli': r['text_original'],
                        'teks_preprocessed': r['text_preprocessed'],
                        'sentimen': r['prediction'],
                        'model': r['model_used']
                    })
            
            if full_results:
                df_download = pd.DataFrame(full_results)
                csv = df_download.to_csv(index=False, encoding='utf-8-sig')
                st.download_button(
                    label="⬇️ Download Hasil (CSV)",
                    data=csv,
                    file_name="hasil_sentimen.csv",
                    mime="text/csv",
                    use_container_width=False
                )

# ═══════════════════════════════════════════════════════
# TAB 3: STATISTIK MODEL
# ═══════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-title">📊 Statistik Performa Model</div>', unsafe_allow_html=True)
    
    stats_data, err = call_api('/stats')
    
    if err or not stats_data:
        st.error("❌ Tidak dapat mengambil statistik model.")
    else:
        stats = stats_data['stats']
        dataset_info = stats.get('dataset', {})
        models_stats = stats.get('models', {})
        
        # Dataset info
        st.markdown("#### 📁 Informasi Dataset")
        col_d1, col_d2, col_d3, col_d4 = st.columns(4)
        with col_d1:
            st.markdown(f'<div class="metric-box"><div class="metric-value">{dataset_info.get("total", 0)}</div><div class="metric-label">Total Data</div></div>', unsafe_allow_html=True)
        with col_d2:
            st.markdown(f'<div class="metric-box"><div class="metric-value">{dataset_info.get("train_size", 0)}</div><div class="metric-label">Data Latih</div></div>', unsafe_allow_html=True)
        with col_d3:
            st.markdown(f'<div class="metric-box"><div class="metric-value">{dataset_info.get("test_size", 0)}</div><div class="metric-label">Data Uji</div></div>', unsafe_allow_html=True)
        with col_d4:
            dist = dataset_info.get('distribution', {})
            st.markdown(f'<div class="metric-box"><div class="metric-value">3</div><div class="metric-label">Kelas Sentimen</div></div>', unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Distribusi dataset
        if dist:
            col_dist, col_acc = st.columns(2)
            
            with col_dist:
                st.markdown("#### 📊 Distribusi Label Dataset")
                dist_df = pd.DataFrame({
                    'Label': list(dist.keys()),
                    'Jumlah': list(dist.values())
                })
                fig_dist = px.bar(
                    dist_df,
                    x='Label',
                    y='Jumlah',
                    color='Label',
                    color_discrete_map={
                        'positif': '#10b981',
                        'negatif': '#ef4444',
                        'netral': '#6b7280'
                    },
                    text='Jumlah'
                )
                fig_dist.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font_color='rgba(255,255,255,0.8)',
                    height=280,
                    margin=dict(l=0, r=0, t=10, b=0),
                    showlegend=False,
                    xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
                    yaxis=dict(gridcolor='rgba(255,255,255,0.05)')
                )
                st.plotly_chart(fig_dist, use_container_width=True)
            
            with col_acc:
                st.markdown("#### 🏆 Perbandingan Akurasi Model")
                acc_data = {
                    'Model': ['Naive Bayes', 'Logistic Regression', 'SVM'],
                    'Akurasi (%)': [
                        models_stats.get('naive_bayes', {}).get('accuracy', 0),
                        models_stats.get('logistic_regression', {}).get('accuracy', 0),
                        models_stats.get('svm', {}).get('accuracy', 0)
                    ]
                }
                fig_acc = px.bar(
                    acc_data,
                    x='Akurasi (%)',
                    y='Model',
                    orientation='h',
                    color='Akurasi (%)',
                    color_continuous_scale='Viridis',
                    text='Akurasi (%)'
                )
                fig_acc.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font_color='rgba(255,255,255,0.8)',
                    height=280,
                    margin=dict(l=0, r=0, t=10, b=0),
                    coloraxis_showscale=False,
                    xaxis=dict(range=[0, 100], gridcolor='rgba(255,255,255,0.1)'),
                    yaxis=dict(gridcolor='rgba(255,255,255,0.05)')
                )
                st.plotly_chart(fig_acc, use_container_width=True)
        
        # Laporan klasifikasi per model
        st.markdown("#### 📋 Laporan Klasifikasi Detail")
        model_tabs = st.tabs(["🔵 Naive Bayes", "🟣 Logistic Regression", "🟠 SVM"])
        
        for i, (model_key, tab) in enumerate(zip(
            ['naive_bayes', 'logistic_regression', 'svm'], 
            model_tabs
        )):
            with tab:
                if model_key in models_stats:
                    report = models_stats[model_key]['report']
                    acc = models_stats[model_key]['accuracy']
                    
                    st.markdown(f"""
                    <div class="card" style="text-align:center; margin-bottom:20px">
                        <div style="font-size:0.8rem; color:rgba(255,255,255,0.4)">AKURASI KESELURUHAN</div>
                        <div style="font-size:2.5rem; font-weight:800; color:#a78bfa">{acc}%</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Per-class metrics
                    rows = []
                    for label in ['positif', 'negatif', 'netral']:
                        if label in report:
                            r = report[label]
                            rows.append({
                                'Kelas': f"{get_sentiment_emoji(label)} {label.capitalize()}",
                                'Precision': f"{r['precision']:.3f}",
                                'Recall': f"{r['recall']:.3f}",
                                'F1-Score': f"{r['f1-score']:.3f}",
                                'Support': int(r['support'])
                            })
                    
                    if rows:
                        df_report = pd.DataFrame(rows)
                        st.dataframe(df_report, use_container_width=True, hide_index=True)
