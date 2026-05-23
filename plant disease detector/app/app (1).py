import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import time
import io
from datetime import datetime

# ─── PAGE CONFIG ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Plant Disease Detector",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── CLASS NAMES ────────────────────────────────────────────────────────────
CLASS_NAMES = [
    "Apple — Apple scab","Apple — Black rot","Apple — Cedar apple rust","Apple — Healthy",
    "Blueberry — Healthy","Cherry — Powdery mildew","Cherry — Healthy",
    "Corn — Cercospora leaf spot / Gray leaf spot","Corn — Common rust",
    "Corn — Northern leaf blight","Corn — Healthy","Grape — Black rot",
    "Grape — Esca (Black measles)","Grape — Leaf blight (Isariopsis leaf spot)",
    "Grape — Healthy","Orange — Haunglongbing (Citrus greening)",
    "Peach — Bacterial spot","Peach — Healthy","Pepper bell — Bacterial spot",
    "Pepper bell — Healthy","Potato — Early blight","Potato — Late blight",
    "Potato — Healthy","Raspberry — Healthy","Soybean — Healthy",
    "Squash — Powdery mildew","Strawberry — Leaf scorch","Strawberry — Healthy",
    "Tomato — Bacterial spot","Tomato — Early blight","Tomato — Late blight",
    "Tomato — Leaf mold","Tomato — Septoria leaf spot",
    "Tomato — Spider mites (Two-spotted spider mite)","Tomato — Target spot",
    "Tomato — Tomato Yellow Leaf Curl Virus","Tomato — Tomato mosaic virus",
    "Tomato — Healthy",
]

TREATMENTS = {
    "Apple — Apple scab":"Apply fungicides containing myclobutanil or captan. Remove and destroy infected leaves. Prune trees to improve air circulation. Apply dormant oil spray in early spring.",
    "Apple — Black rot":"Prune out dead or infected wood. Apply copper-based fungicides. Remove mummified fruits. Ensure good air circulation and avoid wounding trees.",
    "Apple — Cedar apple rust":"Apply fungicides with myclobutanil or mancozeb at bud break. Remove nearby juniper/cedar hosts if possible. Apply preventive sprays every 7–10 days during wet weather.",
    "Apple — Healthy":"Your plant looks healthy! Maintain regular watering, fertilization, and pest monitoring to keep it thriving.",
    "Blueberry — Healthy":"Your plant looks healthy! Continue proper watering and acidic soil maintenance.",
    "Cherry — Powdery mildew":"Apply sulfur-based or potassium bicarbonate fungicides. Improve air circulation by pruning. Avoid overhead irrigation. Remove infected shoots promptly.",
    "Cherry — Healthy":"Your plant looks healthy! Maintain proper care and monitor regularly.",
    "Corn — Cercospora leaf spot / Gray leaf spot":"Plant resistant hybrids. Apply fungicides with strobilurins or triazoles at early disease onset. Rotate crops and till infected residue.",
    "Corn — Common rust":"Plant resistant varieties. Apply fungicides (propiconazole, azoxystrobin) if disease is severe. Early season planting can help avoid peak rust periods.",
    "Corn — Northern leaf blight":"Use resistant hybrids. Apply foliar fungicides at early tasseling if needed. Practice crop rotation and residue management.",
    "Corn — Healthy":"Your plant looks healthy! Maintain proper spacing and monitor for pests.",
    "Grape — Black rot":"Apply fungicides with mancozeb or myclobutanil. Remove infected berries and mummified fruit. Prune for air circulation. Apply preventive sprays before rain events.",
    "Grape — Esca (Black measles)":"Prune infected wood and apply wound sealants. Avoid water stress. No effective chemical cure; focus on prevention and removing infected vines.",
    "Grape — Leaf blight (Isariopsis leaf spot)":"Apply copper-based fungicides. Remove infected leaves. Improve vineyard air circulation. Avoid excessive moisture on leaves.",
    "Grape — Healthy":"Your plant looks healthy! Monitor regularly and maintain proper vineyard practices.",
    "Orange — Haunglongbing (Citrus greening)":"There is no cure for HLB. Remove and destroy infected trees to prevent spread. Control Asian citrus psyllid with insecticides. Plant certified disease-free nursery stock.",
    "Peach — Bacterial spot":"Apply copper-based bactericides. Use resistant varieties. Avoid overhead irrigation. Prune for air circulation and remove infected plant material.",
    "Peach — Healthy":"Your plant looks healthy! Continue regular care and monitoring.",
    "Pepper bell — Bacterial spot":"Apply copper-based bactericides. Use certified disease-free seed. Avoid working with plants when wet. Remove and destroy infected plant debris.",
    "Pepper bell — Healthy":"Your plant looks healthy! Maintain consistent watering and fertilization.",
    "Potato — Early blight":"Apply fungicides with chlorothalonil or mancozeb. Remove infected lower leaves. Ensure adequate plant nutrition, especially nitrogen. Practice crop rotation.",
    "Potato — Late blight":"Apply fungicides with metalaxyl or chlorothalonil immediately. Remove and destroy infected plant material. Avoid overhead irrigation. Plant resistant varieties.",
    "Potato — Healthy":"Your plant looks healthy! Monitor for early signs of blight, especially in wet conditions.",
    "Raspberry — Healthy":"Your plant looks healthy! Prune old canes and maintain good airflow.",
    "Soybean — Healthy":"Your plant looks healthy! Maintain proper field management practices.",
    "Squash — Powdery mildew":"Apply sulfur or potassium bicarbonate fungicides. Improve air circulation. Avoid overhead watering. Remove severely infected leaves.",
    "Strawberry — Leaf scorch":"Apply fungicides with captan or myclobutanil. Remove infected leaves. Avoid overhead irrigation. Ensure good drainage and air circulation.",
    "Strawberry — Healthy":"Your plant looks healthy! Keep runners managed and maintain good soil health.",
    "Tomato — Bacterial spot":"Apply copper-based bactericides preventively. Avoid working with plants when wet. Use disease-free transplants. Remove and destroy infected plant material.",
    "Tomato — Early blight":"Apply fungicides with chlorothalonil or mancozeb. Remove lower infected leaves. Mulch around plants to prevent soil splash. Ensure adequate plant nutrition.",
    "Tomato — Late blight":"Apply fungicides with metalaxyl or chlorothalonil. Remove infected plant material immediately. Avoid overhead irrigation. Destroy all infected plants to prevent spread.",
    "Tomato — Leaf mold":"Improve greenhouse ventilation. Apply fungicides with chlorothalonil. Reduce humidity and avoid wetting leaves. Use resistant varieties.",
    "Tomato — Septoria leaf spot":"Apply fungicides with chlorothalonil or mancozeb. Remove infected lower leaves. Avoid overhead watering. Mulch to reduce soil splash.",
    "Tomato — Spider mites (Two-spotted spider mite)":"Apply miticides or insecticidal soap. Increase humidity around plants. Introduce predatory mites. Remove heavily infested leaves and keep plants well-watered.",
    "Tomato — Target spot":"Apply fungicides with azoxystrobin or chlorothalonil. Remove infected leaves. Ensure good air circulation. Avoid overhead irrigation.",
    "Tomato — Tomato Yellow Leaf Curl Virus":"Control whitefly vectors with insecticides or reflective mulches. Remove and destroy infected plants. Use resistant varieties. Install insect-proof screens in greenhouses.",
    "Tomato — Tomato mosaic virus":"No cure available. Remove and destroy infected plants. Disinfect tools with bleach. Control aphid vectors. Use resistant varieties and certified disease-free seeds.",
    "Tomato — Healthy":"Your plant looks healthy! Maintain regular watering, staking, and monitoring for early pest signs.",
}

SEVERITY = {"scab":"Moderate","rot":"High","rust":"Moderate","mildew":"Moderate","blight":"High","spot":"Moderate","mold":"Moderate","virus":"Critical","greening":"Critical","mites":"Moderate","scorch":"Moderate","measles":"High"}
def severity_for(name):
    n=name.lower()
    if "healthy" in n: return "None"
    for k,v in SEVERITY.items():
        if k in n: return v
    return "Moderate"
SEVERITY_COLOR={"None":"#7fd49b","Moderate":"#e8c068","High":"#e09a3a","Critical":"#e07070"}
SEVERITY_POS={"None":8,"Moderate":38,"High":68,"Critical":94}

# ─── EMERALD + GOLD · GLASS + AURORA CSS ────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Cormorant+Garamond:wght@400;500;600;700&display=swap');

  :root {
    --bg: #06231a;
    --bg-2: #04170f;
    --emerald: #0d7a5f;
    --emerald-2: #14a37a;
    --gold: #c9a84c;
    --gold-2: #e8c971;
    --cream: #f5f0e0;
    --muted: #9ec0b0;
    --line: rgba(201,168,76,0.18);
    --glass: rgba(245,240,224,0.04);
    --glass-2: rgba(245,240,224,0.07);
  }

  html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    color: var(--cream) !important;
    font-family: 'Inter', sans-serif !important;
    overflow-x: hidden;
  }
  [data-testid="stAppViewContainer"]::before {
    content:''; position: fixed; inset: 0; pointer-events: none; z-index: 0;
    background:
      radial-gradient(700px 500px at 10% 0%,  rgba(20,163,122,0.22), transparent 60%),
      radial-gradient(800px 600px at 95% 10%, rgba(201,168,76,0.18), transparent 60%),
      radial-gradient(700px 700px at 50% 110%, rgba(13,122,95,0.20), transparent 60%);
    animation: aurora 22s ease-in-out infinite alternate;
    filter: blur(20px);
  }
  @keyframes aurora {
    0%   { transform: translate3d(0,0,0) scale(1);}
    50%  { transform: translate3d(-30px,20px,0) scale(1.05);}
    100% { transform: translate3d(20px,-15px,0) scale(1);}
  }

  [data-testid="stHeader"] { background: transparent !important; }
  [data-testid="stSidebar"] { display: none !important; }
  .block-container { padding: 1.6rem 2.4rem 3rem 2.4rem !important; max-width: 1380px !important; position: relative; z-index: 1; }
  #MainMenu, footer, header { visibility: hidden; }
  [data-testid="stToolbar"] { display: none; }

  ::-webkit-scrollbar { width: 6px; }
  ::-webkit-scrollbar-track { background: var(--bg-2); }
  ::-webkit-scrollbar-thumb { background: var(--emerald); border-radius: 3px; }

  /* Glass base */
  .glass {
    background: var(--glass);
    border: 1px solid var(--line);
    border-radius: 20px;
    backdrop-filter: blur(18px) saturate(140%);
    -webkit-backdrop-filter: blur(18px) saturate(140%);
    box-shadow: 0 8px 40px rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,255,255,0.04);
  }

  /* HERO */
  .hero { position: relative; padding: 1.6rem 1.9rem; margin-bottom: 1.5rem; overflow: hidden; }
  .hero::before {
    content:''; position:absolute; inset:-1px; border-radius:20px; padding:1px;
    background: linear-gradient(120deg, transparent 20%, rgba(201,168,76,0.55) 50%, transparent 80%);
    background-size: 300% 100%;
    -webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
    -webkit-mask-composite: xor; mask-composite: exclude;
    animation: shimmer 6s linear infinite;
  }
  @keyframes shimmer { 0%{background-position:200% 0;} 100%{background-position:-100% 0;} }
  .hero-row { display:flex; align-items:center; justify-content:space-between; gap:1rem; flex-wrap:wrap; }
  .hero-left { display:flex; align-items:center; gap:1.1rem; }
  .hero-badge {
    width:58px; height:58px; display:grid; place-items:center; font-size:1.9rem;
    border-radius:16px;
    background: linear-gradient(135deg, rgba(20,163,122,0.25), rgba(201,168,76,0.18));
    border:1px solid rgba(201,168,76,0.35);
    box-shadow: 0 0 30px rgba(20,163,122,0.25), inset 0 1px 0 rgba(255,255,255,0.08);
  }
  .hero-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 2.4rem; font-weight: 600; line-height: 1.05; letter-spacing: -0.01em;
    background: linear-gradient(90deg, var(--cream) 0%, var(--gold-2) 70%);
    -webkit-background-clip:text; background-clip:text; color:transparent;
  }
  .hero-subtitle { color: var(--muted); font-size: 0.9rem; margin-top: 0.3rem; letter-spacing: 0.01em; }
  .hero-stats { display:flex; gap:0.5rem; flex-wrap:wrap; }
  .stat-chip {
    background: var(--glass-2);
    border:1px solid var(--line);
    border-radius:999px; padding:0.4rem 0.85rem;
    color: var(--cream); font-size:0.74rem; font-weight:500; letter-spacing:0.04em;
    display:inline-flex; align-items:center; gap:0.4rem;
    backdrop-filter: blur(10px);
  }
  .stat-chip .dot { width:6px; height:6px; border-radius:50%; background: var(--emerald-2); box-shadow:0 0 10px var(--emerald-2); animation: pulse 2.4s ease-in-out infinite;}
  .stat-chip.gold { border-color: rgba(201,168,76,0.35); color: var(--gold-2); }

  /* BYLINE (top credit) */
  .byline {
    display:inline-flex; align-items:center; gap:0.5rem;
    margin-top: 0.7rem; padding: 0.32rem 0.85rem;
    background: var(--glass); border:1px solid var(--line);
    border-radius: 999px; font-size: 0.72rem; color: var(--muted);
    letter-spacing: 0.06em;
    backdrop-filter: blur(10px);
  }
  .byline b { color: var(--gold-2); font-weight: 600; }
  .byline .sep { color: var(--gold); opacity: 0.6; margin: 0 0.2rem; }

  /* UPLOAD CARD */
  .upload-card { padding: 1.4rem 1.4rem 1.2rem 1.4rem; }
  .upload-head { display:flex; align-items:center; gap:0.65rem; margin-bottom: 0.4rem; }
  .upload-head .ico {
    width:36px; height:36px; border-radius:11px; display:grid; place-items:center; font-size:1rem;
    background: linear-gradient(135deg, rgba(20,163,122,0.25), rgba(201,168,76,0.15));
    border:1px solid var(--line);
  }
  .upload-title { color: var(--cream); font-weight:600; font-size:1.05rem; letter-spacing:-0.01em; }
  .upload-desc { color: var(--muted); font-size:0.82rem; margin: 0.1rem 0 1rem 0; }

  /* Tabs */
  .stTabs [data-baseweb="tab-list"] {
    gap: 0.4rem; background: transparent; border-bottom: 1px solid var(--line);
    margin-bottom: 0.8rem;
  }
  .stTabs [data-baseweb="tab"] {
    background: transparent !important; color: var(--muted) !important;
    border-radius: 10px 10px 0 0 !important; padding: 0.5rem 0.9rem !important;
    font-size: 0.85rem !important; font-weight: 500 !important;
  }
  .stTabs [aria-selected="true"] {
    color: var(--gold-2) !important;
    border-bottom: 2px solid var(--gold) !important;
  }

  [data-testid="stFileUploader"] { background:transparent !important; border:none !important; padding:0 !important; }
  [data-testid="stFileUploaderDropzone"] {
    background: var(--glass) !important;
    border:1.5px dashed rgba(201,168,76,0.35) !important;
    border-radius:14px !important;
    transition: all .3s ease; backdrop-filter: blur(10px);
  }
  [data-testid="stFileUploaderDropzone"]:hover {
    border-color: var(--gold) !important;
    background: rgba(201,168,76,0.06) !important;
    box-shadow: 0 0 0 4px rgba(201,168,76,0.08), 0 0 30px rgba(201,168,76,0.12);
  }
  [data-testid="stFileUploaderDropzoneInstructions"] > div > span { color: var(--cream) !important; font-weight:500; }
  [data-testid="stFileUploaderDropzoneInstructions"] > div > small { color: var(--muted) !important; }
  [data-testid="stFileUploader"] section button {
    background: linear-gradient(135deg, var(--gold), #a88a36) !important;
    color: #1a1208 !important; border: none !important;
    border-radius: 10px !important; font-weight: 700 !important; letter-spacing: 0.02em;
    transition: all .2s ease;
  }
  [data-testid="stFileUploader"] section button:hover { transform: translateY(-1px); box-shadow: 0 6px 20px rgba(201,168,76,0.35); }
  .uploadedFile { background: var(--glass-2) !important; border-radius:10px !important; border:1px solid var(--line) !important; }

  .format-chips { display:flex; flex-wrap:wrap; gap:0.35rem; margin-top: 0.9rem; }
  .format-chip {
    font-size:0.7rem; color: var(--muted);
    background: var(--glass); border:1px solid var(--line);
    border-radius:6px; padding:0.22rem 0.55rem; letter-spacing:0.04em;
  }

  .leaf-img-container {
    margin-top: 0.9rem; border:1px solid var(--line); border-radius:14px;
    overflow:hidden; animation: fadeIn 0.5s ease;
    box-shadow: 0 10px 32px rgba(0,0,0,0.4);
  }
  .leaf-img-container img { width:100%; display:block; }
  .leaf-img-label {
    color: var(--gold-2); font-size:0.85rem; font-weight:600;
    margin: 0.9rem 0 0.4rem 0; letter-spacing:0.06em; text-transform: uppercase;
  }

  .tips-card { margin-top: 1rem; padding: 1rem 1.1rem; }
  .tips-title { color: var(--gold-2); font-weight:600; font-size:0.82rem; margin-bottom:0.55rem; letter-spacing:0.08em; text-transform:uppercase;}
  .tips-list { list-style:none; padding:0; margin:0; }
  .tips-list li { color: var(--cream); font-size:0.85rem; line-height:1.55; padding-left:1.2rem; position:relative; margin-bottom:0.35rem; opacity: 0.85;}
  .tips-list li::before { content:'✦'; position:absolute; left:0; color: var(--gold); }

  /* RESULTS */
  .diagnosis-card { padding: 1.5rem 1.7rem; margin-bottom: 1rem; animation: slideUp 0.5s ease; position:relative; overflow:hidden; }
  .diagnosis-card::after {
    content:''; position:absolute; top:-40px; right:-40px; width:200px; height:200px;
    background: radial-gradient(circle, rgba(201,168,76,0.18), transparent 70%);
    pointer-events:none;
  }
  .diagnosis-label, .confidence-label {
    color: var(--gold-2); font-size:0.72rem; font-weight:600;
    letter-spacing:0.14em; text-transform:uppercase;
  }
  .diagnosis-text {
    font-family: 'Cormorant Garamond', serif;
    font-size: 2.1rem; font-weight: 600; color: #ff9999;
    letter-spacing:-0.01em; margin: 0.4rem 0 0.25rem 0; line-height:1.05;
    animation: fadeIn 0.7s ease;
  }
  .diagnosis-text.healthy { color: var(--emerald-2) !important; }
  .diagnosis-sub { color: var(--muted); font-size:0.85rem; }
  .confidence-val {
    font-family: 'Cormorant Garamond', serif;
    font-size: 2.4rem; font-weight: 600; color: var(--cream);
    margin: 0.2rem 0 0.5rem 0; letter-spacing:-0.02em;
  }
  .confidence-bar-bg { background: rgba(0,0,0,0.3); border-radius:99px; height:6px; width:100%; overflow:hidden; border:1px solid var(--line);}
  .confidence-bar-fill {
    background: linear-gradient(90deg, var(--emerald-2), var(--gold));
    border-radius:99px; height:100%;
    animation: growBar 1.1s cubic-bezier(.22,1,.36,1) forwards;
    box-shadow: 0 0 12px rgba(201,168,76,0.4);
  }

  /* SEVERITY GAUGE */
  .gauge-wrap { margin-top: 1rem; }
  .gauge-track {
    position: relative; height: 8px; border-radius: 999px;
    background: linear-gradient(90deg, #7fd49b 0%, #e8c068 35%, #e09a3a 70%, #e07070 100%);
    box-shadow: inset 0 1px 2px rgba(0,0,0,0.4), 0 0 14px rgba(201,168,76,0.18);
    border: 1px solid var(--line);
  }
  .gauge-marker {
    position: absolute; top: 50%;
    width: 18px; height: 18px; border-radius: 50%;
    background: var(--cream); border: 2px solid var(--gold);
    box-shadow: 0 0 14px rgba(245,240,224,0.7), 0 2px 8px rgba(0,0,0,0.5);
    transform: translate(-50%, -50%);
    animation: markerIn 1.1s cubic-bezier(.22,1,.36,1);
  }
  @keyframes markerIn { from { opacity: 0; transform: translate(-50%, -50%) scale(0.2);} to { opacity:1; transform: translate(-50%, -50%) scale(1);} }
  .gauge-scale {
    display:flex; justify-content:space-between; margin-top:0.55rem;
    font-size: 0.62rem; color: var(--muted); letter-spacing: 0.12em; text-transform: uppercase;
  }
  .gauge-caption { color: var(--muted); font-size: 0.82rem; margin-top: 0.6rem;}
  .gauge-caption b { font-weight: 600; }

  .treatment-card {
    padding: 1.2rem 1.4rem; margin-bottom:1rem; animation: slideUp 0.6s ease;
    border-left: 3px solid var(--gold) !important;
  }
  .treatment-title { color: var(--gold-2); font-weight:600; font-size:0.78rem; margin-bottom:0.5rem; letter-spacing:0.12em; text-transform:uppercase;}
  .treatment-text { color: var(--cream); font-size:0.93rem; line-height:1.7; opacity:0.92;}

  .warning-banner {
    padding:0.85rem 1.1rem; color: var(--gold-2); font-size:0.82rem; margin-bottom:1rem;
    display:flex; align-items:flex-start; gap:0.6rem; animation: slideUp 0.7s ease;
    border-left: 3px solid var(--gold) !important;
  }

  .predictions-card { padding:1.2rem 1.4rem 0.6rem 1.4rem; animation: slideUp 0.8s ease; }
  .predictions-title { color: var(--gold-2); font-weight:600; font-size:0.78rem; margin-bottom:1rem; letter-spacing:0.12em; text-transform:uppercase;}
  .pred-rank {
    background: linear-gradient(135deg, rgba(20,163,122,0.3), rgba(201,168,76,0.2));
    color: var(--gold-2); width:30px; height:30px; border-radius:50%;
    display:flex; align-items:center; justify-content:center;
    font-size:0.78rem; font-weight:700; border:1px solid var(--line);
    font-family: 'Cormorant Garamond', serif;
  }
  .pred-name { color: var(--cream); font-size:0.9rem; padding-top:5px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;}
  .pred-bar-bg { background: rgba(0,0,0,0.3); border-radius:99px; height:5px; width:100%; overflow:hidden; margin-top:5px; border:1px solid var(--line);}
  .pred-bar-fill { background: linear-gradient(90deg, var(--emerald-2), var(--gold)); border-radius:99px; height:100%; animation: growBar 1s ease forwards;}
  .pred-pct { color: var(--gold-2); font-size:0.9rem; font-weight:700; text-align:right; padding-top:6px; font-family:'Cormorant Garamond', serif;}

  /* HISTORY */
  .history-card { padding: 1rem 1.1rem; margin-top: 1rem; }
  .history-title { color: var(--gold-2); font-weight:600; font-size:0.78rem; margin-bottom:0.7rem; letter-spacing:0.12em; text-transform:uppercase;}
  .history-empty { color: var(--muted); font-size:0.82rem; opacity:0.7; text-align:center; padding:0.4rem;}
  .history-item { display:flex; justify-content:space-between; align-items:center; padding:0.5rem 0.2rem; border-bottom:1px dashed var(--line); font-size:0.82rem;}
  .history-item:last-child { border-bottom: none; }
  .history-item .name { color: var(--cream); }
  .history-item .pct { font-family: 'Cormorant Garamond', serif; font-weight:600; }

  /* CREDITS (bottom signature) */
  .credits {
    margin-top: 1.6rem; padding: 1.5rem 1.2rem 1.2rem 1.2rem; text-align:center;
    border-top: 1px solid var(--line); position: relative;
  }
  .credits-label {
    color: var(--gold-2); font-size: 0.7rem; font-weight: 600;
    letter-spacing: 0.28em; text-transform: uppercase; margin-bottom: 0.7rem;
    opacity: 0.85;
  }
  .credits-names {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.55rem; font-weight: 500; letter-spacing: 0.02em;
    background: linear-gradient(90deg, var(--cream) 0%, var(--gold-2) 50%, var(--cream) 100%);
    -webkit-background-clip: text; background-clip: text; color: transparent;
  }
  .credits-sep { color: var(--gold); margin: 0 0.9rem; opacity: 0.6; font-weight: 300; }

  .app-footer {
    border-top:1px solid var(--line); margin-top:1.4rem; padding-top:1rem;
    display:flex; gap:1.4rem; justify-content:center; flex-wrap:wrap;
  }
  .footer-item { color: var(--muted); font-size:0.76rem; display:flex; align-items:center; gap:0.35rem; opacity:0.8;}

  .empty-state { text-align:center; padding:3.5rem 1rem; }
  .empty-icon { font-size:3.6rem; margin-bottom:0.9rem; opacity:0.85; filter: drop-shadow(0 0 20px rgba(201,168,76,0.3));}
  .empty-text {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.4rem; color: var(--cream); font-weight:500;
  }
  .empty-sub { font-size:0.85rem; color: var(--muted); margin-top:0.5rem;}

  .stSpinner > div { border-top-color: var(--gold) !important; }

  .stDownloadButton button, .stButton button {
    background: var(--glass-2) !important; color: var(--cream) !important;
    border:1px solid var(--line) !important; border-radius:10px !important;
    font-weight:500 !important; transition: all .2s ease;
    backdrop-filter: blur(10px);
  }
  .stDownloadButton button:hover, .stButton button:hover {
    border-color: var(--gold) !important; color: var(--gold-2) !important;
    transform: translateY(-1px); box-shadow: 0 6px 20px rgba(201,168,76,0.18);
  }

  /* camera input */
  [data-testid="stCameraInput"] button {
    background: linear-gradient(135deg, var(--emerald), var(--emerald-2)) !important;
    color: var(--cream) !important; border:none !important;
    border-radius: 10px !important; font-weight: 600 !important;
  }

  .divider {
    height:1px; margin: 1rem 0;
    background: linear-gradient(90deg, transparent, var(--line), transparent);
  }

  @keyframes fadeIn { from { opacity:0; } to { opacity:1; } }
  @keyframes slideUp { from { opacity:0; transform: translateY(14px); } to { opacity:1; transform: translateY(0); } }
  @keyframes growBar { from { width:0%; } }
  @keyframes pulse { 0%,100% { opacity:1;} 50% { opacity:0.4;} }
  .analyzing-pulse { animation: pulse 1.4s ease-in-out infinite; color: var(--gold-2); font-size:0.88rem; text-align:center; margin-top:0.5rem; letter-spacing:0.06em;}

  /* DRIFTING LEAF PARTICLES */
  .leaf-field {
    position: fixed; inset: 0; pointer-events: none; z-index: 0; overflow: hidden;
  }
  .leaf-field span {
    position: absolute; display: block;
    opacity: 0.12; filter: drop-shadow(0 0 6px rgba(201,168,76,0.3));
    animation: drift linear infinite;
    top: -10vh;
  }
  @keyframes drift {
    0%   { transform: translateY(0) translateX(0) rotate(0deg); opacity: 0; }
    10%  { opacity: 0.16; }
    90%  { opacity: 0.12; }
    100% { transform: translateY(120vh) translateX(60px) rotate(360deg); opacity: 0; }
  }
</style>
""", unsafe_allow_html=True)


# ─── DRIFTING LEAF PARTICLES ────────────────────────────────────────────────
_particles = "".join([
    f'<span style="left:{(i*7.3) % 100:.1f}%; '
    f'animation-duration:{15 + (i*1.7) % 18:.1f}s; '
    f'animation-delay:-{(i*2.1) % 20:.1f}s; '
    f'font-size:{0.9 + (i % 5)*0.25:.2f}rem;">🍃</span>'
    for i in range(18)
])
st.markdown(f'<div class="leaf-field">{_particles}</div>', unsafe_allow_html=True)


# ─── MODEL LOADING ──────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    base = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(base, "..", "models", "best_model .keras"),
        os.path.join(base, "..", "models", "plant_disease_model.keras"),
        os.path.join(base, "..", "models", "basiccnn", "best_model.keras"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return tf.keras.models.load_model(path)
    raise FileNotFoundError("No model file found in models/ directory.")


def predict(image: Image.Image, model):
    img = image.convert("RGB").resize((224, 224))
    arr = np.expand_dims(np.array(img, dtype=np.float32), 0)
    preds = model.predict(arr, verbose=0)[0]
    top3_idx = preds.argsort()[-3:][::-1]
    return [(CLASS_NAMES[i], float(preds[i]) * 100) for i in top3_idx]


# ─── SAMPLE LEAVES ──────────────────────────────────────────────────────────
def list_samples():
    base = os.path.dirname(os.path.abspath(__file__))
    candidates = [os.path.join(base, "samples"), os.path.join(base, "..", "samples")]
    for d in candidates:
        if os.path.isdir(d):
            files = sorted([f for f in os.listdir(d) if f.lower().endswith((".jpg",".jpeg",".png",".bmp",".webp",".tiff"))])[:4]
            return [(os.path.join(d, f), os.path.splitext(f)[0].replace("_"," ").title()) for f in files]
    return []


# ─── PDF REPORT ─────────────────────────────────────────────────────────────
def build_pdf_report(image: Image.Image, plant, disease, top_pct, severity, treatment, top3):
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.pdfgen import canvas
    from reportlab.lib.utils import ImageReader

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    W, H = A4

    # Background
    c.setFillColorRGB(0.024, 0.137, 0.102)
    c.rect(0, 0, W, H, fill=1, stroke=0)

    # Gold accent bar
    c.setFillColorRGB(0.788, 0.659, 0.298)
    c.rect(0, H-12, W, 12, fill=1, stroke=0)

    # Header
    c.setFillColorRGB(0.96, 0.94, 0.88)
    c.setFont("Helvetica-Bold", 22)
    c.drawString(25*mm, H-30*mm, "Plant Disease Diagnosis Report")
    c.setFillColorRGB(0.62, 0.75, 0.69)
    c.setFont("Helvetica", 10)
    c.drawString(25*mm, H-37*mm, f"Generated {datetime.now().strftime('%B %d, %Y · %H:%M')}")

    # Leaf thumbnail
    thumb = image.convert("RGB").copy()
    thumb.thumbnail((600, 600))
    img_reader = ImageReader(thumb)
    iw, ih = thumb.size
    ratio = min(60*mm / iw, 60*mm / ih)
    dw, dh = iw*ratio, ih*ratio
    img_x = W - 25*mm - dw
    img_y = H - 50*mm - dh
    c.setStrokeColorRGB(0.788, 0.659, 0.298)
    c.setLineWidth(1)
    c.rect(img_x-2, img_y-2, dw+4, dh+4, fill=0, stroke=1)
    c.drawImage(img_reader, img_x, img_y, dw, dh)

    # Diagnosis box
    y = H - 60*mm
    c.setFillColorRGB(0.788, 0.659, 0.298)
    c.setFont("Helvetica-Bold", 9)
    c.drawString(25*mm, y, "DIAGNOSIS")
    y -= 8*mm
    c.setFillColorRGB(0.96, 0.94, 0.88)
    c.setFont("Helvetica-Bold", 18)
    c.drawString(25*mm, y, f"{plant}")
    y -= 7*mm
    c.setFillColorRGB(0.5, 0.85, 0.6) if "Healthy" in disease else c.setFillColorRGB(1.0, 0.6, 0.6)
    c.drawString(25*mm, y, f"{disease}")

    y -= 16*mm
    c.setFillColorRGB(0.788, 0.659, 0.298); c.setFont("Helvetica-Bold", 9); c.drawString(25*mm, y, "CONFIDENCE")
    c.drawString(70*mm, y, "SEVERITY")
    y -= 7*mm
    c.setFillColorRGB(0.96, 0.94, 0.88); c.setFont("Helvetica-Bold", 16)
    c.drawString(25*mm, y, f"{top_pct:.1f}%")
    c.drawString(70*mm, y, severity)

    # Confidence bar
    y -= 8*mm
    bar_w, bar_h = 60*mm, 3*mm
    c.setFillColorRGB(0.05, 0.1, 0.08); c.rect(25*mm, y, bar_w, bar_h, fill=1, stroke=0)
    c.setFillColorRGB(0.788, 0.659, 0.298)
    c.rect(25*mm, y, bar_w*(top_pct/100), bar_h, fill=1, stroke=0)

    # Treatment
    y -= 18*mm
    c.setFillColorRGB(0.788, 0.659, 0.298); c.setFont("Helvetica-Bold", 9); c.drawString(25*mm, y, "RECOMMENDED TREATMENT")
    y -= 8*mm
    c.setFillColorRGB(0.96, 0.94, 0.88); c.setFont("Helvetica", 10)
    from textwrap import wrap
    for line in wrap(treatment, width=95):
        c.drawString(25*mm, y, line); y -= 5*mm

    # Top 3
    y -= 10*mm
    c.setFillColorRGB(0.788, 0.659, 0.298); c.setFont("Helvetica-Bold", 9); c.drawString(25*mm, y, "TOP 3 PREDICTIONS")
    y -= 8*mm
    for i,(n,p) in enumerate(top3, 1):
        c.setFillColorRGB(0.96, 0.94, 0.88); c.setFont("Helvetica", 10)
        c.drawString(25*mm, y, f"{i}.  {n}")
        c.setFillColorRGB(0.788, 0.659, 0.298); c.setFont("Helvetica-Bold", 10)
        c.drawRightString(W-25*mm, y, f"{p:.1f}%")
        y -= 6*mm

    # Footer
    c.setFillColorRGB(0.62, 0.75, 0.69); c.setFont("Helvetica-Oblique", 8)
    c.drawString(25*mm, 18*mm, "Plant Disease Detector · MobileNetV2 · PlantVillage Dataset · 38 classes · 96.7% accuracy")
    c.drawString(25*mm, 14*mm, "This is an AI prediction — confirm critical decisions with a certified plant pathologist.")
    c.setFillColorRGB(0.788, 0.659, 0.298); c.setFont("Helvetica-Bold", 8)
    c.drawString(25*mm, 9*mm, "Crafted by Muhammad Rumman Aslam  ·  Zunairah Abdul Rehman")
    c.setFillColorRGB(0.788, 0.659, 0.298); c.rect(0, 0, W, 4, fill=1, stroke=0)

    c.showPage(); c.save()
    buf.seek(0)
    return buf.getvalue()


# ─── SESSION STATE ──────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []
if "sample_choice" not in st.session_state:
    st.session_state.sample_choice = None


# ─── HEADER ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="glass hero">
  <div class="hero-row">
    <div class="hero-left">
      <div class="hero-badge">🌿</div>
      <div>
        <div class="hero-title">Plant Disease Detector</div>
        <div class="hero-subtitle">An elegant AI diagnosis for your garden — 38 disease classes, MobileNetV2</div>
        <div class="byline">
          ✦ Crafted by <b>Muhammad Rumman Aslam</b> <span class="sep">·</span> <b>Zunairah Abdul Rehman</b>
        </div>
      </div>
    </div>
    <div class="hero-stats">
      <span class="stat-chip"><span class="dot"></span>Model Online</span>
      <span class="stat-chip gold">✦ 96.7% Accuracy</span>
      <span class="stat-chip">🧠 MobileNetV2</span>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)


# ─── LAYOUT ─────────────────────────────────────────────────────────────────
left_col, right_col = st.columns([1, 1.55], gap="large")

# ── LEFT: input
with left_col:
    st.markdown("""
    <div class="glass upload-card">
      <div class="upload-head">
        <div class="ico">🌱</div>
        <div>
          <div class="upload-title">Capture a Leaf</div>
          <div class="upload-desc">Upload, snap a photo, or try a sample</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    tab_upload, tab_camera, tab_sample = st.tabs(["📁 Upload", "📷 Camera", "✦ Samples"])

    uploaded = None
    with tab_upload:
        uploaded = st.file_uploader(
            label="Upload leaf image",
            type=["jpg", "jpeg", "png", "bmp", "webp", "tiff"],
            accept_multiple_files=False,
            label_visibility="collapsed",
            key="file_uploader",
        )
        st.markdown("""
          <div class="format-chips">
            <span class="format-chip">JPG</span><span class="format-chip">PNG</span>
            <span class="format-chip">BMP</span><span class="format-chip">WEBP</span>
            <span class="format-chip">TIFF</span>
          </div>
        """, unsafe_allow_html=True)

    with tab_camera:
        cam = st.camera_input("Take a leaf photo", label_visibility="collapsed", key="cam_input")
        if cam is not None and uploaded is None:
            uploaded = cam

    with tab_sample:
        samples = list_samples()
        if not samples:
            st.markdown(
                '<div class="history-empty">'
                'No sample leaves found. Add 2–4 images to a samples/ folder '
                'next to app.py to populate this gallery.'
                '</div>',
                unsafe_allow_html=True,
            )
        else:
            cols = st.columns(min(len(samples), 4))
            for i,(path,name) in enumerate(samples):
                with cols[i]:
                    st.image(path, use_container_width=True)
                    if st.button(name, key=f"sample_{i}", use_container_width=True):
                        st.session_state.sample_choice = path
            if st.session_state.sample_choice and uploaded is None:
                uploaded = st.session_state.sample_choice   # pass the path string

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # Normalize input → PIL image (fixes sample-click crash)
    image_for_display = None
    pil_image = None
    if uploaded is not None:
        try:
            pil_image = Image.open(uploaded)
            pil_image.load()
            image_for_display = pil_image.copy()
        except Exception as e:
            st.error(f"Could not read image: {e}")
            uploaded = None
            pil_image = None
            image_for_display = None

    # Image preview + tips
    if uploaded and image_for_display is not None:
        st.markdown('<div class="leaf-img-label">Uploaded Leaf</div>', unsafe_allow_html=True)
        st.markdown('<div class="leaf-img-container">', unsafe_allow_html=True)
        st.image(image_for_display, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="glass tips-card">
          <div class="tips-title">Capture Tips</div>
          <ul class="tips-list">
            <li>Use natural daylight, avoid harsh shadows</li>
            <li>Center one leaf in the frame</li>
            <li>Plain background works best</li>
            <li>Keep the camera steady and in focus</li>
          </ul>
        </div>
        """, unsafe_allow_html=True)

    # History
    if st.session_state.history:
        st.markdown('<div class="glass history-card">', unsafe_allow_html=True)
        st.markdown('<div class="history-title">Recent Scans</div>', unsafe_allow_html=True)
        for h in reversed(st.session_state.history[-5:]):
            sev_c = SEVERITY_COLOR.get(h["severity"], "#c9a84c")
            st.markdown(
                f'<div class="history-item">'
                f'<span class="name">{h["plant"]} — {h["disease"]}</span>'
                f'<span class="pct" style="color:{sev_c}">{h["confidence"]:.0f}%</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
        st.markdown('</div>', unsafe_allow_html=True)

# ── RIGHT: results
with right_col:
    if uploaded is None or pil_image is None:
        st.markdown("""
        <div class="glass empty-state">
          <div class="empty-icon">🍃</div>
          <div class="empty-text">Awaiting your leaf</div>
          <div class="empty-sub">Upload, capture, or pick a sample to begin</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        with st.spinner(""):
            st.markdown('<div class="analyzing-pulse">✦ Analyzing leaf ✦</div>', unsafe_allow_html=True)
            time.sleep(0.3)
            try:
                model = load_model()
                top3 = predict(pil_image, model)
            except Exception as e:
                st.error(f"Error: {e}")
                st.stop()

        top_name, top_pct = top3[0]
        is_healthy = "healthy" in top_name.lower()
        treatment = TREATMENTS.get(top_name, "Consult a local plant pathologist for tailored advice.")
        severity = severity_for(top_name)
        sev_color = SEVERITY_COLOR.get(severity, "#c9a84c")
        sev_pos = SEVERITY_POS.get(severity, 50)
        diag_class = "healthy" if is_healthy else ""
        plant, _, disease = top_name.partition(" — ")
        disease = disease or top_name

        # Save history
        snap = {"plant": plant or "Plant", "disease": disease, "confidence": top_pct, "severity": severity}
        if not st.session_state.history or st.session_state.history[-1] != snap:
            st.session_state.history.append(snap)

        # Diagnosis card
        st.markdown(f"""
        <div class="glass diagnosis-card">
          <div style="display:flex; justify-content:space-between; gap:1.5rem; flex-wrap:wrap;">
            <div style="flex:1.4; min-width:240px;">
              <div class="diagnosis-label">✦ Diagnosis · {plant or "Plant"}</div>
              <div class="diagnosis-text {diag_class}">{disease}</div>
              <div class="diagnosis-sub">Identified from the uploaded leaf image</div>
              <div class="gauge-wrap">
                <div class="gauge-track"><div class="gauge-marker" style="left:{sev_pos}%;"></div></div>
                <div class="gauge-scale"><span>None</span><span>Moderate</span><span>High</span><span>Critical</span></div>
                <div class="gauge-caption">Severity · <b style="color:{sev_color}">{severity}</b></div>
              </div>
            </div>
            <div style="flex:1; min-width:180px;">
              <div class="confidence-label">Confidence</div>
              <div class="confidence-val">{top_pct:.1f}%</div>
              <div class="confidence-bar-bg"><div class="confidence-bar-fill" style="width:{top_pct}%"></div></div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Treatment
        title_label = "Care Recommendation" if is_healthy else "Recommended Treatment"
        st.markdown(f"""
        <div class="glass treatment-card">
          <div class="treatment-title">✦ {title_label}</div>
          <div class="treatment-text">{treatment}</div>
        </div>
        """, unsafe_allow_html=True)

        # Note
        st.markdown("""
        <div class="glass warning-banner">
          <span>⚠</span>
          <span>This is an AI prediction. For high-stakes treatment decisions, confirm with a certified plant pathologist.</span>
        </div>
        """, unsafe_allow_html=True)

        # Top 3
        st.markdown('<div class="glass predictions-card">', unsafe_allow_html=True)
        st.markdown('<div class="predictions-title">✦ Top 3 Predictions</div>', unsafe_allow_html=True)
        max_pct = top3[0][1] if top3[0][1] > 0 else 1
        for rank, (name, pct) in enumerate(top3, 1):
            bar_w = (pct / max_pct) * 100
            display = name if len(name) <= 42 else name[:40] + "…"
            c1, c2, c3 = st.columns([0.08, 0.72, 0.20])
            with c1:
                st.markdown(f'<div class="pred-rank">{rank}</div>', unsafe_allow_html=True)
            with c2:
                st.markdown(
                    f'<div class="pred-name">{display}</div>'
                    f'<div class="pred-bar-bg"><div class="pred-bar-fill" style="width:{bar_w}%"></div></div>',
                    unsafe_allow_html=True,
                )
            with c3:
                st.markdown(f'<div class="pred-pct">{pct:.1f}%</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Downloads
        try:
            pdf_bytes = build_pdf_report(pil_image, plant or "Plant", disease, top_pct, severity, treatment, top3)
            dlc1, dlc2 = st.columns(2)
            with dlc1:
                st.download_button(
                    "⬇ Download PDF Report",
                    data=pdf_bytes,
                    file_name=f"diagnosis_{(plant or 'plant').lower().replace(' ', '_')}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
            with dlc2:
                txt = (
                    f"PLANT DISEASE DIAGNOSIS\n\nPlant: {plant}\nDiagnosis: {disease}\n"
                    f"Confidence: {top_pct:.2f}%\nSeverity: {severity}\n\nTreatment:\n{treatment}\n\n"
                    "Top 3:\n" + "\n".join([f"  {i+1}. {n} — {p:.2f}%" for i,(n,p) in enumerate(top3)])
                    + "\n\n— Crafted by Muhammad Rumman Aslam & Zunairah Abdul Rehman"
                )
                st.download_button(
                    "⬇ Download Text Report",
                    data=txt,
                    file_name=f"diagnosis_{(plant or 'plant').lower().replace(' ', '_')}.txt",
                    mime="text/plain",
                    use_container_width=True,
                )
        except ImportError:
            st.info("Install `reportlab` for PDF reports: `pip install reportlab`")


# ─── BOTTOM CREDITS ─────────────────────────────────────────────────────────
st.markdown("""
<div class="credits">
  <div class="credits-label">✦ Designed &amp; Built By ✦</div>
  <div class="credits-names">
    Muhammad Rumman Aslam <span class="credits-sep">·</span> Zunairah Abdul Rehman
  </div>
</div>
""", unsafe_allow_html=True)

# ─── TECH FOOTER ────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-footer">
  <div class="footer-item">✦ TensorFlow &amp; MobileNetV2</div>
  <div class="footer-item">✦ PlantVillage Dataset</div>
  <div class="footer-item">✦ 38 Disease Classes</div>
  <div class="footer-item">✦ 96.7% Accuracy</div>
</div>
""", unsafe_allow_html=True)
