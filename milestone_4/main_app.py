# --- IMPORTS ---
import streamlit as st
import os
import sys

# Ensure modules can be imported
sys.path.append(os.path.dirname(__file__))

# Import custom modules
import datasetloader
import app_file

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="🧠 Strategic Sentiment Intelligence Platform",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- CUSTOM STYLING ---
st.markdown("""
    <style>
    /* Global background */
    [data-testid="stAppViewContainer"] {
        background-color: #0f172a;
        color: #f8fafc;
        font-family: 'Inter', sans-serif;
    }

    /* Sidebar gradient + border */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a, #1e293b, #334155);
        color: white;
        border-right: 2px solid #3b82f6;
        box-shadow: 4px 0px 12px rgba(0, 0, 0, 0.4);
    }

    /* Sidebar headers */
    [data-testid="stSidebar"] h2 {
        color: #f1f5f9;
        text-align: center;
        font-weight: 800;
        letter-spacing: 0.5px;
        margin-bottom: 20px;
    }

    /* Radio button styling */
    div[role="radiogroup"] > label {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        padding: 10px 15px;
        margin-bottom: 8px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        color: #cbd5e1;
        transition: all 0.3s ease;
        cursor: pointer;
        font-weight: 500;
    }
    div[role="radiogroup"] > label:hover {
        background: rgba(255, 255, 255, 0.12);
        color: white;
        transform: scale(1.03);
    }
    div[role="radiogroup"] > label[data-checked="true"] {
        background: linear-gradient(90deg, #3b82f6, #2563eb);
        color: white !important;
        font-weight: 600;
        box-shadow: 0 0 15px rgba(59, 130, 246, 0.5);
        border: none;
    }

    /* Sidebar buttons */
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #3b82f6, #2563eb);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.6em 1em;
        font-weight: 600;
        font-size: 1em;
        transition: 0.3s;
        box-shadow: 0px 0px 8px rgba(59,130,246,0.3);
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #2563eb, #1d4ed8);
        box-shadow: 0px 0px 12px rgba(59,130,246,0.6);
        transform: scale(1.03);
    }

    /* Sidebar caption */
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] span {
        color: #94a3b8;
        font-size: 0.9rem;
    }

    /* Divider */
    hr {
        border: 0;
        border-top: 1px solid rgba(255,255,255,0.15);
        margin: 15px 0;
    }

    /* Main title styling */
    h1 {
        font-size: 2.4em !important;
        color: #ffffff !important;
        text-align: center !important;
        margin-top: 0.5em;
        margin-bottom: 0.2em;
    }

    .subtitle {
        text-align: center;
        color: #94a3b8;
        font-size: 1.1em;
        margin-bottom: 1.2em;
    }
    </style>
""", unsafe_allow_html=True)

# --- SIDEBAR NAVIGATION ---
st.sidebar.markdown("<h2>🧭 Navigation Panel</h2>", unsafe_allow_html=True)
app_selection = st.sidebar.radio(
    "Choose a Module:",
    ["📰 New Dataset", "📈 Data Visualization"],
)

st.sidebar.markdown("<hr>", unsafe_allow_html=True)
st.sidebar.caption(
    "Developed by <b>Vineet</b> | Powered by <b>Streamlit</b> ⚡",
    unsafe_allow_html=True,
)

# --- MAIN CONTENT HANDLER ---
if app_selection == "📰 New Dataset":
    st.markdown("""
        <h1>📰 Strategic News Dataset Loader & Analyzer</h1>
        <p class='subtitle'>
            Fetch live global news, perform AI-powered sentiment analysis, and send instant reports to Slack.
        </p>
    """, unsafe_allow_html=True)
    datasetloader.run_dataset_loader()

elif app_selection == "📈 Data Visualization":
    st.markdown("""
        <h1>🧠 Sentiment Forecast & Analytics Dashboard</h1>
        <p class='subtitle'>
            View aggregated insights, sentiment trends, and Prophet-based forecasts from all datasets.
        </p>
    """, unsafe_allow_html=True)
    app_file.run_data_visualization()
