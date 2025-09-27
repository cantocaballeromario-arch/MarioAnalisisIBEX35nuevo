# =============================================
# 🚀 Streamlit App: TOP BUY IBEX 35 con Técnicos + Fundamentales + Noticias + Riesgo + Gráficos
# =============================================

import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import time
import math
import feedparser
import urllib.parse
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns
import ta
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models, expected_returns
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
st.set_page_config(page_title="Análisis IBEX 35", layout="wide")
st.title("📊 Análisis de Empresas - IBEX 35")

# -------------------------
# Parámetros
# -------------------------
TECH_THRESHOLD = 2
FUND_THRESHOLD = 2
NEWS_THRESHOLD = 0.05
NEWS_PER_SOURCE = 5
RSI_MAX_FOR_BUY = 50.0
PE_MAX = 35.0
DTE_MAX = 1.5
ROE_MIN = 0.05
PM_MIN = 0.03

# -------------------------
# Empresas IBEX 35
# -------------------------
EMPRESAS = {
    "ANA.MC": "Acciona",
    "ACX.MC": "Acerinox",
    "ACS.MC": "ACS",
    "AENA.MC": "Aena",
    "ALM.MC": "Almirall",
    "AMS.MC": "Amadeus",
    "SAB.MC": "Banco Sabadell",
    "SAN.MC": "Banco Santander",
    "BBVA.MC": "BBVA",
    "CABK.MC": "CaixaBank",
    "CLNX.MC": "Cellnex",
    "CIE.MC": "CIE Automotive",
    "COL.MC": "Colonial",
    "ENG.MC": "Enagás",
    "ELE.MC": "Endesa",
    "FER.MC": "Ferrovial",
    "GRF.MC": "Grifols",
    "IBE.MC": "Iberdrola",
    "ITX.MC": "Inditex",
    "IDR.MC": "Indra",
    "MAP.MC": "Mapfre",
    "MEL.MC": "Meliá Hotels",
    "NTGY.MC": "Naturgy",
    "PHM.MC": "PharmaMar",
    "RED.MC": "Redeia",
    "REP.MC": "Repsol",
    "ROVI.MC": "Rovi",
    "SLR.MC": "Solaria",
    "TEF.MC": "Telefónica",
    "UNI.MC": "Unicaja Banco",
    "VIS.MC": "Viscofan",
    "LOG.MC": "Logista"
}

# -------------------------
# Sidebar
# -------------------------
st.sidebar.header("Opciones de filtro")
tickers_seleccionados = st.sidebar.multiselect(
    "Selecciona tickers (por defecto todos):",
    options=list(EMPRESAS.keys()),
    default=list(EMPRESAS.keys())
)

periodo = st.sidebar.selectbox(
    "Periodo histórico:",
    ["3mo", "6mo", "1y", "2y"]
)

# -------------------------
# Inicializar VADER
# -------------------------
analyzer = SentimentIntensityAnalyzer()

def obtener_feed_google_news(query):
    query_enc = urllib.parse.quote(query)
    url_es = f"https://news.google.com/rss/search?q={query_enc}&hl=es&gl=ES&ceid=ES:ES"
    url_en = f"https://news.google.com/rss/search?q={query_enc}&hl=en&gl=US&ceid=US:en"
    return [url_es, url_en]

def evaluar_noticias(ticker, n=NEWS_PER_SOURCE):
    nombre = EMPRESAS.get(ticker)
    if not nombre:
        return 0
    score_total = 0
    feeds = obtener_feed_google_news(nombre)
    for feed_url in feeds:
        feed = feedparser.parse(feed_url)
        for entry in feed.entries[:n]:
            vs = analyzer.polarity_scores(entry.title)
            score_total += vs['compound']
    total_items = len(feeds) * n
    return score_total / total_items if total_items > 0 else 0

def es_bullish_engulfing(df):
    if len(df) < 2: return False
    prev = df.iloc[-2]
    last = df.iloc[-1]
    try:
        return prev['Close'] < prev['Open'] and last['Close'] > last['Open'] and last['Close'] > prev['Open'] and last['Open'] < prev['Close']
    except:
        return False

def evaluar_fundamentales(ticker):
    try:
        info = yf.Ticker(ticker).info
    except:
        return {"score": 0, "details": {}, "raw": {}}
    score = 0
    details = {}
    pe = info.get('trailingPE') or info.get('forwardPE')
    if pe and 0 < pe < PE_MAX: score += 1; details['PE_ok']=True
    else: details['PE_ok']=False
    dte = info.get('debtToEquity') or info.get('totalDebt')
    try: dte_val = float(dte) if dte else None
    except: dte_val=None
    if dte_val is not None and dte_val < DTE_MAX: score+=1; details['D/E_ok']=True
    else: details['D/E_ok']=False
    roe = info.get('returnOnEquity') or info.get('returnOnInvestment')
    try: roe_val=float(roe) if roe else None
    except: roe_val=None
    if roe_val and roe_val > ROE_MIN: score+=1; details['ROE_ok']=True
    else: details['ROE_ok']=False
    pm = info.get('profitMargins')
    try: pm_val=float(pm) if pm else None
    except: pm_val=None
    if pm_val and pm_val > PM_MIN: score+=1; details['PM_ok']=True
    else: details['PM_ok']=False
    return {"score": score, "details": details, "raw": info}

def evaluar_tecnica(ticker, period=periodo, interval="1d"):
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
        if df.empty or 'Close' not in df.columns: return None
        close = df['Close'].astype(float)
        ema20 = close.ewm(span=20, adjust=False).mean()
        ema50 = close.ewm(span=50, adjust=False).mean()
        rsi = ta.momentum.RSIIndicator(close, window=14).rsi()
        macd_obj = ta.trend.MACD(close)
        macd, macd_signal = macd_obj.macd(), macd_obj.macd_signal()
        bull = es_bullish_engulfing(df)
        conditions = [rsi.iloc[-1]<RSI_MAX_FOR_BUY, ema20.iloc[-1]>=ema50.iloc[-1], macd.iloc[-1]>macd_signal.iloc[-1], bull]
        tech_score = sum(bool(c) for c in conditions)
        return {'RSI': rsi.iloc[-1],'EMA20': ema20.iloc[-1],'EMA50': ema50.iloc[-1],'MACD': macd.iloc[-1],'MACD_signal': macd_signal.iloc[-1],'Bullish': bull,'tech_score': tech_score,'hist': df}
    except:
        return None

def calcular_metricas(ticker):
    try:
        data = yf.download(ticker, period="1y", interval="1d")['Close'].dropna()
        rets = data.pct_change().dropna()
        sharpe = (rets.mean()/rets.std()) * np.sqrt(252)
        max_dd = ((data / data.cummax()) - 1).min()
        return round(sharpe,2), round(max_dd,2)
    except:
        return None, None

# -------------------------
# Main
# -------------------------
st.header("📈 Resultados por Empresa")

resultados = []
for t in tickers_seleccionados:
    tech = evaluar_tecnica(t)
    fund = evaluar_fundamentales(t)
    news_score = evaluar_noticias(t)
    if not tech: continue
    tech_score = tech['tech_score']
    fund_score = fund['score']
    sharpe, dd = calcular_metricas(t)
    etiqueta = None
    if tech_score>=TECH_THRESHOLD and fund_score>=FUND_THRESHOLD and news_score>=NEWS_THRESHOLD:
        etiqueta="STRONG BUY"
    elif tech_score>=TECH_THRESHOLD or fund_score>=FUND_THRESHOLD or news_score>0:
        etiqueta="BUY"

    resultados.append({
        'Ticker': t,
        'Empresa': EMPRESAS[t],
        'TechScore': tech_score,
        'FundScore': fund_score,
        'NewsScore': round(news_score,3),
        'RSI': round(tech['RSI'],2),
        'Sharpe': sharpe,
        'Drawdown': dd,
        'Etiqueta': etiqueta
    })

    # Gráfico línea de precios
    st.subheader(f"{EMPRESAS[t]} ({t})")
    st.line_chart(tech['hist']['Close'])

    # Gráfico Tech/Fund/News
    st.bar_chart(pd.DataFrame({'Tech':[tech_score],'Fund':[fund_score],'News':[round(news_score,3)]}))

# -------------------------
# Ranking final
# -------------------------
if resultados:
    df = pd.DataFrame(resultados)
    df = df.sort_values(by=["TechScore","FundScore","NewsScore"], ascending=False).reset_index(drop=True)
    st.header("🏆 Ranking de Recomendaciones")
    st.dataframe(df.style.background_gradient(cmap="RdYlGn", subset=["TechScore","FundScore","NewsScore"]))

    # CSV descarga
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Descargar resultados en CSV", data=csv, file_name="ranking_empresas.csv", mime="text/csv")

    # Heatmap
    st.header("🔥 Heatmap de Scores")
    scores_df = df[['Ticker','TechScore','FundScore','NewsScore']].set_index('Ticker')
    fig, ax = plt.subplots(figsize=(10,6))
    sns.heatmap(scores_df, annot=True, cmap="RdYlGn", ax=ax)
    st.pyplot(fig)

else:
    st.warning("No se encontraron datos para los tickers seleccionados.")
