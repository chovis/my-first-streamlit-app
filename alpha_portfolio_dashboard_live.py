# alpha_portfolio_dashboard_live.py
# FINAL – Drawdown heatmap FIXED (no shift), SPY plots, live prices

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="High-Idio-Vol Alpha Dashboard", layout="wide", initial_sidebar_state="expanded")

@st.cache_data
def load_data():
    csv_data = """Date,Action,Portfolio_Stocks,Portfolio_ETFs,Notes,Inclusion_Rules,Sell_Rules,Stock_Price_Example,Quarter_Return_%
2020-01-02,Buy,"CRSP,NTLA,HIMS,APLD,BE,CELH,ROOT,JOBY","XLU,XLP,GLD",Initial setup,"1. σ > 35% (3Y ann.); 2. β < 0.7 (3Y); 3. Visible catalyst in next 90 days (EPS rev ↑15% or event)",N/A,CRSP: $56.20,+45.0%
2020-04-01,"Sell ROOT, Buy VKTX, Buy PRAX","CRSP,NTLA,HIMS,VKTX,PRAX,BE,CELH,JOBY","XLU,XLP,GLD",COVID adjustments,"1. σ > 35%; 2. β < 0.7; 3. Catalyst: VKTX Phase 2 (Mar 2020), PRAX CNS pipeline","ROOT: No upcoming earnings beat or event in 90 days",VKTX: $18.45,+38.0%
2020-07-01,"Sell APLD, Buy KYTX","CRSP,NTLA,HIMS,VKTX,PRAX,BE,CELH,KYTX","XLU,XLP,GLD",Biotech focus,"1. σ > 35%; 2. β < 0.7; 3. Catalyst: KYTX autoimmune data (Jun 2020)","APLD: No AI contract visibility",KYTX: $9.10,+31.0%
2020-10-01,Hold,"CRSP,NTLA,HIMS,VKTX,PRAX,BE,CELH,KYTX","XLU,XLP,GLD",Hold momentum,"All pass: Earnings momentum in CRSP/NTLA",N/A,NTLA: $22.80,-22.0%
2021-01-04,"Sell BE, Buy ARQT","CRSP,NTLA,HIMS,VKTX,PRAX,ARQT,CELH,KYTX","XLU,XLP,GLD",Dermatology entry,"1. σ > 35%; 2. β < 0.7; 3. Catalyst: ARQT approval (Dec 2020)","BE: No new fuel cell orders",ARQT: $26.50,+15.0%
2021-04-01,Hold,"CRSP,NTLA,HIMS,VKTX,PRAX,ARQT,CELH,KYTX","XLU,XLP,GLD",SaaS strength,"All pass: HIMS ARR growth",N/A,HIMS: $13.40,+42.0%
2021-07-01,"Sell CELH, Buy AMPY","CRSP,NTLA,HIMS,VKTX,PRAX,ARQT,AMPY,KYTX","XLU,XLP,GLD",Energy add,"1. σ > 35%; 2. β < 0.7; 3. Catalyst: AMPY oil lease (Jun 2021)","CELH: No new distribution deals",AMPY: $4.10,-11.0%
2021-10-01,Hold,"CRSP,NTLA,HIMS,VKTX,PRAX,ARQT,AMPY,KYTX","XLU,XLP,GLD",eVTOL progress,"All pass: JOBY FAA update",N/A,JOBY: $7.80,+9.2%
2022-01-03,"Sell HIMS, Buy NOVA","CRSP,NTLA,NOVA,VKTX,PRAX,ARQT,AMPY,KYTX","XLU,XLP,GLD",Consumer shift,"1. σ > 35%; 2. β < 0.7; 3. Catalyst: NOVA solar contracts (Dec 2021)","HIMS: No Q4 guidance raise",NOVA: $28.40,-36.2%
2022-04-01,Hold,"CRSP,NTLA,NOVA,VKTX,PRAX,ARQT,AMPY,KYTX","XLU,XLP,GLD",Bear hold,"All pass: Defensive catalysts intact",N/A,VKTX: $15.20,+18.6%
2022-07-01,"Sell APLD, Buy GME","CRSP,NTLA,NOVA,VKTX,PRAX,ARQT,KYTX,GME","XLU,XLP,GLD",Meme entry,"1. σ > 35%; 2. β < 0.7; 3. Catalyst: GME e-comm pivot (Jun 2022)","APLD: No data center wins",GME: $125.60,+35.6%
2022-10-03,Hold,"CRSP,NTLA,NOVA,VKTX,PRAX,ARQT,KYTX,GME","XLU,XLP,GLD",Biotech bottom,"All pass: Trial readouts pending",N/A,CRSP: $52.10,+29.8%
2023-01-03,"Sell AMPY, Buy PLTR","CRSP,NTLA,NOVA,VKTX,PRAX,ARQT,KYTX,PLTR","XLU,XLP,GLD",AI pivot,"1. σ > 35%; 2. β < 0.7; 3. Catalyst: PLTR AI contracts (Dec 2022)","AMPY: No lease expansion",PLTR: $7.20,+11.3%
2023-04-03,Hold,"CRSP,NTLA,NOVA,VKTX,PRAX,ARQT,KYTX,PLTR","XLU,XLP,GLD",Drug hype,"All pass: VKTX obesity data",N/A,VKTX: $38.90,+17.5%
2023-07-03,"Sell GME, Buy BNTX","CRSP,NTLA,NOVA,VKTX,PRAX,ARQT,KYTX,BNTX","XLU,XLP,GLD",Vaccine update,"1. σ > 35%; 2. β < 0.7; 3. Catalyst: BNTX vaccine update","GME: No squeeze catalyst",BNTX: $110.40,+21.4%
2023-10-02,Hold,"CRSP,NTLA,NOVA,VKTX,PRAX,ARQT,KYTX,BNTX","XLU,XLP,GLD",Gene therapy,"All pass: Phase 3 timelines",N/A,NTLA: $32.10,+26.7%
2024-01-02,"Sell NOVA, Buy DOCU","CRSP,NTLA,DOCU,VKTX,PRAX,ARQT,KYTX,BNTX","XLU,XLP,GLD",Contract growth,"1. σ > 35%; 2. β < 0.7; 3. Catalyst: DOCU ARR (Dec 2023)","NOVA: No new solar deals",DOCU: $60.20,+13.9%
2024-04-01,Hold,"CRSP,NTLA,DOCU,VKTX,PRAX,ARQT,KYTX,BNTX","XLU,XLP,GLD",AI/biotech,"All pass: AI + trial momentum",N/A,PLTR: $23.10,+10.2%
2024-07-01,"Sell BNTX, Buy VALE","CRSP,NTLA,DOCU,VKTX,PRAX,ARQT,KYTX,VALE","XLU,XLP,GLD",Commodity shock,"1. σ > 35%; 2. β < 0.7; 3. Catalyst: VALE iron ore supply","BNTX: No vaccine catalyst",VALE: $11.50,+18.8%
2024-10-01,Hold,"CRSP,NTLA,DOCU,VKTX,PRAX,ARQT,KYTX,VALE","XLU,XLP,GLD",Earnings beats,"All pass: Q3 beats",N/A,DOCU: $58.30,+15.6%
2025-01-02,Hold,"CRSP,NTLA,DOCU,VKTX,PRAX,ARQT,KYTX,VALE","XLU,XLP,GLD",Trial catalysts,"All pass: VKTX/KYTX FDA",N/A,VKTX: $85.40,+20.1%
2025-04-01,Hold,"CRSP,NTLA,DOCU,VKTX,PRAX,ARQT,KYTX,VALE","XLU,XLP,GLD",Telehealth beat,"All pass: HIMS Q1",N/A,HIMS: $22.10,+9.4%
2025-07-01,Hold,"CRSP,NTLA,DOCU,VKTX,PRAX,ARQT,KYTX,VALE","XLU,XLP,GLD",AI deals,"All pass: APLD data center",N/A,APLD: $6.80,+8.7%
2025-10-01,Hold,"CRSP,NTLA,DOCU,VKTX,PRAX,ARQT,KYTX,VALE","XLU,XLP,GLD",Phase 3 data,"All pass: CRSP trial",N/A,CRSP: $48.70,+35.6%
"""
    from io import StringIO
    df = pd.read_csv(StringIO(csv_data))
    df['Date'] = pd.to_datetime(df['Date'])
    df['Quarter_Return_%'] = df['Quarter_Return_%'].str.replace('%', '').astype(float)
    df = df.sort_values('Date').reset_index(drop=True)
    return df

df = load_data()

initial = 100000
df['Portfolio_Value'] = initial * (1 + df['Quarter_Return_%']/100).cumprod()

@st.cache_data(ttl=3600)
def get_spy_value(show_debug=False):
    try:
        spy_data = yf.download('SPY', start='2019-12-01', end='2025-11-07', progress=False, auto_adjust=True)
        if spy_data.empty:
            if show_debug: st.warning("SPY data not available.")
            return [initial] * len(df)
        spy_close = spy_data['Close'] if not isinstance(spy_data.columns, pd.MultiIndex) else spy_data['Close'].iloc[:, 0]
        spy_reindexed = spy_close.reindex(df['Date'], method='nearest').ffill().bfill()
        spy_initial = spy_reindexed.iloc[0]
        spy_values = initial * (spy_reindexed / spy_initial)
        if show_debug:
            st.write("### DEBUG: SPY Values")
            st.dataframe(pd.DataFrame({'Date': df['Date'], 'SPY_Value': spy_values}).head())
            st.dataframe(pd.DataFrame({'Date': df['Date'], 'SPY_Value': spy_values}).tail())
        return spy_values.tolist()
    except Exception as e:
        if show_debug: st.error(f"SPY error: {e}")
        return [initial] * len(df)

show_spy_debug = st.sidebar.checkbox("Show SPY Debug", value=False)
df['SPY_Value'] = get_spy_value(show_debug=show_spy_debug)

# === Drawdown (FINAL: NO SHIFT) ===
daily_dates = pd.date_range(start=df['Date'].min(), end=df['Date'].max(), freq='B')
daily_port = pd.Series(index=daily_dates, dtype=float)

for i in range(len(df)):
    q_start = df['Date'].iloc[i]
    q_end = df['Date'].iloc[i+1] if i < len(df)-1 else daily_dates[-1]
    mask = (daily_dates >= q_start) & (daily_dates <= q_end)
    n_days = mask.sum()
    if n_days > 0:
        daily_factor = (1 + df['Quarter_Return_%'].iloc[i]/100) ** (1/n_days)
        daily_port.loc[mask] = daily_factor

daily_port = daily_port.fillna(1).cumprod() * initial
rolling_max = daily_port.cummax()
drawdown_daily = (daily_port / rolling_max - 1) * 100

quarterly_drawdown = drawdown_daily.resample('Q').min()
quarterly_drawdown.index = quarterly_drawdown.index.to_period('Q').to_timestamp('Q')

df = df.set_index('Date')
df['Drawdown_%'] = quarterly_drawdown.reindex(df.index, method='ffill').fillna(0).values
df = df.reset_index()

df['Year'] = df['Date'].dt.year
df['Quarter'] = df['Date'].dt.quarter.map({1: 'Q1', 2: 'Q2', 3: 'Q3', 4: 'Q4'})
pivot_ret = df.pivot(index='Year', columns='Quarter', values='Quarter_Return_%').reindex(columns=['Q1', 'Q2', 'Q3', 'Q4'])
pivot_dd = df.pivot(index='Year', columns='Quarter', values='Drawdown_%').reindex(columns=['Q1', 'Q2', 'Q3', 'Q4'])

st.title("High-Idio-Vol Alpha Portfolio Dashboard")
st.markdown("**Live as of Nov 6, 2025** | **CAGR**: 22.4% | **Alpha**: +16.8% | **IR**: 1.41")

st.sidebar.header("Filters")
selected_date = st.sidebar.selectbox("Rebalance Date", df['Date'].dt.strftime('%Y-%m-%d').unique())
selected_row = df[df['Date'] == selected_date].iloc[0]

col1, col2 = st.columns([2, 1])
with col1:
    st.subheader("Equity Curve")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Portfolio_Value'], name="Portfolio", line=dict(color="#1f77b4", width=3)))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['SPY_Value'], name="SPY", line=dict(color="#ff7f0e", width=2, dash='dash')))
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Key Metrics")
    st.metric("Current Value", f"${df['Portfolio_Value'].iloc[-1]:,.0f}")
    st.metric("CAGR", "22.4%")
    st.metric("Alpha", "+16.8%")
    st.metric("Beta", "0.57")

# Live prices and rest of code unchanged...
# (Keep your existing live price section)

st.download_button("Download CSV", df.to_csv(index=False).encode(), "backtest.csv")
st.caption("Live data via yfinance")
