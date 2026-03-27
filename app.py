"""
SmallCap Dual-Engine — Quant Rankings + Qualitative Sector Analysis
Tab 1: Quantitative Rankings (Established Compounders / Momentum Efficiency)
Tab 2: Fund Sector Flow (dropdown → month-by-month sector changes)
Tab 3: Sector Consensus (cross-fund heatmap + readings)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import timedelta
import warnings, io

try:
    import matplotlib
except ImportError:
    pass

warnings.filterwarnings("ignore")
st.set_page_config(page_title="SmallCap Quants", layout="wide", page_icon="📊")

# ═══════════════════════════════════════════
# FUND QUALITATIVE PROFILES
# ═══════════════════════════════════════════
PROVEN = {
    "Bandhan Small Cap Fund-Reg(G)": {"PE": 16.78, "PB": 2.07, "Stance": "Deep Value",
        "Label": "Deep Value — Buy & Hold",
        "Detail": "PE 16.8x / PB 2.07x — lowest in the group. Finance-heavy at value prices. Minimal sector rotation over 12 months. Conviction-based, low-churn approach. Buys cheap, sits and waits."},
    "Bank of India Small Cap Fund-Reg(G)": {"PE": 32.82, "PB": 4.02, "Stance": "Balanced Momentum",
        "Label": "Cyclical Rotation — Defensive to Offensive",
        "Detail": "Biggest single move: +8pp into banking while cutting healthcare -4pp and finance -4pp. Classic cyclical rotation from defensive to rate-sensitive sectors."},
    "Edelweiss Small Cap Fund-Reg(G)": {"PE": 31.21, "PB": 4.13, "Stance": "Risk-Adjusted Growth",
        "Label": "Diversified All-Weather Portfolio",
        "Detail": "Most balanced allocation — no sector above 13%. Added to banks, autos, chemicals while trimming capital goods. Lowest concentration risk in the group."},
    "Canara Rob Small Cap Fund-Reg(G)": {"PE": 28.33, "PB": 3.64, "Stance": "Quality / Hedged",
        "Label": "Financial Barbell + Defensive Hedge",
        "Detail": "Finance + Banking = 24.3%, highest combined financial exposure. FMCG at 6.1% as defensive ballast. Barbell: high-beta financials for upside, staples for downside protection."},
    "Invesco India Smallcap Fund-Reg(G)": {"PE": 39.47, "PB": 4.24, "Stance": "Ultra-Growth Premium",
        "Label": "Structural Growth — Healthcare + Consumer",
        "Detail": "Healthcare at 21.8% — highest single-sector bet of ANY fund (+6.6pp). Retailing 9.4%. Banking +8pp. Betting on India's structural consumption story. Highest PE (39.5x)."},
    "Mahindra Manulife Small Cap Fund-Reg(G)": {"PE": 29.50, "PB": 3.85, "Stance": "Diversified Growth",
        "Label": "Diversified Multi-Sector",
        "Detail": "Healthcare 14.8%, autos 8.1%, finance 7.9%, iron & steel 6.9%, banking 7%. Broad-based exposure across manufacturing, healthcare, and financials with no extreme concentration."},
}

MOMENTUM_INFO = {
    "TRUSTMF Small Cap Fund-Reg(G)": {"PE": 37.31, "PB": 4.87, "Stance": "Terminal Value Premium",
        "Label": "Concentrated Cyclical — Capex + Credit",
        "Detail": "Capital goods 12.2% + banking 11.6% (+7.7pp). Healthcare trimmed -4pp. Bet on capex cycle + credit growth."},
    "Union Small Cap Fund-Reg(G)": {"PE": 39.27, "PB": 5.24, "Stance": "High-Growth Momentum",
        "Label": "Capex Cycle Maximalist",
        "Detail": "Capital goods at 18.7% — highest of any fund, +7.1pp. Banking +7.2pp, autos +4pp. All-in on India's capex cycle."},
    "DSP Small Cap Fund-Reg(G)": {"PE": 28.39, "PB": 3.52, "Stance": "Concentrated Core",
        "Label": "Domestic Manufacturing Contrarian",
        "Detail": "Autos 16.8% (highest, +5.4pp), chemicals 11.2%, FMCG 10.5%. AVOIDS banking — only fund with no significant banking weight."},
    "Aditya Birla SL Small Cap Fund(G)": {"PE": 27.71, "PB": 4.10, "Stance": "Operating Leverage",
        "Label": "Rate Cut Beneficiary",
        "Detail": "Added +7pp banking and +6pp healthcare while cutting capital goods -5pp. Positioning for operating leverage from lower rates."},
    "Mirae Asset Small Cap Fund-Reg(G)": {"PE": 24.80, "PB": 3.92, "Stance": "GARP",
        "Label": "GARP — Broad Based Builder",
        "Detail": "Added to 6+ sectors simultaneously. Broadest build-up. Lowest PE (24.8x) among momentum funds. Buying growth without overpaying."},
    "Invesco India Smallcap Fund-Reg(G)": {"PE": 39.47, "PB": 4.24, "Stance": "Ultra-Growth Premium",
        "Label": "Structural Growth — Healthcare + Consumer",
        "Detail": "Healthcare 21.8%, Retailing 9.4%, Banking +8pp. Structural consumption conviction."},
    "Mahindra Manulife Small Cap Fund-Reg(G)": {"PE": 29.50, "PB": 3.85, "Stance": "Diversified Growth",
        "Label": "Diversified Multi-Sector",
        "Detail": "Healthcare 14.8%, autos 8.1%, iron & steel 6.9%. Balanced across manufacturing and services."},
}

ALL_INFO = {**PROVEN, **MOMENTUM_INFO}
ALL_QUAL_FUNDS = list(dict.fromkeys(list(PROVEN.keys()) + list(MOMENTUM_INFO.keys())))

SECTOR_READINGS = {
    "Bank": "Strongest crowd trade. Nearly EVERY fund added 5-8pp. Unanimous positioning for rate cuts + credit growth.",
    "Healthcare": "High average but split. Invesco betting huge (21.8%), others steady or trimming. Valuation debate.",
    "Finance": "Core structural holding. Stable across all funds — treated as base allocation, not tactical.",
    "Capital Goods": "Diverging. Union (18.7%) and TRUSTMF heavy; Invesco/Aditya Birla exiting. Capex conviction split.",
    "Automobile & Ancillaries": "Growing consensus on domestic auto + EV supply chain. DSP heaviest (16.8%).",
    "Chemicals": "Selective China+1 additions. DSP and Edelweiss building; others stable.",
    "FMCG": "Defensive hedge. Held by DSP (10.5%) and Canara Rob. Absent from aggressive momentum funds.",
    "Realty": "Housing cycle play. Concentrated in Bandhan and Invesco. Not universal.",
    "Construction Materials": "Small steady positions. No strong conviction either direction.",
    "IT": "Small positions. Selective small-cap IT services exposure.",
    "Retailing": "Invesco-only conviction bet (9.4%). Not consensus.",
    "Consumer Durables": "Small but growing. Several funds adding 2-3pp.",
    "Business Services": "Diverging. Mirae adding (+5pp), others trimming.",
    "Iron & Steel": "Small value positions. Commodity view split.",
    "Textile": "Tiny, fading positions. Not a live theme.",
    "Electricals": "Niche positions. Manufacturing/capex adjacent.",
}

MONTHS = ["Jan_25", "Jun_25", "Sep_25", "Dec_25", "Feb_26"]
MONTH_LABELS = ["Jan 2025", "Jun 2025", "Sep 2025", "Dec 2025", "Feb 2026"]


def short(f):
    return (f.replace("Small Cap", "SC").replace("Smallcap", "SC")
             .replace("Fund-Reg(G)", "").replace("Fund(G)", "").strip())


# ═══════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════
@st.cache_data(show_spinner="Loading SmallCap Data...")
def load_data():
    raw = pd.read_excel("smallcapfinalrank.xlsx")
    fund_names = raw.iloc[1, 1:].dropna().tolist()
    nav = raw.iloc[3:, : len(fund_names) + 1].copy()
    nav.columns = ["Date"] + fund_names
    nav = nav[pd.to_datetime(nav["Date"], errors="coerce").notna()].copy()
    nav["Date"] = pd.to_datetime(nav["Date"])
    nav = nav.sort_values("Date").reset_index(drop=True)
    for f in fund_names:
        nav[f] = pd.to_numeric(nav[f], errors="coerce")

    try:
        bench = pd.read_csv("Nifty_500_TRI_Combined.csv")
        bench["Date"] = pd.to_datetime(bench["Date"])
        nav = pd.merge(nav, bench[["Date", "Benchmark"]], on="Date", how="left")
        nav["Benchmark"] = nav["Benchmark"].ffill()
        bench_name = "Nifty 500 TRI"
    except Exception:
        try:
            bench = pd.read_csv("BSE SMAALCAP.xlsx - Abakkus Small Cap Fund-Reg(G)1.csv", skiprows=2)
            bench["Date"] = pd.to_datetime(bench["Date"])
            bench = bench.rename(columns={"Close Price": "Benchmark"})
            nav = pd.merge(nav, bench[["Date", "Benchmark"]], on="Date", how="left")
            nav["Benchmark"] = nav["Benchmark"].ffill()
            bench_name = "BSE Small-Cap TRI"
        except Exception:
            valid = [f for f in fund_names if nav[f].notna().sum() > 252]
            nav["Benchmark"] = nav[valid].pct_change().mean(axis=1).add(1).cumprod() * 100
            bench_name = "SmallCap Index (Proxy)"

    ar = pd.read_excel("smallcap_aum.xlsx")
    aum = ar.iloc[3:].copy()
    aum.columns = ["Fund", "Month_End", "AUM", "AAUM", "Avg_AUM"]
    aum["AUM"] = pd.to_numeric(aum["AUM"], errors="coerce")
    aum = aum[aum["AUM"].notna()]
    aum["Month_End"] = pd.to_datetime(aum["Month_End"], errors="coerce")
    aum_latest = aum.sort_values("Month_End", ascending=False).groupby("Fund").first().reset_index()[["Fund", "AUM"]]

    return nav, fund_names, aum_latest, bench_name


@st.cache_data(show_spinner="Loading Sector Data...")
def load_sectors():
    df_raw = pd.read_excel("SECTORALLCOATIONSMALLCAP.xlsx")
    d = df_raw.iloc[2:].copy()
    d.columns = ["Fund", "Sector", "Feb_26", "Dec_25", "Sep_25", "Jun_25", "Jan_25",
                  "c7", "c8", "c9", "c10", "c11"]
    d = d[["Fund", "Sector", "Feb_26", "Dec_25", "Sep_25", "Jun_25", "Jan_25"]]
    d = d.dropna(subset=["Fund", "Sector"])
    d = d[~d["Fund"].str.contains("Accord", na=False)]
    d = d[d["Sector"] != "Sector"]
    for c in MONTHS:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    return d


# ═══════════════════════════════════════════
# QUANT METRICS ENGINE
# ═══════════════════════════════════════════
@st.cache_data(show_spinner="Computing Factors...")
def compute_all(_nav, fund_names, aum_latest):
    nav = _nav.copy()
    monthly = nav.set_index("Date").resample("ME").last()
    mret = monthly.pct_change().dropna(how="all")
    br = mret["Benchmark"]

    results = []
    for fund
