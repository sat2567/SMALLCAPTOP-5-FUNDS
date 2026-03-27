import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import timedelta
import warnings, io

# Dependency Check for styling
try:
    import matplotlib
except ImportError:
    st.error("Missing dependency: 'matplotlib'. Please run 'pip install matplotlib'.")

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

def short(f):
    return (f.replace("Small Cap", "SC").replace("Smallcap", "SC")
             .replace("Fund-Reg(G)", "").replace("Fund(G)", "").strip())

# ═══════════════════════════════════════════
# DATA LOAD
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
    aum_latest = aum.sort_values("Month_End", ascending=False).groupby("Fund").first().reset_index()[["Fund", "AUM"]]

    return nav, fund_names, aum_latest, bench_name

# ═══════════════════════════════════════════
# METRICS ENGINE
# ═══════════════════════════════════════════
@st.cache_data(show_spinner="Computing Factors...")
def compute_all(_nav, fund_names, aum_latest):
    nav = _nav.copy()
    monthly = nav.set_index("Date").resample("ME").last()
    mret = monthly.pct_change().dropna(how="all")
    br = mret["Benchmark"]

    results = []
    for fund in fund_names:
        fd = nav[["Date", fund]].dropna()
        if len(fd) < 120: continue 
        
        fr = mret[fund].dropna()
        ci = fr.index.intersection(br.dropna().index)
        if len(ci) < 6: continue
        fr, brc = fr.loc[ci], br.loc[ci]

        # ENGINE 1
        mn = fd.set_index("Date")[fund].resample("ME").last().dropna()
        cagrs_5y = [(((mn.iloc[i]/mn.iloc[i-60])**(1/5))-1)*100 for i in range(60, len(mn))]
        r5_mean = np.mean(cagrs_5y) if cagrs_5y else None
        
        upside_m = brc[brc > 0]
        downside_m = brc[brc < 0]
        up_cap = (fr.loc[upside_m.index].mean() / upside_m.mean()) * 100 if len(upside_m) > 3 else None
        down_cap = (fr.loc[downside_m.index].mean() / downside_m.mean()) * 100 if len(downside_m) > 3 else None

        prices = fd[fund].values
        peaks = np.maximum.accumulate(prices)
        dd_series = (prices - peaks) / peaks
        ulcer_index = np.sqrt(np.mean(dd_series**2)) * 100 
        max_dd = dd_series.min() * 100

        # ENGINE 2
        ld = fd.iloc[-1]["Date"]
        rec_1y = fd[fd["Date"] >= ld - timedelta(days=365)]
        vol = rec_1y[fund].pct_change().dropna().std() * np.sqrt(252) * 100 if len(rec_1y) > 50 else None
        
        def get_abs_ret(days):
            target = ld - timedelta(days=days)
            past_nav_df = fd[fd["Date"] <= target]
            if past_nav_df.empty: return None
            past = past_nav_df.iloc[-1][fund]
            return (prices[-1] / past - 1) * 100

        ret_6m = get_abs_ret(180)
        ret_1y = get_abs_ret(365)

        # Dampened Volatility adjustment
        mom_6m_score = ret_6m / np.sqrt(vol) if (vol and vol > 0 and ret_6m is not None) else None
        mom_1y_score = ret_1y / np.sqrt(vol) if (vol and vol > 0 and ret_1y is not None) else None

        results.append({
            "Fund": fund, "Track_Yrs": len(fd)/252, 
            "Roll_5Y": r5_mean, "Up_Cap": up_cap, "Down_Cap": down_cap, 
            "Ulcer_Index": ulcer_index, "Max_DD": max_dd,
            "Ret_6M": ret_6m, "Ret_1Y": ret_1y, "Vol": vol,
            "Mom_6M_RA": mom_6m_score, "Mom_1Y_RA": mom_1y_score
        })

    return pd.DataFrame(results).merge(aum_latest, on="Fund", how="left")

# ═══════════════════════════════════════════
# RANKING LOGIC (FIXED)
# ═══════════════════════════════════════════
def pctrank(s, asc=True):
    v = s.notna()
    if not v.any(): return s
    r = s[v].rank(ascending=asc, pct=True) * 100
    out = pd.Series(np.nan, index=s.index)
    out[v] = r
    return out

def rank_funds(df, w_est, w_mom):
    # Engine 1
    est = df[df["Track_Yrs"] >= 3].copy()
    if not est.empty:
        est["S_R5"] = pctrank(est["Roll_5Y"])
        est["S_UC"] = pctrank(est["Up_Cap"])
        est["S_DC"] = pctrank(est["Down_Cap"], False)
        est["S_UI"] = pctrank(est["Ulcer_Index"], False)
        
        est["Score"] = (est["S_R5"].fillna(0)*w_est[0] + est["S_UC"].fillna(0)*w_est[1] + 
                        est["S_DC"].fillna(0)*w_est[2] + est["S_UI"].fillna(0)*w_est[3]) / sum(w_est)
        # Use 'Int64' to allow NaNs or handle ranking safely
        est["Rank"] = est["Score"].rank(ascending=False, method="min")

    # Engine 2
    mom = df[df["Mom_6M_RA"].notna()].copy()
    if not mom.empty:
        mom["S_M6"] = pctrank(mom["Mom_6M_RA"])
        mom["S_M1"] = pctrank(mom["Mom_1Y_RA"])
        mom["Score"] = (mom["S_M6"].fillna(0)*w_mom[0] + mom["S_M1"].fillna(0)*w_mom[1]) / sum(w_mom)
        mom["Rank"] = mom["Score"].rank(ascending=False, method="min")

    return est, mom

# ═══════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════
def main():
    nav, fund_names, aum_latest, b_name = load_data()
    df_raw = compute_all(nav, fund_names, aum_latest)

    st.title("🚀 SmallCap Dual-Engine Quants")

    with st.sidebar:
        st.subheader("🏛️ Engine 1 Weights")
        w_cagr = st.slider("5Y Rolling CAGR", 0, 100, 30)
        w_up = st.slider("Upside Capture", 0, 100, 20)
        w_down = st.slider("Downside Capture", 0, 100, 30)
        w_ulcer = st.slider("Ulcer Index", 0, 100, 20)
        
        st.divider()
        st.subheader("🏎️ Engine 2 Weights")
        m6 = st.slider("6M Dampened RA", 0, 100, 50)
        m1 = st.slider("1Y Dampened RA", 0, 100, 50)

    est, mom = rank_funds(df_raw, [w_cagr, w_up, w_down, w_ulcer], [m6, m1])

    view = st.radio("Ranking Engine", ["🏛️ Established Compounders", "🏎️ Momentum Efficiency"], horizontal=True)
    target = est if "Established" in view else mom

    if not target.empty:
        disp = target.sort_values("Rank").copy()
        
        # Keep original name for qualitative mapping
        disp["Orig_Fund"] = disp["Fund"]
        
        if "Established" in view:
            cols = ["Rank", "Fund", "Score", "Up_Cap", "Down_Cap", "Ulcer_Index", "Roll_5Y", "AUM", "Orig_Fund"]
            names = ["Rank", "Fund", "Score", "Upside %", "Downside %", "DD Stress", "5Y CAGR", "AUM (Cr)", "Orig_Fund"]
        else:
            cols = ["Rank", "Fund", "Score", "Ret_6M", "Ret_1Y", "Vol", "AUM", "Orig_Fund"]
            names = ["Rank", "Fund", "Score", "6M Ret%", "1Y Ret%", "Vol%", "AUM (Cr)", "Orig_Fund"]
            
        disp = disp[cols]
        disp.columns = names
        
        # Shorten names for table display
        disp["Fund"] = disp["Fund"].apply(short)

        num_cols = [c for c in disp.columns if c not in ["Rank", "Fund", "AUM (Cr)", "Orig_Fund"]]

        # Styled Table (excluding the hidden Orig_Fund column)
        styled = (disp.drop(columns=["Orig_Fund"]).style
            .background_gradient(subset=["Score"], cmap="RdYlGn")
            .format("{:.0f}", subset=["Rank"])
            .format("{:.1f}", subset=num_cols)
            .format("{:.0f}", subset=["AUM (Cr)"], na_rep="—")
        )

        # ── TWO COLUMN LAYOUT ──
        col_table, col_info = st.columns([7, 3])
        
        with col_table:
            st.dataframe(styled, use_container_width=True, height=600, hide_index=True)

        with col_info:
            st.markdown(f"#### Fund Strategies ({'Momentum' if 'Momentum' in view else 'Established'})")
            
            # Create a scrollable container for the strategies
            st.markdown("<div style='height: 560px; overflow-y: auto; padding-right: 10px;'>", unsafe_allow_html=True)
            
            funds_displayed = 0
            for _, row in disp.iterrows():
                orig_name = row["Orig_Fund"]
                short_name = row["Fund"]
                
                info = ALL_INFO.get(orig_name, {})
                
                if info:
                    st.markdown(f"**{short_name}**")
                    st.caption(f"🛡️ **Stance:** {info.get('Label', '—')}")
                    st.markdown(f"<p style='font-size: 13.5px; margin-top: -10px;'>{info.get('Detail', '—')}</p>", unsafe_allow_html=True)
                    st.markdown("---")
                    funds_displayed += 1
            
            if funds_displayed == 0:
                st.info("No qualitative strategy descriptions available for the filtered list of funds yet.")
                
            st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
