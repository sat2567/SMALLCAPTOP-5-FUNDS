import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import timedelta
import warnings, io

# Dependency Check: Ensure matplotlib is installed for .style.background_gradient
try:
    import matplotlib
except ImportError:
    st.error("Missing dependency: 'matplotlib'. Please run 'pip install matplotlib' to enable table colors.")

warnings.filterwarnings("ignore")
st.set_page_config(page_title="SmallCap Quants", layout="wide", page_icon="📊")

# ═══════════════════════════════════════════
# DATA LOAD (BSE SmallCap TRI Benchmark)
# ═══════════════════════════════════════════
@st.cache_data(show_spinner="Loading SmallCap Data...")
def load_data():
    # 1. Load Fund NAVs
    raw = pd.read_excel("smallcapfinalrank.xlsx")
    fund_names = raw.iloc[1, 1:].dropna().tolist()
    nav = raw.iloc[3:, : len(fund_names) + 1].copy()
    nav.columns = ["Date"] + fund_names
    nav = nav[pd.to_datetime(nav["Date"], errors="coerce").notna()].copy()
    nav["Date"] = pd.to_datetime(nav["Date"])
    nav = nav.sort_values("Date").reset_index(drop=True)
    for f in fund_names:
        nav[f] = pd.to_numeric(nav[f], errors="coerce")

    # 2. Load BSE Small-Cap TRI (Specific SmallCap Benchmark)
    try:
        # Header starts at row 3 (Index Name, Date, Close Price)
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

    # 3. Load AUM
    ar = pd.read_excel("smallcap_aum.xlsx")
    aum = ar.iloc[3:].copy()
    aum.columns = ["Fund", "Month_End", "AUM", "AAUM", "Avg_AUM"]
    aum_latest = aum.sort_values("Month_End", ascending=False).groupby("Fund").first().reset_index()[["Fund", "AUM"]]

    return nav, fund_names, aum_latest, bench_name

# ═══════════════════════════════════════════
# METRICS ENGINE (Dampened Volatility Factor)
# ═══════════════════════════════════════════
@st.cache_data(show_spinner="Computing Dual-Engine Factors...")
def compute_all(_nav, fund_names, aum_latest):
    nav = _nav.copy()
    monthly = nav.set_index("Date").resample("ME").last()
    mret = monthly.pct_change().dropna(how="all")
    br = mret["Benchmark"]

    results = []
    for fund in fund_names:
        fd = nav[["Date", fund]].dropna()
        if len(fd) < 120: continue # Min track record
        
        fr = mret[fund].dropna()
        ci = fr.index.intersection(br.dropna().index)
        if len(ci) < 6: continue
        fr, brc = fr.loc[ci], br.loc[ci]

        # --- ENGINE 1: ESTABLISHED ---
        # 1. 5Y Rolling CAGR
        mn = fd.set_index("Date")[fund].resample("ME").last().dropna()
        cagrs_5y = [(((mn.iloc[i]/mn.iloc[i-60])**(1/5))-1)*100 for i in range(60, len(mn))]
        r5_mean = np.mean(cagrs_5y) if cagrs_5y else None
        
        # 2. Capture Ratios (vs SmallCap Benchmark)
        upside_m = brc[brc > 0]
        downside_m = brc[brc < 0]
        up_cap = (fr.loc[upside_m.index].mean() / upside_m.mean()) * 100 if len(upside_m) > 3 else None
        down_cap = (fr.loc[downside_m.index].mean() / downside_m.mean()) * 100 if len(downside_m) > 3 else None

        # 3. Drawdown Intensity (Ulcer Index)
        prices = fd[fund].values
        peaks = np.maximum.accumulate(prices)
        dd_series = (prices - peaks) / peaks
        ulcer_index = np.sqrt(np.mean(dd_series**2)) * 100 
        max_dd = dd_series.min() * 100

        # --- ENGINE 2: DAMPENED MOMENTUM ---
        ld = fd.iloc[-1]["Date"]
        rec_1y = fd[fd["Date"] >= ld - timedelta(days=365)]
        vol = rec_1y[fund].pct_change().dropna().std() * np.sqrt(252) * 100 if len(rec_1y) > 50 else None
        
        # Calculate Absolute Returns
        def get_abs_ret(days):
            target = ld - timedelta(days=days)
            past = fd.iloc[(fd["Date"] - target).abs().argsort()[:1]][fund].values[0]
            return (prices[-1] / past - 1) * 100

        ret_6m = get_abs_ret(180)
        ret_1y = get_abs_ret(365)

        # Dampened Volatility Factor: Return / sqrt(Vol)
        # Prevents volatility from becoming the predominant factor
        mom_6m_score = ret_6m / np.sqrt(vol) if (vol and vol > 0) else None
        mom_1y_score = ret_1y / np.sqrt(vol) if (vol and vol > 0) else None

        results.append({
            "Fund": fund, "Track_Yrs": len(fd)/252, 
            "Roll_5Y": r5_mean, "Up_Cap": up_cap, "Down_Cap": down_cap, 
            "Ulcer_Index": ulcer_index, "Max_DD": max_dd,
            "Ret_6M": ret_6m, "Ret_1Y": ret_1y, "Vol": vol,
            "Mom_6M_RA": mom_6m_score, "Mom_1Y_RA": mom_1y_score
        })

    return pd.DataFrame(results).merge(aum_latest, on="Fund", how="left")

# ═══════════════════════════════════════════
# RANKING LOGIC
# ═══════════════════════════════════════════
def pctrank(s, asc=True):
    v = s.notna()
    r = s[v].rank(ascending=asc, pct=True) * 100
    out = pd.Series(np.nan, index=s.index)
    out[v] = r
    return out

def rank_funds(df, w_est, w_mom):
    # Engine 1 (> 3Yrs)
    est = df[df["Track_Yrs"] >= 3].copy()
    if not est.empty:
        est["S_R5"] = pctrank(est["Roll_5Y"])
        est["S_UC"] = pctrank(est["Up_Cap"])      # High is good
        est["S_DC"] = pctrank(est["Down_Cap"], False) # Low is good
        est["S_UI"] = pctrank(est["Ulcer_Index"], False) # Low is good
        
        est["Score"] = (est["S_R5"]*w_est[0] + est["S_UC"]*w_est[1] + 
                        est["S_DC"]*w_est[2] + est["S_UI"]*w_est[3]) / sum(w_est)
        est["Rank"] = est["Score"].rank(ascending=False, method="min").astype(int)

    # Engine 2 (All funds >= 6M)
    mom = df[df["Ret_6M"].notna()].copy()
    if not mom.empty:
        mom["S_M6"] = pctrank(mom["Mom_6M_RA"])
        mom["S_M1"] = pctrank(mom["Mom_1Y_RA"])
        mom["Score"] = (mom["S_M6"]*w_mom[0] + mom["S_M1"]*w_mom[1]) / sum(w_mom)
        mom["Rank"] = mom["Score"].rank(ascending=False, method="min").astype(int)

    return est, mom

# ═══════════════════════════════════════════
# UI
# ═══════════════════════════════════════════
def main():
    nav, fund_names, aum_latest, bench_name = load_data()
    df_raw = compute_all(nav, fund_names, aum_latest)

    st.title("🚀 SmallCap Dual-Engine Quants")
    st.info(f"Comparing against **{bench_name}** | Engine 2 uses **Dampened Volatility** Adjustment.")

    with st.sidebar:
        st.subheader("🏛️ Engine 1 (Full-Cycle)")
        w_cagr = st.slider("5Y Rolling CAGR", 0, 100, 30)
        w_up = st.slider("Upside Capture", 0, 100, 20)
        w_down = st.slider("Downside Capture", 0, 100, 30)
        w_ulcer = st.slider("Ulcer Index (DD Stress)", 0, 100, 20)
        
        st.divider()
        st.subheader("🏎️ Engine 2 (Momentum)")
        m6 = st.slider("6M Dampened RA", 0, 100, 50)
        m1 = st.slider("1Y Dampened RA", 0, 100, 50)

    est, mom = rank_funds(df_raw, [w_cagr, w_up, w_down, w_ulcer], [m6, m1])

    view = st.radio("Select Ranking Engine", ["🏛️ Established Compounders", "🏎️ Momentum Efficiency"], horizontal=True)
    target = est if "Established" in view else mom

    if not target.empty:
        disp = target.sort_values("Rank").copy()
        
        # Select relevant cols
        if "Established" in view:
            cols = ["Rank", "Fund", "Score", "Up_Cap", "Down_Cap", "Ulcer_Index", "Roll_5Y", "AUM"]
            names = ["Rank", "Fund", "Score", "Upside %", "Downside %", "DD Stress", "5Y CAGR", "AUM (Cr)"]
        else:
            cols = ["Rank", "Fund", "Score", "Ret_6M", "Ret_1Y", "Vol", "AUM"]
            names = ["Rank", "Fund", "Score", "6M Ret%", "1Y Ret%", "Vol%", "AUM (Cr)"]
            
        disp = disp[cols]
        disp.columns = names
        
        st.dataframe(
            disp.style.background_gradient(subset=["Score"], cmap="RdYlGn")
            .format("{:.1f}", subset=disp.columns.drop(["Rank", "Fund", "AUM (Cr)"]))
            .format("{:.0f}", subset=["AUM (Cr)"]),
            use_container_width=True, height=600, hide_index=True
        )

        # Analysis Chart
        st.divider()
        st.subheader("Visual Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Capture Ratio Matrix** (Top-Left = Best)")
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(x=est["Down_Cap"], y=est["Up_Cap"], mode="markers+text", 
                                     text=est["Fund"].apply(lambda x: x[:8]),
                                     marker=dict(size=est["Score"]/3, color=est["Score"], colorscale="Viridis")))
            fig1.update_layout(xaxis_title="Downside Capture %", yaxis_title="Upside Capture %", height=400)
            st.plotly_chart(fig1, use_container_width=True)
            
        with col2:
            st.write("**Return vs Volatility** (Engine 2 Logic)")
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=mom["Vol"], y=mom["Ret_1Y"], mode="markers", 
                                     marker=dict(size=12, color=mom["Score"], colorscale="Electric")))
            fig2.update_layout(xaxis_title="Annualized Volatility %", yaxis_title="1Y Absolute Return %", height=400)
            st.plotly_chart(fig2, use_container_width=True)

if __name__ == "__main__":
    main()
