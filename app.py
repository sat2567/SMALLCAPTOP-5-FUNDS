import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import timedelta
import warnings, io

warnings.filterwarnings("ignore")
st.set_page_config(page_title="SmallCap Quants", layout="wide", page_icon="📊")

# ═══════════════════════════════════════════
# DATA LOAD (Updated for SmallCap Benchmark)
# ═══════════════════════════════════════════
@st.cache_data(show_spinner="Loading data...")
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

    # 2. Load BSE Small-Cap TRI Benchmark (from your CSV)
    try:
        # Note: Adjusting for the 2-row header offset in your specific CSV
        bench = pd.read_csv("BSE SMAALCAP.xlsx - Abakkus Small Cap Fund-Reg(G)1.csv", skiprows=2)
        bench["Date"] = pd.to_datetime(bench["Date"])
        bench = bench.rename(columns={"Close Price": "Benchmark"})
        nav = pd.merge(nav, bench[["Date", "Benchmark"]], on="Date", how="left")
        nav["Benchmark"] = nav["Benchmark"].ffill()
        bench_name = "BSE Small-Cap TRI"
    except Exception:
        # Fallback to Category Average if file is missing
        valid = [f for f in fund_names if nav[f].notna().sum() > 252]
        nav["Benchmark"] = nav[valid].pct_change().mean(axis=1).add(1).cumprod() * 100
        bench_name = "SmallCap Category Avg"

    # 3. Load AUM & Expense Ratio
    ar = pd.read_excel("smallcap_aum.xlsx")
    aum = ar.iloc[3:].copy()
    aum.columns = ["Fund", "Month_End", "AUM", "AAUM", "Avg_AUM"]
    aum_latest = aum.sort_values("Month_End", ascending=False).groupby("Fund").first().reset_index()[["Fund", "AUM"]]

    er_raw = pd.read_excel("smallcap_expense_ratio.xlsx")
    er_recs, cur = [], None
    for _, row in er_raw.iterrows():
        v = str(row.iloc[0])
        if v.startswith("Scheme Name:"):
            cur = v.replace("Scheme Name: ", "").strip()
        elif cur:
            try:
                er_recs.append({"Fund": cur, "ER": float(row.iloc[1])})
                cur = None
            except: pass
    df_er = pd.DataFrame(er_recs).drop_duplicates(subset="Fund", keep="first")
    
    return nav, fund_names, aum_latest, df_er, bench_name

# ═══════════════════════════════════════════
# METRICS ENGINE (Added Upside & DD Factors)
# ═══════════════════════════════════════════
@st.cache_data(show_spinner="Computing factors for all funds...")
def compute_all(_nav, fund_names, aum_latest, df_er):
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

        # --- ENGINE 1 FACTORS ---
        # 1. Rolling 5Y CAGR
        mn = fd.set_index("Date")[fund].resample("ME").last().dropna()
        cagrs_5y = [(((mn.iloc[i]/mn.iloc[i-60])**(1/5))-1)*100 for i in range(60, len(mn))]
        r5_mean = np.mean(cagrs_5y) if cagrs_5y else None
        
        # 2. Capture Ratios (Upside & Downside)
        um, dm = brc[brc > 0], brc[brc < 0]
        up_cap = (fr.loc[um.index].mean() / um.mean()) * 100 if len(um) > 2 else None
        down_cap = (fr.loc[dm.index].mean() / dm.mean()) * 100 if len(dm) > 2 else None

        # 3. Drawdown Intensity (Ulcer Index approach)
        prices = fd[fund].values
        peaks = np.maximum.accumulate(prices)
        drawdowns = (prices - peaks) / peaks
        # Penalizes deep and long-lasting drawdowns
        dd_intensity = np.sqrt(np.mean(drawdowns**2)) * 100 
        max_dd = drawdowns.min() * 100

        # 4. Alpha Hit Rate (12M Rolling)
        alphas_12m = [fr.iloc[i-12:i].sum() - brc.loc[fr.iloc[i-12:i].index].sum() for i in range(12, len(fr))]
        alpha_hit = np.mean([a > 0 for a in alphas_12m]) * 100 if alphas_12m else None

        # --- ENGINE 2 FACTORS (Momentum) ---
        ld = fd.iloc[-1]["Date"]
        rec = fd[fd["Date"] >= ld - timedelta(days=365)]
        vol = rec[fund].pct_change().dropna().std() * np.sqrt(252) * 100 if len(rec) > 50 else None
        
        # Helper to get past NAV
        def get_ret(days):
            target_date = ld - timedelta(days=days)
            idx = (fd["Date"] - target_date).abs().argsort()[:1]
            past_nav = fd.iloc[idx][fund].values[0]
            return (prices[-1] / past_nav - 1) * 100

        ret_6m = get_ret(180)
        ret_1y = get_ret(365)

        results.append({
            "Fund": fund, "Track_Yrs": len(fd)/252, "Months": len(fr),
            "Roll_5Y": r5_mean, "Up_Cap": up_cap, "Down_Cap": down_cap, 
            "Alpha_Hit": alpha_hit, "DD_Intensity": dd_intensity, "Max_DD": max_dd,
            "Ret_6M": ret_6m, "Ret_1Y": ret_1y, "Vol": vol,
            "Mom_6M_RA": ret_6m/vol if vol and vol > 0 else None,
            "Mom_1Y_RA": ret_1y/vol if vol and vol > 0 else None
        })

    df = pd.DataFrame(results)
    return df.merge(aum_latest, on="Fund", how="left").merge(df_er, on="Fund", how="left")

# ═══════════════════════════════════════════
# RANKING SYSTEM
# ═══════════════════════════════════════════
def pctrank(s, asc=True):
    v = s.notna()
    r = s[v].rank(ascending=asc, pct=True) * 100
    out = pd.Series(np.nan, index=s.index)
    out[v] = r
    return out

def rank_all(df, w_lt, w_mom):
    d = df.copy()
    
    # Engine 1: Established (>3 Years)
    est = d[d["Track_Yrs"] >= 3].copy()
    if not est.empty:
        est["S_R5"] = pctrank(est["Roll_5Y"])
        est["S_UC"] = pctrank(est["Up_Cap"])      # Higher Upside = Better
        est["S_DC"] = pctrank(est["Down_Cap"], False) # Lower Downside = Better
        est["S_AH"] = pctrank(est["Alpha_Hit"])
        est["S_DI"] = pctrank(est["DD_Intensity"], False) # Lower Intensity = Better

        cols = ["S_R5", "S_UC", "S_DC", "S_AH", "S_DI"]
        total_w = sum(w_lt) if sum(w_lt) > 0 else 1
        est["Score"] = est[cols].mul(w_lt).sum(axis=1) / total_w
        est["Rank"] = est["Score"].rank(ascending=False, method="min").astype(int)

    # Engine 2: Momentum (All funds > 6 Months)
    mom = d[d["Months"] >= 6].copy()
    if not mom.empty:
        mom["S_M6"] = pctrank(mom["Mom_6M_RA"])
        mom["S_M1"] = pctrank(mom["Mom_1Y_RA"])
        total_m = sum(w_mom) if sum(w_mom) > 0 else 1
        mom["Score"] = (mom["S_M6"] * w_mom[0] + mom["S_M1"] * w_mom[1]) / total_m
        mom["Rank"] = mom["Score"].rank(ascending=False, method="min").astype(int)

    return est, mom

def short(f):
    return f.replace("Small Cap", "SC").replace("Fund-Reg(G)", "").strip()

# ═══════════════════════════════════════════
# MAIN INTERFACE
# ═══════════════════════════════════════════
def main():
    nav, fund_names, aum_latest, df_er, b_name = load_data()
    df_raw = compute_all(nav, fund_names, aum_latest, df_er)

    st.title("📊 SmallCap Quants: Dual-Engine Model")
    st.caption(f"Benchmark: **{b_name}** | Latest Data: {nav['Date'].max().strftime('%d %b %Y')}")

    with st.sidebar:
        st.header("🏛️ Engine 1: Compounder Weights")
        w_r5 = st.slider("5Y Rolling CAGR", 0, 50, 25)
        w_uc = st.slider("Upside Capture", 0, 50, 15)
        w_dc = st.slider("Downside Capture", 0, 50, 20)
        w_ah = st.slider("Alpha Consistency", 0, 50, 20)
        w_di = st.slider("Drawdown Intensity", 0, 50, 20)
        
        st.divider()
        st.header("🚀 Engine 2: Momentum Weights")
        m_6m = st.slider("6M RA Momentum", 0, 100, 50)
        m_1y = st.slider("1Y RA Momentum", 0, 100, 50)

    est, mom = rank_all(df_raw, [w_r5, w_uc, w_dc, w_ah, w_di], [m_6m, m_1y])

    tab1, tab2 = st.tabs(["🏆 Leaderboards", "🔍 Capture Analysis"])

    with tab1:
        engine_choice = st.radio("Choose Ranking Engine:", ["Established Compounders", "Momentum Strategy"], horizontal=True)
        target = est if "Established" in engine_choice else mom
        
        if target.empty:
            st.warning("Insufficient data for the selected parameters.")
        else:
            # Formatting Display
            disp = target.sort_values("Rank").copy()
            disp["Fund"] = disp["Fund"].apply(short)
            
            # Select columns based on engine
            if "Established" in engine_choice:
                cols = ["Rank", "Fund", "Score", "Up_Cap", "Down_Cap", "Roll_5Y", "DD_Intensity", "Max_DD", "AUM"]
                rename = ["Rank", "Fund", "Score", "Upside %", "Downside %", "5Y CAGR", "DD Stress", "Max DD%", "AUM (Cr)"]
            else:
                cols = ["Rank", "Fund", "Score", "Mom_6M_RA", "Mom_1Y_RA", "Ret_1Y", "Vol", "AUM"]
                rename = ["Rank", "Fund", "Score", "6M/Vol", "1Y/Vol", "1Y Ret%", "Vol%", "AUM (Cr)"]
            
            disp = disp[cols]
            disp.columns = rename
            
            st.dataframe(
                disp.style.background_gradient(subset=["Score"], cmap="RdYlGn")
                .format("{:.1f}", subset=disp.columns.drop(["Rank", "Fund", "AUM (Cr)"]))
                .format("{:.0f}", subset=["AUM (Cr)"]),
                use_container_width=True, height=600, hide_index=True
            )

    with tab2:
        st.subheader("Capture Efficiency Map")
        st.write("Targeting the top-left quadrant (High Upside, Low Downside)")
        
        fig = go.Figure()
        fig.add_hline(y=100, line_dash="dash", line_color="gray")
        fig.add_vline(x=100, line_dash="dash", line_color="gray")
        
        fig.add_trace(go.Scatter(
            x=est["Down_Cap"], y=est["Up_Cap"], mode="markers+text",
            text=est["Fund"].apply(short),
            textposition="top center",
            marker=dict(size=est["Score"]/2, color=est["Score"], colorscale="Portland", showscale=True),
            hovertemplate="<b>%{text}</b><br>Upside Capture: %{y:.1f}%<br>Downside Capture: %{x:.1f}%"
        ))
        
        fig.update_layout(
            xaxis_title="Downside Capture (Lower is Better)",
            yaxis_title="Upside Capture (Higher is Better)",
            height=600
        )
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
