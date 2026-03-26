"""
SmallCap Fund Ranking App
Dual-Engine Layout: Long-Term Compounders + Short-Term Momentum
Uses real Nifty 500 TRI Combined historical data as Benchmark
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import timedelta
import warnings, io

warnings.filterwarnings("ignore")

st.set_page_config(page_title="SmallCap Rankings", layout="wide", page_icon="📊")


# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────
@st.cache_data(show_spinner="Loading fund & benchmark data...")
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

    # ── Real Nifty 500 TRI Benchmark Integration ──
    try:
        # Load the combined benchmark file directly
        bench = pd.read_csv("Nifty_500_TRI_Combined.csv")
        
        # Clean the data
        bench["Date"] = pd.to_datetime(bench["Date"])
        
        # Merge into our main NAV dataframe
        nav = pd.merge(nav, bench, on="Date", how="left")
        
        # Forward fill the benchmark to cover any mismatched market holidays
        nav["Benchmark"] = nav["Benchmark"].ffill()
        
    except Exception as e:
        # Failsafe: If the CSV is missing, revert to the Category Average proxy
        st.sidebar.warning("⚠️ Could not load Nifty_500_TRI_Combined.csv. Using Category Average as Benchmark.")
        valid = [f for f in fund_names if nav[f].notna().sum() > 252]
        nav["Benchmark"] = nav[valid].mean(axis=1)

    # AUM
    ar = pd.read_excel("smallcap_aum.xlsx")
    aum = ar.iloc[3:].copy()
    aum.columns = ["Fund", "Month_End", "AUM", "AAUM", "Avg_AUM"]
    aum["AUM"] = pd.to_numeric(aum["AUM"], errors="coerce")
    aum = aum[aum["AUM"].notna()]
    aum["Month_End"] = pd.to_datetime(aum["Month_End"], errors="coerce")
    aum_latest = (
        aum.sort_values("Month_End", ascending=False)
        .groupby("Fund").first().reset_index()[["Fund", "AUM"]]
    )

    # Expense Ratio
    er_raw = pd.read_excel("smallcap_expense_ratio.xlsx")
    er_recs = []
    cur = None
    for _, row in er_raw.iterrows():
        v = str(row.iloc[0])
        if v.startswith("Scheme Name:"):
            cur = v.replace("Scheme Name: ", "").strip()
        elif cur:
            try:
                er_recs.append({"Fund": cur, "ER": float(row.iloc[1])})
                cur = None
            except:
                pass
    df_er = pd.DataFrame(er_recs).drop_duplicates(subset="Fund", keep="first")

    return nav, fund_names, aum_latest, df_er


# ─────────────────────────────────────────────
# METRICS ENGINE
# ─────────────────────────────────────────────
@st.cache_data(show_spinner="Computing metrics against Nifty 500 TRI...")
def compute_metrics(_nav, fund_names, aum_latest, df_er):
    nav = _nav.copy()
    latest_date = nav["Date"].max()
    monthly = nav.set_index("Date").resample("ME").last()
    monthly_ret = monthly.pct_change().dropna(how="all")
    bench_ret = monthly_ret["Benchmark"]

    results = []
    for fund in fund_names:
        fd = nav[["Date", fund]].dropna()
        if len(fd) < 252:
            continue

        fund_monthly = monthly_ret[fund].dropna()
        common_idx = fund_monthly.index.intersection(bench_ret.dropna().index)
        if len(common_idx) < 12:
            continue
        f_ret = fund_monthly.loc[common_idx]
        b_ret = bench_ret.loc[common_idx]

        daily_fd = fd.set_index("Date")[fund]
        
        # ── Track Record (Age in Years) ──
        track_record_years = len(fd) / 252

        # ── Rolling CAGR ──
        def rolling_cagr(series, years, window_months):
            m_nav = series.resample("ME").last().dropna()
            if len(m_nav) < window_months:
                return None, None, None, None
            cagrs = []
            for i in range(window_months, len(m_nav)):
                s_nav = m_nav.iloc[i - window_months]
                e_nav = m_nav.iloc[i]
                if s_nav > 0:
                    cagrs.append(((e_nav / s_nav) ** (1 / years) - 1) * 100)
            if not cagrs:
                return None, None, None, None
            a = np.array(cagrs)
            return np.mean(a), np.median(a), np.min(a), np.max(a)

        r3_mean, r3_med, r3_min, r3_max = rolling_cagr(daily_fd, 3, 36)
        r5_mean, r5_med, r5_min, r5_max = rolling_cagr(daily_fd, 5, 60)

        # ── Benchmark win rate ──
        def win_rate(fund_s, bench_s, years, wm):
            fm = fund_s.resample("ME").last().dropna()
            bm = bench_s.resample("ME").last().dropna()
            ci = fm.index.intersection(bm.index)
            fm, bm = fm.loc[ci], bm.loc[ci]
            if len(fm) < wm:
                return None
            wins = total = 0
            for i in range(wm, len(fm)):
                fs, fe = fm.iloc[i - wm], fm.iloc[i]
                bs, be = bm.iloc[i - wm], bm.iloc[i]
                if fs > 0 and bs > 0:
                    if (fe / fs) ** (1 / years) > (be / bs) ** (1 / years):
                        wins += 1
                    total += 1
            return wins / total * 100 if total > 0 else None

        bench_daily = nav.set_index("Date")["Benchmark"]
        win_3y = win_rate(daily_fd, bench_daily, 3, 36)
        win_5y = win_rate(daily_fd, bench_daily, 5, 60)

        # ── Capture Ratios (Against Nifty 500) ──
        down_m = b_ret[b_ret < 0]
        up_m = b_ret[b_ret > 0]
        dc = (f_ret.loc[down_m.index].mean() / down_m.mean()) * 100 if len(down_m) > 3 else None
        uc = (f_ret.loc[up_m.index].mean() / up_m.mean()) * 100 if len(up_m) > 3 else None
        cap_ratio = uc / dc if (uc and dc and dc != 0) else None

        # ── Sortino ──
        rf_m = 0.06 / 12
        excess = f_ret - rf_m
        neg = excess[excess < 0]
        ds = np.sqrt(np.mean(neg ** 2)) if len(neg) > 3 else None
        sortino = (f_ret.mean() - rf_m) / ds if ds and ds > 0 else None

        # ── Drawdowns ──
        prices = fd[fund].values
        cummax = np.maximum.accumulate(prices)
        dd = (prices - cummax) / cummax
        max_dd_full = dd.min() * 100

        # ── Sharpe, Calmar, Info Ratio ──
        ann_ret = f_ret.mean() * 12
        vol_ann = f_ret.std() * np.sqrt(12)
        sharpe = (ann_ret - 0.06) / vol_ann if vol_ann > 0 else None
        calmar = abs(ann_ret * 100 / max_dd_full) if max_dd_full != 0 else None
        active = f_ret - b_ret.loc[f_ret.index]
        te = active.std() * np.sqrt(12)
        info_ratio = (active.mean() * 12) / te if te > 0 else None

        # ── Momentum: 1Y & 6M return ──
        latest_nav = fd.iloc[-1][fund]
        ld = fd.iloc[-1]["Date"]
        
        mask_12m = (nav["Date"] >= ld - timedelta(days=375)) & (nav["Date"] <= ld - timedelta(days=355))
        n12 = nav[mask_12m].dropna(subset=[fund])
        ret_1y = None
        if len(n12) > 0:
            nav_12m = n12.iloc[(n12["Date"] - (ld - timedelta(days=365))).abs().argsort().iloc[0]][fund]
            ret_1y = (latest_nav / nav_12m - 1) * 100
            
        mask_6m = (nav["Date"] >= ld - timedelta(days=195)) & (nav["Date"] <= ld - timedelta(days=165))
        n6 = nav[mask_6m].dropna(subset=[fund])
        ret_6m = None
        if len(n6) > 0:
            nav_6m = n6.iloc[(n6["Date"] - (ld - timedelta(days=180))).abs().argsort().iloc[0]][fund]
            ret_6m = (latest_nav / nav_6m - 1) * 100

        # ── Volatility (Short Term Risk) ──
        rec_1y = fd[fd["Date"] >= ld - timedelta(days=365)]
        vol_1y = rec_1y[fund].pct_change().dropna().std() * np.sqrt(252) * 100 if len(rec_1y) > 50 else None

        results.append({
            "Fund": fund, "Track_Record_Years": track_record_years,
            "Ret_6M": ret_6m, "Ret_1Y": ret_1y, "Vol_1Y": vol_1y,
            "Roll_3Y_Mean": r3_mean, "Roll_5Y_Mean": r5_mean, "Win_3Y": win_3y,
            "Down_Cap": dc, "Up_Cap": uc, "Cap_Ratio": cap_ratio,
            "Sortino": sortino, "Max_DD": max_dd_full,
            "Sharpe": sharpe, "Calmar": calmar, "Info_Ratio": info_ratio,
        })

    df = pd.DataFrame(results)
    df = df.merge(aum_latest, on="Fund", how="left")
    df = df.merge(df_er, on="Fund", how="left")
    return df


# ─────────────────────────────────────────────
# DUAL RANKING ENGINE
# ─────────────────────────────────────────────
def pctrank(s, asc=True):
    v = s.notna()
    r = s[v].rank(ascending=asc, pct=True) * 100
    out = pd.Series(np.nan, index=s.index)
    out[v] = r
    return out

def rank_dual_engine(df, w_lt, w_st):
    d = df.copy()

    # --- ENGINE 1: LONG-TERM SCORE ---
    d["S_R3"] = pctrank(d["Roll_3Y_Mean"])
    d["S_R5"] = pctrank(d["Roll_5Y_Mean"])
    d["S_W3"] = pctrank(d["Win_3Y"])
    d["S_DC"] = pctrank(d["Down_Cap"], asc=False)
    d["S_UC"] = pctrank(d["Up_Cap"])
    d["S_SO"] = pctrank(d["Sortino"])
    d["S_DD"] = pctrank(d["Max_DD"])
    d["S_CA"] = pctrank(d["Calmar"])
    d["S_IR"] = pctrank(d["Info_Ratio"])
    d["S_ER"] = pctrank(d["ER"], asc=False)

    def comp_lt(row):
        if row.get("Track_Record_Years", 0) < 3.0: return np.nan
        tw = ts = 0
        for c, wt in w_lt.items():
            if wt > 0 and pd.notna(row.get(c)):
                ts += row[c] * wt
                tw += wt
        return ts / tw if tw > 0 else np.nan

    d["Score_LT"] = d.apply(comp_lt, axis=1)
    d["Rank_LT"] = d["Score_LT"].rank(ascending=False, method="min").astype("Int64")

    # --- ENGINE 2: SHORT-TERM / MOMENTUM SCORE ---
    d["S_6M"] = pctrank(d["Ret_6M"])
    d["S_1Y"] = pctrank(d["Ret_1Y"])
    d["S_V1"] = pctrank(d["Vol_1Y"], asc=False) 

    def comp_st(row):
        tw = ts = 0
        for c, wt in w_st.items():
            if wt > 0 and pd.notna(row.get(c)):
                ts += row[c] * wt
                tw += wt
        return ts / tw if tw > 0 else np.nan

    d["Score_ST"] = d.apply(comp_st, axis=1)
    d["Rank_ST"] = d["Score_ST"].rank(ascending=False, method="min").astype("Int64")

    def get_sig(x):
        if pd.isna(x): return "N/A"
        if x >= 78: return "Elite"
        if x >= 62: return "Strong"
        if x >= 48: return "Above Avg"
        if x >= 35: return "Average"
        return "Below Avg"

    d["Signal_LT"] = d["Score_LT"].apply(get_sig)
    d["Signal_ST"] = d["Score_ST"].apply(get_sig)

    return d


def short(f):
    return (f.replace("Small Cap", "SC").replace("Smallcap", "SC")
             .replace("Fund-Reg(G)", "").replace("Fund(G)", "").strip())


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    nav, fund_names, aum_latest, df_er = load_data()
    df_raw = compute_metrics(nav, fund_names, aum_latest, df_er)

    st.title("📊 SmallCap Fund Rankings: Dual-Engine")
    st.caption(f"{len(df_raw)} funds · Benchmark: Real Nifty 500 TRI · Data through {nav['Date'].max().strftime('%d %b %Y')}")

    # ── Sidebar weights ──
    with st.sidebar:
        st.header("⚙️ Ranking Engine Weights")
        
        with st.expander("🏛️ Long-Term Weights (Funds > 3 Yrs)", expanded=True):
            st.caption("Used to rank established funds.")
            lt_w0 = st.slider("Rolling 3Y CAGR", 0, 30, 15)
            lt_w1 = st.slider("Rolling 5Y CAGR", 0, 30, 15)
            lt_w2 = st.slider("3Y Win Rate vs Nifty 500", 0, 20, 5)
            lt_w3 = st.slider("Downside Capture", 0, 30, 20)
            lt_w4 = st.slider("Upside Capture", 0, 20, 10)
            lt_w5 = st.slider("Sortino Ratio", 0, 25, 15)
            lt_w6 = st.slider("Max Drawdown", 0, 20, 10)
            lt_w7 = st.slider("Calmar Ratio", 0, 15, 5)
            lt_w8 = st.slider("Info Ratio", 0, 15, 3)
            lt_w9 = st.slider("Expense Ratio", 0, 15, 2)
            
            w_lt = {
                "S_R3": lt_w0, "S_R5": lt_w1, "S_W3": lt_w2, "S_DC": lt_w3, "S_UC": lt_w4,
                "S_SO": lt_w5, "S_DD": lt_w6, "S_CA": lt_w7, "S_IR": lt_w8, "S_ER": lt_w9,
            }

        with st.expander("🚀 Short-Term Weights (All Funds)", expanded=True):
            st.caption("Used to rank momentum and newer funds.")
            st_w0 = st.slider("6-Month Momentum", 0, 100, 45)
            st_w1 = st.slider("1-Year Momentum", 0, 100, 40)
            st_w2 = st.slider("1-Year Volatility (Risk)", 0, 50, 15)
            
            w_st = {
                "S_6M": st_w0, "S_1Y": st_w1, "S_V1": st_w2
            }

    df = rank_dual_engine(df_raw, w_lt, w_st)

    # ── Two main tabs ──
    tab_rank, tab_fund = st.tabs(["🏆 Leaderboards", "🔎 Fund Deep-Dive"])

    # ═══════════════════════════════════════════
    # TAB 1: RANKINGS
    # ═══════════════════════════════════════════
    with tab_rank:
        
        board_type = st.radio("Select Leaderboard View:", 
                              ["🏛️ Long-Term Compounders (> 3 Years Old)", "🚀 Short-Term Momentum (All Funds including Emerging)"], 
                              horizontal=True)
        st.divider()

        def color_signal(val):
            colors = {"Elite": "background-color: #dcfce7; color: #166534;", "Strong": "background-color: #e0f2fe; color: #075985;", "Above Avg": "background-color: #fef9c3; color: #854d0e;", "Average": "background-color: #fed7aa; color: #9a3412;", "Below Avg": "background-color: #fecaca; color: #991b1b;"}
            return colors.get(val, "")
        def color_score(val):
            if pd.isna(val): return ""
            if val >= 70: return "background-color: #dcfce7; font-weight: 700;"
            if val >= 50: return "background-color: #e0f2fe;"
            if val >= 35: return "background-color: #fef9c3;"
            return "background-color: #fecaca;"
        def color_momentum(val):
            if pd.isna(val): return ""
            if val > 30: return "color: #16a34a; font-weight: 600;"
            if val > 10: return "color: #ca8a04;"
            return "color: #dc2626;"


        if "Long-Term" in board_type:
            df_view = df[df["Rank_LT"].notna()].sort_values("Rank_LT").copy()
            
            c1, c2, c3 = st.columns(3)
            c1.metric("🥇 Top Compounder", short(df_view.iloc[0]["Fund"]), f"Score {df_view.iloc[0]['Score_LT']:.1f}")
            c2.metric("Median Downside Capture", f"{df_view['Down_Cap'].median():.0f}%")
            c3.metric("Funds Ranked", len(df_view))
            st.caption("Newer funds (< 3 years) are safely hidden from this view to ensure data integrity.")
            
            disp = df_view[[
                "Rank_LT", "Fund", "Score_LT", "Signal_LT", "Track_Record_Years",
                "Roll_3Y_Mean", "Roll_5Y_Mean", "Win_3Y", "Down_Cap", "Up_Cap", "Sortino", "Max_DD", "AUM"
            ]].copy()
            disp.columns = ["Rank", "Fund", "Score", "Signal", "Age (Yrs)", "Roll 3Y%", "Roll 5Y%", "Win Rate 3Y%", "Down Cap%", "Up Cap%", "Sortino", "Max DD%", "AUM (Cr)"]
            disp["Fund"] = disp["Fund"].apply(short)
            disp = disp.round({"Score": 1, "Age (Yrs)": 1, "Roll 3Y%": 1, "Roll 5Y%": 1, "Win Rate 3Y%": 1, "Down Cap%": 0, "Up Cap%": 0, "Sortino": 2, "Max DD%": 1, "AUM (Cr)": 0})
            
            styled = disp.style.map(color_signal, subset=["Signal"]).map(color_score, subset=["Score"]).format(na_rep="—") \
                .set_properties(**{"text-align": "center", "font-size": "13px"}).set_properties(subset=["Fund"], **{"text-align": "left", "font-weight": "500"})
            
            st.dataframe(styled, use_container_width=True, height=600, hide_index=True)

        else:
            df_view = df.sort_values("Rank_ST").copy()
            
            c1, c2, c3 = st.columns(3)
            c1.metric("🔥 Top Momentum Fund", short(df_view.iloc[0]["Fund"]), f"Score {df_view.iloc[0]['Score_ST']:.1f}")
            c2.metric("Median 1Y Return", f"{df_view['Ret_1Y'].median():.1f}%")
            c3.metric("Funds Ranked", len(df_view))
            st.caption("Ranked purely on 6-Month Return, 1-Year Return, and 1-Year Volatility. Includes newer emerging funds.")
            
            disp = df_view[[
                "Rank_ST", "Fund", "Score_ST", "Signal_ST", "Track_Record_Years",
                "Ret_6M", "Ret_1Y", "Vol_1Y", "AUM"
            ]].copy()
            disp.columns = ["Rank", "Fund", "Score", "Signal", "Age (Yrs)", "6M Ret%", "1Y Ret%", "1Y Volatility%", "AUM (Cr)"]
            disp["Fund"] = disp["Fund"].apply(short)
            disp = disp.round({"Score": 1, "Age (Yrs)": 1, "6M Ret%": 1, "1Y Ret%": 1, "1Y Volatility%": 1, "AUM (Cr)": 0})
            
            styled = disp.style.map(color_signal, subset=["Signal"]).map(color_score, subset=["Score"]).map(color_momentum, subset=["6M Ret%", "1Y Ret%"]).format(na_rep="—") \
                .set_properties(**{"text-align": "center", "font-size": "13px"}).set_properties(subset=["Fund"], **{"text-align": "left", "font-weight": "500"})
            
            st.dataframe(styled, use_container_width=True, height=600, hide_index=True)


    # ═══════════════════════════════════════════
    # TAB 2: FUND DEEP-DIVE
    # ═══════════════════════════════════════════
    with tab_fund:
        fund_list = df.sort_values("Fund")["Fund"].tolist()
        selected = st.selectbox("Search for any fund...", fund_list, format_func=short)

        r = df[df["Fund"] == selected].iloc[0]

        st.subheader(f"🔎 {short(selected)}")
        
        colA, colB, colC = st.columns(3)
        with colA:
            st.markdown(f"**Age:** {r['Track_Record_Years']:.1f} Years")
        with colB:
            lt_score = f"{r['Score_LT']:.1f}/100" if pd.notna(r['Score_LT']) else "Not Old Enough"
            st.markdown(f"**🏛️ Long-Term Score:** {lt_score}  (Rank: #{r['Rank_LT'] if pd.notna(r['Rank_LT']) else '-'})")
        with colC:
            st.markdown(f"**🚀 Momentum Score:** {r['Score_ST']:.1f}/100  (Rank: #{r['Rank_ST']})")

        st.divider()

        st.markdown("#### Performance Profile")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("6M Return (Momentum)", f"{r['Ret_6M']:.1f}%" if pd.notna(r["Ret_6M"]) else "—")
        c2.metric("1Y Return (Momentum)", f"{r['Ret_1Y']:.1f}%" if pd.notna(r["Ret_1Y"]) else "—")
        c3.metric("Rolling 3Y CAGR", f"{r['Roll_3Y_Mean']:.1f}%" if pd.notna(r["Roll_3Y_Mean"]) else "—")
        c4.metric("Rolling 5Y CAGR", f"{r['Roll_5Y_Mean']:.1f}%" if pd.notna(r["Roll_5Y_Mean"]) else "—")

        st.markdown("#### Risk Profile against Nifty 500 TRI")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Downside Capture", f"{r['Down_Cap']:.0f}%" if pd.notna(r["Down_Cap"]) else "—")
        c2.metric("Max Drawdown", f"{r['Max_DD']:.1f}%" if pd.notna(r["Max_DD"]) else "—")
        c3.metric("3Y Win Rate", f"{r['Win_3Y']:.1f}%" if pd.notna(r["Win_3Y"]) else "—")
        c4.metric("Sortino Ratio", f"{r['Sortino']:.2f}" if pd.notna(r["Sortino"]) else "—")

        st.divider()

        st.markdown("#### NAV vs Benchmark History & Drawdown")

        fund_nav = nav[["Date", selected, "Benchmark"]].dropna(subset=[selected]).copy()
        fund_nav = fund_nav.rename(columns={selected: "NAV"})
        
        # Scale benchmark to start at the exact same base value (100) as the Fund NAV for visual comparison
        if not fund_nav.empty and fund_nav.iloc[0]["Benchmark"] > 0:
            initial_nav = fund_nav.iloc[0]["NAV"]
            initial_bench = fund_nav.iloc[0]["Benchmark"]
            fund_nav["Scaled_Benchmark"] = fund_nav["Benchmark"] * (initial_nav / initial_bench)
        else:
            fund_nav["Scaled_Benchmark"] = fund_nav["Benchmark"]

        fund_nav["Peak"] = fund_nav["NAV"].cummax()
        fund_nav["DD"] = (fund_nav["NAV"] - fund_nav["Peak"]) / fund_nav["Peak"] * 100

        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            vertical_spacing=0.06, row_heights=[0.65, 0.35],
            subplot_titles=("NAV vs Nifty 500 (Rebased)", "Fund Drawdown (%)"),
        )

        fig.add_trace(go.Scatter(
            x=fund_nav["Date"], y=fund_nav["NAV"],
            fill="tozeroy", fillcolor="rgba(59,130,246,0.07)",
            line=dict(color="#3b82f6", width=1.5), name="NAV",
            hovertemplate="%{x|%d %b %Y}<br>₹%{y:.2f}<extra></extra>",
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=fund_nav["Date"], y=fund_nav["Scaled_Benchmark"],
            line=dict(color="#64748b", width=1.5, dash="dot"), name="Nifty 500 Proxy",
            hovertemplate="Benchmark<extra></extra>",
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=fund_nav["Date"], y=fund_nav["DD"],
            fill="tozeroy", fillcolor="rgba(239,68,68,0.1)",
            line=dict(color="#ef4444", width=1), name="Drawdown",
            hovertemplate="%{x|%d %b %Y}<br>%{y:.1f}%<extra></extra>",
        ), row=2, col=1)

        fig.update_layout(
            height=450, showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=50, r=20, t=30, b=30),
            hovermode="x unified",
        )
        fig.update_xaxes(gridcolor="rgba(0,0,0,0.05)")
        fig.update_yaxes(gridcolor="rgba(0,0,0,0.05)")
        st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
