"""
SmallCap Fund Ranking App
Simple two-page layout: Rankings + Fund Deep-Dive
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
@st.cache_data(show_spinner="Loading fund data...")
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
@st.cache_data(show_spinner="Computing metrics across all rolling windows...")
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

        # ── Capture Ratios ──
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

        f_3y = f_ret.tail(36)
        neg_3y = (f_3y - rf_m)
        neg_3y = neg_3y[neg_3y < 0]
        ds_3y = np.sqrt(np.mean(neg_3y ** 2)) if len(neg_3y) > 3 else None
        sortino_3y = (f_3y.mean() - rf_m) / ds_3y if ds_3y and ds_3y > 0 else None

        # ── Drawdowns ──
        prices = fd[fund].values
        cummax = np.maximum.accumulate(prices)
        dd = (prices - cummax) / cummax
        max_dd_full = dd.min() * 100
        current_dd = (prices[-1] - prices.max()) / prices.max() * 100

        r3y = fd[fd["Date"] >= latest_date - timedelta(days=365 * 3)]
        max_dd_3y = None
        if len(r3y) > 50:
            p3 = r3y[fund].values
            max_dd_3y = ((p3 - np.maximum.accumulate(p3)) / np.maximum.accumulate(p3)).min() * 100

        # ── Sharpe, Calmar, Info Ratio ──
        ann_ret = f_ret.mean() * 12
        vol_ann = f_ret.std() * np.sqrt(12)
        sharpe = (ann_ret - 0.06) / vol_ann if vol_ann > 0 else None
        calmar = abs(ann_ret * 100 / max_dd_full) if max_dd_full != 0 else None

        active = f_ret - b_ret.loc[f_ret.index]
        te = active.std() * np.sqrt(12)
        info_ratio = (active.mean() * 12) / te if te > 0 else None

        # ── 1Y return ──
        latest_nav = fd.iloc[-1][fund]
        ld = fd.iloc[-1]["Date"]
        mask_12m = (nav["Date"] >= ld - timedelta(days=375)) & (nav["Date"] <= ld - timedelta(days=355))
        n12 = nav[mask_12m].dropna(subset=[fund])
        ret_1y = None
        if len(n12) > 0:
            nav_12m = n12.iloc[(n12["Date"] - (ld - timedelta(days=365))).abs().argsort().iloc[0]][fund]
            ret_1y = (latest_nav / nav_12m - 1) * 100

        # ── Volatility ──
        rec_1y = fd[fd["Date"] >= ld - timedelta(days=365)]
        vol_1y = rec_1y[fund].pct_change().dropna().std() * np.sqrt(252) * 100 if len(rec_1y) > 50 else None

        results.append({
            "Fund": fund, "Ret_1Y": ret_1y,
            "Roll_3Y_Mean": r3_mean, "Roll_3Y_Median": r3_med,
            "Roll_3Y_Min": r3_min, "Roll_3Y_Max": r3_max, "Win_3Y": win_3y,
            "Roll_5Y_Mean": r5_mean, "Roll_5Y_Median": r5_med,
            "Roll_5Y_Min": r5_min, "Roll_5Y_Max": r5_max, "Win_5Y": win_5y,
            "Down_Cap": dc, "Up_Cap": uc, "Cap_Ratio": cap_ratio,
            "Sortino": sortino, "Sortino_3Y": sortino_3y,
            "Max_DD": max_dd_full, "Max_DD_3Y": max_dd_3y, "Current_DD": current_dd,
            "Sharpe": sharpe, "Calmar": calmar, "Info_Ratio": info_ratio,
            "Vol_1Y": vol_1y, "Ann_Ret": ann_ret * 100,
        })

    df = pd.DataFrame(results)
    df = df.merge(aum_latest, on="Fund", how="left")
    df = df.merge(df_er, on="Fund", how="left")
    df["Sortino_vs_Cat"] = df["Sortino"] - df["Sortino"].median()
    return df


# ─────────────────────────────────────────────
# RANKING
# ─────────────────────────────────────────────
def pctrank(s, asc=True):
    v = s.notna()
    r = s[v].rank(ascending=asc, pct=True) * 100
    out = pd.Series(np.nan, index=s.index)
    out[v] = r
    return out


def rank_funds(df, w):
    d = df.copy()
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

    wm = {
        "S_R3": w[0], "S_R5": w[1], "S_W3": w[2], "S_DC": w[3], "S_UC": w[4],
        "S_SO": w[5], "S_DD": w[6], "S_CA": w[7], "S_IR": w[8], "S_ER": w[9],
    }

    def comp(row):
        tw = ts = 0
        for c, wt in wm.items():
            if wt > 0 and pd.notna(row.get(c)):
                ts += row[c] * wt
                tw += wt
        return ts / tw if tw > 0 else np.nan

    d["Score"] = d.apply(comp, axis=1)
    d["Rank"] = d["Score"].rank(ascending=False, method="min").astype(int)
    d = d.sort_values("Rank").reset_index(drop=True)
    d["Signal"] = d["Score"].apply(
        lambda x: "Elite" if x >= 78 else (
            "Strong" if x >= 62 else (
                "Above Avg" if x >= 48 else (
                    "Average" if x >= 35 else "Below Avg"))))
    return d


def short(f):
    return (f.replace("Small Cap", "SC").replace("Smallcap", "SC")
             .replace("Fund-Reg(G)", "").replace("Fund(G)", "").strip())


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
nav, fund_names, aum_latest, df_er = load_data()
df_raw = compute_metrics(nav, fund_names, aum_latest, df_er)

st.title("📊 SmallCap Fund Rankings")
st.caption(f"{len(df_raw)} funds · Data through {nav['Date'].max().strftime('%d %b %Y')} · Benchmark: Equal-weight proxy")

# ── Sidebar weights ──
with st.sidebar:
    st.header("Factor Weights")
    st.caption("Drag sliders to change how much each factor matters. Weights auto-normalise.")
    w0 = st.slider("Rolling 3Y CAGR", 0, 30, 15)
    w1 = st.slider("Rolling 5Y CAGR", 0, 30, 15)
    w2 = st.slider("3Y Win Rate vs Benchmark", 0, 20, 5)
    w3 = st.slider("Downside Capture", 0, 30, 20)
    w4 = st.slider("Upside Capture", 0, 20, 10)
    w5 = st.slider("Sortino Ratio", 0, 25, 15)
    w6 = st.slider("Max Drawdown", 0, 20, 10)
    w7 = st.slider("Calmar Ratio", 0, 15, 5)
    w8 = st.slider("Information Ratio", 0, 15, 3)
    w9 = st.slider("Expense Ratio", 0, 15, 2)

    total = w0 + w1 + w2 + w3 + w4 + w5 + w6 + w7 + w8 + w9
    if total == 0:
        total = 1
    weights = [x / total for x in [w0, w1, w2, w3, w4, w5, w6, w7, w8, w9]]

df = rank_funds(df_raw, weights)

# ── Two tabs ──
tab_rank, tab_fund = st.tabs(["🏆 Rankings", "🔎 Fund Deep-Dive"])

# ═══════════════════════════════════════════
# TAB 1: RANKINGS
# ═══════════════════════════════════════════
with tab_rank:
    # Top-line numbers
    c1, c2, c3, c4 = st.columns(4)
    top = df.iloc[0]
    c1.metric("🥇 Top Fund", short(top["Fund"]), f"Score {top['Score']:.1f}")
    c2.metric("Median Downside Capture", f"{df['Down_Cap'].median():.0f}%")
    c3.metric("Median Sortino", f"{df['Sortino'].median():.2f}")
    c4.metric("Median Max Drawdown", f"{df['Max_DD'].median():.1f}%")

    st.divider()

    # Build display table
    display = df[[
        "Rank", "Fund", "Score", "Signal",
        "Roll_3Y_Mean", "Roll_5Y_Mean", "Win_3Y",
        "Down_Cap", "Up_Cap", "Cap_Ratio",
        "Sortino", "Max_DD", "Calmar",
        "AUM", "ER",
    ]].copy()

    display.columns = [
        "Rank", "Fund", "Score", "Signal",
        "Roll 3Y%", "Roll 5Y%", "Win Rate 3Y%",
        "Down Cap%", "Up Cap%", "Cap Ratio",
        "Sortino", "Max DD%", "Calmar",
        "AUM (Cr)", "Exp Ratio%",
    ]

    display["Fund"] = display["Fund"].apply(short)

    # Round
    for c in display.columns:
        if c in ("Rank", "Fund", "Signal"):
            continue
        display[c] = pd.to_numeric(display[c], errors="coerce")

    display = display.round({
        "Score": 1, "Roll 3Y%": 1, "Roll 5Y%": 1, "Win Rate 3Y%": 1,
        "Down Cap%": 0, "Up Cap%": 0, "Cap Ratio": 2,
        "Sortino": 2, "Max DD%": 1, "Calmar": 2,
        "AUM (Cr)": 0, "Exp Ratio%": 2,
    })

    # Color the signal column
    def color_signal(val):
        colors = {
            "Elite": "background-color: #dcfce7; color: #166534;",
            "Strong": "background-color: #e0f2fe; color: #075985;",
            "Above Avg": "background-color: #fef9c3; color: #854d0e;",
            "Average": "background-color: #fed7aa; color: #9a3412;",
            "Below Avg": "background-color: #fecaca; color: #991b1b;",
        }
        return colors.get(val, "")

    def color_down_cap(val):
        if pd.isna(val): return ""
        if val < 80: return "color: #16a34a; font-weight: 600;"
        if val < 100: return "color: #ca8a04;"
        return "color: #dc2626; font-weight: 600;"

    def color_up_cap(val):
        if pd.isna(val): return ""
        if val > 100: return "color: #16a34a; font-weight: 600;"
        return "color: #ca8a04;"

    def color_sortino(val):
        if pd.isna(val): return ""
        if val > 0.15: return "color: #16a34a;"
        if val > 0: return "color: #ca8a04;"
        return "color: #dc2626;"

    def color_dd(val):
        if pd.isna(val): return ""
        if val > -40: return "color: #16a34a;"
        if val > -55: return "color: #ca8a04;"
        return "color: #dc2626; font-weight: 600;"

    def color_score(val):
        if pd.isna(val): return ""
        if val >= 70: return "background-color: #dcfce7; font-weight: 700;"
        if val >= 50: return "background-color: #e0f2fe;"
        if val >= 35: return "background-color: #fef9c3;"
        return "background-color: #fecaca;"

    styled = display.style.applymap(color_signal, subset=["Signal"]) \
        .applymap(color_down_cap, subset=["Down Cap%"]) \
        .applymap(color_up_cap, subset=["Up Cap%"]) \
        .applymap(color_sortino, subset=["Sortino"]) \
        .applymap(color_dd, subset=["Max DD%"]) \
        .applymap(color_score, subset=["Score"]) \
        .format(na_rep="—") \
        .set_properties(**{"text-align": "center", "font-size": "13px"}) \
        .set_properties(subset=["Fund"], **{"text-align": "left", "font-weight": "500"})

    st.dataframe(styled, use_container_width=True, height=700, hide_index=True)

    # Download
    buf = io.BytesIO()
    display.to_excel(buf, index=False, engine="openpyxl")
    st.download_button("📥 Download Rankings", buf.getvalue(),
                       "smallcap_rankings.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


# ═══════════════════════════════════════════
# TAB 2: FUND DEEP-DIVE
# ═══════════════════════════════════════════
with tab_fund:
    fund_list = df.sort_values("Rank")["Fund"].tolist()
    selected = st.selectbox("Pick a fund", fund_list, format_func=short)

    r = df[df["Fund"] == selected].iloc[0]

    # ── Header ──
    st.subheader(f"#{int(r['Rank'])}  {short(selected)}")

    sig_emoji = {"Elite": "🟢", "Strong": "🔵", "Above Avg": "🟡", "Average": "🟠", "Below Avg": "🔴"}
    st.markdown(f"**Signal:** {sig_emoji.get(r['Signal'], '⚪')} {r['Signal']}  ·  **Score:** {r['Score']:.1f}/100")

    st.divider()

    # ── Key Metrics Grid ──
    st.markdown("#### Key Metrics")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rolling 3Y CAGR", f"{r['Roll_3Y_Mean']:.1f}%" if pd.notna(r["Roll_3Y_Mean"]) else "—",
              f"Min {r['Roll_3Y_Min']:.1f}% / Max {r['Roll_3Y_Max']:.1f}%" if pd.notna(r["Roll_3Y_Min"]) else None)
    c2.metric("Rolling 5Y CAGR", f"{r['Roll_5Y_Mean']:.1f}%" if pd.notna(r["Roll_5Y_Mean"]) else "—",
              f"Min {r['Roll_5Y_Min']:.1f}% / Max {r['Roll_5Y_Max']:.1f}%" if pd.notna(r["Roll_5Y_Min"]) else None)
    c3.metric("3Y Win Rate", f"{r['Win_3Y']:.1f}%" if pd.notna(r["Win_3Y"]) else "—",
              "vs benchmark")
    c4.metric("5Y Win Rate", f"{r['Win_5Y']:.1f}%" if pd.notna(r["Win_5Y"]) else "—",
              "vs benchmark")

    c1, c2, c3, c4 = st.columns(4)
    dc = r["Down_Cap"]
    dc_delta = "✓ Below 80" if pd.notna(dc) and dc < 80 else ("Below 100" if pd.notna(dc) and dc < 100 else "Above 100 ⚠")
    c1.metric("Downside Capture", f"{dc:.0f}%" if pd.notna(dc) else "—", dc_delta)
    uc = r["Up_Cap"]
    uc_delta = "✓ Above 100" if pd.notna(uc) and uc > 100 else "Below 100"
    c2.metric("Upside Capture", f"{uc:.0f}%" if pd.notna(uc) else "—", uc_delta)
    c3.metric("Capture Ratio (U/D)", f"{r['Cap_Ratio']:.2f}" if pd.notna(r["Cap_Ratio"]) else "—",
              "Above 1.2 is great" if pd.notna(r["Cap_Ratio"]) and r["Cap_Ratio"] > 1.2 else None)
    c4.metric("1Y Return", f"{r['Ret_1Y']:.1f}%" if pd.notna(r["Ret_1Y"]) else "—")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Sortino Ratio", f"{r['Sortino']:.3f}" if pd.notna(r["Sortino"]) else "—",
              f"vs Cat: {r['Sortino_vs_Cat']:+.3f}" if pd.notna(r["Sortino_vs_Cat"]) else None)
    c2.metric("Max Drawdown (All)", f"{r['Max_DD']:.1f}%" if pd.notna(r["Max_DD"]) else "—")
    c3.metric("Max Drawdown (3Y)", f"{r['Max_DD_3Y']:.1f}%" if pd.notna(r["Max_DD_3Y"]) else "—")
    c4.metric("Calmar Ratio", f"{r['Calmar']:.2f}" if pd.notna(r["Calmar"]) else "—")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Sharpe Ratio", f"{r['Sharpe']:.2f}" if pd.notna(r["Sharpe"]) else "—")
    c2.metric("Information Ratio", f"{r['Info_Ratio']:.2f}" if pd.notna(r["Info_Ratio"]) else "—")
    c3.metric("AUM", f"₹{r['AUM']:.0f} Cr" if pd.notna(r["AUM"]) else "—")
    c4.metric("Expense Ratio", f"{r['ER']:.2f}%" if pd.notna(r["ER"]) else "—")

    st.divider()

    # ── NAV + Drawdown Chart ──
    st.markdown("#### NAV History & Drawdown")

    fund_nav = nav[["Date", selected]].dropna().copy()
    fund_nav = fund_nav.rename(columns={selected: "NAV"})
    fund_nav["Peak"] = fund_nav["NAV"].cummax()
    fund_nav["DD"] = (fund_nav["NAV"] - fund_nav["Peak"]) / fund_nav["Peak"] * 100

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.06, row_heights=[0.65, 0.35],
        subplot_titles=("NAV (₹)", "Drawdown (%)"),
    )

    fig.add_trace(go.Scatter(
        x=fund_nav["Date"], y=fund_nav["NAV"],
        fill="tozeroy", fillcolor="rgba(59,130,246,0.07)",
        line=dict(color="#3b82f6", width=1.5), name="NAV",
        hovertemplate="%{x|%d %b %Y}<br>₹%{y:.2f}<extra></extra>",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=fund_nav["Date"], y=fund_nav["DD"],
        fill="tozeroy", fillcolor="rgba(239,68,68,0.1)",
        line=dict(color="#ef4444", width=1), name="Drawdown",
        hovertemplate="%{x|%d %b %Y}<br>%{y:.1f}%<extra></extra>",
    ), row=2, col=1)

    fig.update_layout(
        height=450, showlegend=False,
        margin=dict(l=50, r=20, t=30, b=30),
        hovermode="x unified",
    )
    fig.update_xaxes(gridcolor="rgba(0,0,0,0.05)")
    fig.update_yaxes(gridcolor="rgba(0,0,0,0.05)")
    st.plotly_chart(fig, use_container_width=True)

    # ── Capture Ratio Context ──
    st.markdown("#### Where does this fund sit?")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Downside vs Upside Capture**")
        cap_df = df.dropna(subset=["Down_Cap", "Up_Cap"])
        colors = ["#22c55e" if f == selected else "#94a3b8" for f in cap_df["Fund"]]
        sizes = [14 if f == selected else 8 for f in cap_df["Fund"]]

        fig = go.Figure()
        fig.add_hline(y=100, line=dict(color="rgba(0,0,0,0.15)", dash="dash", width=1))
        fig.add_vline(x=80, line=dict(color="rgba(0,0,0,0.15)", dash="dash", width=1))
        fig.add_trace(go.Scatter(
            x=cap_df["Down_Cap"], y=cap_df["Up_Cap"],
            mode="markers", marker=dict(size=sizes, color=colors),
            text=[short(f) for f in cap_df["Fund"]],
            hovertemplate="<b>%{text}</b><br>Down: %{x:.0f}%<br>Up: %{y:.0f}%<extra></extra>",
        ))
        fig.update_layout(
            xaxis_title="Downside Capture %", yaxis_title="Upside Capture %",
            height=350, margin=dict(l=50, r=20, t=20, b=40), showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown("**Factor Score Profile**")
        score_map = {
            "S_R3": "Rolling 3Y", "S_R5": "Rolling 5Y", "S_DC": "Low Down Cap",
            "S_UC": "High Up Cap", "S_SO": "Sortino", "S_DD": "Low Drawdown",
            "S_CA": "Calmar", "S_IR": "Info Ratio", "S_ER": "Low Cost",
        }
        avail = {k: v for k, v in score_map.items() if pd.notna(r.get(k))}
        if avail:
            vals = [r[k] for k in avail]
            labels = list(avail.values())
            vals.append(vals[0])
            labels.append(labels[0])

            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=vals, theta=labels, fill="toself",
                fillcolor="rgba(59,130,246,0.1)",
                line=dict(color="#3b82f6", width=2),
                marker=dict(size=5),
            ))
            fig.update_layout(
                polar=dict(radialaxis=dict(range=[0, 100], tickfont=dict(size=10)),
                           angularaxis=dict(tickfont=dict(size=11))),
                height=350, margin=dict(l=40, r=40, t=20, b=20), showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)

    # ── Rolling Return Range ──
    st.markdown("#### Rolling Return Range vs Peers")

    r3_data = df.dropna(subset=["Roll_3Y_Mean"]).sort_values("Roll_3Y_Mean", ascending=True).tail(15)
    highlight = [("#3b82f6" if f == selected else "#d1d5db") for f in r3_data["Fund"]]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=[short(f) for f in r3_data["Fund"]],
        x=r3_data["Roll_3Y_Max"] - r3_data["Roll_3Y_Min"],
        base=r3_data["Roll_3Y_Min"],
        orientation="h",
        marker=dict(color=["rgba(59,130,246,0.15)" if f == selected else "rgba(0,0,0,0.04)"
                           for f in r3_data["Fund"]]),
        name="Range",
        hovertemplate="Min: %{base:.1f}% · Max: %{customdata:.1f}%<extra></extra>",
        customdata=r3_data["Roll_3Y_Max"],
    ))
    fig.add_trace(go.Scatter(
        y=[short(f) for f in r3_data["Fund"]],
        x=r3_data["Roll_3Y_Mean"],
        mode="markers",
        marker=dict(color=highlight, size=9, symbol="diamond"),
        name="Mean",
        hovertemplate="%{x:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        title="Rolling 3Y CAGR — Min / Mean / Max",
        showlegend=False, height=420, margin=dict(l=50, r=20, t=40, b=30),
    )
    fig.update_xaxes(title_text="CAGR (%)")
    st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
