"""
SmallCap Fund Ranking — 3-Factor Quantitative Model

The 3 factors that matter most for small-cap funds, backed by research:

1. ROLLING 5-YEAR CAGR
   Why: Small caps are cyclical. Point-to-point returns lie. Rolling 5Y
   computes the CAGR for EVERY possible 5-year window the fund has lived
   through, then averages them. This is the single best proxy for
   "what return would a real investor actually experience over a full cycle?"
   Source: Morningstar's "Rolling Returns: A Better Measure of Performance"

2. DOWNSIDE CAPTURE RATIO
   Why: The #1 killer of small-cap returns is deep drawdowns. A fund that
   falls 50% needs 100% gain to recover — that's years of compounding lost.
   Downside capture measures what % of the benchmark's losses the fund
   absorbs. Below 80 is good. Below 50 is elite. This single metric
   captures risk management, stock selection quality, and portfolio
   construction — all in one number.
   Source: Sortino & Van der Meer (1991), "Downside risk"; 
           Israelsen (2005), "A refinement to the Sharpe ratio"

3. ALPHA HIT RATE (% of rolling 12M windows with positive alpha)
   Why: A fund can show great 5-year alpha but generated it all in one
   lucky year. Alpha Hit Rate asks: "In what percentage of ALL possible
   12-month windows did this fund actually beat the benchmark?"
   A fund with 70%+ hit rate generates alpha consistently, not randomly.
   This is the purest test of repeatable skill vs luck.
   Source: Cremers & Petajisto (2009), "How Active Is Your Fund Manager?"
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import timedelta
import warnings, io

warnings.filterwarnings("ignore")
st.set_page_config(page_title="SmallCap 3-Factor", layout="wide", page_icon="📊")


# ═══════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════
@st.cache_data(show_spinner="Loading data...")
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
        nav = pd.merge(nav, bench, on="Date", how="left")
        nav["Benchmark"] = nav["Benchmark"].ffill()
    except Exception:
        st.sidebar.warning("Nifty 500 TRI CSV not found — using category average as proxy.")
        valid = [f for f in fund_names if nav[f].notna().sum() > 252]
        nav["Benchmark"] = nav[valid].mean(axis=1)

    ar = pd.read_excel("smallcap_aum.xlsx")
    aum = ar.iloc[3:].copy()
    aum.columns = ["Fund", "Month_End", "AUM", "AAUM", "Avg_AUM"]
    aum["AUM"] = pd.to_numeric(aum["AUM"], errors="coerce")
    aum = aum[aum["AUM"].notna()]
    aum["Month_End"] = pd.to_datetime(aum["Month_End"], errors="coerce")
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
            except:
                pass
    df_er = pd.DataFrame(er_recs).drop_duplicates(subset="Fund", keep="first")
    return nav, fund_names, aum_latest, df_er


# ═══════════════════════════════════════════
# 3-FACTOR COMPUTATION
# ═══════════════════════════════════════════
@st.cache_data(show_spinner="Computing 3-factor model...")
def compute_factors(_nav, fund_names, aum_latest, df_er):
    nav = _nav.copy()
    monthly = nav.set_index("Date").resample("ME").last()
    mret = monthly.pct_change().dropna(how="all")
    bench = mret["Benchmark"]

    results = []
    for fund in fund_names:
        fd = nav[["Date", fund]].dropna()
        if len(fd) < 252:
            continue
        fr = mret[fund].dropna()
        ci = fr.index.intersection(bench.dropna().index)
        if len(ci) < 12:
            continue
        fr = fr.loc[ci]
        br = bench.loc[ci]

        track_yrs = len(fd) / 252
        daily = fd.set_index("Date")[fund]

        # ────────────────────────────────────────
        # FACTOR 1: ROLLING 5-YEAR CAGR
        # ────────────────────────────────────────
        mn = daily.resample("ME").last().dropna()
        cagrs_5y = []
        for i in range(60, len(mn)):
            s, e = mn.iloc[i - 60], mn.iloc[i]
            if s > 0:
                cagrs_5y.append(((e / s) ** (1 / 5) - 1) * 100)

        r5_mean = np.mean(cagrs_5y) if cagrs_5y else None
        r5_median = np.median(cagrs_5y) if cagrs_5y else None
        r5_min = np.min(cagrs_5y) if cagrs_5y else None
        r5_max = np.max(cagrs_5y) if cagrs_5y else None
        r5_windows = len(cagrs_5y)

        # Also compute 3Y for display
        cagrs_3y = []
        for i in range(36, len(mn)):
            s, e = mn.iloc[i - 36], mn.iloc[i]
            if s > 0:
                cagrs_3y.append(((e / s) ** (1 / 3) - 1) * 100)
        r3_mean = np.mean(cagrs_3y) if cagrs_3y else None

        # ────────────────────────────────────────
        # FACTOR 2: DOWNSIDE CAPTURE RATIO
        # ────────────────────────────────────────
        down_months = br[br < 0]
        down_cap = None
        if len(down_months) > 3:
            fund_down = fr.loc[down_months.index]
            down_cap = (fund_down.mean() / down_months.mean()) * 100

        # Also compute upside capture for display
        up_months = br[br > 0]
        up_cap = None
        if len(up_months) > 3:
            fund_up = fr.loc[up_months.index]
            up_cap = (fund_up.mean() / up_months.mean()) * 100

        # ────────────────────────────────────────
        # FACTOR 3: ALPHA HIT RATE
        # ────────────────────────────────────────
        alphas = []
        for i in range(12, len(fr)):
            fund_12m = fr.iloc[i - 12 : i].sum()
            bench_12m = br.loc[fr.iloc[i - 12 : i].index].sum()
            alphas.append(fund_12m - bench_12m)

        alpha_hit = np.mean([a > 0 for a in alphas]) * 100 if alphas else None
        avg_alpha = np.mean(alphas) * 100 if alphas else None
        alpha_windows = len(alphas)

        # ── Supporting metrics for deep-dive (not used in ranking) ──
        prices = fd[fund].values
        cummax = np.maximum.accumulate(prices)
        dd = (prices - cummax) / cummax
        max_dd = dd.min() * 100
        current_dd = (prices[-1] - prices.max()) / prices.max() * 100

        ann_ret = fr.mean() * 12 * 100
        rf_m = 0.06 / 12
        excess = fr - rf_m
        neg = excess[excess < 0]
        ds = np.sqrt(np.mean(neg ** 2)) if len(neg) > 3 else None
        sortino = (fr.mean() - rf_m) / ds if ds and ds > 0 else None

        latest_nav = fd.iloc[-1][fund]
        ld = fd.iloc[-1]["Date"]
        def get_ret(days):
            mask = (nav["Date"] >= ld - timedelta(days=days+15)) & (nav["Date"] <= ld - timedelta(days=days-15))
            ns = nav[mask].dropna(subset=[fund])
            if len(ns) == 0: return None
            nv = ns.iloc[(ns["Date"] - (ld - timedelta(days=days))).abs().argsort().iloc[0]][fund]
            return (latest_nav / nv - 1) * 100
        ret_1y = get_ret(365)

        results.append({
            "Fund": fund, "Track_Yrs": track_yrs,
            # The 3 factors
            "Roll_5Y": r5_mean, "Roll_5Y_Med": r5_median,
            "Roll_5Y_Min": r5_min, "Roll_5Y_Max": r5_max, "Roll_5Y_Windows": r5_windows,
            "Down_Cap": down_cap,
            "Alpha_Hit": alpha_hit, "Avg_Alpha": avg_alpha, "Alpha_Windows": alpha_windows,
            # Supporting
            "Roll_3Y": r3_mean, "Up_Cap": up_cap,
            "Max_DD": max_dd, "Current_DD": current_dd,
            "Sortino": sortino, "Ann_Ret": ann_ret, "Ret_1Y": ret_1y,
        })

    df = pd.DataFrame(results)
    df = df.merge(aum_latest, on="Fund", how="left")
    df = df.merge(df_er, on="Fund", how="left")
    return df


# ═══════════════════════════════════════════
# RANKING
# ═══════════════════════════════════════════
def pctrank(s, asc=True):
    v = s.notna()
    r = s[v].rank(ascending=asc, pct=True) * 100
    out = pd.Series(np.nan, index=s.index)
    out[v] = r
    return out


def rank_funds(df, w1, w2, w3):
    d = df.copy()

    d["S_Roll5Y"] = pctrank(d["Roll_5Y"])
    d["S_DownCap"] = pctrank(d["Down_Cap"], asc=False)  # lower = better
    d["S_AlphaHit"] = pctrank(d["Alpha_Hit"])

    total = w1 + w2 + w3
    if total == 0:
        total = 1

    def comp(row):
        if row.get("Track_Yrs", 0) < 3:
            return np.nan
        tw = ts = 0
        for col, wt in [("S_Roll5Y", w1), ("S_DownCap", w2), ("S_AlphaHit", w3)]:
            if pd.notna(row.get(col)):
                ts += row[col] * (wt / total)
                tw += wt / total
        return ts / tw if tw > 0 else np.nan

    d["Score"] = d.apply(comp, axis=1)
    d["Rank"] = d["Score"].rank(ascending=False, method="min").astype("Int64")

    def signal(x):
        if pd.isna(x): return "Too Young"
        if x >= 75: return "Elite"
        if x >= 55: return "Strong"
        if x >= 40: return "Average"
        return "Weak"
    d["Signal"] = d["Score"].apply(signal)

    return d.sort_values("Rank", na_position="last").reset_index(drop=True)


def short(f):
    return (f.replace("Small Cap", "SC").replace("Smallcap", "SC")
             .replace("Fund-Reg(G)", "").replace("Fund(G)", "").strip())


def fmt(v, d=1, suf=""):
    return f"{v:.{d}f}{suf}" if pd.notna(v) else "—"


# ═══════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════
def main():
    nav, fund_names, aum_latest, df_er = load_data()
    df_raw = compute_factors(nav, fund_names, aum_latest, df_er)

    st.title("📊 SmallCap 3-Factor Ranking")
    st.caption(f"{len(df_raw)} funds · Data through {nav['Date'].max().strftime('%d %b %Y')}")

    # ── Sidebar ──
    with st.sidebar:
        st.header("Factor Weights")
        st.caption("How much each factor matters. Equal weight = 33/33/33.")
        w1 = st.slider("Rolling 5Y CAGR", 10, 60, 35, help="Long-term compounding ability across full market cycles")
        w2 = st.slider("Downside Capture", 10, 60, 35, help="Capital protection when the market falls")
        w3 = st.slider("Alpha Hit Rate", 10, 60, 30, help="Consistency of beating the benchmark")

        st.divider()
        st.markdown("**Effective split**")
        t = w1 + w2 + w3
        st.markdown(f"- Rolling 5Y: **{w1/t*100:.0f}%**")
        st.markdown(f"- Downside Cap: **{w2/t*100:.0f}%**")
        st.markdown(f"- Alpha Hit: **{w3/t*100:.0f}%**")

    df = rank_funds(df_raw, w1, w2, w3)

    tab_rank, tab_deep = st.tabs(["🏆 Rankings", "🔎 Fund Deep-Dive"])

    # ═══════════════════════════════════
    # TAB 1: RANKINGS
    # ═══════════════════════════════════
    with tab_rank:

        ranked = df[df["Score"].notna()]
        unranked = df[df["Score"].isna()]

        c1, c2, c3 = st.columns(3)
        if len(ranked) > 0:
            top = ranked.iloc[0]
            c1.metric("🥇 #1 Fund", short(top["Fund"]), f"Score {top['Score']:.1f}")
        c2.metric("Median Downside Capture", f"{ranked['Down_Cap'].median():.0f}%")
        c3.metric("Median Alpha Hit Rate", f"{ranked['Alpha_Hit'].median():.0f}%")

        st.divider()

        # ── The 3-Factor Table ──
        disp = ranked[[
            "Rank", "Fund", "Score", "Signal",
            "Roll_5Y", "Down_Cap", "Alpha_Hit",
            "Roll_3Y", "Up_Cap", "Max_DD", "Sortino", "AUM", "ER",
        ]].copy()
        disp.columns = [
            "Rank", "Fund", "Score", "Signal",
            "Roll 5Y CAGR%", "Down Capture%", "Alpha Hit Rate%",
            "Roll 3Y%", "Up Capture%", "Max DD%", "Sortino", "AUM (Cr)", "Exp Ratio%",
        ]
        disp["Fund"] = disp["Fund"].apply(short)
        disp = disp.round({
            "Score": 1, "Roll 5Y CAGR%": 1, "Down Capture%": 0, "Alpha Hit Rate%": 1,
            "Roll 3Y%": 1, "Up Capture%": 0, "Max DD%": 1, "Sortino": 2, "AUM (Cr)": 0, "Exp Ratio%": 2,
        })

        def c_sig(val):
            m = {"Elite": "background-color:#dcfce7;color:#166534;",
                 "Strong": "background-color:#e0f2fe;color:#075985;",
                 "Average": "background-color:#fef9c3;color:#854d0e;",
                 "Weak": "background-color:#fecaca;color:#991b1b;"}
            return m.get(val, "")

        def c_score(val):
            if pd.isna(val): return ""
            if val >= 70: return "background-color:#dcfce7;font-weight:700;"
            if val >= 50: return "background-color:#e0f2fe;"
            if val >= 35: return "background-color:#fef9c3;"
            return "background-color:#fecaca;"

        def c_dc(val):
            if pd.isna(val): return ""
            if val < 50: return "color:#16a34a;font-weight:700;"
            if val < 80: return "color:#16a34a;"
            if val < 100: return "color:#ca8a04;"
            return "color:#dc2626;font-weight:600;"

        def c_ah(val):
            if pd.isna(val): return ""
            if val >= 65: return "color:#16a34a;font-weight:600;"
            if val >= 50: return "color:#ca8a04;"
            return "color:#dc2626;"

        styled = (disp.style
            .map(c_sig, subset=["Signal"])
            .map(c_score, subset=["Score"])
            .map(c_dc, subset=["Down Capture%"])
            .map(c_ah, subset=["Alpha Hit Rate%"])
            .format(na_rep="—")
            .set_properties(**{"text-align": "center", "font-size": "13px"})
            .set_properties(subset=["Fund"], **{"text-align": "left", "font-weight": "500"})
        )
        st.dataframe(styled, use_container_width=True, height=600, hide_index=True)

        if len(unranked) > 0:
            with st.expander(f"📎 {len(unranked)} funds excluded (track record < 3 years)"):
                st.dataframe(
                    unranked[["Fund", "Track_Yrs", "Ret_1Y", "AUM"]].assign(
                        Fund=unranked["Fund"].apply(short),
                    ).round(1).rename(columns={"Track_Yrs": "Age (Yrs)", "Ret_1Y": "1Y Ret%", "AUM": "AUM (Cr)"}),
                    hide_index=True, use_container_width=True,
                )

        with st.expander("📖 Why these 3 factors?"):
            st.markdown("""
**1. Rolling 5-Year CAGR** — _"What return would a real investor actually get?"_

Point-to-point returns are misleading. A fund measured from March 2020 (COVID bottom) looks
incredible; measured from Jan 2018 it might look mediocre. Rolling 5Y computes the CAGR for
**every possible** 5-year window the fund has experienced, then averages them. This smooths
out timing luck and shows the true compounding ability across full bull-bear cycles. 5 years
is the right window because Indian small caps typically complete one full cycle in 4–6 years.

**2. Downside Capture Ratio** — _"How much does it bleed when the market crashes?"_

This is the single most important risk metric for small caps. When the Nifty falls 10% in a
month, a fund with 40% downside capture only falls 4%. That gap compounds enormously — a 50%
loss needs a 100% gain to recover. Funds with low downside capture have better stock selection
(avoiding junk), better position sizing, and better sell discipline. Below 50% is elite, below
80% is good, above 100% means the fund amplifies crashes.

**3. Alpha Hit Rate** — _"Is the outperformance skill or luck?"_

A fund can show great 5-year alpha but may have generated it all in one lucky year. Alpha Hit
Rate computes alpha for **every rolling 12-month window** and asks: in what percentage did the
fund actually beat the benchmark? A fund with 70%+ hit rate is generating alpha consistently
— that's skill. Below 50% means the fund underperforms the benchmark more often than not,
regardless of what the headline return says.
""")

        buf = io.BytesIO()
        df.round(2).to_excel(buf, index=False, engine="openpyxl")
        st.download_button("📥 Download", buf.getvalue(), "smallcap_3factor.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    # ═══════════════════════════════════
    # TAB 2: FUND DEEP-DIVE
    # ═══════════════════════════════════
    with tab_deep:
        selected = st.selectbox("Pick a fund", df.sort_values("Rank", na_position="last")["Fund"].tolist(),
                                format_func=short)
        r = df[df["Fund"] == selected].iloc[0]

        st.subheader(short(selected))
        rank_txt = f"Rank #{r['Rank']}" if pd.notna(r["Rank"]) else "Not ranked (< 3 years)"
        sig = r["Signal"]
        st.markdown(f"**{rank_txt}** · **{sig}** · Score: **{fmt(r['Score'],1)}**/100 · Age: {r['Track_Yrs']:.1f} yrs")
        st.divider()

        # ── The 3 Core Factors ──
        st.markdown("#### The 3 Factors")
        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric("Rolling 5Y CAGR", fmt(r["Roll_5Y"], 1, "%"),
                       f"Range: {fmt(r['Roll_5Y_Min'],1)}% to {fmt(r['Roll_5Y_Max'],1)}%" if pd.notna(r["Roll_5Y_Min"]) else None)
            if pd.notna(r["Roll_5Y"]):
                cat_med = df[df["Score"].notna()]["Roll_5Y"].median()
                diff = r["Roll_5Y"] - cat_med
                st.caption(f"Category median: {cat_med:.1f}% · You: {'+' if diff > 0 else ''}{diff:.1f}%")

        with c2:
            dc = r["Down_Cap"]
            label = "✅ Elite (<50)" if pd.notna(dc) and dc < 50 else (
                "✅ Good (<80)" if pd.notna(dc) and dc < 80 else (
                "⚠️ Okay (<100)" if pd.notna(dc) and dc < 100 else "❌ Amplifies losses"))
            st.metric("Downside Capture", fmt(dc, 0, "%"), label)
            if pd.notna(dc):
                st.caption(f"When benchmark falls 10%, this fund falls ~{dc/10:.1f}%")

        with c3:
            ah = r["Alpha_Hit"]
            label = "✅ Consistent (>65%)" if pd.notna(ah) and ah >= 65 else (
                "⚠️ Average (50-65%)" if pd.notna(ah) and ah >= 50 else "❌ Inconsistent")
            st.metric("Alpha Hit Rate", fmt(ah, 1, "%"), label)
            if pd.notna(ah) and pd.notna(r["Alpha_Windows"]):
                st.caption(f"Positive alpha in {ah:.0f}% of {int(r['Alpha_Windows'])} rolling 12M windows")

        st.divider()

        # ── Supporting Context ──
        st.markdown("#### Supporting Context")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("1Y Return", fmt(r["Ret_1Y"], 1, "%"))
        c2.metric("Roll 3Y CAGR", fmt(r["Roll_3Y"], 1, "%"))
        c3.metric("Upside Capture", fmt(r["Up_Cap"], 0, "%"))
        c4.metric("Sortino", fmt(r["Sortino"], 2))
        c5.metric("Max Drawdown", fmt(r["Max_DD"], 1, "%"))

        c1, c2, c3 = st.columns(3)
        c1.metric("Current Drawdown", fmt(r["Current_DD"], 1, "%"))
        c2.metric("AUM", f"₹{r['AUM']:.0f} Cr" if pd.notna(r["AUM"]) else "—")
        c3.metric("Expense Ratio", fmt(r["ER"], 2, "%"))

        st.divider()

        # ── NAV + Benchmark Chart ──
        st.markdown("#### NAV vs Benchmark")
        fn = nav[["Date", selected, "Benchmark"]].dropna(subset=[selected]).copy()
        fn = fn.rename(columns={selected: "NAV"})
        if not fn.empty and fn.iloc[0]["Benchmark"] > 0:
            fn["Bench"] = fn["Benchmark"] * (fn.iloc[0]["NAV"] / fn.iloc[0]["Benchmark"])
        else:
            fn["Bench"] = fn["Benchmark"]
        fn["Peak"] = fn["NAV"].cummax()
        fn["DD"] = (fn["NAV"] - fn["Peak"]) / fn["Peak"] * 100

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                            row_heights=[0.65, 0.35])
        fig.add_trace(go.Scatter(x=fn["Date"], y=fn["NAV"], fill="tozeroy",
                                  fillcolor="rgba(59,130,246,0.06)",
                                  line=dict(color="#3b82f6", width=1.5), name="Fund"), row=1, col=1)
        fig.add_trace(go.Scatter(x=fn["Date"], y=fn["Bench"],
                                  line=dict(color="#94a3b8", width=1.5, dash="dot"), name="Benchmark"), row=1, col=1)
        fig.add_trace(go.Scatter(x=fn["Date"], y=fn["DD"], fill="tozeroy",
                                  fillcolor="rgba(239,68,68,0.08)",
                                  line=dict(color="#ef4444", width=1), name="Drawdown"), row=2, col=1)
        fig.update_layout(height=420, margin=dict(l=50, r=20, t=20, b=30), hovermode="x unified",
                          legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        fig.update_xaxes(gridcolor="rgba(0,0,0,0.05)")
        fig.update_yaxes(gridcolor="rgba(0,0,0,0.05)")
        st.plotly_chart(fig, use_container_width=True)

        # ── Peer Comparison Charts ──
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("#### Where you sit — Downside vs Alpha")
            cdf = df.dropna(subset=["Down_Cap", "Alpha_Hit"])
            fig2 = go.Figure()
            fig2.add_hline(y=50, line=dict(color="rgba(0,0,0,0.1)", dash="dash", width=1))
            fig2.add_vline(x=80, line=dict(color="rgba(0,0,0,0.1)", dash="dash", width=1))
            # Ideal zone shading
            fig2.add_shape(type="rect", x0=0, x1=80, y0=50, y1=100,
                           fillcolor="rgba(34,197,94,0.05)", line=dict(width=0))
            fig2.add_trace(go.Scatter(
                x=cdf["Down_Cap"], y=cdf["Alpha_Hit"], mode="markers+text",
                marker=dict(
                    size=[16 if f == selected else 9 for f in cdf["Fund"]],
                    color=["#3b82f6" if f == selected else "#d1d5db" for f in cdf["Fund"]],
                    line=dict(width=[2 if f == selected else 0 for f in cdf["Fund"]], color="#1d4ed8"),
                ),
                text=["" if f != selected else short(f) for f in cdf["Fund"]],
                textposition="top center", textfont=dict(size=12, color="#1d4ed8"),
                hovertemplate="<b>%{customdata}</b><br>Down Cap: %{x:.0f}%<br>Alpha Hit: %{y:.1f}%<extra></extra>",
                customdata=[short(f) for f in cdf["Fund"]],
            ))
            fig2.update_layout(xaxis_title="Downside Capture % (lower = better)",
                               yaxis_title="Alpha Hit Rate % (higher = better)",
                               height=380, margin=dict(l=50, r=20, t=20, b=50), showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)

        with col_b:
            st.markdown("#### Rolling 5Y CAGR — Range vs Peers")
            r5df = df.dropna(subset=["Roll_5Y"]).sort_values("Roll_5Y", ascending=True).tail(15)
            fig3 = go.Figure()
            fig3.add_trace(go.Bar(
                y=[short(f) for f in r5df["Fund"]],
                x=r5df["Roll_5Y_Max"] - r5df["Roll_5Y_Min"],
                base=r5df["Roll_5Y_Min"], orientation="h",
                marker=dict(color=["rgba(59,130,246,0.15)" if f == selected else "rgba(0,0,0,0.04)"
                                   for f in r5df["Fund"]]),
                hovertemplate="Min: %{base:.1f}% · Max: %{customdata:.1f}%<extra></extra>",
                customdata=r5df["Roll_5Y_Max"],
            ))
            fig3.add_trace(go.Scatter(
                y=[short(f) for f in r5df["Fund"]],
                x=r5df["Roll_5Y"], mode="markers",
                marker=dict(
                    color=["#3b82f6" if f == selected else "#d1d5db" for f in r5df["Fund"]],
                    size=10, symbol="diamond",
                ),
            ))
            fig3.update_layout(showlegend=False, height=380,
                               margin=dict(l=50, r=20, t=20, b=30))
            fig3.update_xaxes(title_text="CAGR %")
            st.plotly_chart(fig3, use_container_width=True)


if __name__ == "__main__":
    main()
