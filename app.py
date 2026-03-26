"""
SmallCap Fund Ranking — Dual-Engine Model
Benchmark: Nifty 500 TRI

ENGINE 1 — ESTABLISHED COMPOUNDERS (> 3 years)
  Factor 1: Rolling 5Y CAGR — full-cycle compounding
  Factor 2: Downside Capture — capital protection
  Factor 3: Alpha Hit Rate (12M rolling) — consistency of skill

ENGINE 2 — RISK-ADJUSTED MOMENTUM (All Funds >= 6 Months)
  Factor 1: 6M Risk-Adjusted Momentum (6M Return / Annualised Volatility)
  Factor 2: 1Y Risk-Adjusted Momentum (1Y Return / Annualised Volatility)
"""

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
# DATA
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
        bench_name = "Nifty 500 TRI"
    except Exception:
        valid = [f for f in fund_names if nav[f].notna().sum() > 252]
        nav["Benchmark"] = nav[valid].mean(axis=1)
        bench_name = "Category Average (proxy)"

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
    return nav, fund_names, aum_latest, df_er, bench_name


# ═══════════════════════════════════════════
# METRICS ENGINE
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
        if len(fd) < 120:  # need at least ~6 months
            continue
        fr = mret[fund].dropna()
        ci = fr.index.intersection(br.dropna().index)
        if len(ci) < 6:
            continue
        fr, brc = fr.loc[ci], br.loc[ci]

        track_yrs = len(fd) / 252
        daily = fd.set_index("Date")[fund]
        months_available = len(fr)

        # ══════════════════════════════════
        # ESTABLISHED FUND FACTORS (>3 yrs)
        # ══════════════════════════════════

        # Factor 1: Rolling 5Y CAGR
        mn = daily.resample("ME").last().dropna()
        cagrs_5y = []
        for i in range(60, len(mn)):
            s, e = mn.iloc[i - 60], mn.iloc[i]
            if s > 0:
                cagrs_5y.append(((e / s) ** (1 / 5) - 1) * 100)
        r5_mean = np.mean(cagrs_5y) if cagrs_5y else None
        
        # Factor 2: Downside Capture
        dm = brc[brc < 0]
        down_cap = (fr.loc[dm.index].mean() / dm.mean()) * 100 if len(dm) >= 3 else None

        # Factor 3: Alpha Hit Rate — 12M rolling
        alphas_12m = []
        for i in range(12, len(fr)):
            alphas_12m.append(fr.iloc[i - 12 : i].sum() - brc.loc[fr.iloc[i - 12 : i].index].sum())
        alpha_hit_12m = np.mean([a > 0 for a in alphas_12m]) * 100 if alphas_12m else None

        # ══════════════════════════════════
        # MOMENTUM FACTORS (All Funds)
        # ══════════════════════════════════
        latest_nav = fd.iloc[-1][fund]
        ld = fd.iloc[-1]["Date"]

        def get_ret(days):
            mask = (nav["Date"] >= ld - timedelta(days=days + 15)) & (nav["Date"] <= ld - timedelta(days=days - 15))
            ns = nav[mask].dropna(subset=[fund])
            if len(ns) == 0:
                return None
            nv = ns.iloc[(ns["Date"] - (ld - timedelta(days=days))).abs().argsort().iloc[0]][fund]
            return (latest_nav / nv - 1) * 100

        ret_1y = get_ret(365)
        ret_6m = get_ret(180)
        
        # Volatility Calculation for Risk-Adjustment
        rec = fd[fd["Date"] >= ld - timedelta(days=365)]
        if len(rec) < 100:  # If fund is less than 1 yr old, use 6M data to annualise vol
            rec = fd[fd["Date"] >= ld - timedelta(days=180)]
            
        vol = rec[fund].pct_change().dropna().std() * np.sqrt(252) * 100 if len(rec) > 50 else None

        # Risk-Adjusted Momentum = Return / Volatility
        mom_6m_ra = (ret_6m / vol) if (ret_6m is not None and vol) else None
        mom_1y_ra = (ret_1y / vol) if (ret_1y is not None and vol) else None

        # ══════════════════════════════════
        # SUPPORTING METRICS (display only)
        # ══════════════════════════════════
        prices = fd[fund].values
        cummax = np.maximum.accumulate(prices)
        dd = (prices - cummax) / cummax
        max_dd = dd.min() * 100
        current_dd = (prices[-1] - prices.max()) / prices.max() * 100

        rf_m = 0.06 / 12
        excess = fr - rf_m
        neg = excess[excess < 0]
        ds = np.sqrt(np.mean(neg ** 2)) if len(neg) > 3 else None
        sortino = (fr.mean() - rf_m) / ds if ds and ds > 0 else None

        results.append({
            "Fund": fund, "Track_Yrs": track_yrs, "Months": months_available,
            # Established
            "Roll_5Y": r5_mean, "Down_Cap": down_cap, "Alpha_Hit_12M": alpha_hit_12m,
            # Momentum
            "Ret_6M": ret_6m, "Ret_1Y": ret_1y, "Vol": vol,
            "Mom_6M_RA": mom_6m_ra, "Mom_1Y_RA": mom_1y_ra,
            # Supporting
            "Max_DD": max_dd, "Current_DD": current_dd, "Sortino": sortino,
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


def rank_all(df, w_lt, w_mom):
    d = df.copy()

    # ── ENGINE 1: ESTABLISHED (>3 years) ──
    est = d[d["Track_Yrs"] >= 3].copy()
    est["S_R5"] = pctrank(est["Roll_5Y"])
    est["S_DC"] = pctrank(est["Down_Cap"], asc=False)
    est["S_AH"] = pctrank(est["Alpha_Hit_12M"])

    t1 = sum(w_lt)
    if t1 == 0: t1 = 1

    def comp_lt(row):
        tw = ts = 0
        for col, wt in zip(["S_R5", "S_DC", "S_AH"], w_lt):
            if pd.notna(row.get(col)):
                ts += row[col] * (wt / t1)
                tw += wt / t1
        return ts / tw if tw > 0 else np.nan

    est["Score"] = est.apply(comp_lt, axis=1)
    est["Rank"] = est["Score"].rank(ascending=False, method="min").astype(int)
    est["Engine"] = "Established"

    def sig_lt(x):
        if pd.isna(x): return "N/A"
        if x >= 75: return "Elite"
        if x >= 55: return "Strong"
        if x >= 40: return "Average"
        return "Weak"
    est["Signal"] = est["Score"].apply(sig_lt)

    # ── ENGINE 2: MOMENTUM STRATEGY (All funds >= 6 months) ──
    mom = d[d["Months"] >= 6].copy()

    if len(mom) > 0:
        mom["S_M6"] = pctrank(mom["Mom_6M_RA"])
        mom["S_M1"] = pctrank(mom["Mom_1Y_RA"])

        t2 = sum(w_mom)
        if t2 == 0: t2 = 1

        def comp_mom(row):
            tw = ts = 0
            for col, wt in zip(["S_M6", "S_M1"], w_mom):
                if pd.notna(row.get(col)):
                    ts += row[col] * (wt / t2)
                    tw += wt / t2
            return ts / tw if tw > 0 else np.nan

        mom["Score"] = mom.apply(comp_mom, axis=1)
        mom["Rank"] = mom["Score"].rank(ascending=False, method="min").astype(int)
        mom["Engine"] = "Momentum"

        def sig_mom(x):
            if pd.isna(x): return "N/A"
            if x >= 75: return "Hot Trend"
            if x >= 55: return "Building"
            if x >= 40: return "Stagnant"
            return "Downtrend"
        mom["Signal"] = mom["Score"].apply(sig_mom)

    # Too young (< 6 months)
    too_young = d[d["Months"] < 6].copy() if "Months" in d.columns else pd.DataFrame()
    if len(too_young) > 0:
        too_young["Score"] = np.nan
        too_young["Rank"] = np.nan
        too_young["Signal"] = "Too Young"
        too_young["Engine"] = "Too Young"

    return est, mom, too_young


def short(f):
    return (f.replace("Small Cap", "SC").replace("Smallcap", "SC")
             .replace("Fund-Reg(G)", "").replace("Fund(G)", "").strip())


def fmt(v, d=1, suf=""):
    return f"{v:.{d}f}{suf}" if pd.notna(v) else "—"


# ═══════════════════════════════════════════
# STYLING
# ═══════════════════════════════════════════
def c_sig(val):
    m = {
        "Elite": "background-color:#dcfce7;color:#166534;",
        "Strong": "background-color:#e0f2fe;color:#075985;",
        "Average": "background-color:#fef9c3;color:#854d0e;",
        "Weak": "background-color:#fecaca;color:#991b1b;",
        "Hot Trend": "background-color:#dcfce7;color:#166534;",
        "Building": "background-color:#e0f2fe;color:#075985;",
        "Stagnant": "background-color:#fef9c3;color:#854d0e;",
        "Downtrend": "background-color:#fecaca;color:#991b1b;",
    }
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

def c_mom(val):
    if pd.isna(val): return ""
    if val > 2.0: return "color:#16a34a;font-weight:700;"
    if val > 1.0: return "color:#ca8a04;"
    return "color:#dc2626;"


# ═══════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════
def main():
    nav, fund_names, aum_latest, df_er, bench_name = load_data()
    df_raw = compute_all(nav, fund_names, aum_latest, df_er)

    st.title("📊 SmallCap Quants: Dual-Engine")
    st.caption(f"{len(df_raw)} funds · Benchmark: **{bench_name}** · Data through {nav['Date'].max().strftime('%d %b %Y')}")

    # Sidebar
    with st.sidebar:
        st.header("🏛️ Established Engine")
        st.caption("Fundamentals & Full-Cycle (>3 yrs)")
        lt1 = st.slider("Rolling 5Y CAGR", 10, 60, 35, key="lt1")
        lt2 = st.slider("Downside Capture", 10, 60, 35, key="lt2")
        lt3 = st.slider("Alpha Hit Rate (12M)", 10, 60, 30, key="lt3")

        st.divider()
        st.header("🚀 Momentum Engine")
        st.caption("Risk-Adjusted Velocity (All Funds)")
        mom1 = st.slider("6M Return / Volatility", 0, 100, 50, key="mom1")
        mom2 = st.slider("1Y Return / Volatility", 0, 100, 50, key="mom2")

    est, mom, too_young = rank_all(df_raw, [lt1, lt2, lt3], [mom1, mom2])

    tab_rank, tab_deep = st.tabs(["🏆 Leaderboards", "🔎 Fund Deep-Dive"])

    # ═══════════════════════════════════
    # TAB 1: RANKINGS
    # ═══════════════════════════════════
    with tab_rank:
        board = st.radio("Select Strategy Engine:", [
            "🏛️ Established Compounders (Fundamentals)",
            "🚀 Risk-Adjusted Momentum (Trend Following)",
        ], horizontal=True)
        st.divider()

        if "Established" in board:
            c1, c2, c3, c4 = st.columns(4)
            if len(est) > 0:
                top = est.sort_values("Rank").iloc[0]
                c1.metric("🥇 Top Compounder", short(top["Fund"]), f"Score {top['Score']:.1f}")
            c2.metric("Median Down Capture", f"{est['Down_Cap'].median():.0f}%")
            c3.metric("Median Alpha Hit", f"{est['Alpha_Hit_12M'].median():.0f}%")
            c4.metric("Funds Ranked", len(est))

            disp = est.sort_values("Rank")[[
                "Rank", "Fund", "Score", "Signal", "Track_Yrs",
                "Roll_5Y", "Down_Cap", "Alpha_Hit_12M",
                "Max_DD", "Sortino", "AUM", "ER",
            ]].copy()
            disp.columns = [
                "Rank", "Fund", "Score", "Signal", "Age (Yrs)",
                "Roll 5Y%", "Down Cap%", "Alpha Hit%",
                "Max DD%", "Sortino", "AUM (Cr)", "Exp Ratio%",
            ]
            disp["Fund"] = disp["Fund"].apply(short)
            disp = disp.round({
                "Score": 1, "Age (Yrs)": 1, "Roll 5Y%": 1, "Down Cap%": 0, "Alpha Hit%": 1,
                "Max DD%": 1, "Sortino": 2, "AUM (Cr)": 0, "Exp Ratio%": 2,
            })

            styled = (disp.style
                .map(c_sig, subset=["Signal"])
                .map(c_score, subset=["Score"])
                .map(c_dc, subset=["Down Cap%"])
                .format(na_rep="—")
                .set_properties(**{"text-align": "center", "font-size": "13px"})
                .set_properties(subset=["Fund"], **{"text-align": "left", "font-weight": "500"})
            )
            st.dataframe(styled, use_container_width=True, height=600, hide_index=True)

        else:  # Momentum
            if len(mom) == 0:
                st.info("No funds found with sufficient data for momentum scoring.")
            else:
                c1, c2, c3, c4 = st.columns(4)
                top = mom.sort_values("Rank").iloc[0]
                c1.metric("🔥 Top Momentum Fund", short(top["Fund"]), f"Score {top['Score']:.1f}")
                c2.metric("Median 6M/Vol Ratio", f"{mom['Mom_6M_RA'].median():.2f}")
                c3.metric("Median 1Y/Vol Ratio", f"{mom['Mom_1Y_RA'].median():.2f}")
                c4.metric("Funds Ranked", len(mom))

                disp = mom.sort_values("Rank")[[
                    "Rank", "Fund", "Score", "Signal", "Track_Yrs",
                    "Ret_6M", "Ret_1Y", "Vol", "Mom_6M_RA", "Mom_1Y_RA", "AUM"
                ]].copy()
                disp.columns = [
                    "Rank", "Fund", "Score", "Signal", "Age (Yrs)",
                    "6M Ret%", "1Y Ret%", "Vol%", "6M / Vol", "1Y / Vol", "AUM (Cr)"
                ]
                disp["Fund"] = disp["Fund"].apply(short)
                disp = disp.round({
                    "Score": 1, "Age (Yrs)": 1, "6M Ret%": 1, "1Y Ret%": 1,
                    "Vol%": 1, "6M / Vol": 2, "1Y / Vol": 2, "AUM (Cr)": 0
                })

                styled = (disp.style
                    .map(c_sig, subset=["Signal"])
                    .map(c_score, subset=["Score"])
                    .map(c_mom, subset=["6M / Vol", "1Y / Vol"])
                    .format(na_rep="—")
                    .set_properties(**{"text-align": "center", "font-size": "13px"})
                    .set_properties(subset=["Fund"], **{"text-align": "left", "font-weight": "500"})
                )
                st.dataframe(styled, use_container_width=True, height=600, hide_index=True)

                with st.expander("📖 How the Risk-Adjusted Momentum Strategy Works"):
                    st.markdown("""
                    Instead of just buying the fund with the highest recent returns (which might just be recklessly volatile), this engine divides **Absolute Return** by **Annualised Volatility**. 
                    
                    * **6M / Vol Ratio:** Evaluates short-term trend efficiency. A score of 1.5 means for every 1% of volatility, the fund generated 1.5% in returns over 6 months.
                    * **1Y / Vol Ratio:** Evaluates mid-term trend efficiency.
                    
                    This allows you to dynamically rank **all funds** (both 10-year veterans and 8-month-old emerging funds) on an absolutely level playing field based purely on current price velocity and risk management.
                    """)

        # Download
        all_funds = pd.concat([est, mom, too_young], ignore_index=True) if len(mom) > 0 else est
        buf = io.BytesIO()
        all_funds.round(2).to_excel(buf, index=False, engine="openpyxl")
        st.download_button("📥 Download all data", buf.getvalue(), "smallcap_quant_dual.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    # ═══════════════════════════════════
    # TAB 2: FUND DEEP-DIVE
    # ═══════════════════════════════════
    with tab_deep:
        # Use Momentum DF to ensure all valid funds are searchable
        fund_list = mom.sort_values("Rank", na_position="last")["Fund"].tolist()
        selected = st.selectbox("Search for a fund", fund_list, format_func=short)

        # Pull data from both engines
        r_mom = mom[mom["Fund"] == selected].iloc[0]
        r_est = est[est["Fund"] == selected].iloc[0] if selected in est["Fund"].values else None

        st.subheader(short(selected))

        # Header Stats
        colA, colB, colC = st.columns(3)
        with colA:
            st.markdown(f"**Age:** {r_mom['Track_Yrs']:.1f} Years")
        with colB:
            if r_est is not None:
                st.markdown(f"**🏛️ Established Score:** {r_est['Score']:.1f}/100  (Rank #{int(r_est['Rank'])})")
            else:
                st.markdown("**🏛️ Established Score:** Not old enough (< 3 Yrs)")
        with colC:
            st.markdown(f"**🚀 Momentum Score:** {r_mom['Score']:.1f}/100  (Rank #{int(r_mom['Rank'])})")

        st.divider()

        st.markdown("#### Performance vs Risk Metrics")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("6M Return", fmt(r_mom['Ret_6M'], 1, "%"))
        c2.metric("1Y Return", fmt(r_mom['Ret_1Y'], 1, "%"))
        c3.metric("Annualised Volatility", fmt(r_mom['Vol'], 1, "%"))
        c4.metric("Sortino Ratio", fmt(r_mom['Sortino'], 2))
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("6M Momentum / Vol", fmt(r_mom['Mom_6M_RA'], 2))
        c2.metric("1Y Momentum / Vol", fmt(r_mom['Mom_1Y_RA'], 2))
        c3.metric("Downside Capture", fmt(r_mom['Down_Cap'], 0, "%") if pd.notna(r_mom['Down_Cap']) else "—")
        c4.metric("Max Drawdown", fmt(r_mom['Max_DD'], 1, "%"))

        st.divider()

        # NAV Chart
        st.markdown("#### NAV vs Nifty 500 TRI")
        fn = nav[["Date", selected, "Benchmark"]].dropna(subset=[selected]).copy()
        fn = fn.rename(columns={selected: "NAV"})
        if not fn.empty and fn.iloc[0]["Benchmark"] > 0:
            fn["Bench"] = fn["Benchmark"] * (fn.iloc[0]["NAV"] / fn.iloc[0]["Benchmark"])
        else:
            fn["Bench"] = fn["Benchmark"]
        fn["Peak"] = fn["NAV"].cummax()
        fn["DD"] = (fn["NAV"] - fn["Peak"]) / fn["Peak"] * 100

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06, row_heights=[0.65, 0.35])
        fig.add_trace(go.Scatter(x=fn["Date"], y=fn["NAV"], fill="tozeroy", fillcolor="rgba(59,130,246,0.06)",
                                  line=dict(color="#3b82f6", width=1.5), name="Fund"), row=1, col=1)
        fig.add_trace(go.Scatter(x=fn["Date"], y=fn["Bench"],
                                  line=dict(color="#94a3b8", width=1.5, dash="dot"), name="Nifty 500 TRI"), row=1, col=1)
        fig.add_trace(go.Scatter(x=fn["Date"], y=fn["DD"], fill="tozeroy", fillcolor="rgba(239,68,68,0.08)",
                                  line=dict(color="#ef4444", width=1), name="Drawdown"), row=2, col=1)
        fig.update_layout(height=420, margin=dict(l=50, r=20, t=20, b=30), hovermode="x unified",
                          legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        fig.update_xaxes(gridcolor="rgba(0,0,0,0.05)")
        fig.update_yaxes(gridcolor="rgba(0,0,0,0.05)")
        st.plotly_chart(fig, use_container_width=True)

        # Peer Momentum Map
        st.markdown("#### Momentum Efficiency Map")
        cdf2 = mom.dropna(subset=["Mom_6M_RA", "Mom_1Y_RA"])
        fig2 = go.Figure()
        fig2.add_hline(y=1.0, line=dict(color="rgba(0,0,0,0.1)", dash="dash", width=1))
        fig2.add_vline(x=1.0, line=dict(color="rgba(0,0,0,0.1)", dash="dash", width=1))
        fig2.add_shape(type="rect", x0=1.0, x1=max(cdf2["Mom_6M_RA"].max(), 2), y0=1.0, y1=max(cdf2["Mom_1Y_RA"].max(), 2),
                       fillcolor="rgba(34,197,94,0.05)", line=dict(width=0))
        fig2.add_trace(go.Scatter(
            x=cdf2["Mom_6M_RA"], y=cdf2["Mom_1Y_RA"], mode="markers",
            marker=dict(
                size=[18 if f == selected else 9 for f in cdf2["Fund"]],
                color=["#3b82f6" if f == selected else "#d1d5db" for f in cdf2["Fund"]],
            ),
            text=[short(f) for f in cdf2["Fund"]],
            hovertemplate="<b>%{text}</b><br>6M/Vol: %{x:.2f}<br>1Y/Vol: %{y:.2f}<extra></extra>",
        ))
        fig2.update_layout(xaxis_title="6M Momentum / Volatility (Higher = Better)",
                           yaxis_title="1Y Momentum / Volatility",
                           height=400, margin=dict(l=50, r=20, t=20, b=50), showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)


if __name__ == "__main__":
    main()
