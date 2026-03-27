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
    for fund in fund_names:
        fd = nav[["Date", fund]].dropna()
        if len(fd) < 120:
            continue

        fr = mret[fund].dropna()
        ci = fr.index.intersection(br.dropna().index)
        if len(ci) < 6:
            continue
        fr, brc = fr.loc[ci], br.loc[ci]

        mn = fd.set_index("Date")[fund].resample("ME").last().dropna()
        cagrs_5y = [((mn.iloc[i] / mn.iloc[i - 60]) ** (1 / 5) - 1) * 100 for i in range(60, len(mn))]
        r5_mean = np.mean(cagrs_5y) if cagrs_5y else None
        cagrs_3y = [((mn.iloc[i] / mn.iloc[i - 36]) ** (1 / 3) - 1) * 100 for i in range(36, len(mn))]
        r3_mean = np.mean(cagrs_3y) if cagrs_3y else None

        up_m = brc[brc > 0]
        dn_m = brc[brc < 0]
        up_cap = (fr.loc[up_m.index].mean() / up_m.mean()) * 100 if len(up_m) > 3 else None
        down_cap = (fr.loc[dn_m.index].mean() / dn_m.mean()) * 100 if len(dn_m) > 3 else None

        prices = fd[fund].values
        peaks = np.maximum.accumulate(prices)
        dd_series = (prices - peaks) / peaks
        ulcer = np.sqrt(np.mean(dd_series ** 2)) * 100
        max_dd = dd_series.min() * 100

        ld = fd.iloc[-1]["Date"]
        rec_1y = fd[fd["Date"] >= ld - timedelta(days=365)]
        vol = rec_1y[fund].pct_change().dropna().std() * np.sqrt(252) * 100 if len(rec_1y) > 50 else None

        def get_ret(days):
            past = fd[fd["Date"] <= ld - timedelta(days=days)]
            if past.empty:
                return None
            return (prices[-1] / past.iloc[-1][fund] - 1) * 100

        ret_6m = get_ret(180)
        ret_1y = get_ret(365)

        mom_6m = ret_6m / np.sqrt(vol) if (vol and vol > 0 and ret_6m is not None) else None
        mom_1y = ret_1y / np.sqrt(vol) if (vol and vol > 0 and ret_1y is not None) else None

        results.append({
            "Fund": fund, "Track_Yrs": len(fd) / 252,
            "Roll_3Y": r3_mean, "Roll_5Y": r5_mean, "Up_Cap": up_cap, "Down_Cap": down_cap,
            "Cap_Ratio": (up_cap / down_cap if (up_cap and down_cap and down_cap != 0) else None),
            "Ulcer_Index": ulcer, "Max_DD": max_dd,
            "Ret_6M": ret_6m, "Ret_1Y": ret_1y, "Vol": vol,
            "Mom_6M_RA": mom_6m, "Mom_1Y_RA": mom_1y,
        })

    return pd.DataFrame(results).merge(aum_latest, on="Fund", how="left")


# ═══════════════════════════════════════════
# RANKING
# ═══════════════════════════════════════════
def pctrank(s, asc=True):
    v = s.notna()
    if not v.any():
        return s
    r = s[v].rank(ascending=asc, pct=True) * 100
    out = pd.Series(np.nan, index=s.index)
    out[v] = r
    return out


def rank_funds(df, w_est, w_mom):
    est = df[df["Track_Yrs"] >= 3].copy()
    if not est.empty:
        est["S_R3"] = pctrank(est["Roll_3Y"])
        est["S_R5"] = pctrank(est["Roll_5Y"])
        est["S_UC"] = pctrank(est["Up_Cap"])
        est["S_DC"] = pctrank(est["Down_Cap"], False)
        est["S_UI"] = pctrank(est["Ulcer_Index"], False)
        est["Score"] = (est["S_R3"].fillna(0) * w_est[0] + est["S_R5"].fillna(0) * w_est[1] +
                        est["S_UC"].fillna(0) * w_est[2] + est["S_DC"].fillna(0) * w_est[3] +
                        est["S_UI"].fillna(0) * w_est[4]) / sum(w_est)
        est["Rank"] = est["Score"].rank(ascending=False, method="min")

    mom = df[df["Mom_6M_RA"].notna()].copy()
    if not mom.empty:
        mom["S_M6"] = pctrank(mom["Mom_6M_RA"])
        mom["S_M1"] = pctrank(mom["Mom_1Y_RA"])
        mom["Score"] = (mom["S_M6"].fillna(0) * w_mom[0] + mom["S_M1"].fillna(0) * w_mom[1]) / sum(w_mom)
        mom["Rank"] = mom["Score"].rank(ascending=False, method="min")

    return est, mom


# ═══════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════
def main():
    nav, fund_names, aum_latest, b_name = load_data()
    df_raw = compute_all(nav, fund_names, aum_latest)

    try:
        sector_data = load_sectors()
        has_sectors = True
    except Exception:
        has_sectors = False

    st.title("🚀 SmallCap Dual-Engine")

    # Sidebar
    with st.sidebar:
        st.subheader("🏛️ Compounder Weights")
        w_cagr3 = st.slider("3Y Rolling CAGR", 0, 100, 20)
        w_cagr5 = st.slider("5Y Rolling CAGR", 0, 100, 25)
        w_up = st.slider("Upside Capture", 0, 100, 15)
        w_down = st.slider("Downside Capture", 0, 100, 25)
        w_ulcer = st.slider("Ulcer Index", 0, 100, 15)

        st.divider()
        st.subheader("🏎️ Momentum Weights")
        m6 = st.slider("6M Dampened RA", 0, 100, 50)
        m1 = st.slider("1Y Dampened RA", 0, 100, 50)

    est, mom = rank_funds(df_raw, [w_cagr3, w_cagr5, w_up, w_down, w_ulcer], [m6, m1])

    # ── Tabs ──
    if has_sectors:
        tab_quant, tab_sector, tab_consensus = st.tabs([
            "📊 Quantitative Rankings",
            "🔬 Fund Sector Flow",
            "🔎 Sector Consensus",
        ])
    else:
        tab_quant = st.tabs(["📊 Quantitative Rankings"])[0]

    # ═══════════════════════════════════
    # TAB 1: QUANTITATIVE RANKINGS
    # ═══════════════════════════════════
    with tab_quant:
        view = st.radio("Ranking Engine", [
            "🏛️ Established Compounders (Funds > 3 Yrs)",
            "🏎️ Momentum Efficiency"
        ], horizontal=True)

        target = est if "Established" in view else mom

        if target.empty:
            st.warning("No funds qualify for this engine.")
        else:
            disp = target.sort_values("Rank").copy()

            if "Established" in view:
                cols = ["Rank", "Fund", "Score", "Roll_3Y", "Roll_5Y", "Up_Cap", "Down_Cap", "Cap_Ratio", "Ulcer_Index", "AUM"]
                names = ["Rank", "Fund", "Score", "3Y CAGR", "5Y CAGR", "Up Cap%", "Down Cap%", "Up/Down", "Ulcer Index", "AUM (Cr)"]
            else:
                cols = ["Rank", "Fund", "Score", "Ret_6M", "Ret_1Y", "Vol", "Up_Cap", "Down_Cap", "Cap_Ratio", "AUM"]
                names = ["Rank", "Fund", "Score", "6M Ret%", "1Y Ret%", "Vol%", "Up Cap%", "Down Cap%", "Up/Down", "AUM (Cr)"]

            disp = disp[cols].copy()
            disp.columns = names
            disp["Fund"] = disp["Fund"].apply(short)

            num_cols = [c for c in names if c not in ("Rank", "Fund", "AUM (Cr)", "Up/Down")]

            def c_dc(val):
                if pd.isna(val): return ""
                if val < 50: return "color:#16a34a;font-weight:700;"
                if val < 80: return "color:#16a34a;"
                if val < 100: return "color:#ca8a04;"
                return "color:#dc2626;font-weight:600;"

            def c_uc(val):
                if pd.isna(val): return ""
                if val > 120: return "color:#16a34a;font-weight:700;"
                if val > 100: return "color:#16a34a;"
                return "color:#ca8a04;"

            def c_cr(val):
                if pd.isna(val): return ""
                if val > 1.3: return "background-color:#dcfce7;color:#166534;font-weight:700;"
                if val > 1.1: return "background-color:#e0f2fe;color:#075985;"
                if val > 1.0: return "color:#ca8a04;"
                return "background-color:#fecaca;color:#991b1b;"

            styled = (disp.style
                .background_gradient(subset=["Score"], cmap="RdYlGn")
                .map(c_dc, subset=["Down Cap%"])
                .map(c_uc, subset=["Up Cap%"])
                .map(c_cr, subset=["Up/Down"])
                .format("{:.0f}", subset=["Rank"])
                .format("{:.1f}", subset=num_cols)
                .format("{:.2f}", subset=["Up/Down"])
                .format("{:.0f}", subset=["AUM (Cr)"])
                .format(na_rep="—")
            )
            st.dataframe(styled, use_container_width=True, height=600, hide_index=True)

            if "Established" in view:
                st.caption("Benchmark: BSE SmallCap TRI  ·  Upside/Downside Capture measured against benchmark  ·  Only funds with 3+ years track record")

    # ═══════════════════════════════════
    # TAB 2: FUND SECTOR FLOW
    # ═══════════════════════════════════
    if has_sectors:
        with tab_sector:
            st.markdown("## Fund Strategy Profiles & Sector Flow")

            # ── Complete Strategy Table ──
            st.markdown("#### All Fund Profiles at a Glance")

            profile_rows = []
            for fund in ALL_QUAL_FUNDS:
                info_p = ALL_INFO.get(fund, {})
                tags = []
                if fund in PROVEN:
                    tags.append("🏛️ Compounder")
                if fund in MOMENTUM_INFO:
                    tags.append("🏎️ Momentum")
                profile_rows.append({
                    "Fund": short(fund),
                    "Category": " + ".join(tags),
                    "PE": info_p.get("PE"),
                    "PB": info_p.get("PB"),
                    "Valuation Stance": info_p.get("Stance", ""),
                })

            pdf = pd.DataFrame(profile_rows)

            def c_pe(val):
                if pd.isna(val): return ""
                if val < 25: return "background-color:#dcfce7;color:#166534;font-weight:600;"
                if val < 32: return "background-color:#fef9c3;color:#854d0e;"
                return "background-color:#fecaca;color:#991b1b;font-weight:600;"

            def c_pb(val):
                if pd.isna(val): return ""
                if val < 3: return "background-color:#dcfce7;color:#166534;font-weight:600;"
                if val < 4.2: return "background-color:#fef9c3;color:#854d0e;"
                return "background-color:#fecaca;color:#991b1b;"

            def c_cat(val):
                if "Compounder" in str(val) and "Momentum" in str(val):
                    return "background-color:#ede9fe;color:#5b21b6;"
                if "Compounder" in str(val):
                    return "background-color:#dcfce7;color:#166534;"
                if "Momentum" in str(val):
                    return "background-color:#e0f2fe;color:#075985;"
                return ""

            styled_p = (pdf.style
                .map(c_pe, subset=["PE"])
                .map(c_pb, subset=["PB"])
                .map(c_cat, subset=["Category"])
                .format({"PE": "{:.1f}x", "PB": "{:.2f}x"}, na_rep="—")
                .set_properties(**{"text-align": "center", "font-size": "13px"})
                .set_properties(subset=["Fund"], **{"text-align": "left", "font-weight": "600"})
            )
            st.dataframe(styled_p, use_container_width=True, height=420, hide_index=True)

            st.caption("PE color: 🟢 Value (<25x) · 🟡 Growth (25-32x) · 🔴 Premium (>32x)  |  PB color: 🟢 <3x · 🟡 3-4.2x · 🔴 >4.2x")

            with st.expander("ℹ️ What do the Valuation Stances mean?"):
                st.markdown("""
| Stance | Meaning |
|---|---|
| **Deep Value** | Buying stocks at significant discount to intrinsic value. Lowest PE/PB in the group. Willing to wait years for re-rating. |
| **GARP** | Growth At Reasonable Price — wants growth but refuses to overpay. Balances earnings momentum with valuation discipline. |
| **Operating Leverage** | Targeting companies where small revenue increases lead to outsized profit jumps — typically capital-intensive businesses with high fixed costs. |
| **Concentrated Core** | High-conviction portfolio in 2-3 sector themes. Less diversified, but deeper research per holding. |
| **Quality / Hedged** | Combines high-quality growth holdings with defensive positions (FMCG, consumer staples) to cushion downside. |
| **Risk-Adjusted Growth** | Growth-oriented but with strict risk controls — no single sector dominates, wide diversification. |
| **Balanced Momentum** | Blends value and momentum — rotates between defensive and cyclical sectors based on market regime. |
| **Terminal Value Premium** | Paying premium valuations for companies with long-duration growth — betting that future earnings justify today's high PE. |
| **High-Growth Momentum** | Chasing the strongest recent performers. Highest PE/PB, highest churn. Works in bull markets, risky in corrections. |
| **Ultra-Growth Premium** | Highest conviction growth bets at the most expensive valuations. Concentrated in structural themes like healthcare/consumer. |
| **Diversified Growth** | Broad exposure across multiple sectors and themes. No extreme bets. Moderate PE/PB. |
""")

            st.divider()

            # ── Fund Dropdown ──
            st.markdown("#### Sector Flow — Pick a Fund")

            selected = st.selectbox("Select Fund", ALL_QUAL_FUNDS, format_func=short)

            info = ALL_INFO.get(selected, {})
            fd = sector_data[sector_data["Fund"] == selected].copy()

            # Category tag
            tags = []
            if selected in PROVEN:
                tags.append("🏛️ Proven Compounder")
            if selected in MOMENTUM_INFO:
                tags.append("🏎️ Momentum Leader")
            cat = " + ".join(tags)

            # Header
            st.markdown(f"### {short(selected)}")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Category", cat)
            c2.metric("PE Ratio", f"{info.get('PE', '—')}x")
            c3.metric("PB Ratio", f"{info.get('PB', '—')}x")
            c4.metric("Stance", info.get("Stance", "—"))

            # Significant sectors only
            fd_sig = fd[fd[MONTHS].max(axis=1) >= 1.5].copy()
            fd_sig = fd_sig.sort_values("Feb_26", ascending=False)

            if len(fd_sig) == 0:
                st.warning("No sector data available.")
            else:
                # ── Line chart: sector flow ──
                st.markdown("#### Sector allocation over time")
                top_n = fd_sig.head(8)
                colors = ["#3b82f6", "#22c55e", "#f59e0b", "#ef4444", "#8b5cf6",
                          "#06b6d4", "#f97316", "#ec4899"]

                fig = go.Figure()
                for i, (_, row) in enumerate(top_n.iterrows()):
                    vals = [row[m] if pd.notna(row[m]) else 0 for m in MONTHS]
                    fig.add_trace(go.Scatter(
                        x=MONTH_LABELS, y=vals, mode="lines",
                        name=row["Sector"],
                        line=dict(width=2.5, color=colors[i % len(colors)]),
                        hovertemplate="%{y:.1f}%<extra>%{fullData.name}</extra>",
                    ))
                fig.update_layout(
                    height=380, margin=dict(l=50, r=20, t=20, b=40),
                    yaxis_title="Allocation %", hovermode="x unified",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                                font=dict(size=11)),
                )
                fig.update_xaxes(gridcolor="rgba(0,0,0,0.05)")
                fig.update_yaxes(gridcolor="rgba(0,0,0,0.05)")
                st.plotly_chart(fig, use_container_width=True)

                # ── Heatmapped table ──
                st.markdown("#### Month-by-month breakdown")
                table_rows = []
                for _, row in fd_sig.iterrows():
                    jan, feb = row["Jan_25"], row["Feb_26"]
                    if pd.notna(jan) and pd.notna(feb):
                        diff = feb - jan
                        if diff > 3: trend = f"↑↑ +{diff:.1f}pp"
                        elif diff > 1: trend = f"↑ +{diff:.1f}pp"
                        elif diff < -3: trend = f"↓↓ {diff:.1f}pp"
                        elif diff < -1: trend = f"↓ {diff:.1f}pp"
                        else: trend = f"→ {diff:+.1f}pp"
                    elif pd.notna(feb):
                        trend = "New"
                    else:
                        trend = "—"

                    table_rows.append({
                        "Sector": row["Sector"],
                        "Jan 2025": round(jan, 1) if pd.notna(jan) else None,
                        "Jun 2025": round(row["Jun_25"], 1) if pd.notna(row["Jun_25"]) else None,
                        "Sep 2025": round(row["Sep_25"], 1) if pd.notna(row["Sep_25"]) else None,
                        "Dec 2025": round(row["Dec_25"], 1) if pd.notna(row["Dec_25"]) else None,
                        "Feb 2026": round(feb, 1) if pd.notna(feb) else None,
                        "12M Trend": trend,
                    })

                tdf = pd.DataFrame(table_rows)

                def c_alloc(val):
                    if pd.isna(val): return ""
                    if val >= 15: return "background-color:#2563eb;color:white;font-weight:700;"
                    if val >= 10: return "background-color:#60a5fa;color:white;"
                    if val >= 7: return "background-color:#93c5fd;"
                    if val >= 4: return "background-color:#bfdbfe;"
                    if val >= 2: return "background-color:#dbeafe;"
                    return ""

                def c_trend(val):
                    if "↑↑" in str(val): return "background-color:#166534;color:white;font-weight:700;"
                    if "↑" in str(val): return "background-color:#dcfce7;color:#166534;"
                    if "↓↓" in str(val): return "background-color:#991b1b;color:white;font-weight:700;"
                    if "↓" in str(val): return "background-color:#fecaca;color:#991b1b;"
                    return "color:#9ca3af;"

                alloc_cols = ["Jan 2025", "Jun 2025", "Sep 2025", "Dec 2025", "Feb 2026"]
                styled = (tdf.style
                    .map(c_alloc, subset=alloc_cols)
                    .map(c_trend, subset=["12M Trend"])
                    .format(na_rep="—")
                    .set_properties(**{"text-align": "center", "font-size": "13px"})
                    .set_properties(subset=["Sector"], **{"text-align": "left", "font-weight": "600"})
                    .set_properties(subset=["12M Trend"], **{"font-weight": "600"})
                )
                st.dataframe(styled, use_container_width=True, height=500, hide_index=True)

                # ── Biggest moves ──
                st.markdown("#### Biggest moves (Jan 2025 → Feb 2026)")
                moves = []
                for _, row in fd_sig.iterrows():
                    if pd.notna(row["Jan_25"]) and pd.notna(row["Feb_26"]):
                        diff = row["Feb_26"] - row["Jan_25"]
                        if abs(diff) > 1.5:
                            moves.append((row["Sector"], diff, row["Feb_26"]))
                moves.sort(key=lambda x: -abs(x[1]))

                if moves:
                    cols = st.columns(min(len(moves), 5))
                    for i, (sector, diff, current) in enumerate(moves[:5]):
                        with cols[i]:
                            st.metric(sector, f"{current:.1f}%", f"{'+' if diff > 0 else ''}{diff:.1f}pp")
                else:
                    st.caption("No major sector shifts (>1.5pp) detected.")

        # ═══════════════════════════════════
        # TAB 3: SECTOR CONSENSUS
        # ═══════════════════════════════════
        with tab_consensus:
            st.markdown("## Sector Consensus")
            st.markdown("Where are all 10 funds converging and diverging?")

            top_sectors = (sector_data[sector_data["Fund"].isin(ALL_QUAL_FUNDS)]
                            .groupby("Sector")["Feb_26"].mean()
                            .sort_values(ascending=False).head(15).index.tolist())

            cons = []
            for sector in top_sectors:
                allocs, changes = [], []
                cnt = 0
                for fund in ALL_QUAL_FUNDS:
                    fd = sector_data[(sector_data["Fund"] == fund) & (sector_data["Sector"] == sector)]
                    if len(fd) > 0 and pd.notna(fd.iloc[0]["Feb_26"]):
                        allocs.append(fd.iloc[0]["Feb_26"])
                        cnt += 1
                        if pd.notna(fd.iloc[0]["Jan_25"]):
                            changes.append(fd.iloc[0]["Feb_26"] - fd.iloc[0]["Jan_25"])

                avg_a = np.mean(allocs) if allocs else 0
                avg_c = np.mean(changes) if changes else 0
                if avg_c > 2: direction = "Strong Addition ↑↑"
                elif avg_c > 0.5: direction = "Adding ↑"
                elif avg_c < -2: direction = "Strong Reduction ↓↓"
                elif avg_c < -0.5: direction = "Trimming ↓"
                else: direction = "Stable →"

                cons.append({
                    "Sector": sector, "Avg Alloc%": round(avg_a, 1), "Funds": cnt,
                    "Avg 12M Change": round(avg_c, 1), "Direction": direction,
                })

            cdf = pd.DataFrame(cons)

            # Bar chart
            cdf_s = cdf.sort_values("Avg Alloc%", ascending=True)
            bar_colors = []
            for _, r in cdf_s.iterrows():
                ch = r["Avg 12M Change"]
                if ch > 2: bar_colors.append("#22c55e")
                elif ch > 0.5: bar_colors.append("#86efac")
                elif ch < -2: bar_colors.append("#ef4444")
                elif ch < -0.5: bar_colors.append("#fca5a5")
                else: bar_colors.append("#94a3b8")

            fig = go.Figure()
            fig.add_trace(go.Bar(
                y=cdf_s["Sector"], x=cdf_s["Avg Alloc%"], orientation="h",
                marker=dict(color=bar_colors),
                text=[f"{v:.1f}%" for v in cdf_s["Avg Alloc%"]],
                textposition="outside",
            ))
            fig.update_layout(height=420, margin=dict(l=10, r=40, t=10, b=30),
                              xaxis_title="Avg allocation %", showlegend=False)
            fig.update_xaxes(gridcolor="rgba(0,0,0,0.05)")
            st.plotly_chart(fig, use_container_width=True)
            st.caption("🟢 Funds adding · 🔴 Funds trimming · ⚪ Stable")

            # Consensus table
            def c_dir(val):
                if "Strong Addition" in str(val): return "background-color:#166534;color:white;font-weight:700;"
                if "Adding" in str(val): return "background-color:#dcfce7;color:#166534;"
                if "Strong Reduction" in str(val): return "background-color:#991b1b;color:white;font-weight:700;"
                if "Trimming" in str(val): return "background-color:#fecaca;color:#991b1b;"
                return "color:#9ca3af;"

            def c_ch(val):
                if pd.isna(val): return ""
                if val > 2: return "color:#16a34a;font-weight:700;"
                if val > 0: return "color:#16a34a;"
                if val < -2: return "color:#dc2626;font-weight:700;"
                if val < 0: return "color:#dc2626;"
                return ""

            styled = (cdf.style
                .map(c_dir, subset=["Direction"])
                .map(c_ch, subset=["Avg 12M Change"])
                .format(na_rep="—")
                .set_properties(**{"text-align": "center", "font-size": "13px"})
                .set_properties(subset=["Sector"], **{"text-align": "left", "font-weight": "600"})
            )
            st.dataframe(styled, use_container_width=True, height=500, hide_index=True)

            # Cross-fund heatmap
            st.markdown("#### Cross-fund allocation heatmap — Feb 2026")
            heat_sectors = top_sectors[:12]
            heat_data = []
            for sector in heat_sectors:
                row_d = {"Sector": sector}
                for fund in ALL_QUAL_FUNDS:
                    fd = sector_data[(sector_data["Fund"] == fund) & (sector_data["Sector"] == sector)]
                    val = fd.iloc[0]["Feb_26"] if len(fd) > 0 and pd.notna(fd.iloc[0]["Feb_26"]) else 0
                    row_d[short(fund)] = round(val, 1)
                heat_data.append(row_d)

            hdf = pd.DataFrame(heat_data)
            f_short = [short(f) for f in ALL_QUAL_FUNDS]
            z = hdf[f_short].values

            fig2 = go.Figure(data=go.Heatmap(
                z=z, x=f_short, y=hdf["Sector"],
                colorscale=[[0, "#f8fafc"], [0.2, "#dbeafe"], [0.4, "#93c5fd"],
                            [0.6, "#3b82f6"], [0.8, "#1d4ed8"], [1.0, "#1e3a5f"]],
                text=z, texttemplate="%{text:.1f}",
                textfont=dict(size=11),
                colorbar=dict(title="Alloc %", thickness=15),
            ))
            fig2.update_layout(
                height=440, margin=dict(l=10, r=10, t=10, b=10),
                xaxis=dict(tickangle=-45, tickfont=dict(size=11)),
                yaxis=dict(tickfont=dict(size=11), autorange="reversed"),
            )
            st.plotly_chart(fig2, use_container_width=True)

if __name__ == "__main__":
    main()
