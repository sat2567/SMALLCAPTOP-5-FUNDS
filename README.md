# SmallCap Fund Ranking Engine 🔬

Advanced multi-factor ranking system for Indian small-cap mutual funds using institutional-grade quantitative metrics.

## Metrics Used

| Factor | What It Measures | Why It Matters |
|--------|-----------------|----------------|
| **Rolling 3Y/5Y CAGR** | Average return across ALL possible 3/5-year windows | Smooths out lucky timing; shows consistency |
| **Benchmark Win Rate** | % of rolling windows where fund beat the benchmark | True alpha generation across market cycles |
| **Downside Capture** | How much the fund falls when market falls | Capital protection is #1 in small caps |
| **Upside Capture** | How much the fund gains when market rises | Asymmetric capture = compounding edge |
| **Sortino Ratio** | Return per unit of *downside* risk only | Better than Sharpe for skewed small-cap returns |
| **Maximum Drawdown** | Worst peak-to-trough loss ever | Tests risk management under real stress |
| **Calmar Ratio** | Annual return / Max Drawdown | Balances return ambition with crash severity |
| **Information Ratio** | Active return / Tracking Error vs benchmark | Consistency of outperformance |
| **Expense Ratio** | Annual fund charges | Direct drag on returns |

## Setup & Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Place data files in the same folder as `app.py`
- `smallcapfinalrank.xlsx` — NAV history
- `smallcap_aum.xlsx` — AUM data
- `smallcap_expense_ratio.xlsx` — Expense ratios

### 3. Run the app
```bash
streamlit run app.py
```

### 4. Deploy on Streamlit Cloud
1. Push all files to a GitHub repo
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo and set `app.py` as the main file
4. Deploy!

## How the Scoring Works

1. Each fund is evaluated on 10 quantitative factors
2. Each factor is converted to a **percentile rank** (0–100) across all funds
3. Percentiles are combined using **user-configurable weights** (sidebar sliders)
4. Final composite score determines the ranking

### Default Weight Distribution
- **Rolling Returns**: 35% (3Y: 15%, 5Y: 15%, Win Rate: 5%)
- **Capture Ratios**: 30% (Downside: 20%, Upside: 10%)
- **Risk-Adjusted**: 30% (Sortino: 15%, Max DD: 10%, Calmar: 5%)
- **Other**: 5% (Info Ratio: 3%, Expense: 2%)

### Benchmark
Equal-weight composite of all small-cap funds in the universe (proxy for Nifty Smallcap 250 TRI).

## App Tabs

| Tab | Contents |
|-----|----------|
| 🏆 Rankings | Full composite ranking table with all key metrics |
| 📊 Rolling Returns | Rolling 3Y/5Y CAGR range charts + benchmark win rates |
| 🛡️ Capture Ratios | Downside vs Upside scatter plot + capture ratio bars |
| 📈 Risk Metrics | Sortino distribution, max drawdown comparison, full risk table |
| 🔎 Fund Deep-Dive | Individual fund analysis with NAV chart, drawdown, radar profile |
