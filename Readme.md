# 🎯 Alpha Zero: The Indicator Challenge

> **Test if technical indicators can predict profitable trades**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

Alpha Zero is a rigorous testing framework for evaluating whether technical indicators and market features can consistently identify profitable trading opportunities. Using clustering and forward-scan backtesting, it challenges the myth that technical analysis can predict future price movements.

---

## 🌿 Branch Information

This repository has two main branches:

- **`streamlit`**: Demo version for trying the app on Streamlit Community Cloud (limited to 50k rows, 60 candle max lookahead, 4 features max)
- **`unlimited`** (recommended): Full-featured version without constraints - use this for serious research and local development

Clone the unlimited branch for maximum flexibility:
```bash
git clone -b unlimited https://github.com/ashoktzr/Alpha_Zero.git
```

For detailed branch differences, see [BRANCHES.md](BRANCHES.md).

---

## 💡 The Challenge

**Can you find a feature combination that achieves >50% precision with TP/SL > 1?**

Most traders believe technical indicators can predict profitable trades. This tool lets you test that hypothesis using:
- Machine learning clustering (K-Means/HDBSCAN)
- Rigorous historical simulation
- Statistical analysis of win rates

**Spoiler**: With realistic reward/risk ratios, achieving consistent profitability is virtually impossible due to market randomness.

---

## ✨ Key Features

### 📊 **Feature Engineering**
- **Technical Indicators**: RSI, MACD, Bollinger Bands, ATR, Stochastic, Williams %R, CCI, ADX
- **Market Structure**: Pivot distances, MA angles, autocorrelation, log returns
- **Configurable Parameters**: Customize all indicator settings

### 🧠 **Clustering**
- **Methods**: K-Means, HDBSCAN
- **Scaling Options**: Standard, MinMax, Robust, Rolling Z-Score, None
- **PCA Visualization**: 2D projection of clusters
- **Unlimited features**: No limit on feature selection (unlimited branch)

### 🔍 **Forward Scan**
- **Dual-direction simulation**: Both long and short analyzed simultaneously
- **Configurable lookahead**: Prevent data leakage
- **Visual verification**: Inspect individual trades

### 📈 **Results Analysis**
cd Alpha_Zero

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run the App

```bash
streamlit run app/gui_streamlit.py
```

### Data Options

**Option 1: Download from Binance (Built-in)**
- Select "Download from Binance" in the app
- Choose symbol, timeframe, days back
- Data fetched automatically

**Option 2: Upload CSV**
- Required columns: `timestamp`, `open`, `high`, `low`, `close`, `volume`
- Works best with Binance-format data

---

## 📖 How It Works

### 1. **Data & Indicators**
Load OHLCV data and compute technical indicators

### 2. **Feature Selection**
Choose features from:
- **Indicators**: RSI, MACD, Stochastic, CCI, ADX, etc.
- **Structure**: Log returns, pivots, MA angles, volume

### 3. **Clustering**
Group similar market conditions using K-Means or HDBSCAN
- **Why scaling?** Prevents feature dominance (volume vs RSI)
- **Visualization**: See if clusters are well-separated
- **Green dots** = TP hit first
- **Red dots** = SL hit first
- **Gray dots** = Neither hit
- **Spatial separation** = features working to separate wins from losses
- **Overlapping colors** = random outcomes

![Cluster PCA Visualization](docs/screenshot_cluster_pca.png)
*PCA projection showing how market conditions group into distinct clusters*

![Outcome Distribution](docs/screenshot_outcome_distribution.png)
*Trade outcome distribution showing True Positives (wins) vs False Positives (losses/no action)*

### Trade Inspection
![Trade Inspection](docs/screenshot_trade_inspection.png)
*Individual trade visualization with entry, TP, and SL levels for both Long and Short directions*

### Success Criteria
Find a cluster with:
- **Precision > 50%** (win rate)
- **Coverage ≥ 3%** (enough trades)
- **TP/SL > 1** (realistic reward/risk)

---

## 📁 Project Structure

```
Alpha_Zero/
│
├── app/
│   ├── gui_streamlit.py       # Main entry point
│   ├── pipeline.py            # Core orchestration
│   ├── indicators.py          # Technical indicators
│   ├── features.py            # Feature engineering
│   ├── cluster_analysis.py    # Clustering algorithms
│   ├── forward_scan.py        # TP/SL detection
│   ├── visualization.py       # Plotly charts
│   ├── config_types.py        # Configuration dataclasses
│   ├── profiles.py            # Profile management
│   └── tabs/
│       ├── pipeline_page.py   # Main workflow tab
│       ├── inspection_page.py # Trade inspection
│       └── sidebar_config.py  # Configuration panel
│
├── configs/                   # User profiles (gitignored)
├── requirements.txt
└── Readme.md
```

---

## 🤝 Contributing

Found a combination that beats the odds? Have improvements?

- Open issues on GitHub
- Submit pull requests
- Report bugs or feature requests

---

## ⚠️ Disclaimer

**This software is for educational and research purposes only.**

- Trading carries significant financial risk
- Past performance doesn't guarantee future results
- Technical indicators are backward-looking
- Always do your own research

The authors are not responsible for any financial losses.

---

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

---

**Challenge the myth. Test with data. Trade with caution.** 🎲
