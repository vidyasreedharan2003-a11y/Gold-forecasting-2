import streamlit as st
import numpy    as np
import pandas   as pd
import joblib
import warnings
warnings.filterwarnings("ignore")

from tensorflow.keras.models import load_model

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title = "gold Forecast",
    page_icon  = "📈",
    layout     = "wide"
)

# ── Constants ────────────────────────────────────────────────
MODEL_DIR = "gold_streamlit/model"
DATA_PATH = "gold_streamlit/data/gold.csv"
LOOKBACK  = 60
HORIZON   = 10
TARGET    = "Target_Return"


# ── Load artefacts (cached — loads only once per session) ────
@st.cache_resource
def load_artefacts():
    model         = load_model(f"{MODEL_DIR}/gold_bilstm_model.keras")
    scaler        = joblib.load(f"{MODEL_DIR}/gold_scaler.pkl")
    target_scaler = joblib.load(f"{MODEL_DIR}/gold_target_scaler.pkl")
    features      = (pd.read_csv(f"{MODEL_DIR}/gold_features.csv")
                       ["feature"].tolist())
    return model, scaler, target_scaler, features


# ── Feature pipeline (same as training pipeline) ─────────────
@st.cache_data
def load_and_prepare():
    """Load CSV → add indicators → convert to returns."""

    # ── Load ────────────────────────────────────────────────-
    df = pd.read_csv(DATA_PATH)
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
    df.set_index("Date", inplace=True)
    df.sort_index(inplace=True)
    df = df[["Open", "High", "Low", "Close", "Volume"]]
    df = df.apply(pd.to_numeric, errors="coerce")
    df["Volume"] = df["Volume"].replace(0, float("nan"))
    df.dropna(inplace=True)

    d = df.copy()

    # ── Trend ────────────────────────────────────────────────
    d["EMA_12"]      = d["Close"].ewm(span=12, adjust=False).mean()
    d["EMA_26"]      = d["Close"].ewm(span=26, adjust=False).mean()
    d["MACD"]        = d["EMA_12"] - d["EMA_26"]
    d["MACD_Signal"] = d["MACD"].ewm(span=9, adjust=False).mean()
    d["MACD_Hist"]   = d["MACD"] - d["MACD_Signal"]
    d["MA_10"]       = d["Close"].rolling(10).mean()
    d["MA_50"]       = d["Close"].rolling(50).mean()
    d["MA_Cross"]    = d["MA_10"] - d["MA_50"]

    # ── Momentum ─────────────────────────────────────────────
    delta         = d["Close"].diff()
    gain          = delta.clip(lower=0)
    loss          = -delta.clip(upper=0)
    rs            = (gain.rolling(14).mean() /
                     (loss.rolling(14).mean() + 1e-10))
    d["RSI"]      = 100 - (100 / (1 + rs))
    d["RSI_Norm"] = d["RSI"] / 100.0
    d["ROC_5"]    = d["Close"].pct_change(5)  * 100
    d["ROC_10"]   = d["Close"].pct_change(10) * 100
    d["ROC_20"]   = d["Close"].pct_change(20) * 100
    low14         = d["Low"].rolling(14).min()
    high14        = d["High"].rolling(14).max()
    d["Stoch_K"]  = (d["Close"] - low14) / (high14 - low14 + 1e-10) * 100
    d["Stoch_D"]  = d["Stoch_K"].rolling(3).mean()

    # ── Volatility ───────────────────────────────────────────
    d["BB_Mid"]   = d["Close"].rolling(20).mean()
    d["BB_Std"]   = d["Close"].rolling(20).std()
    d["BB_Upper"] = d["BB_Mid"] + 2 * d["BB_Std"]
    d["BB_Lower"] = d["BB_Mid"] - 2 * d["BB_Std"]
    d["BB_Width"] = (d["BB_Upper"] - d["BB_Lower"]) / (d["BB_Mid"]   + 1e-10)
    d["BB_Pct"]   = (d["Close"]   - d["BB_Lower"]) / (d["BB_Upper"] - d["BB_Lower"] + 1e-10)
    hl            = d["High"] - d["Low"]
    hc            = (d["High"] - d["Close"].shift(1)).abs()
    lc            = (d["Low"]  - d["Close"].shift(1)).abs()
    tr            = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    d["ATR"]      = tr.rolling(14).mean()
    d["ATR_Pct"]  = d["ATR"] / (d["Close"] + 1e-10)
    d["HV_20"]    = d["Close"].pct_change().rolling(20).std() * np.sqrt(252)

    # ── Volume ───────────────────────────────────────────────
    obv = [0]
    for i in range(1, len(d)):
        if   d["Close"].iloc[i] > d["Close"].iloc[i - 1]:
            obv.append(obv[-1] + d["Volume"].iloc[i])
        elif d["Close"].iloc[i] < d["Close"].iloc[i - 1]:
            obv.append(obv[-1] - d["Volume"].iloc[i])
        else:
            obv.append(obv[-1])
    d["OBV"]        = obv
    d["OBV_EMA"]    = pd.Series(obv, index=d.index).ewm(span=10).mean().values
    d["Vol_MA20"]   = d["Volume"].rolling(20).mean()
    d["Vol_Ratio"]  = d["Volume"] / (d["Vol_MA20"] + 1e-10)

    # ── Price structure ──────────────────────────────────────
    d["HL_Pct"]       = (d["High"] - d["Low"]) / (d["Close"] + 1e-10)
    d["Gap_Pct"]      = (d["Open"] - d["Close"].shift(1)) / (d["Close"].shift(1) + 1e-10)
    d["Upper_Shadow"] = (d["High"] - d[["Open","Close"]].max(axis=1)) / (d["Close"] + 1e-10)
    d["Lower_Shadow"] = (d[["Open","Close"]].min(axis=1) - d["Low"])  / (d["Close"] + 1e-10)

    # ── Stationary conversion ────────────────────────────────
    d["Open_Return"]   = d["Open"].pct_change()
    d["High_Return"]   = d["High"].pct_change()
    d["Low_Return"]    = d["Low"].pct_change()
    d["Close_Return"]  = d["Close"].pct_change()
    d["Volume_Log"]    = np.log(d["Volume"] / d["Volume"].shift(1))
    d["Target_Return"] = d["Close"].pct_change().shift(-1)

    drop = ["Open","High","Low","Close","Volume",
            "EMA_12","EMA_26","MA_10","MA_50",
            "BB_Mid","BB_Std","BB_Upper","BB_Lower",
            "ATR","OBV","Vol_MA20"]
    d.drop(columns=[c for c in drop if c in d.columns], inplace=True)
    d.dropna(inplace=True)

    return d, df   # d = stationary, df = raw (for last price)


# ── Forecast function ─────────────────────────────────────────
def run_forecast(model, scaler, target_scaler, features):
    df_stat, df_raw = load_and_prepare()

    all_cols    = features + [TARGET]
    df_clean    = df_stat[all_cols].dropna()
    full_scaled = scaler.transform(df_clean.values)   # transform only

    last_seq    = full_scaled[-LOOKBACK:, :-1]
    inp         = last_seq.reshape(1, LOOKBACK, len(features))

    pred_sc     = model.predict(inp, verbose=0)
    pred_ret    = target_scaler.inverse_transform(
                      pred_sc.reshape(-1, 1)).flatten()

    last_price  = float(df_raw["Close"].iloc[-1])
    last_date   = df_raw.index[-1]

    prices = [last_price]
    for r in pred_ret:
        prices.append(prices[-1] * (1 + r))
    pred_prices = np.array(prices[1:])

    future_dates = pd.bdate_range(
        start=last_date + pd.Timedelta(days=1),
        periods=HORIZON
    )

    result = pd.DataFrame({
        "Date"            : [d.strftime("%d %b %Y") for d in future_dates],
        "Predicted Close" : pred_prices.round(2),
        "Return"          : pred_ret.round(6),
        "Change %"        : (pred_ret * 100).round(3),
        "Direction"       : ["▲ Up" if r >= 0 else "▼ Down"
                             for r in pred_ret],
    })

    return result, last_price, last_date, df_raw

st.title("📈 gold — BiLSTM Forecast")
st.caption("Bidirectional LSTM · 25 Features · 10-Day Multi-Step Forecast")
st.divider()

# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    horizon_display = st.slider(
        "Forecast days to display", 1, HORIZON, HORIZON)
    show_chart   = st.checkbox("Show price chart",   value=True)
    show_returns = st.checkbox("Show returns chart", value=True)
    show_history = st.checkbox("Show price history", value=True)
    st.divider()
    st.caption("Model: gold_BiLSTM")
    st.caption(f"Lookback : {LOOKBACK} days")
    st.caption(f"Horizon  : {HORIZON} days")
    st.caption("Features : 25")

# ── Load artefacts ───────────────────────────────────────────
with st.spinner("Loading model artefacts..."):
    model, scaler, target_scaler, features = load_artefacts()

st.success(f"Model ready — {len(features)} features loaded", icon="✅")

# ── Run forecast button ──────────────────────────────────────
if st.button("🚀 Run Forecast", type="primary", use_container_width=True):

    with st.spinner("Running forecast pipeline..."):
        result, last_price, last_date, df_raw = run_forecast(
            model, scaler, target_scaler, features
        )

    # ── Metric cards ─────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)

    last_ret     = result["Return"].iloc[0]
    total_change = ((result["Predicted Close"].iloc[-1] - last_price)
                     / last_price * 100)
    up_days      = (result["Return"] >= 0).sum()

    col1.metric(
        "Last Known Close",
        f"₹{last_price:,.2f}",
        str(last_date.date())
    )
    col2.metric(
        "Day 1 Forecast",
        f"₹{result['Predicted Close'].iloc[0]:,.2f}",
        f"{result['Change %'].iloc[0]:+.2f}%"
    )
    col3.metric(
        f"Day {horizon_display} Forecast",
        f"₹{result['Predicted Close'].iloc[horizon_display-1]:,.2f}",
        f"{total_change:+.2f}% total"
    )
    col4.metric(
        "Up Days",
        f"{up_days} / {HORIZON}",
        "bullish" if up_days >= 6 else "bearish"
    )

    st.divider()

    # ── Forecast table ────────────────────────────────────────
    st.subheader("📋 Forecast Table")

    display_df = result.head(horizon_display).copy()

    def colour_direction(val):
        if "Up"   in str(val): return "color: #3FB950; font-weight: bold"
        if "Down" in str(val): return "color: #F85149; font-weight: bold"
        return ""

    def colour_change(val):
        try:
            return ("color: #3FB950" if float(val) >= 0
                    else "color: #F85149")
        except:
            return ""

    styled = (display_df.style
              .applymap(colour_direction, subset=["Direction"])
              .applymap(colour_change,    subset=["Change %", "Return"])
              .format({"Predicted Close": "₹{:,.2f}",
                       "Return"         : "{:+.4%}",
                       "Change %"       : "{:+.3f}%"}))
    st.dataframe(styled, use_container_width=True, hide_index=True)

    # ── Price chart ───────────────────────────────────────────
    if show_chart:
        st.subheader("📉 Forecast Price Chart")

        import matplotlib.pyplot as plt
        import matplotlib.dates  as mdates

        fig, ax = plt.subplots(figsize=(12, 4))
        fig.patch.set_facecolor("#0D1117")
        ax.set_facecolor("#161B22")

        # Recent 90 days historical
        recent = df_raw["Close"].iloc[-90:]
        ax.plot(recent.index, recent.values,
                color="#58A6FF", lw=1.5,
                label="Historical Close")

        # Forecast ribbon
        future_dates_plot = pd.bdate_range(
            start=df_raw.index[-1] + pd.Timedelta(days=1),
            periods=horizon_display
        )
        fp = result["Predicted Close"].values[:horizon_display]
        unc = np.array([abs(fp[i]) * (0.008 + i * 0.002)
                        for i in range(len(fp))])

        ax.plot(future_dates_plot, fp,
                color="#F78166", lw=2.5, ls="--",
                label=f"{horizon_display}-Day Forecast", zorder=5)
        ax.fill_between(future_dates_plot,
                        fp - unc, fp + unc,
                        color="#F78166", alpha=0.2)
        ax.axvline(df_raw.index[-1], color="#E3B341",
                   ls=":", lw=1.5, alpha=0.8)

        ax.tick_params(colors="#C9D1D9", labelsize=9)
        ax.set_ylabel("Price (₹)", color="#C9D1D9")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        plt.setp(ax.xaxis.get_majorticklabels(),
                 rotation=25, ha="right", color="#C9D1D9")
        for sp in ax.spines.values():
            sp.set_edgecolor("#30363D")
        ax.grid(ls="--", alpha=0.18, color="#6E7681")
        ax.legend(fontsize=9, facecolor="#161B22",
                  labelcolor="#C9D1D9")
        st.pyplot(fig, use_container_width=True)
        plt.close()

    # ── Returns bar chart ─────────────────────────────────────
    if show_returns:
        st.subheader("📊 Predicted Daily Returns")

        fig2, ax2 = plt.subplots(figsize=(12, 3))
        fig2.patch.set_facecolor("#0D1117")
        ax2.set_facecolor("#161B22")

        ret_vals = result["Change %"].values[:horizon_display]
        colors   = ["#3FB950" if v >= 0 else "#F85149"
                    for v in ret_vals]
        ax2.bar(range(1, len(ret_vals) + 1),
                ret_vals, color=colors, width=0.6, alpha=0.85)
        ax2.axhline(0, color="#6E7681", lw=0.8)

        for i, v in enumerate(ret_vals):
            ax2.text(i + 1, v + np.sign(v) * 0.01,
                     f"{v:+.2f}%", ha="center",
                     va="bottom" if v >= 0 else "top",
                     fontsize=8, color="#C9D1D9")

        ax2.set_xlabel("Forecast Day", color="#C9D1D9")
        ax2.set_ylabel("Return (%)",   color="#C9D1D9")
        ax2.set_xticks(range(1, len(ret_vals) + 1))
        ax2.tick_params(colors="#C9D1D9", labelsize=9)
        for sp in ax2.spines.values():
            sp.set_edgecolor("#30363D")
        ax2.grid(axis="y", ls="--", alpha=0.18, color="#6E7681")
        st.pyplot(fig2, use_container_width=True)
        plt.close()

    # ── Price history ─────────────────────────────────────────
    if show_history:
        st.subheader("🕰️ Full Price History")
        hist_df = df_raw[["Close"]].rename(
            columns={"Close": "Close Price (₹)"})
        st.line_chart(hist_df, color="#58A6FF",
                      use_container_width=True)

    # ── Download button ───────────────────────────────────────
    st.divider()
    csv = result.to_csv(index=False).encode("utf-8")
    st.download_button(
        label     = "⬇️ Download Forecast as CSV",
        data      = csv,
        file_name = "gold_forecast.csv",
        mime        = "text/csv",
        use_container_width=True
    )
