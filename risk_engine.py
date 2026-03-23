import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import t

# --- UI SETUP ---
st.set_page_config(page_title="Monte Carlo Simulator", layout="wide", page_icon="🎲")

# CSS to hide the automatic sidebar navigation and style metrics
st.markdown("""
    <style>
    /* Hides the auto-generated page list in the sidebar */
    section[data-testid="stSidebarNav"] {
        display: none !important;
    }

    .main { background-color: #0e1117; }
    .stMetric { 
        background-color: #161b22; 
        padding: 15px; 
        border-radius: 10px; 
        border: 1px solid #30363d; 
    }
    [data-testid="stMetricValue"] { font-size: 1.6rem !important; }

    /* Slight adjustment to top padding for the 'About' link */
    .block-container {
        padding-top: 3rem;
    }
    </style>
    """, unsafe_allow_html=True)


def human_format(num):
    """Helper to convert large numbers into readable strings with overflow protection."""
    magnitude = 0
    labels = ['', 'K', 'M', 'B', 'T']
    
    # Handle the "Too high to display" logic
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
        if magnitude >= len(labels):
            return "Too high to display!"
            
    return '{}{}'.format('{:,.1f}'.format(num).rstrip('0').rstrip('.'), labels[magnitude])

# Navigation link kept at the top of the main body
st.page_link("pages/about.py", label="About", icon="ℹ️")

st.title("🎲 Monte Carlo Portfolio Simulator")
st.caption("This tool uses past ticker data to simulate thousands of possible market paths to show how long term investment outcomes can vary due to uncertainty.")

# --- SIDEBAR: INPUTS ---
with st.sidebar:
    st.header("1. Portfolio Strategy")
    ticker_input = st.text_input("Input any tickers (comma separated)", "SPY, QQQ, GLD, BTC-USD")
    tickers = [ticker.strip().upper() for ticker in ticker_input.split(",") if ticker.strip()]

    weights = []
    cols = st.columns(2)
    for i, ticker in enumerate(tickers):
        w = cols[i % 2].number_input(f"{ticker} (%)", 0, 100, 100 // len(tickers), key=f"w_{ticker}")
        weights.append(w / 100)

    if abs(sum(weights) - 1.0) > 0.001:
        st.error(f"Total weight is {sum(weights) * 100:.1f}%. Please adjust to 100%.")
        st.stop()

    st.header("2. Simulation Params")
    init_inv = st.number_input("Initial Investment ($)", min_value=0, value=100000, step=1000)
    monthly_cont = st.number_input("Monthly Contribution ($)", min_value=0, value=500, step=100)
    years = st.slider("Horizon (Years)", 1, 40, 20)
    inf_rate = st.slider("Annual Inflation (%)", 0.0, 10.0, 3.0) / 100
    stress_test = st.toggle("Simulate Early Market Crash (SOR Risk)", value=False)


# --- DATA ENGINE ---
@st.cache_data(show_spinner=False)
def get_market_data(tickers):
    try:
        # 1. Download data (using 20y to get as much overlap as possible)
        df = yf.download(tickers, period="20y")['Close']
        
        # 2. Robust MultiIndex handling (yfinance returns different shapes)
        if isinstance(df, pd.Series):
            df = df.to_frame(name=tickers[0])
        
        # If columns are MultiIndex (Price, Ticker), flatten them
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(1)
            
        # 3. Calculate Daily Returns
        # log returns are better for additive Monte Carlo paths
        returns_df = np.log(df / df.shift(1))
        
        # 4. The "Overlap" Fix: Drop rows where ANY ticker is NaN
        # This ensures the correlation matrix is calculated on the same days
        clean_returns = returns_df.dropna()
            
        return clean_returns, df.iloc[-1], clean_returns.corr()

    except Exception as e:
        st.error(f"Data Engine Error: {e}")
        return None, None, None

# --- MONTE CARLO ENGINE ---
@st.cache_data(show_spinner=False)
def run_simulation(tickers, weights, years, init_inv, monthly_cont, stress_test):
    returns_df, last_prices, corr = get_market_data(tickers)
    if returns_df is None: return None

    w = np.array(weights)
    port_returns = returns_df.values @ w
    mu, std = np.mean(port_returns), np.std(port_returns)

    n_sims, n_days = 10000, years * 252
    scale = std * np.sqrt((3 - 2) / 3)  # Student-T adjustment
    shocks = t.rvs(df=3, loc=mu, scale=scale, size=(n_days, n_sims))
    ann_vol = (std * np.sqrt(252)) * 100

    if stress_test:
        # Sequence of Returns Risk: Force the worst 10% of days to happen at the start
        crash_len = int(n_days * 0.1)
        shocks[:crash_len, :] = np.sort(shocks[:crash_len, :], axis=0)

    daily_cont = monthly_cont / 21
    wealth_path = np.zeros((n_days, n_sims))
    current_wealth = np.full(n_sims, float(init_inv))

    growth_factors = np.exp(shocks)
    for d in range(n_days):
        current_wealth = (current_wealth * growth_factors[d]) + daily_cont
        wealth_path[d] = current_wealth

    return wealth_path, returns_df, last_prices, corr, ann_vol


# --- EXECUTION ---
if st.sidebar.button("🚀 Run Analysis"):
    res = run_simulation(tickers, weights, years, init_inv, monthly_cont, stress_test)
    if res: st.session_state.sim_data = res

if 'sim_data' in st.session_state:
    wealth_path, returns_df, last_prices, corr, ann_vol = st.session_state.sim_data
    n_days = wealth_path.shape[0]

    view_mode = st.radio("Currency View:", ["Nominal", "Real (Inflation Adjusted)"], horizontal=True)

    if "Real" in view_mode:
        discount_factors = (1 + inf_rate) ** (np.arange(n_days).reshape(-1, 1) / 252)
        display_path = wealth_path / discount_factors
    else:
        display_path = wealth_path

    t1, t2, t3 = st.tabs(["📈 Wealth Projection", "📊 Risk Analysis", "🧬 Correlations"])

    with t1:
        step = 5
        fast_path = display_path[::step, :]
        timeline = np.linspace(0, years, len(fast_path))

        fig = go.Figure()
        p_max = np.max(fast_path, axis=1)
        p_min = np.min(fast_path, axis=1)
        p90 = np.percentile(fast_path, 90, axis=1)
        p50 = np.percentile(fast_path, 50, axis=1)
        p10 = np.percentile(fast_path, 10, axis=1)

        fig.add_trace(
            go.Scatter(x=timeline, y=p_max, name="Absolute BEST Case", line=dict(color='gold', width=1, dash='dot')))
        fig.add_trace(
            go.Scatter(x=timeline, y=p90, name="90th Percentile", line=dict(color='rgba(0, 255, 150, 0.4)', width=1)))
        fig.add_trace(go.Scatter(x=timeline, y=p50, name="Median (Most Likely)", line=dict(color='#00d4ff', width=3)))
        fig.add_trace(
            go.Scatter(x=timeline, y=p10, name="10th Percentile", line=dict(color='rgba(255, 80, 80, 0.4)', width=1)))
        fig.add_trace(go.Scatter(x=timeline, y=p_min, name="Absolute WORST Case", line=dict(color='crimson', width=2)))

        is_log = st.toggle("Use Log Scale",
                           help="Visualizes exponential growth as a straight line. Better for large values.")
        fig.update_layout(template="plotly_dark", height=600, xaxis_title="Years", yaxis_title="Value ($)",
                          hovermode="x unified", yaxis_type="log" if is_log else "linear")
        st.plotly_chart(fig, use_container_width=True)

    with t2:
        final_wealth = display_path[-1, :]
        st.subheader("Final Wealth Distribution")
        fig_hist = px.histogram(final_wealth, nbins=100, template="plotly_dark", color_discrete_sequence=['#00d4ff'])
        st.plotly_chart(fig_hist, use_container_width=True)

        worst_path = display_path[:, np.argmin(display_path[-1, :])]
        peak = np.maximum.accumulate(worst_path)
        drawdown = (worst_path - peak) / peak
        max_dd = np.min(drawdown) * 100
        st.error(f"⚠️ **Worst Case Drawdown:** This simulation faced a peak-to-trough drop of **{max_dd:.1f}%**.")

    with t3:
        st.subheader("Historical Correlation Matrix")
        fig_corr = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", template="plotly_dark")
        st.plotly_chart(fig_corr, use_container_width=True)

    # --- METRICS BAR ---
    st.divider()
    total_contributions = init_inv + (monthly_cont * 12 * years)

    m_col1, m_col2, m_col3 = st.columns(3)

    with m_col1:
        st.metric("Top 10%", f"${human_format(np.percentile(final_wealth, 90))}")
        st.metric("Bottom 10%", f"${human_format(np.percentile(final_wealth, 10))}")

    with m_col2:
        st.metric("Median (50th %)", f"${human_format(np.percentile(final_wealth, 50))}")
        st.metric("Total Capital Invested", f"${human_format(total_contributions)}")

    with m_col3:
        st.metric("Annualized Volatility", f"{ann_vol:.2f}%")
        success_rate = (display_path[-1, :] > total_contributions).mean() * 100
        st.metric("Success Prob.", f"{success_rate:.1f}%")

else:

    st.info("👈 Enter your portfolio details and click 'Run Analysis' to begin.")

