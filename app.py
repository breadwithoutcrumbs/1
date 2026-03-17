import streamlit as st

# --- PASSWORD PROTECTION ---
password = "Qwerty96!@"

if "password_correct" not in st.session_state:
    st.session_state.password_correct = False

# Only show password input if not yet correct
if not st.session_state.password_correct:
    user_input = st.text_input("Enter password to access the dashboard:", type="password")

    if user_input == password:
        st.session_state.password_correct = True  # Mark password as correct
        st.success("✅ Password correct! Loading dashboard...")
    else:
        if user_input:  # Only show warning if user typed something
            st.warning("🔒 Incorrect password.")
        st.stop()  # Stop the rest of the dashboard from rendering

import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from config import FUND_MAPPING, DEFAULT_BENCHMARK, START_DATE
from data import load_price_data, compute_cumulative_return, compute_trailing_return
import io
import plotly.io as pio

# --- Session state defaults for persistence ---
if "selected_funds" not in st.session_state:
    st.session_state.selected_funds = ["AIA US Equity"]

if "benchmark" not in st.session_state:
    st.session_state.benchmark = DEFAULT_BENCHMARK

if "start_date" not in st.session_state:
    st.session_state.start_date = pd.to_datetime(START_DATE)

if "end_date" not in st.session_state:
    st.session_state.end_date = pd.to_datetime("today")


st.set_page_config(page_title="Historial Data", layout="wide")

TABLEAU_COLORS = ["#4E79A7","#F28E2B","#E15759","#76B7B2","#59A14F","#EDC948","#B07AA1","#FF9DA7","#9C755F","#BAB0AC"]
TABLEAU_FONT = "Tableau Regular, Arial, sans-serif"

st.markdown(f"""
<style>
.main {{ font-family: {TABLEAU_FONT}; }}
.stButton>button {{ background-color: #4E79A7; color: white; font-weight: 600; border-radius: 5px; }}
</style>
""", unsafe_allow_html=True)

st.title("Historical Data")
st.sidebar.header("Filters")

fund_names = list(FUND_MAPPING.keys())
selected_funds = st.sidebar.multiselect("Select Funds", fund_names, default=st.session_state.selected_funds)
benchmark = st.sidebar.selectbox("Select Benchmark", fund_names, index=fund_names.index(st.session_state.benchmark))
symbols = {name:FUND_MAPPING[name] for name in selected_funds + [benchmark]}
prices = load_price_data(list(set(symbols.values())), START_DATE)
# --- Time controls ---
min_date = pd.to_datetime(START_DATE)
max_date = prices.index.max()

years = st.sidebar.slider("Invested Time Window (Years)", 1, 15, 11)

# Compute a slider-based default start date
slider_start = max(max_date - pd.DateOffset(years=years), min_date)

use_manual_dates = st.sidebar.checkbox("Use Manual Dates", value=False)

if use_manual_dates:
    start_date = st.sidebar.date_input(
        "Start Date",
        value=slider_start,
        min_value=min_date,
        max_value=max_date
    )

    end_date = st.sidebar.date_input(
        "End Date",
        value=max_date,
        min_value=min_date,
        max_value=max_date
    )
else:
    start_date = slider_start
    end_date = max_date

start_date = pd.to_datetime(start_date)
end_date = pd.to_datetime(end_date)

# --- Update session state after reading user input ---
st.session_state.selected_funds = selected_funds
st.session_state.benchmark = benchmark
st.session_state.start_date = start_date
st.session_state.end_date = end_date


prices = prices.loc[start_date:end_date]
returns = compute_cumulative_return(prices)

# --- Risk metrics ---
daily_returns = prices.pct_change().dropna()

risk_table = []
for name, symbol in symbols.items():
    total_return = returns[symbol].iloc[-1]
    volatility = daily_returns[symbol].std() * (252 ** 0.5) * 100
    ratio = total_return / volatility if volatility != 0 else 0

    risk_table.append({
        "Fund": name,
        "Total Return (%)": round(total_return, 2),
        "Volatility (%)": round(volatility, 2),
        "Return / Risk": round(ratio, 2)
    })

risk_df = pd.DataFrame(risk_table).sort_values("Return / Risk", ascending=False)

# --- Sidebar Quick Stats ---
st.sidebar.markdown("### 📊 Quick Stats")

top3 = risk_df.head(3)
bottom3 = risk_df.tail(3)

top_lines = "<br>".join(
    [f"<span style='color:#2E7D32'>▲ {row['Fund']} ({row['Total Return (%)']}%)</span>"
     for _, row in top3.iterrows()]
)

bottom_lines = "<br>".join(
    [f"<span style='color:#C62828'>▼ {row['Fund']} ({row['Total Return (%)']}%)</span>"
     for _, row in bottom3.iterrows()]
)

st.sidebar.markdown(
    f"""
<div style="font-size:13px; line-height:1.4">
<b>Top Performers</b><br>
{top_lines}
<br><br>
<b>Bottom Performers</b><br>
{bottom_lines}
</div>
""",
    unsafe_allow_html=True
)

end_date = prices.index.max()
prices = prices.loc[start_date:end_date]
returns = compute_cumulative_return(prices)

# Create figure
fig = go.Figure()

for i, (name, symbol) in enumerate(symbols.items()):
    color = TABLEAU_COLORS[i % len(TABLEAU_COLORS)]
    y_data = returns[symbol]

    # Find index of max and min
    max_pos = y_data.values.argmax()
    min_pos = y_data.values.argmin()

    # Create a list of labels (empty by default)
    labels = [""] * len(y_data)
    labels[max_pos] = f"High: {y_data.iloc[max_pos]:.1f}%"
    labels[min_pos] = f"Low: {y_data.iloc[min_pos]:.1f}%"

    # Add the line trace
    fig.add_trace(
        go.Scatter(
            x=y_data.index,
            y=y_data,
            mode="lines+text",
            name=name,
            line=dict(color=color, width=2),
            text=labels,
            textposition="top center",
            hovertemplate=f"<b>{name}</b><br>%{{y:.2f}}<extra></extra>"
        )
    )

# Layout settings
fig.update_layout(
    title=f"Performance from {returns.index.min().date()} to {returns.index.max().date()}",
    xaxis_title="Date",
    yaxis_title="Percentage Change (%)",
    hovermode="x unified",
    legend=dict(orientation="h", y=-0.25),
    template="plotly_white",
    font=dict(family=TABLEAU_FONT, size=14),
    margin=dict(t=80, b=100)
)


fig.update_layout(title=f"Performance from {start_date.date()} to {end_date.date()}",
                  xaxis_title="Date", yaxis_title="Percentage Change (%)",
                  yaxis_tickformat="%d", hovermode="x unified",
                  legend=dict(orientation="h", y=-0.25),
                  template="plotly_white", font=dict(family=TABLEAU_FONT, size=14),
                  margin=dict(t=80, b=100))

st.plotly_chart(fig, use_container_width=True)

# --- Two-column layout ---
left_col, right_col = st.columns([1, 2])  # left narrower, right wider

# ---------------- LEFT COLUMN ----------------
with left_col:
    st.subheader("Risk & Return Metrics")
    
    # Display each fund as a colored mini card
    for _, row in risk_df.iterrows():
        fund = row["Fund"]
        total_return = row["Total Return (%)"]
        volatility = row["Volatility (%)"]
        ratio = row["Return / Risk"]
        
        # Color based on performance
        bg_color = "#2E7D32" if total_return >= 0 else "#C62828"  # green/red
        text_color = "white"
        
        st.markdown(f"""
        <div style="
            background-color:{bg_color};
            color:{text_color};
            padding:10px;
            border-radius:8px;
            margin-bottom:5px;
            box-shadow: 1px 1px 3px rgba(0,0,0,0.2);
            font-size:14px;
        ">
            <b>{fund}</b><br>
            Total Return: {total_return}%<br>
            Volatility: {volatility}%<br>
            Return/Risk: {ratio}
        </div>
        """, unsafe_allow_html=True)

# ---------------- RIGHT COLUMN ----------------
with right_col:
    st.subheader("Historical Data")
    
    # Currency mapping
    FUND_CURRENCY = {
        "AIA US Equity": "USD",
        "Apple Inc": "USD",
        "AIA Singapore Bond Index": "SGD",
        "AIA Acorns of Asia": "SGD",
        # Add other funds as needed
    }

    # Price table (columns are symbols)
    price_table = prices[list(symbols.values())].copy()
    
    # Forward-fill missing prices to avoid blanks
    price_table.fillna(method="ffill", inplace=True)

    # Format columns with currency
    new_price_columns = []
    for name, symbol in symbols.items():
        currency = FUND_CURRENCY.get(name, "$")
        new_name = f"{name} ({currency})"
        new_price_columns.append(new_name)
        price_table[symbol] = price_table[symbol].apply(
            lambda x: f"${x:,.2f}" if currency in ["USD","SGD"] else f"{x:,.2f}"
        )
    price_table.columns = new_price_columns

    # Cumulative returns
    return_table = returns[list(symbols.values())].copy()
    return_table.fillna(method="ffill", inplace=True)
    return_table.columns = [f"{name} Return (%)" for name in symbols.keys()]
    return_table = return_table.round(2)
    for col in return_table.columns:
        return_table[col] = return_table[col].apply(lambda x: f"{x:.2f}%")

    # Combine price + returns
    combined_table = pd.concat([price_table, return_table], axis=1)
    combined_table.index = combined_table.index.strftime("%Y-%m-%d")

    # --- Dark theme styling ---
    st.dataframe(
        combined_table.style
        .set_properties(**{
            "text-align": "right",
            "font-family": "Arial, sans-serif",
            "font-size": "13px",
            "color": "white",                 # text color
            "background-color": "#1E1E1E"     # dark background
        })
        .set_table_styles([{
            "selector": "th",
            "props": [
                ("text-align", "center"),
                ("background-color", "#2E2E2E"),
                ("color", "white"),
                ("font-weight", "bold")
            ]
        }])
        .apply(lambda x: ["background-color: #2A2A2A" if i%2==0 else "#background-color: #1E1E1E" for i in range(len(x))], axis=0)
    , use_container_width=True)
