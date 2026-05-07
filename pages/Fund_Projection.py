import streamlit as st

# --- PASSWORD PROTECTION ---
from auth import check_password
check_password()

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from config import FUND_MAPPING, START_DATE
from data import load_price_data

aia_funds = [name for name in FUND_MAPPING.keys() if name.startswith("AIA")]
non_aia_funds = [name for name in FUND_MAPPING.keys() if not name.startswith("AIA")]
all_funds = aia_funds + non_aia_funds

st.set_page_config(page_title="Fund Projection", layout="wide")
st.title("Fund Projection")

st.markdown("Projection uses the fund's historical CAGR based on available price data.")

# -------------------- SIDEBAR --------------------
# -------------------- FUND PROJECTION PARAMETERS --------------------
with st.sidebar.expander("Parameters", expanded=True):

    # -------------------- FUND SELECTION --------------------
    st.subheader("Funds")

    selected_funds = st.multiselect(
        "Select Funds",
        all_funds,
        default=[all_funds[0]],
        key="fund_selector"
    )

    if not selected_funds:
        st.warning("Please select at least one fund.")
        st.stop()

    # -------------------- INVESTMENT SETTINGS --------------------
    st.subheader("Investment Settings")

    investment_years = st.number_input(
        "Investment Duration (Years)",
        min_value=1,
        value=20
    )

    annual_premium = st.number_input(
        "Annual Investment ($)",
        min_value=0.0,
        value=10000.0
    )

# -------------------- NON-AIA FEES (kept separate but clean) --------------------
with st.sidebar.expander("Non-AIA Fees", expanded=False):

    st.caption("Applies only to Non-AIA funds")

    transaction_cost = st.number_input(
        "Transaction Cost (%)",
        value=0.10,
        step=0.01,
        help="Applied on annual contribution"
    ) / 100

    first_year_free = st.checkbox(
        "First Year Transaction Cost Free",
        value=False
    )

    holding_cost = st.number_input(
        "Holding Cost (%)",
        value=0.03,
        step=0.01,
        help="Applied on fund value annually"
    ) / 100

    gst_rate = st.number_input(
        "GST (%)",
        value=9.0,
        step=0.1,
        help="Goods & Service Tax are applied only on transaction and holding costs in accordance with IRAS regulations."
    ) / 100

# -------------------- ADVANCED --------------------
manual_cagr_settings = {}

with st.sidebar.expander("Advanced", expanded=False):

    st.subheader("CAGR Overrides")

    for fund in selected_funds:

        use_manual = st.checkbox(
            f"Use another CAGR for {fund}",
            key=f"use_manual_{fund}"
        )

        manual_value = None

        if use_manual:
            manual_value = st.number_input(
                f"Manual CAGR (%) - {fund}",
                value=5.0,
                step=0.01,
                key=f"cagr_{fund}"
            ) / 100

        manual_cagr_settings[fund] = {
            "enabled": use_manual,
            "value": manual_value
        }

# -------------------- LOAD DATA --------------------
symbols = [FUND_MAPPING[fund] for fund in selected_funds]
prices = load_price_data(symbols, START_DATE)

if prices.empty:
    st.error("No data available for this fund.")
    st.stop()

prices = prices.dropna()

# -------------------- BONUS FUNCTION --------------------
def get_bonus(year):
    if year == 1:
        return 0.15
    elif year == 2:
        return 0.18
    elif year == 3:
        return 0.20
    elif 4 <= year <= 9:
        return 0.00
    elif 10 <= year <= 19:
        return 0.05
    else:
        return 0.08

# -------------------- MULTI-FUND SIMULATION --------------------

all_projection_tables = {}
all_graph_lines = {}
fund_metrics = {}

fig = go.Figure()

for selected_fund in selected_funds:

    symbol = FUND_MAPPING[selected_fund]
    is_aia = selected_fund.startswith("AIA")

    fund_prices = prices[[symbol]].dropna()

    if fund_prices.empty:
        continue

    # -------------------- CAGR --------------------
    start_price = fund_prices.iloc[0, 0]
    end_price = fund_prices.iloc[-1, 0]

    years_of_data = (fund_prices.index[-1] - fund_prices.index[0]).days / 365.25
    historical_cagr = (end_price / start_price) ** (1 / years_of_data) - 1

    manual_setting = manual_cagr_settings.get(selected_fund)

    if manual_setting and manual_setting["enabled"]:
        cagr = manual_setting["value"]
    else:
        cagr = historical_cagr

    fund_metrics[selected_fund] = cagr

    # -------------------- RESET PER FUND --------------------
    results = []
    profit_results = []
    years_list = []
    bonus_results = []
    charges_results = []
    contribution_results = []

    fund_value = 0
    total_contributions = 0

    # -------------------- SIMULATION --------------------
    for year in range(1, investment_years + 1):

        fund_value += annual_premium
        yearly_contribution = annual_premium
        total_contributions += annual_premium

        bonus = 0
        yearly_charge = 0

        # ================= AIA =================
        if is_aia:

            bonus = annual_premium * get_bonus(year)
            fund_value += bonus

            fund_value *= (1 + cagr)

            if year <= 10:
                annual_charge_rate = 1 - (1 - 0.00325) ** 12
                yearly_charge = fund_value * annual_charge_rate
                fund_value -= yearly_charge

        # ================= NON-AIA =================
        else:

            fund_value *= (1 + cagr)

            transaction_fee = annual_premium * transaction_cost

            if first_year_free and year == 1:
                transaction_fee = 0

            holding_fee = fund_value * holding_cost
            gst = (transaction_fee + holding_fee) * gst_rate

            yearly_charge = transaction_fee + holding_fee + gst

            fund_value -= yearly_charge

        # ---------------- PROFIT AFTER GROWTH ----------------
        profit_base = fund_value - total_contributions

        # ---------------- STORE RESULTS ----------------
        results.append(fund_value)
        profit_results.append(profit_base)
        years_list.append(year)
        bonus_results.append(bonus)
        charges_results.append(yearly_charge)
        contribution_results.append(yearly_contribution)

    # -------------------- SAVE FOR GRAPH --------------------
    all_graph_lines[selected_fund] = results

    # -------------------- SAVE TABLE --------------------
    all_projection_tables[selected_fund] = pd.DataFrame({
    "Year": [0] + years_list,
    "Contribution ($)": [0] + contribution_results,
    "Yearly Bonus ($)": [0] + bonus_results,
    "Profit ($)": [0] + profit_results,
    "Yearly Charges ($)": [0] + charges_results,
    "Fund Value ($)": [annual_premium] + results
})

    # -------------------- ADD TO GRAPH --------------------
    fig.add_trace(
        go.Scatter(
            x=years_list,
            y=results,
            mode="lines+markers",
            name=selected_fund,
            hovertemplate="%{y:$,.2f}<extra>%{fullData.name}</extra>"
        )
    )

# -------------------- GRAPH LAYOUT --------------------
fig.update_layout(
    title="Projected Fund Growth Comparison",
    xaxis_title="Year",
    yaxis_title="Fund Value ($)",
    template="plotly_white"
)

st.plotly_chart(fig, use_container_width=True)

# -------------------- TABLE OUTPUT --------------------
st.subheader("Projection Tables")

for fund_name, df in all_projection_tables.items():

    cagr_value = fund_metrics.get(fund_name)

    if cagr_value is not None:
        st.markdown(f"### {fund_name} — CAGR: {cagr_value*100:.2f}%")
    else:
        st.markdown(f"### {fund_name}")

    st.dataframe(df, use_container_width=True, hide_index=True)


# -------------------- POLICY BREAKDOWN --------------------
has_aia = any(f.startswith("AIA") for f in selected_funds)
has_non_aia = any(not f.startswith("AIA") for f in selected_funds)

with st.expander("Policy Mechanics", expanded=False):

    # -------------------- AIA SECTION --------------------
    if has_aia:

        st.subheader("AIA Funds")

        st.caption("Bonuses and monthly charges apply only to AIA funds.")

        periods = [
            "Year 1", "Year 2", "Year 3",
            "Years 4–9", "Year 10",
            "Years 11–19", "Year 20+"
        ]

        bonus_rates = [15, 18, 20, 0, 5, 5, 8]
        charge_indicator = [1, 1, 1, 1, 1, 0, 0]

        fig2 = go.Figure()

        fig2.add_trace(
            go.Bar(
                x=periods,
                y=bonus_rates,
                text=[f"{b}%" for b in bonus_rates],
                textposition="outside",
                hovertemplate="Yearly Bonus: %{y}%<extra></extra>"
            )
        )

        charge_periods = [p for p, c in zip(periods, charge_indicator) if c == 1]

        fig2.add_trace(
            go.Scatter(
                x=charge_periods,
                y=[-0.7] * len(charge_periods),
                mode="markers",
                hovertemplate="0.325% monthly charge applied<extra></extra>",
                name="Monthly Charges"
            )
        )

        fig2.update_layout(
            height=300,
            template="plotly_white",
            showlegend=False,
            yaxis=dict(range=[-1.2, 22]),
            title="AIA Bonuses & Charges"
        )

        st.plotly_chart(fig2, use_container_width=True)

    # -------------------- NON-AIA SECTION --------------------
    if has_non_aia:

        st.subheader("Non-AIA Funds")

        st.caption("Fee structure applies only to Non-AIA funds.")

        st.markdown(f"""
        - Transaction Cost: {transaction_cost*100:.2f}%
        - Holding Cost: {holding_cost*100:.2f}%
        - GST: {gst_rate*100:.2f}%
        - First Year Free: {"Yes" if first_year_free else "No"}
        """)