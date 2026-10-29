import os

import pandas as pd
import streamlit as st
import altair as alt

try:
    import requests
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    requests = None

try:
    from trading_strategy import (
        analyze_results,
        calculate_moving_averages,
        get_stock_data,
        identify_golden_cross,
        implement_strategy,
    )
    from app import get_backtest_history, save_backtest_history
except ModuleNotFoundError as exc:
    st.error(
        "Missing dependency: the app could not import the trading logic. "
        "Make sure `yfinance` is included in the deployment requirements."
    )
    st.stop()

st.set_page_config(page_title="Portfolio Trading Strategy Dashboard", layout="wide")

AVAILABLE_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "NFLX", "AMD", "IBM"]


@st.cache_data
def load_strategy_data(symbol="MSFT"):
    data = get_stock_data(symbol)
    data = calculate_moving_averages(data)
    data = identify_golden_cross(data)
    positions = implement_strategy(data)
    summary = analyze_results(positions)

    chart_df = data.reset_index().rename(columns={"index": "Date"})
    chart_df["Date"] = pd.to_datetime(chart_df["Date"])
    chart_df = chart_df.sort_values("Date").reset_index(drop=True)

    total_return = 0.0
    if not positions.empty:
        total_return = round(((positions["SellPrice"].sum() / positions["BuyPrice"].sum()) - 1) * 100, 2)

    return chart_df, positions, summary, total_return


st.sidebar.title("Backtest Controls")
selected_symbols = st.sidebar.multiselect(
    "Select stocks (up to 3)",
    AVAILABLE_TICKERS,
    default=["MSFT"],
    max_selections=3,
)
preview_signal_chart = st.sidebar.checkbox("Preview MA / Trade Signals", value=False)
run_backtest = st.sidebar.button("Run backtest")

page = st.sidebar.radio("Navigation", ["Dashboard", "Backtest History"])

if not selected_symbols:
    selected_symbols = ["MSFT"]

if run_backtest:
    portfolio_returns = []
    portfolio_frames = []
    for symbol in selected_symbols:
        chart_df, positions, summary, total_return = load_strategy_data(symbol)
        portfolio_returns.append(total_return)
        chart_df = chart_df[["Date", "Close"]].rename(columns={"Close": symbol})
        portfolio_frames.append(chart_df)

    if portfolio_frames:
        merged_chart = portfolio_frames[0]
        for frame in portfolio_frames[1:]:
            merged_chart = merged_chart.merge(frame, on="Date", how="outer")
    else:
        merged_chart = pd.DataFrame()

    average_return = round(sum(portfolio_returns) / len(portfolio_returns), 2) if portfolio_returns else 0.0
    save_backtest_history(selected_symbols, average_return)
    st.sidebar.success(f"Saved portfolio backtest for {', '.join(selected_symbols)}: {average_return}%")

    selected_symbol = selected_symbols[0]
    chart_df, positions, summary, total_return = load_strategy_data(selected_symbol)
    combined_chart = merged_chart
else:
    try:
        portfolio_returns = []
        portfolio_frames = []
        for symbol in selected_symbols:
            chart_df, positions, summary, total_return = load_strategy_data(symbol)
            portfolio_returns.append(total_return)
            df = chart_df[["Date", "Close"]].rename(columns={"Close": symbol})
            portfolio_frames.append(df)

        if portfolio_frames:
            combined_chart = portfolio_frames[0]
            for frame in portfolio_frames[1:]:
                combined_chart = combined_chart.merge(frame, on="Date", how="outer")
        else:
            combined_chart = pd.DataFrame()

        average_return = round(sum(portfolio_returns) / len(portfolio_returns), 2) if portfolio_returns else 0.0
        selected_symbol = selected_symbols[0]
    except Exception as exc:
        st.error(f"Unable to load trading data: {exc}")
        st.stop()

portfolio_title = ", ".join(selected_symbols)
st.title(f"Portfolio Dashboard: {portfolio_title}")


def build_financial_summary_prompt(selected_stock_list, portfolio_return, win_rate):
    return (
        "You are a professional financial analyst. "
        "Write a concise but professional summary for a portfolio backtest. "
        "Use the information below and keep the tone clear, structured, and investor-focused. "
        "Do not mention that you are an AI or a language model. "
        f"Selected stocks: {', '.join(selected_stock_list)}. "
        f"Average total portfolio return: {portfolio_return}%. "
        f"Portfolio win rate: {win_rate}%. "
        "Summarize the current performance, interpret risk and consistency, and provide a brief recommendation for the next action."
    )


def generate_financial_summary(prompt):
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key and requests is not None:
        try:
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "gpt-4o-mini",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.5,
                },
                timeout=30,
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"].strip()
        except Exception:
            pass

    return (
        "Portfolio assessment: The selected portfolio shows an average total return of "
        f"{prompt.split('Average total portfolio return: ')[1].split('. ')[0]}% with a win rate of "
        f"{prompt.split('Portfolio win rate: ')[1].split('. ')[0]}%. "
        "The strategy indicates reasonable performance, but continued monitoring of execution quality and market conditions is recommended. "
        "The portfolio remains best suited for a disciplined, data-driven review rather than a short-term tactical trade decision."
    )


if page == "Dashboard":
    chart_tab, stats_tab, trades_tab = st.tabs([
        "Price Chart",
        "Trade Statistics",
        "Detailed Trades",
    ])

    with chart_tab:
        st.subheader(f"Combined Portfolio Price Chart — Total Portfolio Return: {average_return}%")
        if combined_chart.empty:
            st.info("Select at least one stock to view the combined chart.")
        else:
            combined_chart = combined_chart.sort_values("Date").reset_index(drop=True)
            chart_lines = []
            colors = ["#2563eb", "#10b981", "#f59e0b"]
            for index, column in enumerate([col for col in combined_chart.columns if col != "Date"]):
                line = (
                    alt.Chart(combined_chart)
                    .mark_line(color=colors[index % len(colors)], strokeWidth=2)
                    .encode(
                        x=alt.X("Date:T", title="Date"),
                        y=alt.Y(f"{column}:Q", title="Price ($)"),
                        tooltip=["Date:T", alt.Tooltip(f"{column}:Q", title=column)],
                    )
                )
                chart_lines.append(line)

            combined_plot = alt.layer(*chart_lines).properties(
                width=1200,
                height=600,
                title=f"Portfolio: {portfolio_title}",
            ).interactive()

            st.altair_chart(combined_plot, use_container_width=True)

        if preview_signal_chart:
            st.subheader("Preview: Price Trend with Moving Averages and Trade Signals")
            first_symbol = selected_symbols[0]
            symbol_df, symbol_positions, _, _ = load_strategy_data(first_symbol)

            buy_points = symbol_positions[["BuyDate", "BuyPrice"]].rename(
                columns={"BuyDate": "Date", "BuyPrice": "Price"}
            )
            sell_points = symbol_positions[["SellDate", "SellPrice"]].rename(
                columns={"SellDate": "Date", "SellPrice": "Price"}
            )
            buy_points["Date"] = pd.to_datetime(buy_points["Date"])
            sell_points["Date"] = pd.to_datetime(sell_points["Date"])

            line_close = (
                alt.Chart(symbol_df)
                .mark_line(color="#2563eb", strokeWidth=2)
                .encode(
                    x=alt.X("Date:T", title="Date"),
                    y=alt.Y("Close:Q", title="Price ($)"),
                    tooltip=["Date:T", "Close:Q"],
                )
            )
            ma50 = (
                alt.Chart(symbol_df)
                .mark_line(color="#10b981", strokeWidth=1.5)
                .encode(
                    x=alt.X("Date:T"),
                    y=alt.Y("MA50:Q"),
                    tooltip=["Date:T", "MA50:Q"],
                )
            )
            ma200 = (
                alt.Chart(symbol_df)
                .mark_line(color="#f59e0b", strokeWidth=1.5)
                .encode(
                    x=alt.X("Date:T"),
                    y=alt.Y("MA200:Q"),
                    tooltip=["Date:T", "MA200:Q"],
                )
            )
            buy_marks = (
                alt.Chart(buy_points)
                .mark_circle(size=70, color="#22c55e")
                .encode(x="Date:T", y="Price:Q", tooltip=["Date:T", "Price:Q"])
            )
            sell_marks = (
                alt.Chart(sell_points)
                .mark_circle(size=70, color="#ef4444")
                .encode(x="Date:T", y="Price:Q", tooltip=["Date:T", "Price:Q"])
            )

            preview_chart = alt.layer(line_close, ma50, ma200, buy_marks, sell_marks).properties(
                width=1200,
                height=500,
                title=f"{first_symbol} Price Trend with Moving Averages and Trade Signals",
            ).interactive()
            st.altair_chart(preview_chart, use_container_width=True)

    with stats_tab:
        st.subheader("Overall Portfolio Statistics")

        metrics = [
            ("Selected Stocks", ", ".join(selected_symbols)),
            ("Average Total Return", f"{average_return}%"),
            ("Stocks in Portfolio", len(selected_symbols)),
        ]

        cols = st.columns(3)
        for index, (label, value) in enumerate(metrics):
            cols[index % 3].metric(label=label, value=value)

        stock_summary = pd.DataFrame(
            [
                {"stock": symbol, "total_return": load_strategy_data(symbol)[3]}
                for symbol in selected_symbols
            ]
        )
        st.dataframe(stock_summary, use_container_width=True, hide_index=True)

    with trades_tab:
        st.subheader("Portfolio Trade Summary")
        trade_rows = []
        for symbol in selected_symbols:
            _, positions, _, total_return = load_strategy_data(symbol)
            for _, trade in positions.iterrows():
                trade_rows.append(
                    {
                        "stock": symbol,
                        "BuyDate": trade["BuyDate"].strftime("%Y-%m-%d"),
                        "BuyPrice": round(float(trade["BuyPrice"]), 4),
                        "SellDate": trade["SellDate"].strftime("%Y-%m-%d"),
                        "SellPrice": round(float(trade["SellPrice"]), 4),
                        "HoldingDays": int(trade["HoldingDays"]),
                        "ProfitPct": round(float(trade["ProfitPct"]), 2),
                        "SellReason": trade["SellReason"],
                    }
                )

        if trade_rows:
            st.dataframe(pd.DataFrame(trade_rows), use_container_width=True, hide_index=True)
        else:
            st.info("No trades available for the selected stocks.")

    st.markdown("---")
    st.subheader("AI Financial Summary")

    prompt = build_financial_summary_prompt(selected_symbols, average_return, 0 if 'win_rate' not in locals() else 0)
    if len(selected_symbols) > 0:
        try:
            portfolio_win_rate = round(
                sum(
                    load_strategy_data(symbol)[2]["win_rate"] for symbol in selected_symbols
                ) / len(selected_symbols),
                2,
            )
        except Exception:
            portfolio_win_rate = 0.0
        prompt = build_financial_summary_prompt(selected_symbols, average_return, portfolio_win_rate)

    summary_text = generate_financial_summary(prompt)
    st.text_area("Financial Summary", summary_text, height=220)

else:
    st.subheader("Backtest History")
    history = pd.DataFrame(get_backtest_history(limit=50))
    if history.empty:
        st.info("No backtests saved yet. Run one from the sidebar to populate the history.")
    else:
        st.dataframe(history, use_container_width=True, hide_index=True)
