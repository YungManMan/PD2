import os
import sqlite3
from datetime import datetime

import pandas as pd
from flask import Flask, render_template, request

from trading_strategy import (
    analyze_results,
    calculate_moving_averages,
    get_stock_data,
    identify_golden_cross,
    implement_strategy,
)

app = Flask(__name__)
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backtest_history.db")


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS backtest_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            stock_symbol TEXT NOT NULL,
            total_return REAL NOT NULL,
            timestamp TEXT NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()


def save_backtest_history(stock_symbol, total_return):
    init_db()
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO backtest_history (stock_symbol, total_return, timestamp) VALUES (?, ?, ?)",
        (str(stock_symbol).upper(), float(total_return), timestamp),
    )
    conn.commit()
    conn.close()


def get_backtest_history(stock_symbol=None, limit=20):
    init_db()
    conn = sqlite3.connect(DB_PATH)
    if stock_symbol:
        rows = conn.execute(
            "SELECT stock_symbol, total_return, timestamp FROM backtest_history WHERE stock_symbol = ? ORDER BY timestamp DESC LIMIT ?",
            (str(stock_symbol).upper(), int(limit)),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT stock_symbol, total_return, timestamp FROM backtest_history ORDER BY timestamp DESC LIMIT ?",
            (int(limit),),
        ).fetchall()
    conn.close()
    return [
        {"stock_symbol": symbol, "total_return": total_return, "timestamp": timestamp}
        for symbol, total_return, timestamp in rows
    ]


AVAILABLE_TICKERS = [
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "NVDA",
    "META",
    "TSLA",
    "NFLX",
    "AMD",
    "IBM",
]
DEFAULT_TICKERS = ["MSFT"]


def normalize_tickers(raw_tickers):
    if raw_tickers is None:
        return DEFAULT_TICKERS[:]

    if isinstance(raw_tickers, str):
        raw_tickers = [raw_tickers]

    selected = []
    for ticker in raw_tickers:
        if ticker is None:
            continue
        symbol = str(ticker).strip().upper()
        if symbol and symbol in AVAILABLE_TICKERS and symbol not in selected:
            selected.append(symbol)

    if not selected:
        return DEFAULT_TICKERS[:]

    return selected[:3]


def get_stock_view(ticker):
    data = get_stock_data(ticker)
    data = calculate_moving_averages(data)
    data = identify_golden_cross(data)
    positions = implement_strategy(data)
    summary = analyze_results(positions)

    chart_df = data.reset_index().rename(columns={"index": "Date"})
    chart_df["Date"] = chart_df["Date"].dt.strftime("%Y-%m-%d")
    chart_df["Close"] = chart_df["Close"].round(4)
    chart_df["MA50"] = chart_df["MA50"].round(4)
    chart_df["MA200"] = chart_df["MA200"].round(4)

    buy_points = []
    sell_points = []

    for _, trade in positions.iterrows():
        buy_points.append({
            "date": trade["BuyDate"].strftime("%Y-%m-%d"),
            "price": round(float(trade["BuyPrice"]), 4),
            "ticker": ticker,
        })
        sell_points.append({
            "date": trade["SellDate"].strftime("%Y-%m-%d"),
            "price": round(float(trade["SellPrice"]), 4),
            "ticker": ticker,
        })

    return {
        "ticker": ticker,
        "chart_data": chart_df,
        "positions": positions,
        "summary": summary,
        "buy_points": buy_points,
        "sell_points": sell_points,
    }


def calculate_portfolio_return(portfolio_positions):
    total_cost = 0.0
    total_proceeds = 0.0

    for positions in portfolio_positions.values():
        if positions.empty:
            continue
        total_cost += float(positions["BuyPrice"].sum())
        total_proceeds += float(positions["SellPrice"].sum())

    if total_cost == 0:
        return 0.0

    return round(((total_proceeds / total_cost) - 1) * 100, 2)


def prepare_data(selected_tickers=None):
    selected_tickers = normalize_tickers(selected_tickers)
    stock_views = {ticker: get_stock_view(ticker) for ticker in selected_tickers}

    combined_chart = None
    all_positions = []
    portfolio_summary = {
        "total_trades": 0,
        "winning_trades": 0,
        "losing_trades": 0,
        "win_rate": 0,
        "average_profit": 0,
        "average_holding_days": 0,
        "target_reached": 0,
        "max_period": 0,
    }

    for ticker in selected_tickers:
        view = stock_views[ticker]
        positions = view["positions"].copy()
        if not positions.empty:
            positions["Ticker"] = ticker
            all_positions.append(positions)
        portfolio_summary["total_trades"] += view["summary"]["total_trades"]
        portfolio_summary["winning_trades"] += view["summary"]["winning_trades"]
        portfolio_summary["losing_trades"] += view["summary"]["losing_trades"]
        portfolio_summary["target_reached"] += view["summary"]["target_reached"]
        portfolio_summary["max_period"] += view["summary"]["max_period"]

        if combined_chart is None:
            combined_chart = view["chart_data"][ ["Date", "Close"] ].rename(columns={"Close": ticker})
        else:
            merged = view["chart_data"][ ["Date", "Close"] ].rename(columns={"Close": ticker})
            combined_chart = combined_chart.merge(merged, on="Date", how="outer")

    combined_chart = combined_chart.sort_values("Date").reset_index(drop=True)
    combined_chart = combined_chart.ffill().bfill()

    if all_positions:
        combined_positions = pd.concat(all_positions, ignore_index=True)
    else:
        combined_positions = []

    total_return = calculate_portfolio_return({ticker: stock_views[ticker]["positions"] for ticker in selected_tickers})
    portfolio_summary["portfolio_return"] = total_return

    if portfolio_summary["total_trades"]:
        portfolio_summary["win_rate"] = round(
            (portfolio_summary["winning_trades"] / portfolio_summary["total_trades"]) * 100,
            2,
        )
        portfolio_summary["average_profit"] = round(
            sum(
                stock_views[ticker]["summary"]["average_profit"] * stock_views[ticker]["summary"]["total_trades"]
                for ticker in selected_tickers
            ) / portfolio_summary["total_trades"],
            2,
        )
        portfolio_summary["average_holding_days"] = round(
            sum(
                stock_views[ticker]["summary"]["average_holding_days"] * stock_views[ticker]["summary"]["total_trades"]
                for ticker in selected_tickers
            ) / portfolio_summary["total_trades"],
            2,
        )

    stock_summaries = []
    for ticker in selected_tickers:
        view = stock_views[ticker]
        stock_summaries.append({
            "ticker": ticker,
            "total_trades": view["summary"]["total_trades"],
            "winning_trades": view["summary"]["winning_trades"],
            "average_profit": view["summary"]["average_profit"],
        })

    buy_points = []
    sell_points = []
    for ticker in selected_tickers:
        buy_points.extend(stock_views[ticker]["buy_points"])
        sell_points.extend(stock_views[ticker]["sell_points"])

    return {
        "selected_tickers": selected_tickers,
        "chart_data": combined_chart,
        "positions": combined_positions,
        "summary": portfolio_summary,
        "stock_summaries": stock_summaries,
        "buy_points": buy_points,
        "sell_points": sell_points,
    }


@app.route("/")
def chart_page():
    selected = []
    for field in ["stock_1", "stock_2", "stock_3"]:
        value = request.args.get(field)
        if value:
            selected.append(value)

    portfolio = prepare_data(selected)
    return render_template(
        "chart.html",
        selected_tickers=portfolio["selected_tickers"],
        all_tickers=AVAILABLE_TICKERS,
        chart_data=portfolio["chart_data"].to_dict("records"),
        buy_points=portfolio["buy_points"],
        sell_points=portfolio["sell_points"],
        positions=portfolio["positions"].to_dict("records") if isinstance(portfolio["positions"], object) and hasattr(portfolio["positions"], "to_dict") else [],
        summary=portfolio["summary"],
    )


@app.route("/stats")
def stats_page():
    selected = []
    for field in ["stock_1", "stock_2", "stock_3"]:
        value = request.args.get(field)
        if value:
            selected.append(value)

    portfolio = prepare_data(selected)
    return render_template(
        "stats.html",
        selected_tickers=portfolio["selected_tickers"],
        summary=portfolio["summary"],
        stock_summaries=portfolio["stock_summaries"],
        all_tickers=AVAILABLE_TICKERS,
    )


@app.route("/trades")
def trades_page():
    selected = []
    for field in ["stock_1", "stock_2", "stock_3"]:
        value = request.args.get(field)
        if value:
            selected.append(value)

    portfolio = prepare_data(selected)
    trades = portfolio["positions"].to_dict("records") if isinstance(portfolio["positions"], object) and hasattr(portfolio["positions"], "to_dict") else []
    for trade in trades:
        trade["BuyDate"] = trade["BuyDate"].strftime("%Y-%m-%d")
        trade["SellDate"] = trade["SellDate"].strftime("%Y-%m-%d")
        trade["BuyPrice"] = round(float(trade["BuyPrice"]), 4)
        trade["SellPrice"] = round(float(trade["SellPrice"]), 4)
        trade["ProfitPct"] = round(float(trade["ProfitPct"]), 2)
    return render_template(
        "trades.html",
        trades=trades,
        selected_tickers=portfolio["selected_tickers"],
        all_tickers=AVAILABLE_TICKERS,
    )


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
