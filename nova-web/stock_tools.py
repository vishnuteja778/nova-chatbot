"""Tools Nova can call to research stocks.

The model never touches the internet itself. It asks for a tool by name with
some arguments (for example get_stock_quote(symbol="TCS.NS")), our code runs
the matching Python function, and the result is sent back to the model as
JSON. The model then reads that data and writes its answer.

Data comes from Yahoo Finance through the free `yfinance` library, plus an
optional web search (Tavily) when TAVILY_API_KEY is set.
"""
import json
import math
import os
import re
from datetime import datetime, timezone

import httpx
import yfinance as yf

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "").strip()

# Ticker symbols look like AAPL, TCS.NS, RELIANCE.BO, BRK-B, ^NSEI, EURUSD=X.
# Anything else is rejected before it reaches Yahoo.
SYMBOL_RE = re.compile(r"^[A-Za-z0-9^][A-Za-z0-9.\-=&^]{0,19}$")
PERIODS = {"1mo", "3mo", "6mo", "1y", "2y", "5y"}


class ToolError(Exception):
    """A problem the model should hear about (bad symbol, no data...)."""


def _symbol(value) -> str:
    s = str(value or "").strip().upper()
    if not SYMBOL_RE.match(s):
        raise ToolError(f"'{value}' doesn't look like a ticker symbol. Use search_symbol first.")
    return s


def _num(value, digits=2):
    """Plain JSON-friendly number, or None for missing/NaN values."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return round(f, digits)


def _pct(value):
    v = _num(value, 4)
    return None if v is None else round(v * 100, 2)


def _clean(d: dict) -> dict:
    """Drop empty fields so the model reads (and we pay for) fewer tokens."""
    return {k: v for k, v in d.items() if v not in (None, "", [], {})}


def _info(symbol: str) -> dict:
    info = yf.Ticker(symbol).info or {}
    if not (info.get("regularMarketPrice") or info.get("currentPrice") or info.get("longName")):
        raise ToolError(f"No data found for {symbol}. Check the symbol with search_symbol "
                        "(Indian stocks need .NS for NSE or .BO for BSE).")
    return info


# ---------- the tools ----------

def search_symbol(query: str) -> dict:
    query = str(query or "").strip()[:80]
    if not query:
        raise ToolError("Give a company name to search for.")
    quotes = yf.Search(query, max_results=6, news_count=0).quotes
    return {"query": query, "matches": [_clean({
        "symbol": q.get("symbol"),
        "name": q.get("longname") or q.get("shortname"),
        "exchange": q.get("exchDisp") or q.get("exchange"),
        "type": q.get("quoteType"),
    }) for q in quotes]}


def get_stock_quote(symbol: str) -> dict:
    symbol = _symbol(symbol)
    i = _info(symbol)
    price = i.get("regularMarketPrice") or i.get("currentPrice")
    return _clean({
        "symbol": symbol,
        "name": i.get("longName") or i.get("shortName"),
        "exchange": i.get("fullExchangeName") or i.get("exchange"),
        "currency": i.get("currency"),
        "price": _num(price),
        "change_pct_today": _num(i.get("regularMarketChangePercent")),
        "previous_close": _num(i.get("regularMarketPreviousClose")),
        "day_range": i.get("regularMarketDayRange"),
        "52_week_low": _num(i.get("fiftyTwoWeekLow")),
        "52_week_high": _num(i.get("fiftyTwoWeekHigh")),
        "market_cap": i.get("marketCap"),
        "pe_trailing": _num(i.get("trailingPE")),
        "pe_forward": _num(i.get("forwardPE")),
        "price_to_book": _num(i.get("priceToBook")),
        "eps_trailing": _num(i.get("trailingEps")),
        "dividend_yield_pct": _num(i.get("dividendYield")),  # Yahoo already gives a percent
        "beta": _num(i.get("beta")),
        "sector": i.get("sector"),
        "industry": i.get("industry"),
        "analyst_rating": i.get("averageAnalystRating"),
        "analyst_target_mean": _num(i.get("targetMeanPrice")),
        "analyst_count": i.get("numberOfAnalystOpinions"),
        "business_summary": (i.get("longBusinessSummary") or "")[:600],
    })


def _rsi(closes, length=14):
    delta = closes.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / length, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / length, adjust=False).mean()
    rs = gain / loss
    return _num((100 - 100 / (1 + rs)).iloc[-1], 1)


def get_price_history(symbol: str, period: str = "1y") -> dict:
    symbol = _symbol(symbol)
    period = period if period in PERIODS else "1y"
    # Download extra history so the 200-day average and 1-year return always work.
    fetch = "5y" if period == "5y" else "2y"
    h = yf.Ticker(symbol).history(period=fetch, auto_adjust=True)
    if h.empty:
        raise ToolError(f"No price history for {symbol}.")
    closes = h["Close"].dropna()
    days = {"1mo": 21, "3mo": 63, "6mo": 126, "1y": 252, "2y": 504, "5y": 1260}[period]
    window = closes.iloc[-days:]
    last = closes.iloc[-1]

    def ret(n):
        return _pct(last / closes.iloc[-n - 1] - 1) if len(closes) > n else None

    def sma(n):
        return _num(closes.iloc[-n:].mean()) if len(closes) >= n else None

    daily = closes.pct_change().dropna().iloc[-days:]
    # About 12 evenly spaced closing prices so the model can "see" the shape.
    step = max(1, len(window) // 12)
    sampled = window.iloc[::step]
    return _clean({
        "symbol": symbol,
        "period": period,
        "last_close": _num(last),
        "last_date": str(closes.index[-1].date()),
        "return_pct": {"1_week": ret(5), "1_month": ret(21), "3_months": ret(63),
                       "6_months": ret(126), "1_year": ret(252)},
        "period_high": _num(window.max()),
        "period_low": _num(window.min()),
        "sma_20": sma(20), "sma_50": sma(50), "sma_200": sma(200),
        "rsi_14": _rsi(closes),
        "annualized_volatility_pct": _num(daily.std() * math.sqrt(252) * 100),
        "max_drawdown_pct": _pct((window / window.cummax() - 1).min()),
        "avg_daily_volume": int(h["Volume"].iloc[-days:].mean()) if "Volume" in h else None,
        "closes": {str(d.date()): _num(v) for d, v in sampled.items()},
    })


def _rows(frame, wanted, years=4):
    """Pick a few named rows from a yfinance statement, newest year first."""
    if frame is None or frame.empty:
        return {}
    out = {}
    for col in list(frame.columns)[:years]:
        year = str(getattr(col, "date", lambda: col)())
        out[year] = _clean({label: _num(frame.at[row, col], 0)
                            for label, row in wanted.items() if row in frame.index})
    return out


def get_financials(symbol: str) -> dict:
    symbol = _symbol(symbol)
    t = yf.Ticker(symbol)
    i = t.info or {}
    income = _rows(t.income_stmt, {
        "revenue": "Total Revenue", "gross_profit": "Gross Profit",
        "operating_income": "Operating Income", "net_income": "Net Income",
        "diluted_eps": "Diluted EPS",
    })
    balance = _rows(t.balance_sheet, {
        "total_assets": "Total Assets", "total_debt": "Total Debt",
        "cash": "Cash And Cash Equivalents", "shareholder_equity": "Stockholders Equity",
    })
    cash = _rows(t.cashflow, {
        "operating_cash_flow": "Operating Cash Flow", "capex": "Capital Expenditure",
        "free_cash_flow": "Free Cash Flow",
    })
    if not (income or balance or cash):
        raise ToolError(f"No financial statements available for {symbol}.")
    return _clean({
        "symbol": symbol,
        "currency": i.get("financialCurrency") or i.get("currency"),
        "note": "Annual figures, newest first, in the reporting currency.",
        "ratios": _clean({
            "gross_margin_pct": _pct(i.get("grossMargins")),
            "operating_margin_pct": _pct(i.get("operatingMargins")),
            "profit_margin_pct": _pct(i.get("profitMargins")),
            "return_on_equity_pct": _pct(i.get("returnOnEquity")),
            "debt_to_equity": _num(i.get("debtToEquity")),
            "current_ratio": _num(i.get("currentRatio")),
            "revenue_growth_yoy_pct": _pct(i.get("revenueGrowth")),
            "earnings_growth_yoy_pct": _pct(i.get("earningsGrowth")),
        }),
        "income_statement": income,
        "balance_sheet": balance,
        "cash_flow": cash,
    })


def get_stock_news(symbol: str) -> dict:
    symbol = _symbol(symbol)
    items = []
    for n in (yf.Ticker(symbol).news or [])[:8]:
        c = n.get("content", n)  # newer yfinance nests everything under "content"
        items.append(_clean({
            "title": c.get("title"),
            "publisher": (c.get("provider") or {}).get("displayName") or c.get("publisher"),
            "date": (c.get("pubDate") or "")[:10],
            "summary": (c.get("summary") or c.get("description") or "")[:300],
            "url": (c.get("canonicalUrl") or {}).get("url") or c.get("link"),
        }))
    if not items:
        raise ToolError(f"No recent news found for {symbol}.")
    return {"symbol": symbol, "news": items}


def analyze_stock(symbol: str) -> dict:
    """Everything for a full analysis in ONE call.

    Each extra round of tool calls resends the whole conversation to the
    model, and Groq's free tier allows only ~8,000 tokens per minute. One
    combined, trimmed call keeps a full analysis inside that budget.
    """
    symbol = _symbol(symbol)
    out = {"symbol": symbol}
    parts = {
        "quote": get_stock_quote,
        "price_history": lambda s: get_price_history(s, "1y"),
        "financials": get_financials,
        "news": get_stock_news,
    }
    for key, fn in parts.items():
        try:
            out[key] = fn(symbol)
        except ToolError as e:
            out[key] = {"error": str(e)}
        except Exception as e:
            out[key] = {"error": f"failed ({type(e).__name__})"}
    if "error" in out["quote"] and all("error" in out[k] for k in parts):
        raise ToolError(out["quote"]["error"])

    # Trim the parts the model needs least, to save tokens.
    q = out["quote"]
    if "business_summary" in q:
        q["business_summary"] = q["business_summary"][:250]
    h = out["price_history"]
    if "closes" in h:
        items = list(h["closes"].items())
        h["closes"] = dict(items[::2])  # ~6 points is enough to see the shape
    f = out["financials"]
    for section in ("income_statement", "balance_sheet", "cash_flow"):
        if section in f:
            f[section] = dict(list(f[section].items())[:3])  # last 3 years
    n = out["news"]
    if "news" in n:
        n["news"] = [{k: v for k, v in item.items() if k != "url"} | {"summary": item.get("summary", "")[:160]}
                     for item in n["news"][:5]]
    return out


def web_search(query: str) -> dict:
    query = str(query or "").strip()[:300]
    if not query:
        raise ToolError("Give something to search for.")
    r = httpx.post("https://api.tavily.com/search", timeout=20, json={
        "api_key": TAVILY_API_KEY, "query": query, "max_results": 5,
        "search_depth": "basic", "topic": "news",
    })
    r.raise_for_status()
    return {"query": query, "results": [_clean({
        "title": x.get("title"), "url": x.get("url"),
        "date": (x.get("published_date") or "")[:16],
        "content": (x.get("content") or "")[:500],
    }) for x in r.json().get("results", [])]}


# ---------- what the model is told about the tools ----------

def _fn(name, description, properties, required):
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": properties, "required": required},
    }}


SYMBOL_ARG = {"symbol": {"type": "string",
                         "description": "Yahoo Finance ticker, e.g. AAPL, TCS.NS (NSE), RELIANCE.BO (BSE)"}}

TOOLS = [
    _fn("search_symbol", "Find the ticker symbol for a company name. Use this whenever you are not "
        "sure of the exact symbol.", {"query": {"type": "string", "description": "Company name"}},
        ["query"]),
    _fn("analyze_stock", "Full research pack in one call: quote and valuation, 1-year price trend "
        "and technicals, 3 years of financials, and latest news. Use this for any general analysis, "
        "outlook or 'should I look at X' question. Prefer it over calling the single tools "
        "one by one.", SYMBOL_ARG, ["symbol"]),
    _fn("get_stock_quote", "Current price, valuation (P/E, P/B, market cap), 52-week range, sector, "
        "analyst rating and a short business description.", SYMBOL_ARG, ["symbol"]),
    _fn("get_price_history", "Price trend and technicals: returns over several periods, moving "
        "averages (20/50/200-day), RSI, volatility, drawdown and sampled closing prices.",
        {**SYMBOL_ARG, "period": {"type": "string", "enum": sorted(PERIODS),
                                  "description": "How far back to look. Default 1y."}}, ["symbol"]),
    _fn("get_financials", "Annual income statement, balance sheet and cash flow for the last few "
        "years, plus margins, ROE, debt-to-equity and growth.", SYMBOL_ARG, ["symbol"]),
    _fn("get_stock_news", "Latest news headlines and summaries about the company.", SYMBOL_ARG,
        ["symbol"]),
]
if TAVILY_API_KEY:
    TOOLS.append(_fn("web_search", "Search the web for recent news and analysis (sector trends, "
                     "regulation, management changes, macro events).",
                     {"query": {"type": "string"}}, ["query"]))

FUNCTIONS = {
    "search_symbol": search_symbol,
    "analyze_stock": analyze_stock,
    "get_stock_quote": get_stock_quote,
    "get_price_history": get_price_history,
    "get_financials": get_financials,
    "get_stock_news": get_stock_news,
    "web_search": web_search,
}
FUNCTIONS = {k: v for k, v in FUNCTIONS.items() if any(t["function"]["name"] == k for t in TOOLS)}

# Short labels shown in the chat while a tool runs.
LABELS = {
    "search_symbol": "Looking up symbol",
    "analyze_stock": "Gathering price, financials & news for",
    "get_stock_quote": "Getting quote",
    "get_price_history": "Analysing price history",
    "get_financials": "Reading financials",
    "get_stock_news": "Reading news",
    "web_search": "Searching the web",
}

MAX_RESULT_CHARS = 6000  # keeps one tool result from flooding the model's context


def run_tool(name: str, raw_args: str) -> str:
    """Run one tool call from the model and return the result as JSON text.

    Errors are returned to the model as {"error": ...} instead of crashing, so
    it can fix its mistake (for example, look up the right symbol and retry).
    """
    fn = FUNCTIONS.get(name)
    if fn is None:
        return json.dumps({"error": f"Unknown tool '{name}'."})
    try:
        args = json.loads(raw_args or "{}")
        if not isinstance(args, dict):
            raise ToolError("Arguments must be a JSON object.")
        result = fn(**{k: v for k, v in args.items() if isinstance(k, str)})
    except ToolError as e:
        result = {"error": str(e)}
    except TypeError:
        result = {"error": f"Wrong arguments for {name}."}
    except json.JSONDecodeError:
        result = {"error": "Arguments were not valid JSON."}
    except Exception as e:  # network trouble, Yahoo changes, rate limits...
        result = {"error": f"{name} failed ({type(e).__name__}). The data source may be busy."}
    result_text = json.dumps(result, default=str, ensure_ascii=False)
    if len(result_text) > MAX_RESULT_CHARS:
        result_text = result_text[:MAX_RESULT_CHARS] + "...(truncated)"
    return result_text


def label_for(name: str, raw_args: str) -> str:
    try:
        args = json.loads(raw_args or "{}")
    except json.JSONDecodeError:
        args = {}
    target = args.get("symbol") or args.get("query") or ""
    return f"{LABELS.get(name, name)} {str(target)[:40]}".strip()


def today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")
