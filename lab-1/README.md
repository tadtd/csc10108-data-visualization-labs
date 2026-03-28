# Shopee Agentic AI Crawler

An agentic AI that crawls product data from **Shopee Vietnam** using:

- **crawl4ai** — stealth browser automation (uses your real Edge profile to bypass anti-bot)
- **LangGraph** — agent orchestration (ReAct pattern)
- **Ollama** — free, local LLM for decision-making and data extraction

The agent autonomously searches Shopee, paginates through results, extracts
structured product data, and saves everything to a CSV file.

## Prerequisites

| Requirement         | Version        | Notes                                   |
|---------------------|----------------|-----------------------------------------|
| Python              | >= 3.13        |                                         |
| Ollama              | latest         | <https://ollama.com/download>           |
| Microsoft Edge      | any recent     | Used for browser profile (cookies etc.) |

## Setup

### 1. Create and activate a virtual environment

```bash
python -m venv .venv

# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1

# Linux / macOS
source .venv/bin/activate
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Install the browser driver

crawl4ai needs a Chromium browser driver. Run **one** of:

```bash
# Option A — crawl4ai's built-in setup
crawl4ai-setup

# Option B — Playwright directly
playwright install chromium
```

### 4. Pull an Ollama model

The default model is `qwen2.5:7b` (good tool-calling support, 32K context).
Pull it with:

```bash
ollama pull qwen2.5:7b
```

Other compatible models (must support tool calling):

- `llama3.1:8b`
- `mistral:7b`
- `qwen2.5:14b` (if you have more VRAM)

### 5. Log in to Shopee in Microsoft Edge

1. Open **Microsoft Edge** normally.
2. Go to <https://shopee.vn> and log in to your account.
3. **Close Edge completely** (check the system tray — Edge may still be
   running in the background; right-click the icon and quit).

> **Why?** The crawler opens Edge using your real browser profile (cookies,
> login session, browsing history). This makes it look like a real user to
> Shopee's anti-bot system. Only one process can use a browser profile at a
> time, so Edge must be closed before running the script.

## Usage

```bash
python main.py --query-file "query.txt"
```

### All arguments

| Argument           | Default                                           | Description                                 |
|--------------------|---------------------------------------------------|---------------------------------------------|
| `--query-file`    | *(required)*                                      | Path to a `.txt` file containing the search query text |
| `--max-pages`      | `3`                                               | Number of result pages to crawl             |
| `--output`         | `products.csv`                                    | Output CSV file path                        |
| `--model`          | `qwen2.5:7b`                                      | Ollama model name (must support tool calling)|
| `--headless`       | `true`                                            | Run browser headlessly (default)           |
| `--no-headless`    | *(flag)*                                         | Run browser with UI (equivalent to `--headless=false`) |
| `--user-data-dir`  | `None`                                            | Optional Edge profile directory for cookie/session reuse |

### Examples

```bash
# Search for laptops, crawl 5 pages, save to laptops.csv
python main.py --user-data-dir "%my_chrome_profile%" --query-file "query.txt" --max-pages 5 --output laptops.csv

# Use a different model
python main.py --query-file "query.txt" --model llama3.1:8b

# Run with UI (helpful for CAPTCHA/debug)
python main.py --query-file "query.txt" --no-headless

# Use a custom browser profile directory
python main.py --query-file "query.txt" --user-data-dir "C:\MyProfile"
```

## Output

The crawler produces a CSV file with these columns:

| Column           | Description                          |
|------------------|--------------------------------------|
| `name`           | Product name                         |
| `price`          | Current/discounted price (VND)       |
| `original_price` | Original price before discount       |
| `discount`       | Discount percentage                  |
| `rating`         | Star rating                          |
| `sold`           | Number of units sold                 |
| `location`       | Seller location                      |
| `shop_name`      | Shop/seller name                     |
| `url`            | Product page URL                     |
| `image_url`      | Product thumbnail URL                |

## Project Structure

```
lab-1/
  main.py              # Entry point — argument parsing + async runner
  crawl/
    __init__.py
    config.py           # BrowserConfig (Edge profile) + CrawlerRunConfig
    schemas.py          # Pydantic model for product data
    tools.py            # LangChain tools: search_shopee, save_products_csv
    agent.py            # LangGraph agent (ReAct pattern with ChatOllama)
  requirements.txt
  README.md
```

## How It Works

```
User runs main.py with --query-file "query.txt"
        │
        ▼
main.py opens a Chromium browser profile via crawl4ai
        │
        ▼
LangGraph agent receives task from system prompt
        │
        ▼
Agent calls search_shopee(page=0) (query is bound from main.py)
        │
        ▼
crawl4ai navigates to shopee.vn/search?keyword=<query>&page=0
  → scrolls page to load all products
  → returns page content as markdown
        │
        ▼
Agent reads markdown, extracts product data
        │
        ▼
Agent calls search_shopee(..., page=1), page=2, etc.
        │
        ▼
Agent calls save_products_csv(json_data, "products.csv")
        │
        ▼
CSV file saved ✓
```

## Troubleshooting

### "Browser profile is locked" / browser won't start

**Cause**: Edge is still running (possibly in background).

**Fix**: Close Edge completely. Check Task Manager or system tray.

### Shopee shows CAPTCHA or blocks the page

**Fix**: Run with `--no-headless` so you can see the browser.
If a CAPTCHA appears, solve it manually — the agent will wait for the page to
load and then continue automatically.

### Ollama model doesn't support tool calling

**Fix**: Use a model that supports tool calling: `qwen2.5:7b`, `llama3.1:8b`,
or `mistral:7b`. Check Ollama's model library for tool-calling support.

### "Connection refused" from Ollama

**Fix**: Make sure Ollama is running. Start it with:

```bash
ollama serve
```

### Empty or incomplete CSV

This can happen if Shopee changes its page structure. Try:

1. Running with `--no-headless` to see what the browser is loading.
2. Reducing `--max-pages` to 1 for a test run.
3. Trying a different Ollama model (larger models extract more reliably).
