# GAIA Agent

This repository contains my submission for the Hugging Face Agents Course Final Assignment.

## Architecture

- **Framework:** `smolagents` (CodeAgent)
- **LLM:** Gemini 2.0 Flash (via OpenAI-compatible endpoint)
- **Tools Used:**
  - `web_search` (DuckDuckGo) for real-time information retrieval.
  - `wikipedia_search` (Custom Tool) for encyclopedic facts.
  - `python_interpreter` for math and data manipulation.

## Features

- **Exact-Match Prompting:** Custom system prompt engineered to strip conversational filler and return only answers.
- **Defensive Formatting:** Post-processing regex ensures dates, names, and lists match the GAIA required formats.
- **Rate-Limiting:** Built-in delays on the web search tool to prevent API blocks.

## How to Run Locally

1. Install requirements: `pip install -r requirements.txt`
2. Set your API key: `export GEMINI_API_KEY="your_key_here"`
3. Launch the UI: `python app.py`
