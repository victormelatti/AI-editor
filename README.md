# AI Agent for University Rankings and PDF Knowledge Extraction

## 📘 Introduction

This AI agent provides an intelligent interface for querying university ranking data and extracting insights from documents (PDFs) using OpenAI's GPT models. It combines multiple capabilities to allow users to:

- Generate and execute SQL queries on a university ranking database (PostgreSQL backend).
- Automatically extract parameters (e.g., metric, year, countries, visualization) from user queries using fine-tuned GPT models.
- Plot trends (line or bar charts) with `plotly` based on ranking metrics across countries and years.
- Retrieve context-aware information from local PDF documents using semantic search powered by FAISS and `sentence-transformers`.
- Fall back to real-time web search (via Tavily API) when neither database nor PDF documents contain the needed information.

All interactions are managed through a smart LangChain-powered agent using **tool selection** based on query content.

## 🧠 Features

### SQL Database Interaction
- Uses a pre-configured Postgres connection.
- Leverages `langchain_experimental.sql.SQLDatabaseChain` for SQL generation.
- Allows analytical queries using aggregation: average, count, sum, max, min.
- Automatically filters and groups by year, country, and metric.
- Ensures chart generation using `plotly`, optionally saving output to disk.

### Document Retrieval (PDFs)
- Loads PDFs like *Interdisciplinary Science Rankings 2025.pdf* and *merged_texts.pdf*.
- Splits, embeds, and indexes them with FAISS and `sentence-transformers`.
- Enables keyword search and summarization based on semantic match.

### Tool Management via LangChain Agent
- Selects from tools: `Call database and Generate Plot`, `PDF Retrieval`, or `Web Search`.
- Routes queries to the right tool using zero-shot reasoning and a fine-tuned GPT-4 model.
- Handles fallback logic gracefully, and always strives for structured output.

## 🛠️ Setup

Install dependencies from `AI_agent_environment`

Environment variables expected:
- `OPENAI_API_KEY`
- `TAVILY_API_KEY`

## 🚀 Usage

You can use the `query_openai_agent()` function to issue natural language queries. Examples:

```python
query_openai_agent("Plot in a line chart the average score number of universities ranked in United States and India from 2021 to 2025.")
query_openai_agent("How many universities were ranked in Italy in 2025?")
query_openai_agent("I want to know the impact of Brexit on the UK from the PDFs only.")
```

## ✅ Output

- A generated SQL query for metric trends.
- A `metric_trend.png` plot.
- An `metric_trend_data.xlsx` spreadsheet.
- Extracted answers from PDF with optional summarization.

---

For any updates or extension (e.g., new PDFs, new tools, fine-tuned model changes), please ensure you update the relevant lists in the script.


### 🧠 How the AI Agent Queries the Database

The AI agent uses a fine-tuned GPT-4 model and LangChain’s SQL tooling to interpret natural language questions and retrieve relevant university ranking data from a PostgreSQL database.

It extracts structured parameters such as metric, time range, countries, aggregation type, and chart preferences using OpenAI’s function calling. These are converted into a SQL query via the `SQLDatabaseChain` module. The query is then executed using SQLAlchemy, and the results are returned as a DataFrame. If visualization is requested, a chart is generated using Plotly.

This process allows accurate and flexible database access from natural language input.

