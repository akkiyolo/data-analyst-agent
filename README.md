# 📊 AI Data Analyst

An AI-powered CSV data analysis assistant built with **Python, Streamlit, LangGraph, LangChain, Pandas, and Groq**.

Upload a CSV file and ask questions about your data in natural language. The agent automatically selects the appropriate analysis tool to perform calculations, filtering, grouping, and statistical analysis.

## ✨ Features

- 📁 Upload CSV files directly through the Streamlit UI
- 💬 Ask questions about your dataset using natural language
- 🤖 AI-powered analysis using Groq
- 🧠 LangGraph agent orchestration
- 💾 Conversation memory using `InMemorySaver`
- 📊 Statistical analysis
- 🔎 Data filtering
- 📈 Group-by analysis
- 📝 Dataset information and preview
- 🔐 Structured analysis tools without arbitrary `eval()`
- ⚡ Interactive Streamlit interface

## 🛠️ Tech Stack

- **Python**
- **Streamlit**
- **LangGraph**
- **LangChain**
- **Groq**
- **Pandas**
- **python-dotenv**

## 🏗️ Architecture

```text
User
 │
 ▼
Streamlit UI
 │
 ├── CSV Upload
 │
 └── Chat Interface
        │
        ▼
   LangGraph Agent
        │
        ▼
     Groq LLM
        │
        ├───────────────┐
        │               │
        ▼               ▼
  get_data_info   filter_data_tool
        │               │
        └───────┬───────┘
                │
                ▼
          analyze_data
                │
                ▼
             Pandas
                │
                ▼
            CSV Data
