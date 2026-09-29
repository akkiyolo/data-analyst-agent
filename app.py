import pandas as pd
import streamlit as st

from typing import Literal

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver


# ============================================================
# CONFIG
# ============================================================

load_dotenv()

st.set_page_config(
    page_title="AI Data Analyst",
    page_icon="📊",
    layout="wide",
)


# ============================================================
# TITLE
# ============================================================

st.title("📊 AI Data Analyst")
st.caption("Ask anything about your CSV document")


# ============================================================
# SESSION STATE
# ============================================================

if "df" not in st.session_state:
    st.session_state.df = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "memory" not in st.session_state:
    st.session_state.memory = InMemorySaver()


# ============================================================
# FILE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload your CSV file",
    type=["csv"],
)


# ============================================================
# LOAD CSV
# ============================================================

if uploaded_file is not None:

    try:
        df = pd.read_csv(uploaded_file)

        st.session_state.df = df

        # Reset chat when a new file is uploaded
        st.session_state.messages = []

        st.success(
            f"Loaded successfully — {df.shape[0]} rows × {df.shape[1]} columns"
        )

    except Exception as e:
        st.error(f"Error loading CSV: {e}")


# ============================================================
# STOP IF NO CSV
# ============================================================

if st.session_state.df is None:

    st.info("👆 Upload a CSV file to start analyzing your data.")

    st.stop()


# ============================================================
# CURRENT DATAFRAME
# ============================================================

_df = st.session_state.df


# ============================================================
# TOOL 1 — GET DATA INFO
# ============================================================

@tool
def get_data_info():
    """
    Get columns, data types, shape, and first 5 rows
    of the uploaded CSV file.
    """

    if _df is None:
        return "No CSV loaded."

    return f"""
Shape:
{_df.shape}

Columns:
{list(_df.columns)}

Columns and Types:
{_df.dtypes.to_string()}

First 5 Rows:
{_df.head().to_string()}
"""


# ============================================================
# TOOL 2 — FILTER DATA
# ============================================================

@tool
def filter_data_tool(condition: str):
    """
    Filter rows using a pandas query condition.

    Example:
    Age > 30 and Department == 'ML Engineer'
    """

    if _df is None:
        return "No CSV loaded."

    try:

        result = _df.query(condition)

        if result.empty:
            return "No rows matched the given condition."

        return f"""
Found {len(result)} rows:

{result.to_string()}
"""

    except Exception as e:

        return f"Error while filtering data: {e}"


# ============================================================
# TOOL 3 — ANALYZE DATA
# ============================================================

@tool
def analyze_data(
    operation: Literal[
        "mean",
        "median",
        "max",
        "min",
        "sum",
        "count",
        "value_counts",
        "highest_row",
        "lowest_row",
    ],
    column: str,
    group_by: str = "",
):
    """
    Analyze a column in the uploaded CSV.

    Supported operations:

    mean
    median
    max
    min
    sum
    count
    value_counts
    highest_row
    lowest_row

    Examples:

    operation="mean", column="salary_inr"

    operation="median", column="salary_inr"

    operation="highest_row", column="salary_inr"

    operation="median",
    column="salary_inr",
    group_by="job_title"
    """

    if _df is None:
        return "No CSV loaded."

    try:

        # ----------------------------------------------------
        # Validate column
        # ----------------------------------------------------

        if column not in _df.columns:

            return (
                f"Column '{column}' was not found.\n\n"
                f"Available columns:\n{list(_df.columns)}"
            )


        # ----------------------------------------------------
        # Grouped analysis
        # ----------------------------------------------------

        if group_by:

            if group_by not in _df.columns:

                return (
                    f"Group-by column '{group_by}' was not found.\n\n"
                    f"Available columns:\n{list(_df.columns)}"
                )

            grouped = _df.groupby(group_by)[column]


            if operation == "mean":

                result = grouped.mean()


            elif operation == "median":

                result = grouped.median()


            elif operation == "max":

                result = grouped.max()


            elif operation == "min":

                result = grouped.min()


            elif operation == "sum":

                result = grouped.sum()


            elif operation == "count":

                result = grouped.count()


            else:

                return (
                    f"Operation '{operation}' "
                    f"does not support group-by analysis."
                )


            return result.sort_values(
                ascending=False
            ).to_string()


        # ----------------------------------------------------
        # Normal analysis
        # ----------------------------------------------------

        if operation == "mean":

            return str(_df[column].mean())


        elif operation == "median":

            return str(_df[column].median())


        elif operation == "max":

            return str(_df[column].max())


        elif operation == "min":

            return str(_df[column].min())


        elif operation == "sum":

            return str(_df[column].sum())


        elif operation == "count":

            return str(_df[column].count())


        elif operation == "value_counts":

            return _df[column].value_counts().to_string()


        elif operation == "highest_row":

            row = _df.loc[
                _df[column].idxmax()
            ]

            return row.to_string()


        elif operation == "lowest_row":

            row = _df.loc[
                _df[column].idxmin()
            ]

            return row.to_string()


        return f"Unknown operation: {operation}"


    except Exception as e:

        return f"Error while analyzing data: {e}"


# ============================================================
# TOOLS
# ============================================================

all_tools = [
    get_data_info,
    filter_data_tool,
    analyze_data,
]


# ============================================================
# LLM
# ============================================================

llm = ChatGroq(
    model="openai/gpt-oss-20b",
)


# ============================================================
# SYSTEM PROMPT
# ============================================================

system_prompt = """
You are an expert data analyst assistant.

Your job is to help the user explore and analyze
the uploaded CSV document.

AVAILABLE TOOLS:

1. get_data_info

Use this when you need to understand:

- column names
- data types
- number of rows
- number of columns
- sample data


2. filter_data_tool

Use this when the user wants to filter
rows based on a condition.


3. analyze_data

Use this for:

- mean
- median
- maximum
- minimum
- sum
- count
- value counts
- highest row
- lowest row

It also supports grouped analysis.

For example:

Question:
"What is the average salary?"

Use:

operation = "mean"
column = "salary_inr"


Question:
"Which employee has the highest salary?"

Use:

operation = "highest_row"
column = "salary_inr"


Question:
"Which job title has the highest median salary?"

Use:

operation = "median"
column = "salary_inr"
group_by = "job_title"


IMPORTANT RULES:

1. Never guess values.

2. Always use the tools to calculate
   answers from the uploaded CSV.

3. If you are unsure about column names,
   call get_data_info first.

4. Never generate arbitrary Python code.

5. Never use eval.

6. Use analyze_data for calculations.

7. Explain the final result clearly.

8. Keep answers concise but useful.

9. If the requested column does not exist,
   tell the user which columns are available.

The uploaded dataframe is the dataset
you should analyze.
"""


# ============================================================
# CREATE AGENT
# ============================================================

if "agent" not in st.session_state:

    st.session_state.agent = create_agent(
        model=llm,
        tools=all_tools,
        system_prompt=system_prompt,
        checkpointer=st.session_state.memory,
    )


agent = st.session_state.agent


# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# ============================================================
# CHAT INPUT
# ============================================================

query = st.chat_input(
    "Ask anything about your document..."
)


# ============================================================
# PROCESS QUERY
# ============================================================

if query:

    # --------------------------------------------------------
    # Add user message
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": query,
        }
    )


    with st.chat_message("user"):

        st.markdown(query)


    # --------------------------------------------------------
    # Agent response
    # --------------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner("Analyzing your data..."):

            try:

                response = agent.invoke(
                    {
                        "messages": [
                            {
                                "role": "user",
                                "content": query,
                            }
                        ]
                    },
                    {
                        "configurable": {
                            "thread_id": "1",
                        }
                    },
                )


                answer = response["messages"][-1].content


                st.markdown(answer)


                # ------------------------------------------------
                # Save assistant response
                # ------------------------------------------------

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                    }
                )


            except Exception as e:

                st.error(
                    f"Error while processing your question: {e}"
                )