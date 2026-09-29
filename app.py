import pandas as pd
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langchain.tools import tool
from dotenv import load_dotenv

load_dotenv()

_df=pd.read_csv("./employees.csv")

@tool
def get_data_info():
  """Get columns, data types, shape, and first 5 rows of the uploaded CSV file"""
  if not _df:
    return "No CSV loaded"

  return f"""
    Shape : {_df.shape},
    Columns & Types : {_df.dtypes.to_string()},
    First 5 Rows : {_df.head().to_string()}
  """

@tool
def filter_data_tool(condition: str):
  """
    Filter rows using a pandas query condition
    Example: 'Age>30 and Department=='ML Engineer'
  """
  if not _df:
      return "No CSV loaded"
  try:
     result=_df.query(condition)
     return f"found {len(result)} rows: {result.to_string()}"
  except Exception as e:
     return f"Error: {e}"


def analyze_data(condition:str):
   """
    Run a pandas expression on the dataframe (referred to as 'df').
    Examples:
    df['Salary'].mean()
    df.groupby('Department')['Salary'].mean()
    df['City'].value_counts()
  """
