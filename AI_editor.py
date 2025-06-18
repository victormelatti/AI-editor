from langchain_openai import ChatOpenAI
from langchain.sql_database import SQLDatabase
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.agents import initialize_agent, AgentType
from sqlalchemy import create_engine
import os
import pandas as pd
import sys
import openai
from langchain.tools import DuckDuckGoSearchRun
from langchain.tools import Tool
from sqlalchemy import text
from sqlalchemy import MetaData

from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper

from langchain.document_loaders import PyMuPDFLoader
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
import regex as re

from langchain_experimental.sql import SQLDatabaseChain
import json
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import datetime
from sqlalchemy import text
from sentence_transformers import SentenceTransformer, util

import secret
from secret import *
import dscc.db_utils
import dscc.folders
conn = dscc.db_utils.get_db_conn(dscc.db_utils.NewAtlanta, username=secret.postgres_user, password=secret.postgres_password)

from dscc.db_utils import NewAtlanta

# modules with functions and descriptions
from utilities import retrieve_pdf_info, format_text
from descriptions import custom_table_info, extract_parameters_from_query

# # relode a module
# import importlib
# import descriptions
# importlib.reload(descriptions)

## Add the API keys
os.environ['OPENAI_API_KEY'] = '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'
os.environ["TAVILY_API_KEY"] = "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"

## Directories
working_directory = os.path.join(dscc.folders.sharepoint_dir, 'NLP_openai','AI editor')
excel_table_folder = os.path.join(working_directory, 'tables') 
figure_folder = os.path.join(working_directory, 'figure')
pdf_path = os.path.join(working_directory, 'docs')

# FAISS storage directory
faiss_index_path = os.path.join(working_directory, "faiss_index")

# Create the folder if it does not exist
for folder in [excel_table_folder, figure_folder]:
    os.makedirs(folder, exist_ok=True)

# connect to DB
from dscc.db_utils import get_conn_string
DATABASE_URL, _ =  get_conn_string(config=NewAtlanta, username=secret.postgres_user, password=secret.postgres_password)

## create FAISS storage or Load the vector store

pdfs = ["Interdisciplinary Science Rankings 2025.pdf", "merged_texts.pdf"]
pdfs_paths = [os.path.join(pdf_path, pdf) for pdf in pdfs]


engine = create_engine(DATABASE_URL, connect_args={"options": "-c search_path=wur"})
metadata = MetaData()
metadata.reflect(bind=engine, views=True)  # Ensure views are included

print(" Check that the tables that the LLM can access are the only ones I specified in custom_table_info", [f"{table}: {table in list(metadata.tables.keys())}" for table in custom_table_info.keys()])


db = SQLDatabase(engine, view_support=True, include_tables=["rnk_wur_2025_latest_vw", "rnk_wur_all_years_latest_vw"], custom_table_info=custom_table_info )
print("Usable tables/views:", db.get_usable_table_names())

# Initialize the language model and toolkit
general_llm = "gpt-4-turbo"
fine_tuened_llm = "ft:gpt-4o-2024-08-06:times-higher-education::AzNdE2kH"

# fine_tuened_llm.invoke("What is the capital of France?")
# replace the llm with the fine-tuned one
# Set the SQLDatabaseChain to return SQL queries only withouth running the query
sql_chain = SQLDatabaseChain.from_llm(ChatOpenAI(model=general_llm), db, return_sql=True, verbose=True) 


#def create_SQL_query(metric: 'id', time_range: [2023, 2025], countries: ['United Kingdom', 'China'], chart_type: 'line', aggregation: 'count', require_visualization = False):
def create_SQL_query(metric = 'id', time_range = [2023, 2025], countries = ['United Kingdom', 'China'], chart_type = 'line', aggregation = 'count', require_visualization = False):
  
    start_year, end_year = time_range
    num_years = end_year - start_year + 1
    aggregation_map = {
        "average": "AVG",
        "count": "COUNT",
        "sum": "SUM",
        "max": "MAX",
        "min": "MIN"
    }
    sql_aggregation = aggregation_map.get(aggregation.lower())
    if not sql_aggregation:
        raise ValueError(f"Unsupported aggregation type: {aggregation}")
        # If counting, switch metric to 'id'
    if aggregation.lower() == "count":
        metric = "id"

    if aggregation.lower() != "count" and num_years:
            filter_condition = (
                f"AND id IN ( "
                f"    SELECT id FROM rnk_wur_all_years_latest_vw "
                f"    WHERE country IN ({countries}) AND subject = 'Overall' "
                f"    AND wur_year BETWEEN {start_year} AND {end_year} "
                f"    GROUP BY id HAVING COUNT(DISTINCT wur_year) = {num_years} "
                f") "
            )
    else:
        filter_condition = ""  # No additional filtering for other aggregation types


    prompt = (
        f"Generate a SQL query for PostgreSQL that returns"
        f"the {sql_aggregation} of '{metric}' (as metric_value) from the table 'rnk_wur_all_years_latest_vw' "
        f"for universities in {countries} where subject = 'Overall' and for the year between {start_year} and {end_year} . "
        f"{filter_condition} "
        f"GROUP BY wur_year, country ORDER BY wur_year ASC, country ASC;"   
        f"The table has a year column named 'wur_year'. "
        f"Group the results by 'wur_year', 'country' and order them in ascending order. "
        f"Return **only** the SQL query as plain text, starting with SELECT. "
        f"Do not include any markdown, explanations, formatting, or additional text—only the SQL query itself."
    )
    
    # Use SQLDatabaseChain to generate the SQL query and request intermediate steps.
    chain_result = sql_chain({"query": prompt})

    print(chain_result['result'])

    sql_query = chain_result['result']
    clean_query = sql_query.replace('"', '').replace('\n', ' ')
    #clean_query = re.sub(r"(?i)^.*?\bSELECT\b", "SELECT", clean_query)
    clean_query = re.sub(r"(?i)^.*?\bSELECT\b", "SELECT", re.sub(r'^```|```$', '', clean_query)).strip()

    print("results: ", clean_query)


    # Execute the generated SQL query.
    with engine.connect() as conn:
        data = conn.execute(text(clean_query))
        # Expecting two columns: one for the year (wur_year) and one for the aggregated value (avg_rank)
        # data = {str(row[0]): row[1] for row in result}
        data = pd.DataFrame(data.fetchall(), columns=data.keys())

    return data


def generate_metric_plot(metric_data, metric = 'id', time_range = [2023, 2025], countries = ['United Kingdom', 'China'], chart_type = 'line', aggregation = 'count', require_visualization = True):
    """
    Generates a ranking trend plot by dynamically retrieving data using SQLDatabaseChain.
    The function obtains data for each year (as generated by the chain), then plots either a line
    or bar chart using Plotly. If the query involves ranking over time, the Y-axis will be inverted.
    Supports multiple countries.
    """

    # Create a DataFrame to store the underlying data
    start_year, end_year = time_range
    metric_name = metric.replace("_", " ").capitalize()
    aggregation_name = aggregation.capitalize()

    df_to_save = metric_data.copy().rename(columns = {'metric_value': f"{aggregation_name} {metric_name}"})

    # Save data to an Excel file
    df_to_save.to_excel(os.path.join(excel_table_folder, "metric_trend_data.xlsx"), index=False)

    # Create the plot using Plotly
    fig = go.Figure()
    #for country in countries:
    for country in countries: 
        fig.add_trace(go.Scatter(x=metric_data[metric_data['country'] == country]['wur_year'], 
                                 y=metric_data[metric_data['country'] == country]['metric_value'],
                                 mode='lines+markers', name=country,
                                 marker=dict(size=8)
                                 ))

    # Customize the layout
    fig.update_layout(
            title=f"{aggregation.capitalize()} {metric.replace('_', ' ').capitalize()} of Universities in {', '.join(countries)} from {start_year} to {end_year}",
            xaxis_title="Year",
            yaxis_title=f"{aggregation.capitalize()} {metric.replace('_', ' ').capitalize()}",
            xaxis=dict(showgrid=True),
            yaxis=dict(showgrid=True, autorange="reversed" if metric == "rank_number" else True),
            template="plotly_white",
            width=800, height=500
        )
    # Save the plot to an HTML file for visualization
    plot_png_path = os.path.join(figure_folder,"metric_trend.png")
    fig.show()
    fig.write_image(plot_png_path, format="png", scale=1)  # scale=2 for higher resolution
    print(plot_png_path)
    return plot_png_path


extract_parameters_from_query = {
                "name": "extract_parameters_from_query",
                "description": (
                    """ Extract parameters from a user query.
                        From the user query, extract the following:
                        - the metric of interest 'metric',
                        - a time period 'time_range' as a list of two years [start_year, end_year],
                        - a list of countries 'countries',
                        - if the user wants a visualization - 'require_visualization',
                        - what type of chart to generate 'chart_type' if the user has specified that a visualization is required (require_visualization=True),
                        - deduce what type of aggregation to apply ('aggregation') to generate a working SQL query that would answer the user query.
                        """
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "metric": {
                            "type": "string",
                            "enum": ["rank_number", "score_number", "id"],
                            "description": "The metric to plot. For 'count', the 'id' column (unique universities) is automatically used."
                        },
                        "time_range": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "description": """Time period for the ranking trend as a list of two years [start_year, end_year]. Current year is 2025, any reference to WUR followed by a number should be intended as a year, i.e WUR23 is 2023, WUR25 is 2025. 
                            If no time range is specified, the current year (2025) must be used is used. If the query mentions a single year, the time range should be [year, year]."""
                        },
                        "countries": {
                            "type": "array", 
                            "items": {"type": "string"}, 
                            "description": "List of countries"
                        },
                        "require_visualization": {
                            "type": "boolean",
                            "default": False,
                            "description": "Whether the user query requires a visualization (plot) or not, option as True or False"
                        },
                        "chart_type": {
                            "type": "string",
                            "enum": ["line", "bar"],
                            "description": "Type of chart to generate as intended by the user"
                        },
                        "aggregation": {
                            "type": "string",
                            "enum": ["average", "count", "sum", "max", "min"],
                            "description": "Aggregation operation to apply (e.g., average, count, maximum, minimum)."
                        }
                    },
                   "required": ["time_range", "countries", "metric",]
                }
            }

function_calling = [extract_parameters_from_query]

def function_calling_db(user_query):
    
    client = openai.OpenAI()

    response = client.chat.completions.create(
        model = fine_tuened_llm, #"gpt-4-turbo", #"gpt-4o", #general_llm, #fine_tuened_llm,  #"gpt-4o", #gpt-4-1106-preview",
        messages=[{"role": "user", "content": user_query}],
        functions = function_calling,
        function_call="auto"
    )

    
    if response.choices[0].message.function_call:
        function_name = response.choices[0].message.function_call.name
        function_args = json.loads(response.choices[0].message.function_call.arguments)
        print(function_args)
        metric_data = create_SQL_query(**function_args)
        if function_args["require_visualization"]:
            plot_path = generate_metric_plot(metric_data, **function_args)

    return metric_data
#------------------------------------------------------

# TOOLS

#------------------------------#------------------------

# Example-based reasoning for tool selection
# Load sentence transformer model for similarity comparison
similarity_model = SentenceTransformer('all-MiniLM-L6-v2')


pdf_retrieval = Tool(
    name="PDF Retrieval",
    func=lambda query: retrieve_pdf_info(query, pdfs_paths, faiss_index_path),
    description="Use this tool to retrieve relevant information from PDF documents about education system of a country or a specific university. Also to retrieve information about news like Brexit or covid for example."
        "If the query asks for details, summaries, or specific content from a PDF, use this tool. "
        "It works best when a user requests text extraction, document insights, or keyword-based searches inside PDFs. "
        "Use this tool for general knowledge, DO NOT USE if the user query specifies a database search."
)

tavily_api = TavilySearchAPIWrapper()
web_search = Tool(
    name="Web Search",
    func=tavily_api.results,
    description=(
        "Use this tool to search for the latest information that is NOT available in the database or from the PDF Retrieval."
        "This includes breaking news, recent updates, and any external knowledge not covered by internal documents or tables. "
        "Ideal for real-time information, current events, and fact-checking. "
        "USE this tool ony if the database queries or PDF searche have not been successful. DO NOT use this tool for database queries or PDF searches."
    )
)

Call_db_and_plotting_tool = Tool(
    name="Call database and Generate Plot",
    func=lambda query: function_calling_db(query),
    description=(
        "This is the PRIMARY tool for ALL database queries about university rankings data, whether they need visualization or not. "
        "This tool COMPLETELY handles: "
        "- Any questions about university counts, scores, or rankings from the database "
        "- Any requests for trends or comparisons over time "
        "- Any visualization needs for ranking data "
        "Examples of queries this tool handles: "
        "- How many universities were ranked in Italy in 2025? "
        "- Plot the average scores for US universities from 2020-2025 "
        "- Compare ranking trends between UK and China "
        "DO NOT try to construct SQL queries or handle database operations separately - this tool manages the entire process."
    )
)

# Create the agent using LangChain's standard framework
agent_executor = initialize_agent(
    tools= [ pdf_retrieval, Call_db_and_plotting_tool, web_search],
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
    llm=ChatOpenAI(model=fine_tuened_llm), #gpt-4-turbo" general_llm, #fine_tuened_llm,
    verbose=True,
    handle_parsing_errors=True
)

# Query the agent with intelligent tool selection
def query_openai_agent(user_query): 
           
    final_query = user_query
    response = agent_executor.run(final_query)
    
    return response


query1 = "Plot in a line chart the average score number of universities ranked in United States and India from 2021 to 2025."
response = query_openai_agent(query1)

query2 = "I want to know the impact of brexit on the UK. use the knowledge from the pdfs you have do not use web search. The response should be 100 words long. Add numerical data that you find in your knoledge to support your answer."
response = query_openai_agent(query2)

query3 = "How many universitis were ranked in in Italy in 2025?"
response = query_openai_agent(query3)
# Example Usage
formatted_response = format_text(response, words_per_line=30)
print(formatted_response)

