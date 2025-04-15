
# # import streamlit as st
# # import snowflake.connector
# # import pandas as pd
# # import numpy as np
# # import os
# # import re
# # import json
# # from prompts import generate_introduction, get_llm_response, call_openai
# # from rag_utils import (
# #     initialize_vector_store,
# #     process_query_with_rag,
# #     get_openai_embedding
# # )
# # from openai import OpenAI
# # import plotly.express as px
# # import plotly.graph_objects as go

# # # Set page config
# # st.set_page_config(
# #     page_title="OEE Manager",
# #     page_icon="📊",
# #     layout="wide"
# # )

# # # OpenAI client
# # # client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
# # client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
# # GPT_MODEL = "gpt-4o"  # the newest OpenAI model is "gpt-4o" which was released May 13, 2024.

# # # Initialize session state variables
# # if 'initialized' not in st.session_state:
# #     st.session_state.initialized = False
# # if 'messages' not in st.session_state:
# #     st.session_state.messages = []
# # if 'full_responses' not in st.session_state:
# #     st.session_state.full_responses = []  # For storing full responses with dataframes and visualizations
# # if 'chat_history' not in st.session_state:
# #     st.session_state.chat_history = []  # For the OpenAI conversation history
# # if 'df' not in st.session_state:
# #     st.session_state.df = None
# # if 'vector_store' not in st.session_state:
# #     st.session_state.vector_store = None
# # if 'embedding_status' not in st.session_state:
# #     st.session_state.embedding_status = "Not Started"
# # if 'show_history' not in st.session_state:
# #     st.session_state.show_history = True  # Toggle for conversation memory
# # if 'debug_mode' not in st.session_state:
# #     st.session_state.debug_mode = False  # Toggle for developer mode
# # if 'table_name' not in st.session_state:
# #     st.session_state.table_name = "OEESHIFTWISE_AI"  # Default table name
# # if 'schema_columns' not in st.session_state:
# #     st.session_state.schema_columns = []  # To track schema changes

# # #new changes made here for multiple tables

# # # New variables for multi-table support
# # if 'tables' not in st.session_state:
# #     st.session_state.tables = {}  # Dictionary to store multiple tables and their schemas
# # if 'table_relationships' not in st.session_state:
# #     st.session_state.table_relationships = []  # Store relationships between tables
# # if 'table_descriptions' not in st.session_state:
# #     st.session_state.table_descriptions = {}  # Store descriptions of tables
# # if 'use_multi_table' not in st.session_state:
# #     st.session_state.use_multi_table = False  # Toggle for multi-table mode
# # if 'relationship_description' not in st.session_state:
# #     st.session_state.relationship_description = ""  # Description of table relationships
# # if 'conversation_dates' not in st.session_state:
# #     st.session_state.conversation_dates = {}  # For tracking conversation dates in history
# # if 'history_expander' not in st.session_state:
# #     st.session_state.history_expander = {}  # For managing history expanders
# # if 'conversation_by_date' not in st.session_state:
# #     st.session_state.conversation_by_date = {}  # For ChatGPT-style history
# # if 'selected_conversation' not in st.session_state:
# #     st.session_state.selected_conversation = None  # For tracking selected conversations


# # # Snowflake Connection Functions
# # # def init_snowflake_connection():
# # #     try:
# # #         conn = snowflake.connector.connect(
# # #             user=st.session_state.snowflake_user,
# # #             password=st.session_state.snowflake_password,
# # #             account=st.session_state.snowflake_account,
# # #             warehouse=st.session_state.snowflake_warehouse,
# # #             database='O3_AI_DB',
# # #             schema='O3_AI_DB_SCHEMA'
# # #         )
# # #         return conn
# # #     except Exception as e:
# # #         st.error(f"Error connecting to Snowflake: {e}")
# # #         return None

# # # Snowflake Connection Functions
# # def init_snowflake_connection():
# #     try:
# #         conn = snowflake.connector.connect(
# #             user=st.secrets["snowflake"]["user"],
# #             password=st.secrets["snowflake"]["password"],
# #             account=st.secrets["snowflake"]["account"],
# #             warehouse=st.secrets["snowflake"]["warehouse"],
# #             database='O3_AI_DB',
# #             schema='O3_AI_DB_SCHEMA'
# #         )
# #         return conn
# #     except Exception as e:
# #         st.error(f"Error connecting to Snowflake: {e}")
# #         return None


# # def execute_snowflake_query(query):
# #     conn = init_snowflake_connection()
# #     if conn:
# #         try:
# #             cursor = conn.cursor()
# #             cursor.execute(query)
# #             result = cursor.fetchall()
# #             columns = [desc[0] for desc in cursor.description]
# #             df = pd.DataFrame(result, columns=columns)
# #             cursor.close()
# #             return df
# #         except Exception as e:
# #             st.error(f"Error executing query: {e}")
# #             return None
# #     return None

# # def get_table_schema(table_name, database='O3_AI_DB', schema='O3_AI_DB_SCHEMA'):
# #     """Get the schema information for a specific table."""
# #     query = f"""
# #     SELECT 
# #         COLUMN_NAME, 
# #         DATA_TYPE,
# #         CHARACTER_MAXIMUM_LENGTH,
# #         NUMERIC_PRECISION,
# #         NUMERIC_SCALE,
# #         IS_NULLABLE,
# #         COLUMN_DEFAULT,
# #         COMMENT
# #     FROM 
# #         {database}.INFORMATION_SCHEMA.COLUMNS
# #     WHERE 
# #         TABLE_SCHEMA = '{schema}'
# #         AND TABLE_NAME = '{table_name}'
# #     ORDER BY 
# #         ORDINAL_POSITION
# #     """
    
# #     try:
# #         df = execute_snowflake_query(query)
# #         if df is not None and not df.empty:
# #             return df
# #         else:
# #             st.warning(f"No schema information found for table {table_name}")
# #             return None
# #     except Exception as e:
# #         st.error(f"Error fetching schema for {table_name}: {e}")
# #         return None

# # def get_table_relationships(database='O3_AI_DB', schema='O3_AI_DB_SCHEMA'):
# #     """Get relationships between tables based on foreign keys."""
# #     query = f"""
# #     SELECT
# #         pc.CONSTRAINT_NAME,
# #         pc.TABLE_NAME as PARENT_TABLE,
# #         kcu.COLUMN_NAME as PARENT_COLUMN,
# #         rc.TABLE_NAME as REFERENCED_TABLE,
# #         kcu2.COLUMN_NAME as REFERENCED_COLUMN
# #     FROM
# #         {database}.INFORMATION_SCHEMA.REFERENTIAL_CONSTRAINTS rc
# #     JOIN
# #         {database}.INFORMATION_SCHEMA.TABLE_CONSTRAINTS pc
# #         ON rc.CONSTRAINT_NAME = pc.CONSTRAINT_NAME
# #         AND rc.CONSTRAINT_SCHEMA = pc.CONSTRAINT_SCHEMA
# #     JOIN
# #         {database}.INFORMATION_SCHEMA.KEY_COLUMN_USAGE kcu
# #         ON pc.CONSTRAINT_NAME = kcu.CONSTRAINT_NAME
# #         AND pc.CONSTRAINT_SCHEMA = kcu.CONSTRAINT_SCHEMA
# #     JOIN
# #         {database}.INFORMATION_SCHEMA.TABLE_CONSTRAINTS rc_tc
# #         ON rc.UNIQUE_CONSTRAINT_NAME = rc_tc.CONSTRAINT_NAME
# #         AND rc.UNIQUE_CONSTRAINT_SCHEMA = rc_tc.CONSTRAINT_SCHEMA
# #     JOIN
# #         {database}.INFORMATION_SCHEMA.KEY_COLUMN_USAGE kcu2
# #         ON rc_tc.CONSTRAINT_NAME = kcu2.CONSTRAINT_NAME
# #         AND rc_tc.CONSTRAINT_SCHEMA = kcu2.CONSTRAINT_SCHEMA
# #     WHERE
# #         pc.TABLE_SCHEMA = '{schema}'
# #     ORDER BY
# #         pc.TABLE_NAME, kcu.COLUMN_NAME
# #     """
    
# #     try:
# #         df = execute_snowflake_query(query)
# #         if df is not None and not df.empty:
# #             relationships = []
# #             for _, row in df.iterrows():
# #                 relationships.append({
# #                     'constraint_name': row['CONSTRAINT_NAME'],
# #                     'parent_table': row['PARENT_TABLE'],
# #                     'parent_column': row['PARENT_COLUMN'],
# #                     'referenced_table': row['REFERENCED_TABLE'],
# #                     'referenced_column': row['REFERENCED_COLUMN']
# #                 })
# #             return relationships
# #         else:
# #             # If no relationships found, this is not an error
# #             return []
# #     except Exception as e:
# #         st.error(f"Error fetching table relationships: {e}")
# #         return []

# # def get_all_tables(database='O3_AI_DB', schema='O3_AI_DB_SCHEMA'):
# #     """Get a list of all tables in the specified schema."""
# #     query = f"""
# #     SELECT 
# #         TABLE_NAME, 
# #         COMMENT
# #     FROM 
# #         {database}.INFORMATION_SCHEMA.TABLES
# #     WHERE 
# #         TABLE_SCHEMA = '{schema}'
# #         AND TABLE_TYPE = 'BASE TABLE'
# #     ORDER BY 
# #         TABLE_NAME
# #     """
    
# #     try:
# #         df = execute_snowflake_query(query)
# #         if df is not None and not df.empty:
# #             tables = []
# #             for _, row in df.iterrows():
# #                 tables.append({
# #                     'name': row['TABLE_NAME'],
# #                     'description': row['COMMENT'] if row['COMMENT'] else f"Table {row['TABLE_NAME']}"
# #                 })
# #             return tables
# #         else:
# #             st.warning(f"No tables found in schema {schema}")
# #             return []
# #     except Exception as e:
# #         st.error(f"Error fetching tables: {e}")
# #         return []

# # def initialize_multi_table_environment(database='O3_AI_DB', schema='O3_AI_DB_SCHEMA', table_list=None):
# #     """Initialize the environment with multiple tables and their relationships."""
# #     with st.spinner("Initializing multi-table environment..."):
# #         # Use the two specific tables: OEESHIFTWISE_AI and OEEBREAKDOWNAI
# #         if not table_list:
# #             table_list = ["OEESHIFTWISE_AI", "OEEBREAKDOWNAI"]
            
# #             # Set table descriptions
# #             st.session_state.table_descriptions["OEESHIFTWISE_AI"] = "OEE Shift-wise data containing shift performance metrics"
# #             st.session_state.table_descriptions["OEEBREAKDOWNAI"] = "OEE Breakdown data containing equipment breakdown and maintenance information"
        
# #         # Get schema for each table
# #         for table_name in table_list:
# #             schema_df = get_table_schema(table_name, database, schema)
# #             if schema_df is not None:
# #                 # Get sample data
# #                 sample_query = f"SELECT * FROM {schema}.{table_name} LIMIT 1000"
# #                 sample_df = execute_snowflake_query(sample_query)
                
# #                 # Store table schema and sample data
# #                 st.session_state.tables[table_name] = {
# #                     'schema': schema_df,
# #                     'sample_data': sample_df,
# #                     'column_info': "\n".join([f"- {col}: {sample_df[col].dtype}" for col in sample_df.columns]) if sample_df is not None else ""
# #                 }
# #             else:
# #                 st.warning(f"Could not fetch schema for {table_name}. The table might not exist yet.")
        
# #         # Define a manual relationship between the tables if they have common fields
# #         # Assuming PUID might be a common field between the two tables
# #         st.session_state.table_relationships = [
# #             {
# #                 'constraint_name': 'MANUAL_RELATIONSHIP',
# #                 'parent_table': 'OEESHIFTWISE_AI',
# #                 'parent_column': 'PUID',
# #                 'referenced_table': 'OEEBREAKDOWNAI',
# #                 'referenced_column': 'PUID'
# #             }
# #         ]
        
# #         # Generate a description of the table relationships for the LLM
# #         relationship_descriptions = []
# #         for rel in st.session_state.table_relationships:
# #             relationship_descriptions.append(
# #                 f"Table {rel['parent_table']} has a relationship with {rel['referenced_table']} " +
# #                 f"through the common field {rel['parent_column']}"
# #             )
        
# #         st.session_state.relationship_description = "\n".join(relationship_descriptions)
        
# #         return len(st.session_state.tables) > 0


# # def determine_visualization_type(user_query, sql_query, result_df):
# #     """Determine the appropriate visualization type based on the query and results."""
# #     try:
# #         vis_prompt = f"""
# #         I need to visualize the following SQL query results for the question: "{user_query}"
        
# #         The SQL query was:
# #         ```sql
# #         {sql_query}
# #         ```
        
# #         The query returned {len(result_df)} rows with the following column names and data types:
# #         {[(col, str(result_df[col].dtype)) for col in result_df.columns]}
        
# #         Based on the user's question and the data returned, determine the most appropriate visualization type.
# #         Respond with a JSON object with the following structure:
# #         {{
# #             "viz_type": "line|bar|scatter|pie|histogram|heatmap|none",
# #             "x_column": "name of column to use for x-axis or categories",
# #             "y_column": "name of column to use for y-axis or values",
# #             "color_column": "name of column to use for color differentiation (optional, can be null)",
# #             "title": "Suggested title for the visualization",
# #             "description": "Brief rationale for why this visualization type is appropriate"
# #         }}
        
# #         Only suggest a visualization if it makes sense for the data and query. If visualization is not appropriate, return "viz_type": "none".
# #         """
        
# #         system_prompt = "You are a data visualization expert that chooses appropriate chart types based on query results. Always respond with valid JSON."
        
# #         # Set response format to JSON for structured response
# #         vis_response = client.chat.completions.create(
# #             model=GPT_MODEL,
# #             messages=[
# #                 {"role": "system", "content": system_prompt},
# #                 {"role": "user", "content": vis_prompt}
# #             ],
# #             response_format={"type": "json_object"}
# #         )
        
# #         # Parse the JSON response
# #         vis_recommendation = json.loads(vis_response.choices[0].message.content)
# #         return vis_recommendation
        
# #     except Exception as e:
# #         st.warning(f"Could not determine visualization type: {str(e)}")
# #         return {"viz_type": "none"}

# # def create_visualization(result_df, vis_recommendation):
# #     """Create an appropriate visualization based on the recommendation."""
# #     try:
# #         viz_type = vis_recommendation.get("viz_type", "none")
# #         if viz_type == "none" or len(result_df) == 0:
# #             return None
            
# #         x_col = vis_recommendation.get("x_column")
# #         y_col = vis_recommendation.get("y_column")
# #         color_col = vis_recommendation.get("color_column")
# #         title = vis_recommendation.get("title", "Data Visualization")
        
# #         # Check if the recommended columns exist in the dataframe
# #         available_cols = result_df.columns.tolist()
# #         if x_col and x_col not in available_cols:
# #             x_col = available_cols[0] if available_cols else None
# #         if y_col and y_col not in available_cols:
# #             y_col = available_cols[1] if len(available_cols) > 1 else available_cols[0] if available_cols else None
# #         if color_col and color_col not in available_cols:
# #             color_col = None
            
# #         if not x_col or not y_col:
# #             return None
            
# #         # Create the appropriate plot based on visualization type
# #         if viz_type == "bar":
# #             # Handle aggregation if needed
# #             if len(result_df) > 25:  # Too many bars becomes unreadable
# #                 # Try to aggregate data if it makes sense
# #                 result_df = result_df.groupby(x_col, as_index=False)[y_col].agg('sum')
                
# #             fig = px.bar(result_df, x=x_col, y=y_col, color=color_col, title=title)
            
# #         elif viz_type == "line":
# #             # Sort by x if it's a datetime or numeric column
# #             if pd.api.types.is_datetime64_any_dtype(result_df[x_col]) or pd.api.types.is_numeric_dtype(result_df[x_col]):
# #                 result_df = result_df.sort_values(by=x_col)
                
# #             fig = px.line(result_df, x=x_col, y=y_col, color=color_col, title=title, markers=True)
            
# #         elif viz_type == "scatter":
# #             fig = px.scatter(result_df, x=x_col, y=y_col, color=color_col, title=title)
            
# #         elif viz_type == "pie":
# #             fig = px.pie(result_df, names=x_col, values=y_col, title=title)
            
# #         elif viz_type == "histogram":
# #             fig = px.histogram(result_df, x=x_col, title=title)
            
# #         elif viz_type == "heatmap":
# #             # Create a pivot table for heatmap
# #             if color_col:
# #                 pivot_df = result_df.pivot_table(values=color_col, index=y_col, columns=x_col, aggfunc='mean')
# #                 fig = px.imshow(pivot_df, title=title)
# #             else:
# #                 return None
# #         else:
# #             return None
            
# #         # Style the figure for better appearance
# #         fig.update_layout(
# #             template="plotly_white",
# #             height=500,
# #             margin=dict(l=50, r=50, t=80, b=50)
# #         )
        
# #         return fig
        
# #     except Exception as e:
# #         st.warning(f"Could not create visualization: {str(e)}")
# #         return None

# # def generate_nlp_summary(user_query, sql_query, result_df):
# #     """Generate a natural language summary of SQL query results."""
# #     try:
# #         with st.spinner("Generating natural language summary..."):
# #             nlp_summary_prompt = f"""
# #             I need a natural language summary of the following SQL query results for the question: "{user_query}"
            
# #             The SQL query was:
# #             ```sql
# #             {sql_query}
# #             ```
            
# #             The query returned {len(result_df)} rows with the following data:
# #             {result_df.to_string(index=False, max_rows=10)}
            
# #             Please provide a 1-2 sentence natural language summary of these results that directly answers the user's question.
# #             Focus on the key metrics, highest/lowest values, or trends as appropriate.
# #             Be specific and include the actual values from the data.
# #             """
            
# #             nlp_summary = call_openai(nlp_summary_prompt, "You are a data analyst summarizing SQL query results in plain language.")
# #             return nlp_summary
# #     except Exception as e:
# #         st.warning(f"Could not generate natural language summary: {str(e)}")
# #         return "I couldn't generate a natural language summary for these results."

# # # UI Components
# # # Title and sidebar
# # st.title("OEE Manager")

# # # Sidebar for Snowflake credentials and OpenAI API key
# # with st.sidebar:
# #     # st.header("Connection Settings")
    
# #     # # Check OpenAI API key
# #     api_key = st.secrets.get("OPENAI_API_KEY")
# #     # if api_key:
# #     #     st.success("OpenAI API Key is configured")
# #     # else:
# #     #     st.error("OpenAI API Key is missing")
# #     #     st.info("Please add your OpenAI API key to use this application")
    
# #     st.header("Chat Settings")
# #     # Toggle for conversation memory
# #     st.session_state.show_history = st.checkbox("Enable conversation memory", value=st.session_state.show_history)
# #     # if st.session_state.show_history:
# #     #     st.success("Conversation memory is enabled")
# #     #     st.info("The chatbot will remember previous messages for context")
# #     # else:
# #     #     st.info("Conversation memory is disabled")
        
# #     # Developer Mode toggle
# #     if 'debug_mode' not in st.session_state:
# #         st.session_state.debug_mode = False
# #     st.session_state.debug_mode = st.checkbox("Developer Mode", value=st.session_state.debug_mode)
# #     # if st.session_state.debug_mode:
# #     #     st.success("Developer Mode is enabled")
# #     #     st.info("SQL queries will be shown in responses")
# #     # else:
# #     #     st.info("Developer Mode is disabled")
# #     #     st.info("SQL queries will be hidden in responses")

# #         # Multi-table Mode toggle
# #     st.divider()
# #     st.session_state.use_multi_table = st.checkbox("Enable Multi-table Mode", value=st.session_state.use_multi_table)
# #     if st.session_state.use_multi_table:
# #         st.success("Multi-table Mode is enabled")
# #         st.info("The assistant will analyze relationships between tables and join them when needed")
        
# #         # Initialize multi-table environment if not already done
# #         if not st.session_state.tables:
# #             if st.button("Initialize Multi-table Environment"):
# #                 with st.spinner("Loading tables and relationships..."):
# #                     if initialize_multi_table_environment():
# #                         st.success(f"Successfully loaded {len(st.session_state.tables)} tables and {len(st.session_state.table_relationships)} relationships")
# #                     else:
# #                         st.error("Failed to initialize multi-table environment")
# #         else:
# #             st.success(f"Loaded {len(st.session_state.tables)} tables and {len(st.session_state.table_relationships)} relationships")
# #     else:
# #         st.info("Multi-table Mode is disabled")
# #         st.info("The assistant will only use the primary table for queries")
    
# #     # Chat History Section - ChatGPT Style
# #     st.header("Chat History")
# #     if not st.session_state.messages:
# #         st.info("No chat history yet. Start a conversation!")
# #     else:
# #         # Group conversations by date
# #         from datetime import datetime, timedelta
        
# #         # Initialize conversation tracking by date
# #         if 'conversation_by_date' not in st.session_state:
# #             st.session_state.conversation_by_date = {}
        
# #         # Get current date, yesterday, and other dates
# #         now = datetime.now()
# #         today = now.strftime("%Y-%m-%d")
# #         yesterday = (now - timedelta(days=1)).strftime("%Y-%m-%d")
        
# #         # For new conversations, initialize with today's date
# #         if len(st.session_state.conversation_by_date) == 0:
# #             st.session_state.conversation_by_date[today] = []
        
# #         # Get all user messages (queries) for display
# #         user_conversations = []
# #         for i, msg in enumerate(st.session_state.messages):
# #             if i % 2 == 0 and i < len(st.session_state.messages):  # User messages (even indices)
# #                 # If this is a new conversation, add it to today's list
# #                 if not any(msg["content"] == conv["query"] for date_convs in st.session_state.conversation_by_date.values() for conv in date_convs):
# #                     if today not in st.session_state.conversation_by_date:
# #                         st.session_state.conversation_by_date[today] = []
# #                     st.session_state.conversation_by_date[today].append({"query": msg["content"], "index": i})
# #                 user_conversations.append({"query": msg["content"], "index": i})
        
# #         # Display conversations grouped by Today, Yesterday, and older dates
        
# #         # First Today's conversations
# #         if today in st.session_state.conversation_by_date and st.session_state.conversation_by_date[today]:
# #             st.markdown("**Today**", unsafe_allow_html=True)
# #             # Create a dark container for today's conversations
# #             with st.container(border=False):
# #                 for conv in st.session_state.conversation_by_date[today]:
# #                     # Truncate long messages
# #                     truncated_query = conv["query"]
# #                     if len(truncated_query) > 40:
# #                         truncated_query = truncated_query[:37] + "..."
                    
# #                     # Create a button with the conversation title
# #                     if st.button(truncated_query, key=f"today_{conv['index']}"):
# #                         st.session_state.selected_conversation = conv["query"]
# #                         # In a real implementation, you could scroll to or show this conversation
        
# #         # Then Yesterday's conversations
# #         if yesterday in st.session_state.conversation_by_date and st.session_state.conversation_by_date[yesterday]:
# #             st.markdown("**Yesterday**", unsafe_allow_html=True)
# #             for conv in st.session_state.conversation_by_date[yesterday]:
# #                 # Truncate long messages
# #                 truncated_query = conv["query"]
# #                 if len(truncated_query) > 40:
# #                     truncated_query = truncated_query[:37] + "..."
                
# #                 # Create a button with the conversation title
# #                 if st.button(truncated_query, key=f"yesterday_{conv['index']}"):
# #                     st.session_state.selected_conversation = conv["query"]
        
# #         # Finally older conversations
# #         older_dates = [date for date in st.session_state.conversation_by_date.keys() 
# #                       if date != today and date != yesterday]
        
# #         for date in sorted(older_dates, reverse=True):
# #             if st.session_state.conversation_by_date[date]:
# #                 # Format the date in a more readable way
# #                 display_date = datetime.strptime(date, "%Y-%m-%d").strftime("%B %d")
# #                 st.markdown(f"**{display_date}**", unsafe_allow_html=True)
                
# #                 for conv in st.session_state.conversation_by_date[date]:
# #                     # Truncate long messages
# #                     truncated_query = conv["query"]
# #                     if len(truncated_query) > 40:
# #                         truncated_query = truncated_query[:37] + "..."
                    
# #                     # Create a button with the conversation title
# #                     if st.button(truncated_query, key=f"older_{date}_{conv['index']}"):
# #                         st.session_state.selected_conversation = conv["query"]
    
# #     # Clear history button
# #     if st.button("Clear Chat History"):
# #         st.session_state.messages = []
# #         st.session_state.chat_history = []
# #         st.session_state.full_responses = []
# #         st.session_state.conversation_dates = {}
# #         st.session_state.conversation_by_date = {}
# #         st.session_state.selected_conversation = None
# #         st.success("Chat history cleared!")
# #         st.rerun()
    
# #     st.header("Snowflake Connection")
    
# #     # if not st.session_state.initialized:
# #     #     st.session_state.snowflake_user = st.text_input("Snowflake Username")
# #     #     st.session_state.snowflake_password = st.text_input("Snowflake Password", type="password")
# #     #     st.session_state.snowflake_account = st.text_input("Snowflake Account")
# #     #     st.session_state.snowflake_warehouse = st.text_input("Snowflake Warehouse")
        
# #     #     connect_button = st.button("Connect")
        
# #     #     if connect_button:
# #     #         # Verify API key
# #     #         if not api_key:
# #     #             st.error("Please provide an OpenAI API key before connecting")
# #     #             st.stop()
                
# #     #         # Now connect to Snowflake
# #     #         conn = init_snowflake_connection()
# #     #         if conn:
# #     #             st.success("Connected to Snowflake!")

# #     # Check Snowflake credentials
# #     snowflake_creds = st.secrets.get("snowflake")
# #     if snowflake_creds and all(k in snowflake_creds for k in ["user", "password", "account", "warehouse"]):
# #         st.success("Snowflake credentials are configured")
        
# #         # Info about current table mode
# #         if st.session_state.use_multi_table:
# #             st.success("Multi-table mode is enabled")
# #             st.info("The chatbot will analyze relationships between tables and select the appropriate table for each query")
# #         else:
# #             st.info("Using single table mode with table: " + st.session_state.table_name)
# #         # Auto-connect to Snowflake if not initialized
# #         if not st.session_state.initialized:
# #             # Initialize connection
# #             with st.spinner("Connecting to Snowflake..."):
# #                 # Verify API key
# #                 if not api_key:
# #                     st.error("Please provide an OpenAI API key before connecting")
# #                     st.stop()
                    
# #                 # Now connect to Snowflake
# #                 conn = init_snowflake_connection()
# #                 if conn:
# #                     st.success("Connected to Snowflake!")

# #                     if st.session_state.use_multi_table:
# #                         # Initialize multi-table environment
# #                         if initialize_multi_table_environment():
# #                             st.success("Initialized multi-table environment successfully!")
                            
# #                             # Combine all sample data for vector store
# #                             all_dfs = []
# #                             for table_name, table_data in st.session_state.tables.items():
# #                                 if table_data['sample_data'] is not None:
# #                                     # Add table name as a column for identification
# #                                     df_copy = table_data['sample_data'].copy()
# #                                     df_copy['source_table'] = table_name
# #                                     all_dfs.append(df_copy)
                            
# #                             if all_dfs:
# #                                 # Concatenate all dataframes for the vector store
# #                                 combined_df = pd.concat(all_dfs, ignore_index=True)
                                
# #                                 # Initialize vector store with combined data
# #                                 with st.spinner("Initializing vector store for semantic search..."):
# #                                     st.session_state.embedding_status = "In Progress"
# #                                     st.session_state.vector_store = initialize_vector_store(combined_df, "all_tables")
# #                                     st.session_state.embedding_status = "Complete"
                                
# #                                 # Generate introduction for all tables
# #                                 intro = f"""
# #                                 Welcome to the OEE Manager! I can help you analyze data from the following tables:
                                
# #                                 {', '.join(st.session_state.tables.keys())}
                                
# #                                 These tables contain manufacturing equipment effectiveness data. I can help you:
# #                                 - Query specific metrics from any table
# #                                 - Join data across tables to answer complex questions
# #                                 - Visualize OEE trends and performance indicators
                                
# #                                 What would you like to know about your manufacturing data?
# #                                 """
# #                                 st.session_state.messages.append({"role": "assistant", "content": intro})
                                
# #                                 # Mark as initialized
# #                                 st.session_state.initialized = True
# #                                 st.rerun()
# #                             else:
# #                                 st.error("Could not load sample data from any tables")
# #                         else:
# #                             st.error("Failed to initialize multi-table environment")
# #                     else:
# #                         # Single table initialization (original behavior)
# #                         # Get sample data for the model to understand the schema
# #                         with st.spinner("Fetching sample data..."):
# #                           sample_query = f"SELECT * FROM {st.session_state.table_name} LIMIT 1000"
# #                           df = execute_snowflake_query(sample_query)
# #                           if df is not None:
# #                            st.session_state.df = df
# #                            # Save schema information for checking changes later
# #                            st.session_state.schema_columns = list(df.columns)
                           
# #                            # Format column info for the LLM
# #                            column_info = "\n".join([f"- {col}: {df[col].dtype}" for col in df.columns])
                                
# #                           # Initialize vector store
# #                           with st.spinner("Initializing vector store for semantic search..."):
# #                                     st.session_state.embedding_status = "In Progress"
# #                                     st.session_state.vector_store = initialize_vector_store(df, st.session_state.table_name)
# #                                     st.session_state.embedding_status = "Complete"  # Store column names
                    
# #                           # Generate introduction about the data
# #                           introduction = generate_introduction(column_info, st.session_state.table_name)
# #                           # schema_info = {col: str(df[col].dtype) for col in df.columns}
# #                           # introduction = generate_introduction(schema_info, table_name=st.session_state.table_name)
# #                           st.session_state.messages.append({"role": "assistant", "content": introduction})
                          
# #                           # Mark as initialized
# #                           st.session_state.initialized = True
# #                           st.rerun()
                          
# #                     # Add introduction to full_responses
# #                     st.session_state.full_responses.append({
# #                         "user_query": "Hi, can you tell me about the OEE data?",
# #                         "text_response": introduction,
# #                         "data": None,
# #                         "visualization": None,
# #                         "visualization_notes": None
# #                     })
# #                     st.session_state.initialized = True
                    
# #                     # Initialize vector store with the data
# #                     progress_placeholder = st.empty()
# #                     progress_placeholder.info("Creating embeddings and initializing vector store...")
# #                     st.session_state.embedding_status = "In Progress"
# #                     st.session_state.vector_store = initialize_vector_store(df)
# #                     st.session_state.embedding_status = "Completed"
# #                     progress_placeholder.success("Embeddings created successfully!")
# #     else:
# #         st.success("Connected to Snowflake!")
# #         st.info(f"Current table: {st.session_state.table_name}")
# #         st.info(f"Embedding Status: {st.session_state.embedding_status}")
        
# #         if st.button("Disconnect"):
# #             st.session_state.initialized = False
# #             st.session_state.messages = []
# #             st.session_state.chat_history = []  # Clear conversation history as well
# #             st.session_state.full_responses = []  # Clear full responses with tables and visualizations
# #             st.session_state.df = None
# #             st.session_state.vector_store = None
# #             st.session_state.embedding_status = "Not Started"
# #             st.rerun()

# # # Chat interface
# # if st.session_state.initialized:
# #     # Display chat messages and full responses (with tables and visualizations)
# #     if len(st.session_state.full_responses) > 0:
# #         # Display messages with full content (tables and visualizations)
# #         for idx, response in enumerate(st.session_state.full_responses):
# #             # Display user message
# #             with st.chat_message("user"):
# #                 st.write(response.get("user_query", ""))
            
# #             # Display assistant response with tables and visualizations
# #             with st.chat_message("assistant"):
# #                 st.write(response.get("text_response", ""))
                
# #                 # Display data table if available
# #                 if response.get("data") is not None:
# #                     st.dataframe(response["data"])
                
# #                 # Display visualization if available
# #                 if response.get("visualization") is not None:
# #                     st.plotly_chart(response["visualization"], use_container_width=True)
# #                     if response.get("visualization_notes"):
# #                         st.caption(response["visualization_notes"])
# #     else:
# #         # Fall back to just displaying text messages if no full responses exist yet
# #         for message in st.session_state.messages:
# #             with st.chat_message(message["role"]):
# #                 st.write(message["content"])
    
# #     # User input
# #     user_query = st.chat_input("What would you like to know about the OEE data?")
# #     if user_query:
# #         # Add to display messages
# #         st.session_state.messages.append({"role": "user", "content": user_query})
# #         # Add to chat history for OpenAI (only if history is enabled)
# #         if st.session_state.show_history:
# #             st.session_state.chat_history.append({"role": "user", "content": user_query})
        
# #         with st.chat_message("user"):
# #             st.write(user_query)
        
# #         with st.spinner("Generating response..."):
# #             # Get column info for context
# #             column_info = {col: str(st.session_state.df[col].dtype) for col in st.session_state.df.columns}
            
# #             # Prepare conversation history if enabled
# #             conversation_history = st.session_state.chat_history if st.session_state.show_history else None
            
# #             # Process query with RAG
# #             if st.session_state.vector_store and st.session_state.embedding_status == "Completed":
# #                 rag_response = process_query_with_rag(
# #                     user_query=user_query,
# #                     vector_store=st.session_state.vector_store,
# #                     table_name=st.session_state.table_name,
# #                     schema_name="O3_AI_DB_SCHEMA",
# #                     database_name="O3_AI_DB",
# #                     column_info=column_info,
# #                     conversation_history=conversation_history
# #                 )
                
# #                 # Extract SQL query from response if available
# #                 if "```sql" in rag_response:
# #                     sql_query = rag_response.split("```sql")[1].split("```")[0].strip()
                    
# #                     # Execute SQL query
# #                     result_df = execute_snowflake_query(sql_query)
                    
# #                     # Format final response
# #                     if result_df is not None:
# #                         # Generate natural language summary
# #                         nlp_summary = generate_nlp_summary(user_query, sql_query, result_df)
                        
# #                         # Format response based on debug mode
# #                         if st.session_state.debug_mode:
# #                             # Show SQL in debug mode
# #                             final_response = f"Based on your question, I generated this SQL query:\n```sql\n{sql_query}\n```\n\n"
# #                             # Add the natural language summary
# #                             final_response += f"{nlp_summary}\n\n"
# #                             # Add the result
# #                             final_response += "Here are the detailed results:\n"
# #                         else:
# #                             # Hide SQL in regular mode - just show the summary
# #                             final_response = f"{nlp_summary}\n\n"
# #                             # Add the result
# #                             final_response += "Here are the detailed results:\n"
                        
# #                         # Determine appropriate visualization
# #                         with st.spinner("Creating visualization..."):
# #                             vis_recommendation = determine_visualization_type(user_query, sql_query, result_df)
# #                             fig = create_visualization(result_df, vis_recommendation)
                        
# #                         with st.chat_message("assistant"):
# #                             st.write(final_response)
                            
# #                             # Display the data table
# #                             st.dataframe(result_df)
                            
# #                             # Display visualization if available
# #                             if fig:
# #                                 st.plotly_chart(fig, use_container_width=True)
# #                                 description = vis_recommendation.get("description", "")
# #                                 if description:
# #                                     st.caption(f"Visualization notes: {description}")
                        
# #                         # Save message without the dataframe (for OpenAI history)
# #                         response_content = final_response + "[Results shown in table format and visualization]"
# #                         st.session_state.messages.append({
# #                             "role": "assistant", 
# #                             "content": response_content
# #                         })
                        
# #                         # Store full response with dataframe and visualization for display
# #                         visualization_notes = ""
# #                         if fig and vis_recommendation.get("description"):
# #                             visualization_notes = f"Visualization notes: {vis_recommendation.get('description')}"
                            
# #                         st.session_state.full_responses.append({
# #                             "user_query": user_query,
# #                             "text_response": final_response,
# #                             "data": result_df,
# #                             "visualization": fig,
# #                             "visualization_notes": visualization_notes,
# #                             "sql_query": sql_query if st.session_state.debug_mode else None
# #                         })
                        
# #                         # Add to chat history for OpenAI (only if history is enabled)
# #                         if st.session_state.show_history:
# #                             st.session_state.chat_history.append({
# #                                 "role": "assistant", 
# #                                 "content": response_content
# #                             })
# #                     else:
# #                         # Format error response based on debug mode
# #                         if st.session_state.debug_mode:
# #                             final_response = f"I generated this SQL query, but there was an error executing it:\n```sql\n{sql_query}\n```"
# #                         else:
# #                             final_response = f"I couldn't retrieve the data you asked for. There might be an issue with the query or connection."
                        
# #                         with st.chat_message("assistant"):
# #                             st.write(final_response)
# #                         st.session_state.messages.append({"role": "assistant", "content": final_response})
                        
# #                         # Store error in full_responses for consistent display
# #                         st.session_state.full_responses.append({
# #                             "user_query": user_query,
# #                             "text_response": final_response,
# #                             "data": None,
# #                             "visualization": None,
# #                             "visualization_notes": None,
# #                             "sql_query": sql_query if st.session_state.debug_mode else None
# #                         })
                        
# #                         # Add to chat history for OpenAI (only if history is enabled)
# #                         if st.session_state.show_history:
# #                             st.session_state.chat_history.append({"role": "assistant", "content": final_response})
# #                 else:
# #                     # If no SQL was generated
# #                     with st.chat_message("assistant"):
# #                         st.write(rag_response)
# #                     st.session_state.messages.append({"role": "assistant", "content": rag_response})
                    
# #                     # Store in full_responses for consistent display
# #                     st.session_state.full_responses.append({
# #                         "user_query": user_query,
# #                         "text_response": rag_response,
# #                         "data": None,
# #                         "visualization": None,
# #                         "visualization_notes": None
# #                     })
                    
# #                     # Add to chat history for OpenAI (only if history is enabled)
# #                     if st.session_state.show_history:
# #                         st.session_state.chat_history.append({"role": "assistant", "content": rag_response})
# #             else:
# #                 # Fallback to regular LLM response if vector store not ready
# #                 llm_response = get_llm_response(
# #                     user_query=user_query,
# #                     table_name=st.session_state.table_name,
# #                     schema_name="O3_AI_DB_SCHEMA",
# #                     database_name="O3_AI_DB",
# #                     column_info=column_info,
# #                     conversation_history=conversation_history
# #                 )
# #                 with st.chat_message("assistant"):
# #                     st.write(llm_response)
# #                 st.session_state.messages.append({"role": "assistant", "content": llm_response})
                
# #                 # Store in full_responses for consistent display
# #                 st.session_state.full_responses.append({
# #                     "user_query": user_query,
# #                     "text_response": llm_response,
# #                     "data": None,
# #                     "visualization": None,
# #                     "visualization_notes": None
# #                 })
                
# #                 # Add to chat history for OpenAI (only if history is enabled)
# #                 if st.session_state.show_history:
# #                     st.session_state.chat_history.append({"role": "assistant", "content": llm_response})
# # else:
# #     st.info("Please connect to Snowflake to use the chatbot.")


































































# import streamlit as st
# import snowflake.connector
# import pandas as pd
# import numpy as np
# import os
# import re
# import json
# from prompts import generate_introduction, get_llm_response, call_openai
# from rag_utils import (
#     initialize_vector_store,
#     process_query_with_rag,
#     get_openai_embedding
# )
# from openai import OpenAI
# import plotly.express as px
# import plotly.graph_objects as go
# from pathlib import Path
# import uuid
# unique_key = f"fig_chart_{uuid.uuid4()}"

# # Safe full paths to avatars inside src folder
# BASE_DIR = Path(__file__).parent
# user_avatar = (BASE_DIR / "user.png").resolve().as_posix()
# assistant_avatar = (BASE_DIR / "Assistant.png").resolve().as_posix()


# # Set page config
# st.set_page_config(
#     page_title="O3 Agent",
#     page_icon="📊",
#     layout="wide"
# )

# # OpenAI client
# # client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
# client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
# GPT_MODEL = "gpt-3.5-turbo"  # the newest OpenAI model is "gpt-4o" which was released May 13, 2024.

# # Initialize session state variables
# if 'initialized' not in st.session_state:
#     st.session_state.initialized = False
# if 'messages' not in st.session_state:
#     st.session_state.messages = []
# if 'full_responses' not in st.session_state:
#     st.session_state.full_responses = []  # For storing full responses with dataframes and visualizations
# if 'chat_history' not in st.session_state:
#     st.session_state.chat_history = []  # For the OpenAI conversation history
# if 'df' not in st.session_state:
#     st.session_state.df = None
# if 'vector_store' not in st.session_state:
#     st.session_state.vector_store = None
# if 'embedding_status' not in st.session_state:
#     st.session_state.embedding_status = "Not Started"
# if 'show_history' not in st.session_state:
#     st.session_state.show_history = True  # Toggle for conversation memory
# if 'debug_mode' not in st.session_state:
#     st.session_state.debug_mode = False  # Toggle for developer mode
# if 'table_name' not in st.session_state:
#     st.session_state.table_name = "OEESHIFTWISE_AI"  # Default table name
# if 'schema_columns' not in st.session_state:
#     st.session_state.schema_columns = []  # To track schema changes

# # Snowflake Connection Functions
# # def init_snowflake_connection():
# #     try:
# #         conn = snowflake.connector.connect(
# #             user=st.session_state.snowflake_user,
# #             password=st.session_state.snowflake_password,
# #             account=st.session_state.snowflake_account,
# #             warehouse=st.session_state.snowflake_warehouse,
# #             database='O3_AI_DB',
# #             schema='O3_AI_DB_SCHEMA'
# #         )
# #         return conn
# #     except Exception as e:
# #         st.error(f"Error connecting to Snowflake: {e}")
# #         return None

# # Snowflake Connection Functions
# def init_snowflake_connection():
#     try:
#         conn = snowflake.connector.connect(
#             user=st.secrets["snowflake"]["user"],
#             password=st.secrets["snowflake"]["password"],
#             account=st.secrets["snowflake"]["account"],
#             warehouse=st.secrets["snowflake"]["warehouse"],
#             # database='O3_AI_DB',
#             # schema='O3_AI_DB_SCHEMA'
#             database="O3_AI_DB",
#             schema="O3_AI_DB_SCHEMA"
           
#         )
#         return conn
#     except Exception as e:
#         st.error(f"Error connecting to Snowflake: {e}")
#         return None


# def execute_snowflake_query(query):
#     conn = init_snowflake_connection()
#     if conn:
#         try:
#             cursor = conn.cursor()
#             cursor.execute(query)
#             result = cursor.fetchall()
#             columns = [desc[0] for desc in cursor.description]
#             df = pd.DataFrame(result, columns=columns)
#             cursor.close()
#             return df
#         except Exception as e:
#             st.error(f"Error executing query: {e}")
#             return None
#     return None

# def determine_visualization_type(user_query, sql_query, result_df):
#     """Determine the appropriate visualization type based on the query and results."""
#     try:
#         vis_prompt = f"""
#         I need to visualize the following SQL query results for the question: "{user_query}"

#         The SQL query was:
#         ```sql
#         {sql_query}
#         ```

#         The query returned {len(result_df)} rows with the following column names and data types:
#         {[(col, str(result_df[col].dtype)) for col in result_df.columns]}

#         Based on the user's question and the data returned, determine the most appropriate visualization type.
#         Respond with a JSON object with the following structure:
#         {{
#             "viz_type": "line|bar|scatter|pie|histogram|heatmap|none",
#             "x_column": "name of column to use for x-axis or categories",
#             "y_column": "name of column to use for y-axis or values",
#             "color_column": "name of column to use for color differentiation (optional, can be null)",
#             "title": "Suggested title for the visualization",
#             "description": "Brief rationale for why this visualization type is appropriate"
#         }}

#         Only suggest a visualization if it makes sense for the data and query. If visualization is not appropriate, return "viz_type": "none".
#         """

#         system_prompt = "You are a data visualization expert that chooses appropriate chart types based on query results. Always respond with valid JSON."

#         # Set response format to JSON for structured response
#         vis_response = client.chat.completions.create(
#             model=GPT_MODEL,
#             messages=[
#                 {"role": "system", "content": system_prompt},
#                 {"role": "user", "content": vis_prompt}
#             ],
#             response_format={"type": "json_object"}
#         )

#         # Parse the JSON response
#         # print(vis_response['usage'])
#         # st.write(vis_response['usage'])
#         vis_recommendation = json.loads(vis_response.choices[0].message.content)
#         return vis_recommendation

#     except Exception as e:
#         st.warning(f"Could not determine visualization type: {str(e)}")
#         return {"viz_type": "none"}

# def create_visualization(result_df, vis_recommendation):
#     """Create an appropriate visualization based on the recommendation."""
#     try:
#         viz_type = vis_recommendation.get("viz_type", "none")
#         if viz_type == "none" or len(result_df) == 0:
#             return None
        
#         # 🌈 Custom color palette
#         custom_palette = ["#242bf0", "#7ECF9A"]


#         x_col = vis_recommendation.get("x_column")
#         y_col = vis_recommendation.get("y_column")
#         color_col = vis_recommendation.get("color_column")
#         title = vis_recommendation.get("title", "Data Visualization")

#         # Check if the recommended columns exist in the dataframe
#         available_cols = result_df.columns.tolist()
#         if x_col and x_col not in available_cols:
#             x_col = available_cols[0] if available_cols else None
#         if y_col and y_col not in available_cols:
#             y_col = available_cols[1] if len(available_cols) > 1 else available_cols[0] if available_cols else None
#         if color_col and color_col not in available_cols:
#             color_col = None

#         if not x_col or not y_col:
#             return None

#         # Create the appropriate plot based on visualization type
#         if viz_type == "bar":
#             # Handle aggregation if needed
#             if len(result_df) > 25:  # Too many bars becomes unreadable
#                 # Try to aggregate data if it makes sense
#                 result_df = result_df.groupby(x_col, as_index=False)[y_col].agg('sum')

#             fig = px.bar(result_df, x=x_col, y=y_col, color=color_col, title=title,color_discrete_sequence=custom_palette)

#         elif viz_type == "line":
#             # Sort by x if it's a datetime or numeric column
#             if pd.api.types.is_datetime64_any_dtype(result_df[x_col]) or pd.api.types.is_numeric_dtype(result_df[x_col]):
#                 result_df = result_df.sort_values(by=x_col)

#             fig = px.line(result_df, x=x_col, y=y_col, color=color_col, title=title, markers=True)

#         elif viz_type == "scatter":
#             fig = px.scatter(result_df, x=x_col, y=y_col, color=color_col, title=title)

#         elif viz_type == "pie":
#             fig = px.pie(result_df, names=x_col, values=y_col, title=title)

#         elif viz_type == "histogram":
#             fig = px.histogram(result_df, x=x_col, title=title,color_discrete_sequence=custom_palette)

#         elif viz_type == "heatmap":
#             # Create a pivot table for heatmap
#             if color_col:
#                 pivot_df = result_df.pivot_table(values=color_col, index=y_col, columns=x_col, aggfunc='mean')
#                 fig = px.imshow(pivot_df, title=title)
#             else:
#                 return None
#         else:
#             return None

#         # Style the figure for better appearance
#         fig.update_layout(
#             template="plotly_dark",
#             height=500,
#             margin=dict(l=50, r=50, t=80, b=50)
#         )

#         return fig

#     except Exception as e:
#         st.warning(f"Could not create visualization: {str(e)}")
#         return None

# def generate_nlp_summary(user_query, sql_query, result_df):
#     """Generate a natural language summary of SQL query results."""
#     try:
#         with st.spinner("Generating natural language summary..."):
#             nlp_summary_prompt = f"""
#             I need a natural language summary of the following SQL query results for the question: "{user_query}"

#             The SQL query was:
#             ```sql
#             {sql_query}
#             ```

#             The query returned {len(result_df)} rows with the following data:
#             {result_df.to_string(index=False, max_rows=10)}

#             Please provide a 1-2 sentence natural language summary of these results that directly answers the user's question.
#             Focus on the key metrics, highest/lowest values, or trends as appropriate.
#             Be specific and include the actual values from the data.
#             """

#             nlp_summary = call_openai(nlp_summary_prompt, "You are a data analyst summarizing SQL query results in plain language.")
#             return nlp_summary
#     except Exception as e:
#         st.warning(f"Could not generate natural language summary: {str(e)}")
#         return "I couldn't generate a natural language summary for these results."

# # UI Components
# # Title and sidebar
# # st.title("OEE MONITORING AI AGENT")
# # st.markdown("""
# # <h1 style='
# #     text-align: center;
# #     font-size: 50px;
# #     font-weight: 900;
# #     letter-spacing: 2px;
# #     color: #1f3b73;
# #     text-shadow: 2px 2px 4px #7ECF9A;
# # '>
# #   OEE MONITORING AI AGENT
# # </h1>
# # """, unsafe_allow_html=True)

# # st.markdown("""
# # <h1 style='
# #     text-align: center;
# #     font-size: 50px;
# #     font-weight: 900;
# #     letter-spacing: 2px;

# #     color: rgb(81,103,246);
# #     text-shadow: 2px 2px 4px #7ECF9A;'
# #   OEE MONITORING AI AGENT
# # </h1>
# # """, unsafe_allow_html=True)



# st.markdown("""
# <h1 style='
#     text-align: center;
#     font-size: 50px;
#     font-weight: 900;
#     letter-spacing: 2px;
#     color: rgb(36, 43, 240);
#     text-shadow: 2px 2px 4px #7ECF9A;
# '>
#   O3 AI AGENT
# </h1>
# """, unsafe_allow_html=True)

# # st.markdown("""
# # <h1 style='text-align: center; font-size: 50px;'>
# #   <span style='color:#1f3b73;'>OEE MONITORING</span>
# #   <span style='color:#7ECF9A;'>AI AGENT</span>
# # </h1>
# # """, unsafe_allow_html=True)



# # Sidebar for Snowflake credentials and OpenAI API key
# with st.sidebar:
#     st.header("Connection Settings")

#     # Check OpenAI API key
#     # api_key = os.environ.get("OPENAI_API_KEY")
#     # api_key = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
#     api_key = st.secrets.get("OPENAI_API_KEY")
#     if api_key:
#         st.success("OpenAI API Key is configured")
#     else:
#         st.error("OpenAI API Key is missing")
#         st.info("Please add your OpenAI API key to use this application")

#     st.header("Chat Settings")
#     # Toggle for conversation memory
#     st.session_state.show_history = st.checkbox("Enable conversation memory", value=st.session_state.show_history)
#     if st.session_state.show_history:
#         st.success("Conversation memory is enabled")
#         st.info("The chatbot will remember previous messages for context")
#     else:
#         st.info("Conversation memory is disabled")

#     # Developer Mode toggle
#     if 'debug_mode' not in st.session_state:
#         st.session_state.debug_mode = False
#     st.session_state.debug_mode = st.checkbox("Developer Mode", value=st.session_state.debug_mode)
#     if st.session_state.debug_mode:
#         st.success("Developer Mode is enabled")
#         st.info("SQL queries will be shown in responses")
#     else:
#         st.info("Developer Mode is disabled")
#         st.info("SQL queries will be hidden in responses")

#     if st.button("Clear Chat History"):
#         st.session_state.messages = []
#         st.session_state.chat_history = []
#         st.session_state.full_responses = []
#         st.success("Chat history cleared!")
#         st.rerun()

#     st.header("Snowflake Connection")

#     # if not st.session_state.initialized:
#     #     st.session_state.snowflake_user = st.text_input("Snowflake Username")
#     #     st.session_state.snowflake_password = st.text_input("Snowflake Password", type="password")
#     #     st.session_state.snowflake_account = st.text_input("Snowflake Account")
#     #     st.session_state.snowflake_warehouse = st.text_input("Snowflake Warehouse")

#     #     connect_button = st.button("Connect")

#     #     if connect_button:
#     #         # Verify API key
#     #         if not api_key:
#     #             st.error("Please provide an OpenAI API key before connecting")
#     #             st.stop()

#     #         # Now connect to Snowflake
#     #         conn = init_snowflake_connection()
#     #         if conn:
#     #             st.success("Connected to Snowflake!")

#     # Check Snowflake credentials
#     snowflake_creds = st.secrets.get("snowflake")
#     if snowflake_creds and all(k in snowflake_creds for k in ["user", "password", "account", "warehouse"]):
#         # st.success("Snowflake credentials are configured")

#         # Auto-connect to Snowflake if not initialized
#         if not st.session_state.initialized:
#             # Initialize connection
#             with st.spinner("Connecting to Snowflake..."):
#                 # Verify API key
#                 if not api_key:
#                     st.error("Please provide an OpenAI API key before connecting")
#                     st.stop()

#                 # Now connect to Snowflake
#                 conn = init_snowflake_connection()
#                 if conn:
#                     st.success("Connected to Snowflake!")
#                 # Get sample data for the model to understand the schema
#                 with st.spinner("Fetching sample data..."):
#                   sample_query = f"SELECT * FROM {st.session_state.table_name} LIMIT 1000"
#                   df = execute_snowflake_query(sample_query)
#                   if df is not None:
#                     st.session_state.df = df
#                     # Save schema information for checking changes later
#                     st.session_state.schema_columns = list(df.columns)  # Store column names

#                     # Generate introduction about the data
#                     schema_info = {col: str(df[col].dtype) for col in df.columns}
#                     introduction = generate_introduction(schema_info, table_name=st.session_state.table_name)
#                     st.session_state.messages.append({"role": "assistant", "content": introduction})

#                     # Add introduction to full_responses
#                     st.session_state.full_responses.append({
#                         "user_query": "Hi, can you tell me about the OEE data?",
#                         "text_response": introduction,
#                         "data": None,
#                         "visualization": None,
#                         "visualization_notes": None
#                     })
#                     st.session_state.initialized = True

#                     # Initialize vector store with the data
#                     progress_placeholder = st.empty()
#                     progress_placeholder.info("Creating embeddings and initializing vector store...")
#                     st.session_state.embedding_status = "In Progress"
#                     st.session_state.vector_store = initialize_vector_store(df)
#                     st.session_state.embedding_status = "Completed"
#                     progress_placeholder.success("Embeddings created successfully!")
#     else:
#         # st.success("Connected to Snowflake!")
#         st.info(f"Current table: {st.session_state.table_name}")
#         st.info(f"Embedding Status: {st.session_state.embedding_status}")

#         if st.button("Disconnect"):
#             st.session_state.initialized = False
#             st.session_state.messages = []
#             st.session_state.chat_history = []  # Clear conversation history as well
#             st.session_state.full_responses = []  # Clear full responses with tables and visualizations
#             st.session_state.df = None
#             st.session_state.vector_store = None
#             st.session_state.embedding_status = "Not Started"
#             st.rerun()

# # Chat interface
# if st.session_state.initialized:
#     # Display chat messages and full responses (with tables and visualizations)
#     if len(st.session_state.full_responses) > 0:
#         # Display messages with full content (tables and visualizations)
#         for idx, response in enumerate(st.session_state.full_responses):
#             # Display user message
#             with st.chat_message("user",avatar=user_avatar):
#                 st.write(response.get("user_query", ""))

#             # Display assistant response with tables and visualizations
#             with st.chat_message("assistant",avatar=assistant_avatar):
#                 st.write(response.get("text_response", ""))

#                 # Display data table if available
#                 if response.get("data") is not None:
#                     st.dataframe(response["data"])

#                 # Display visualization if available
#                 if response.get("visualization") is not None:
#                     st.plotly_chart(response["visualization"], use_container_width=True,key=f"main_fig_chart_{idx}")
#                     if response.get("visualization_notes"):
#                         st.caption(response["visualization_notes"])
#     else:
#         # Fall back to just displaying text messages if no full responses exist yet
#         for message in st.session_state.messages:
#             with st.chat_message(message["role"]):
#                 st.write(message["content"])

#     # User input
#     user_query = st.chat_input("What would you like to know about the OEE data?")
#     if user_query:
#         # Add to display messages
#         st.session_state.messages.append({"role": "user", "content": user_query})
#         # Add to chat history for OpenAI (only if history is enabled)
#         if st.session_state.show_history:
#             st.session_state.chat_history.append({"role": "user", "content": user_query})

#         with st.chat_message("user",avatar=user_avatar):
#             st.write(user_query)

#         with st.spinner("Generating response..."):
#             # Get column info for context
#             column_info = {col: str(st.session_state.df[col].dtype) for col in st.session_state.df.columns}

#             # Prepare conversation history if enabled
#             conversation_history = st.session_state.chat_history if st.session_state.show_history else None

#             # Process query with RAG
#             if st.session_state.vector_store and st.session_state.embedding_status == "Completed":
#                 rag_response = process_query_with_rag(
#                     user_query=user_query,
#                     vector_store=st.session_state.vector_store,
#                     table_name=st.session_state.table_name,
#                     # schema_name="O3_AI_DB_SCHEMA",
#                     # database_name="O3_AI_DB",
#                     schema_name="O3_AI_DB_SCHEMA",
#                     database_name="O3_AI_DB",
#                     column_info=column_info,
#                     conversation_history=conversation_history
#                 )

#                 # Extract SQL query from response if available
#                 if "```sql" in rag_response:
#                     sql_query = rag_response.split("```sql")[1].split("```")[0].strip()

#                     # Execute SQL query
#                     result_df = execute_snowflake_query(sql_query)

#                     # Format final response
#                     if result_df is not None and not result_df.empty:
#                         # Generate natural language summary
#                         nlp_summary = generate_nlp_summary(user_query, sql_query, result_df)

#                         # Format response based on debug mode
#                         if st.session_state.debug_mode:
#                             # Show SQL in debug mode
#                             final_response = f"Based on your question, I generated this SQL query:\n```sql\n{sql_query}\n```\n\n"
#                             # Add the natural language summary
#                             final_response += f"{nlp_summary}\n\n"
#                             # Add the result
#                             final_response += "Here are the detailed results:\n"
#                         else:
#                             # Hide SQL in regular mode - just show the summary
#                             final_response = f"{nlp_summary}\n\n"
#                             # Add the result
#                             final_response += "Here are the detailed results:\n"

#                         # Determine appropriate visualization
#                         with st.spinner("Creating visualization..."):
#                             vis_recommendation = determine_visualization_type(user_query, sql_query, result_df)
#                             # 🔍 Debug: show what the LLM recommended
#                             from datetime import datetime  # 💡 put this at the top of your file if not already

#                             # 📦 Build log entry
#                             log_entry = {
#                               "user_query": user_query,
#                               "sql_query": sql_query,
#                               "vis_recommendation": vis_recommendation
# }

#                             # 🕒 Add timestamp to log
#                             log_entry["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

#                             # 📝 Save it to a file called vis_logs.txt
#                             with open("vis_logs.txt", "a", encoding="utf-8") as log_file:
#                                 log_file.write(json.dumps(log_entry, indent=2))
#                                 log_file.write("\n\n" + "="*80 + "\n\n")


#                             fig = create_visualization(result_df, vis_recommendation)

#                         with st.chat_message("assistant",avatar=assistant_avatar):
#                             st.write(final_response)

#                             # Display the data table
#                             st.dataframe(result_df)

#                             # Display visualization if available
#                             if fig:
#                                 st.plotly_chart(fig, use_container_width=True,key="main_fig_chart")
#                                 description = vis_recommendation.get("description", "")
#                                 if description:
#                                     st.caption(f"Visualization notes: {description}")

#                         # Save message without the dataframe (for OpenAI history)
#                         response_content = final_response + "[Results shown in table format and visualization]"
#                         st.session_state.messages.append({
#                             "role": "assistant", 
#                             "content": response_content
#                         })

#                         # Store full response with dataframe and visualization for display
#                         visualization_notes = ""
#                         if fig and vis_recommendation.get("description"):
#                             visualization_notes = f"Visualization notes: {vis_recommendation.get('description')}"

#                         st.session_state.full_responses.append({
#                             "user_query": user_query,
#                             "text_response": final_response,
#                             "data": result_df,
#                             "visualization": fig,
#                             "visualization_notes": visualization_notes,
#                             "sql_query": sql_query if st.session_state.debug_mode else None
#                         })

#                         # Add to chat history for OpenAI (only if history is enabled)
#                         if st.session_state.show_history:
#                             st.session_state.chat_history.append({
#                                 "role": "assistant", 
#                                 "content": response_content
#                             })

#                     elif result_df is not None and result_df.empty:
#                         no_data_msg='I apologize,no results found regarding your query.'
#                         with st.chat_message("assistant",avatar=assistant_avatar):
#                             st.warning(no_data_msg)  
#                         st.session_state.messages.append({"role":"assistant","content":no_data_msg})
#                         st.session_state.full_responses.append({
#                             "user_query":user_query,
#                             "text_response":no_data_msg,
#                             "data": None,
#                             "visualization_notes": None,
#                             "sql_query": sql_query if st.session_state.debug_mode else None
#                         }) 
#                         if st.session_state.show_history:
#                             st.session_state.chat_history.append({"role":"assistant","content": no_data_msg})         
#                     else:
#                         # Format error response based on debug mode
#                         if st.session_state.debug_mode:
#                             final_response = f"I generated this SQL query, but there was an error executing it:\n```sql\n{sql_query}\n```"
#                         else:
#                             final_response = f"I couldn't retrieve the data you asked for. There might be an issue with the query or connection."

#                         with st.chat_message("assistant",avatar=assistant_avatar):
#                             st.write(final_response)
#                         st.session_state.messages.append({"role": "assistant", "content": final_response})

#                         # Store error in full_responses for consistent display
#                         st.session_state.full_responses.append({
#                             "user_query": user_query,
#                             "text_response": final_response,
#                             "data": None,
#                             "visualization": None,
#                             "visualization_notes": None,
#                             "sql_query": sql_query if st.session_state.debug_mode else None
#                         })

#                         # Add to chat history for OpenAI (only if history is enabled)
#                         if st.session_state.show_history:
#                             st.session_state.chat_history.append({"role": "assistant", "content": final_response})
#                 else:
#                     # If no SQL was generated
#                     with st.chat_message("assistant",avatar=assistant_avatar):
#                         st.write(rag_response)
#                     st.session_state.messages.append({"role": "assistant", "content": rag_response})

#                     # Store in full_responses for consistent display
#                     st.session_state.full_responses.append({
#                         "user_query": user_query,
#                         "text_response": rag_response,
#                         "data": None,
#                         "visualization": None,
#                         "visualization_notes": None
#                     })

#                     # Add to chat history for OpenAI (only if history is enabled)
#                     if st.session_state.show_history:
#                         st.session_state.chat_history.append({"role": "assistant", "content": rag_response})
#             else:
#                 # Fallback to regular LLM response if vector store not ready
#                 llm_response = get_llm_response(
#                     user_query=user_query,
#                     table_name=st.session_state.table_name,
#                     schema_name="O3_AI_DB",
#                     database_name="O3_AI_DB_SCHEMA",
#                     column_info=column_info,
#                     conversation_history=conversation_history
#                 )
#                 with st.chat_message("assistant",avatar=assistant_avatar):
#                     st.write(llm_response)
#                 st.session_state.messages.append({"role": "assistant", "content": llm_response})

#                 # Store in full_responses for consistent display
#                 st.session_state.full_responses.append({
#                     "user_query": user_query,
#                     "text_response": llm_response,
#                     "data": None,
#                     "visualization": None,
#                     "visualization_notes": None
#                 })

#                 # Add to chat history for OpenAI (only if history is enabled)
#                 if st.session_state.show_history:
#                     st.session_state.chat_history.append({"role": "assistant", "content": llm_response})
# else:
#     st.info("Please connect to Snowflake to use the chatbot.")

































# import streamlit as st
# import snowflake.connector
# import pandas as pd
# import numpy as np
# import os
# import re
# import json
# from datetime import datetime

# from prompts import generate_introduction, get_llm_response, call_openai
# from rag_utils import (
#     initialize_vector_store,
#     process_query_with_rag,
#     get_openai_embedding
# )
# from openai import OpenAI
# import plotly.express as px
# import plotly.graph_objects as go
# from pathlib import Path
# import uuid
# unique_key = f"fig_chart_{uuid.uuid4()}"

# # Safe full paths to avatars inside src folder
# BASE_DIR = Path(__file__).parent
# user_avatar = (BASE_DIR / "user.png").resolve().as_posix()
# assistant_avatar = (BASE_DIR / "Assistant.png").resolve().as_posix()


# # Set page config
# st.set_page_config(
#     page_title="OEE MONITORING AGENT",
#     page_icon="📊",
#     layout="wide"
# )

# # OpenAI client
# # client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
# client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
# GPT_MODEL = "gpt-4o"  # the newest OpenAI model is "gpt-4o" which was released May 13, 2024.

# # Initialize session state variables
# if 'initialized' not in st.session_state:
#     st.session_state.initialized = False
# # if 'show_welcome_message' not in st.session_state:
# #     st.session_state.show_welcome_message = True

# if 'messages' not in st.session_state:
#     st.session_state.messages = []
# if 'full_responses' not in st.session_state:
#     st.session_state.full_responses = []  # For storing full responses with dataframes and visualizations
# if 'chat_history' not in st.session_state:
#     st.session_state.chat_history = []  # For the OpenAI conversation history
# if 'df' not in st.session_state:
#     st.session_state.df = None
# if 'vector_store' not in st.session_state:
#     st.session_state.vector_store = None
# if 'embedding_status' not in st.session_state:
#     st.session_state.embedding_status = "Not Started"
# if 'show_history' not in st.session_state:
#     st.session_state.show_history = True  # Toggle for conversation memory
# if 'debug_mode' not in st.session_state:
#     st.session_state.debug_mode = False  # Toggle for developer mode
# if 'table_name' not in st.session_state:
#     st.session_state.table_name = "OEESHIFTWISE_AI"  # Default table name
# if 'schema_columns' not in st.session_state:
#     st.session_state.schema_columns = []  # To track schema changes

# # Snowflake Connection Functions
# # def init_snowflake_connection():
# #     try:
# #         conn = snowflake.connector.connect(
# #             user=st.session_state.snowflake_user,
# #             password=st.session_state.snowflake_password,
# #             account=st.session_state.snowflake_account,
# #             warehouse=st.session_state.snowflake_warehouse,
# #             database='O3_AI_DB',
# #             schema='O3_AI_DB_SCHEMA'
# #         )
# #         return conn
# #     except Exception as e:
# #         st.error(f"Error connecting to Snowflake: {e}")
# #         return None

# # Snowflake Connection Functions
# def init_snowflake_connection():
#     try:
#         conn = snowflake.connector.connect(
#             user=st.secrets["snowflake"]["user"],
#             password=st.secrets["snowflake"]["password"],
#             account=st.secrets["snowflake"]["account"],
#             warehouse=st.secrets["snowflake"]["warehouse"],
#             # database='O3_AI_DB',
#             # schema='O3_AI_DB_SCHEMA'
#             database="O3_AI_DB",
#             schema="O3_AI_DB_SCHEMA"
           
#         )
#         return conn
#     except Exception as e:
#         st.error(f"Error connecting to Snowflake: {e}")
#         return None


# def execute_snowflake_query(query):
#     conn = init_snowflake_connection()
#     if conn:
#         try:
#             cursor = conn.cursor()
#             cursor.execute(query)
#             result = cursor.fetchall()
#             columns = [desc[0] for desc in cursor.description]
#             df = pd.DataFrame(result, columns=columns)
#             cursor.close()
#             return df
#         except Exception as e:
#             st.error(f"Error executing query: {e}")
#             return None
#     return None

# def determine_visualization_type(user_query, sql_query, result_df):
#     """Determine the appropriate visualization type based on the query and results."""
#     try:
#         vis_prompt = f"""
#         I need to visualize the following SQL query results for the question: "{user_query}"

#         The SQL query was:
#         ```sql
#         {sql_query}
#         ```

#         The query returned {len(result_df)} rows with the following column names and data types:
#         {[(col, str(result_df[col].dtype)) for col in result_df.columns]}

#         Based on the user's question and the data returned, determine the most appropriate visualization type.
#         Respond with a JSON object with the following structure:
#         {{
#             "viz_type": "line|bar|scatter|pie|histogram|heatmap|none",
#             "x_column": "name of column to use for x-axis or categories",
#             "y_column": "name of column to use for y-axis or values",
#            "color_column": "name of column to use for color differentiation (optional, can be null)",

#             "title": "Suggested title for the visualization",
#             "description": "Brief rationale for why this visualization type is appropriate"
#         }}

#         Only suggest a visualization if it makes sense for the data and query.
#         """

#         system_prompt = "You are a data visualization expert that chooses appropriate chart types based on query results. Always respond with valid JSON."

#         # Set response format to JSON for structured response
#         vis_response = client.chat.completions.create(
#             model=GPT_MODEL,
#             messages=[
#                 {"role": "system", "content": system_prompt},
#                 {"role": "user", "content": vis_prompt}
#             ],
#             response_format={"type": "json_object"}
#         )

#         # Parse the JSON response
#         # print(vis_response['usage'])
#         # st.write(vis_response['usage'])
#         vis_recommendation = json.loads(vis_response.choices[0].message.content)
#         return vis_recommendation

#     except Exception as e:
#         st.warning(f"Could not determine visualization type: {str(e)}")
#         return {"viz_type": "none"}

# def create_visualization(result_df, vis_recommendation):
#     """Create an appropriate visualization based on the recommendation."""
#     try:
#         viz_type = vis_recommendation.get("viz_type", "none")
#         if viz_type == "none" or len(result_df) == 0:
#             return None
        
#         # 🌈 Custom color palette
#         custom_palette = ["#242bf0", "#7ECF9A"]


#         x_col = vis_recommendation.get("x_column")
#         y_col = vis_recommendation.get("y_column")
#         color_col = vis_recommendation.get("color_column")
#         title = vis_recommendation.get("title", "Data Visualization")

#         # Check if the recommended columns exist in the dataframe
#         available_cols = result_df.columns.tolist()
#         if x_col and x_col not in available_cols:
#             x_col = available_cols[0] if available_cols else None
#         if y_col and y_col not in available_cols:
#             y_col = available_cols[1] if len(available_cols) > 1 else available_cols[0] if available_cols else None
#         if color_col and color_col not in available_cols:
#             color_col = None

#         if not x_col or not y_col:
#             return None

#         # Create the appropriate plot based on visualization type
#         if viz_type == "bar":
#             # Handle aggregation if needed
#             if len(result_df) > 25:  # Too many bars becomes unreadable
#                 # Try to aggregate data if it makes sense
#                 result_df = result_df.groupby(x_col, as_index=False)[y_col].agg('sum')

#             fig = px.bar(result_df, x=x_col, y=y_col, color=color_col, title=title,color_discrete_sequence=custom_palette)
#             # fig = px.bar(result_df, x=x_col, y=y_col, title=title,color_discrete_sequence=custom_palette)

#         elif viz_type == "line":
#             # Sort by x if it's a datetime or numeric column
#             if pd.api.types.is_datetime64_any_dtype(result_df[x_col]) or pd.api.types.is_numeric_dtype(result_df[x_col]):
#                 result_df = result_df.sort_values(by=x_col)

#             fig = px.line(result_df, x=x_col, y=y_col, color=color_col, title=title, markers=True)
#             # fig = px.line(result_df, x=x_col, y=y_col, title=title, markers=True,color_discrete_sequence=custom_palette)

#         elif viz_type == "scatter":
#             fig = px.scatter(result_df, x=x_col, y=y_col, color=color_col, title=title)
#             # fig = px.scatter(result_df, x=x_col, y=y_col, title=title,color_discrete_sequence=custom_palette)

#         elif viz_type == "pie":
#             fig = px.pie(result_df, names=x_col, values=y_col, title=title,color_discrete_sequence=custom_palette)

#         elif viz_type == "histogram":
#             fig = px.histogram(result_df, x=x_col, title=title,color_discrete_sequence=custom_palette)

#         elif viz_type == "heatmap":
#             # Create a pivot table for heatmap
#             if color_col:
#                 pivot_df = result_df.pivot_table(values=color_col, index=y_col, columns=x_col, aggfunc='mean')
#                 fig = px.imshow(pivot_df, title=title)
#             else:
#                 return None
#         else:
#             return None

#         # Style the figure for better appearance
#         fig.update_layout(
#             template="plotly_dark",
#             height=500,
#             margin=dict(l=50, r=50, t=80, b=50)
#         )

#         return fig

#     except Exception as e:
#         st.warning(f"Could not create visualization: {str(e)}")
#         return None

# def generate_nlp_summary(user_query, sql_query, result_df):
#     """Generate a natural language summary of SQL query results."""
#     try:
#         with st.spinner("Generating natural language summary..."):
#             nlp_summary_prompt = f"""
#             I need a natural language summary of the following SQL query results for the question: "{user_query}"

#             The SQL query was:
#             ```sql
#             {sql_query}
#             ```

#             The query returned {len(result_df)} rows with the following data:
#             {result_df.to_string(index=False, max_rows=10)}

#             Please provide a 1-2 sentence natural language summary of these results that directly answers the user's question.
#             Focus on the key metrics, highest/lowest values, or trends as appropriate.
#             Be specific and include the actual values from the data.
#             """

#             nlp_summary = call_openai(nlp_summary_prompt, "You are a data analyst summarizing SQL query results in plain language.")
#             return nlp_summary
#     except Exception as e:
#         st.warning(f"Could not generate natural language summary: {str(e)}")
#         return "I couldn't generate a natural language summary for these results."

# # UI Components
# # Title and sidebar
# # st.title("OEE MONITORING AI AGENT")
# # st.markdown("""
# # <h1 style='
# #     text-align: center;
# #     font-size: 50px;
# #     font-weight: 900;
# #     letter-spacing: 2px;
# #     color: #1f3b73;
# #     text-shadow: 2px 2px 4px #7ECF9A;
# # '>
# #   OEE MONITORING AI AGENT
# # </h1>
# # """, unsafe_allow_html=True)

# # st.markdown("""
# # <h1 style='
# #     text-align: center;
# #     font-size: 50px;
# #     font-weight: 900;
# #     letter-spacing: 2px;

# #     color: rgb(81,103,246);
# #     text-shadow: 2px 2px 4px #7ECF9A;'
# #   OEE MONITORING AI AGENT
# # </h1>
# # """, unsafe_allow_html=True)



# # st.markdown("""
# # <h1 style='
# #     text-align: center;
# #     font-size: 50px;
# #     font-weight: 900;
# #     letter-spacing: 2px;
# #     color: rgb(36, 43, 240);
# #     text-shadow: 2px 2px 4px #7ECF9A;
# # '>
# #   OEE MONITORING AI AGENT
# # </h1>
# # """, unsafe_allow_html=True)

# # st.markdown("""
# # <h1 style='text-align: center; font-size: 50px;'>
# #   <span style='color:#1f3b73;'>OEE MONITORING</span>
# #   <span style='color:#7ECF9A;'>AI AGENT</span>
# # </h1>
# # """, unsafe_allow_html=True)



# # Sidebar for Snowflake credentials and OpenAI API key
# with st.sidebar:
#     st.header("Connection Settings")

#     # Check OpenAI API key
#     # api_key = os.environ.get("OPENAI_API_KEY")
#     # api_key = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
#     api_key = st.secrets.get("OPENAI_API_KEY")
#     if api_key:
#         st.success("OpenAI API Key is configured")
#     else:
#         st.error("OpenAI API Key is missing")
#         st.info("Please add your OpenAI API key to use this application")

#     st.header("Chat Settings")
#     # Toggle for conversation memory
#     st.session_state.show_history = st.checkbox("Enable conversation memory", value=st.session_state.show_history)
#     if st.session_state.show_history:
#         st.success("Conversation memory is enabled")
#         st.info("The chatbot will remember previous messages for context")
#     else:
#         st.info("Conversation memory is disabled")

#     # Developer Mode toggle
#     if 'debug_mode' not in st.session_state:
#         st.session_state.debug_mode = False
#     st.session_state.debug_mode = st.checkbox("Developer Mode", value=st.session_state.debug_mode)
#     if st.session_state.debug_mode:
#         st.success("Developer Mode is enabled")
#         st.info("SQL queries will be shown in responses")
#     else:
#         st.info("Developer Mode is disabled")
#         st.info("SQL queries will be hidden in responses")

#     if st.button("Clear Chat History"):
#         st.session_state.messages = []
#         st.session_state.chat_history = []
#         st.session_state.full_responses = []
#         st.success("Chat history cleared!")
#         st.rerun()

#     st.header("Snowflake Connection")

#     # if not st.session_state.initialized:
#     #     st.session_state.snowflake_user = st.text_input("Snowflake Username")
#     #     st.session_state.snowflake_password = st.text_input("Snowflake Password", type="password")
#     #     st.session_state.snowflake_account = st.text_input("Snowflake Account")
#     #     st.session_state.snowflake_warehouse = st.text_input("Snowflake Warehouse")

#     #     connect_button = st.button("Connect")

#     #     if connect_button:
#     #         # Verify API key
#     #         if not api_key:
#     #             st.error("Please provide an OpenAI API key before connecting")
#     #             st.stop()

#     #         # Now connect to Snowflake
#     #         conn = init_snowflake_connection()
#     #         if conn:
#     #             st.success("Connected to Snowflake!")

#     # Check Snowflake credentials
#     snowflake_creds = st.secrets.get("snowflake")
#     if snowflake_creds and all(k in snowflake_creds for k in ["user", "password", "account", "warehouse"]):
#         # st.success("Snowflake credentials are configured")

#         # Auto-connect to Snowflake if not initialized
#         if not st.session_state.initialized:
#             # Initialize connection
#             with st.spinner("Connecting to Snowflake..."):
#                 # Verify API key
#                 if not api_key:
#                     st.error("Please provide an OpenAI API key before connecting")
#                     st.stop()

#                 # Now connect to Snowflake
#                 conn = init_snowflake_connection()
#                 if conn:
#                     st.success("Connected to Snowflake!")
#                 # Get sample data for the model to understand the schema
#                 with st.spinner("Fetching sample data..."):
#                   sample_query = f"SELECT * FROM {st.session_state.table_name} LIMIT 1000"
#                   df = execute_snowflake_query(sample_query)
#                   if df is not None:
#                     st.session_state.df = df
#                     # Save schema information for checking changes later
#                     st.session_state.schema_columns = list(df.columns)  # Store column names

#                     # Generate introduction about the data
#                     schema_info = {col: str(df[col].dtype) for col in df.columns}
#                     introduction = generate_introduction(schema_info, table_name=st.session_state.table_name)
#                     # st.session_state.messages.append({"role": "assistant", "content": introduction})

#                     # Add introduction to full_responses
#                     st.session_state.full_responses.append({
#                         "user_query": "Hi, how can I help you?",
#                         "text_response":"",
#                         # "data": None,
#                         # "visualization": None,
#                         # "visualization_notes": None
#                     })
#                     st.session_state.initialized = True

#                     # Initialize vector store with the data
#                     progress_placeholder = st.empty()
#                     progress_placeholder.info("Creating embeddings and initializing vector store...")
#                     st.session_state.embedding_status = "In Progress"
#                     st.session_state.vector_store = initialize_vector_store(df)
#                     st.session_state.embedding_status = "Completed"
#                     progress_placeholder.success("Embeddings created successfully!")
#     else:
#         # st.success("Connected to Snowflake!")
#         st.info(f"Current table: {st.session_state.table_name}")
#         st.info(f"Embedding Status: {st.session_state.embedding_status}")

#         if st.button("Disconnect"):
#             st.session_state.initialized = False
#             st.session_state.messages = []
#             st.session_state.chat_history = []  # Clear conversation history as well
#             st.session_state.full_responses = []  # Clear full responses with tables and visualizations
#             st.session_state.df = None
#             st.session_state.vector_store = None
#             st.session_state.embedding_status = "Not Started"
#             st.rerun()

# # # Chat interface
# # if st.session_state.initialized:
# #     # Display chat messages and full responses (with tables and visualizations)
# #     if len(st.session_state.full_responses) > 0:
# #         # Display messages with full content (tables and visualizations)
# #         for idx, response in enumerate(st.session_state.full_responses):
# #             # Display user message
# #             with st.chat_message("user",avatar=user_avatar):
# #                 st.write(response.get("user_query", ""))

# #             # Display assistant response with tables and visualizations
# #             # with st.chat_message("assistant",avatar=assistant_avatar):
# #                 # st.write(response.get("text_response", ""))

# #                 # Display data table if available
# #                 if response.get("data") is not None:
# #                     st.dataframe(response["data"])

# #                 # Display visualization if available
# #                 if response.get("visualization") is not None:
# #                     # st.plotly_chart(response["visualization"], use_container_width=True, key="response_vis_chart")
# #                     st.plotly_chart(response["visualization"], use_container_width=True, key=f"main_fig_chart_{idx}")
# #                     if response.get("visualization_notes"):
# #                         st.caption(response["visualization_notes"])
# #     else:
# #         # Fall back to just displaying text messages if no full responses exist yet
# #         for message in st.session_state.messages:
# #             with st.chat_message(message["role"]):
# #                 st.write(message["content"])

# #     # User input
# #     user_query = st.chat_input("What would you like to know about the OEE data?")
# #     if user_query:
# #         # Add to display messages
# #         st.session_state.messages.append({"role": "user", "content": user_query})
# #         # Add to chat history for OpenAI (only if history is enabled)
# #         if st.session_state.show_history:
# #             st.session_state.chat_history.append({"role": "user", "content": user_query})

# #         with st.chat_message("user",avatar=user_avatar):
# #             st.write(user_query)

# #         with st.spinner("Generating response..."):
# #             # Get column info for context
# #             column_info = {col: str(st.session_state.df[col].dtype) for col in st.session_state.df.columns}

# #             # Prepare conversation history if enabled
# #             conversation_history = st.session_state.chat_history if st.session_state.show_history else None

# #             # Process query with RAG
# #             if st.session_state.vector_store and st.session_state.embedding_status == "Completed":
# #                 rag_response = process_query_with_rag(
# #                     user_query=user_query,
# #                     vector_store=st.session_state.vector_store,
# #                     table_name=st.session_state.table_name,
# #                     # schema_name="O3_AI_DB_SCHEMA",
# #                     # database_name="O3_AI_DB",
# #                     schema_name="O3_AI_DB_SCHEMA",
# #                     database_name="O3_AI_DB",
# #                     column_info=column_info,
# #                     conversation_history=conversation_history
# #                 )

# #                 # Extract SQL query from response if available
# #                 if "```sql" in rag_response:
# #                     sql_query = rag_response.split("```sql")[1].split("```")[0].strip()

# #                     # Execute SQL query
# #                     result_df = execute_snowflake_query(sql_query)

# #                     # Format final response
# #                     if result_df is not None and not result_df.empty:
# #                         # Generate natural language summary
# #                         nlp_summary = generate_nlp_summary(user_query, sql_query, result_df)

# #                         # Format response based on debug mode
# #                         if st.session_state.debug_mode:
# #                             # Show SQL in debug mode
# #                             final_response = f"Based on your question, I generated this SQL query:\n```sql\n{sql_query}\n```\n\n"
# #                             # Add the natural language summary
# #                             final_response += f"{nlp_summary}\n\n"
# #                             # Add the result
# #                             final_response += "Here are the detailed results:\n"
# #                         else:
# #                             # Hide SQL in regular mode - just show the summary
# #                             final_response = f"{nlp_summary}\n\n"
# #                             # Add the result
# #                             final_response += "Here are the detailed results:\n"

# #                         # Determine appropriate visualization
# #                         with st.spinner("Creating visualization..."):
# #                             vis_recommendation = determine_visualization_type(user_query, sql_query, result_df)
# #                             # 🔍 Debug: show what the LLM recommended
# #                             from datetime import datetime  # 💡 put this at the top of your file if not already

# #                             # 📦 Build log entry
# #                             log_entry = {
# #                               "user_query": user_query,
# #                               "sql_query": sql_query,
# #                               "vis_recommendation": vis_recommendation
# # }

# #                             # 🕒 Add timestamp to log
# #                             log_entry["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# #                             # 📝 Save it to a file called vis_logs.txt
# #                             with open("vis_logs.txt", "a", encoding="utf-8") as log_file:
# #                                 log_file.write(json.dumps(log_entry, indent=2))
# #                                 log_file.write("\n\n" + "="*80 + "\n\n")


# #                             fig = create_visualization(result_df, vis_recommendation)

# #                         with st.chat_message("assistant",avatar=assistant_avatar):
# #                             st.write(final_response)

# #                             # Display the data table
# #                             st.dataframe(result_df)

# #                             # Display visualization if available
# #                             if fig:
# #                                 st.plotly_chart(fig, use_container_width=True,key="main_fig_chart")
# #                                 description = vis_recommendation.get("description", "")
# #                                 if description:
# #                                     st.caption(f"Visualization notes: {description}")

# #                         # Save message without the dataframe (for OpenAI history)
# #                         response_content = final_response + "[Results shown in table format and visualization]"
# #                         st.session_state.messages.append({
# #                             "role": "assistant", 
# #                             "content": response_content
# #                         })

# #                         # Store full response with dataframe and visualization for display
# #                         visualization_notes = ""
# #                         if fig and vis_recommendation.get("description"):
# #                             visualization_notes = f"Visualization notes: {vis_recommendation.get('description')}"

# #                         st.session_state.full_responses.append({
# #                             "user_query": user_query,
# #                             "text_response": final_response,
# #                             "data": result_df,
# #                             "visualization": fig,
# #                             "visualization_notes": visualization_notes,
# #                             "sql_query": sql_query if st.session_state.debug_mode else None
# #                         })

# #                         # Add to chat history for OpenAI (only if history is enabled)
# #                         if st.session_state.show_history:
# #                             st.session_state.chat_history.append({
# #                                 "role": "assistant", 
# #                                 "content": response_content
# #                             })

# #                     elif result_df is not None and result_df.empty:
# #                         no_data_msg='I apologize,no results found regarding your query.'
# #                         with st.chat_message("assistant",avatar=assistant_avatar):
# #                             st.warning(no_data_msg)  
# #                         st.session_state.messages.append({"role":"assistant","content":no_data_msg})
# #                         st.session_state.full_responses.append({
# #                             "user_query":user_query,
# #                             "text_response":no_data_msg,
# #                             "data": None,
# #                             "visualization_notes": None,
# #                             "sql_query": sql_query if st.session_state.debug_mode else None
# #                         }) 
# #                         if st.session_state.show_history:
# #                             st.session_state.chat_history.append({"role":"assistant","content": no_data_msg})         
# #                     else:
# #                         # Format error response based on debug mode
# #                         if st.session_state.debug_mode:
# #                             final_response = f"I generated this SQL query, but there was an error executing it:\n```sql\n{sql_query}\n```"
# #                         else:
# #                             final_response = f"I couldn't retrieve the data you asked for. There might be an issue with the query or connection."

# #                         with st.chat_message("assistant",avatar=assistant_avatar):
# #                             st.write(final_response)
# #                         st.session_state.messages.append({"role": "assistant", "content": final_response})

# #                         # Store error in full_responses for consistent display
# #                         st.session_state.full_responses.append({
# #                             "user_query": user_query,
# #                             "text_response": final_response,
# #                             "data": None,
# #                             "visualization": None,
# #                             "visualization_notes": None,
# #                             "sql_query": sql_query if st.session_state.debug_mode else None
# #                         })

# #                         # Add to chat history for OpenAI (only if history is enabled)
# #                         if st.session_state.show_history:
# #                             st.session_state.chat_history.append({"role": "assistant", "content": final_response})
# #                 else:
# #                     # If no SQL was generated
# #                     with st.chat_message("assistant",avatar=assistant_avatar):
# #                         st.write(rag_response)
# #                     st.session_state.messages.append({"role": "assistant", "content": rag_response})

# #                     # Store in full_responses for consistent display
# #                     st.session_state.full_responses.append({
# #                         "user_query": user_query,
# #                         "text_response": rag_response,
# #                         "data": None,
# #                         "visualization": None,
# #                         "visualization_notes": None
# #                     })

# #                     # Add to chat history for OpenAI (only if history is enabled)
# #                     if st.session_state.show_history:
# #                         st.session_state.chat_history.append({"role": "assistant", "content": rag_response})
# #             else:
# #                 # Fallback to regular LLM response if vector store not ready
# #                 llm_response = get_llm_response(
# #                     user_query=user_query,
# #                     table_name=st.session_state.table_name,
# #                     schema_name="O3_AI_DB",
# #                     database_name="O3_AI_DB_SCHEMA",
# #                     column_info=column_info,
# #                     conversation_history=conversation_history
# #                 )
# #                 with st.chat_message("assistant",avatar=assistant_avatar):
# #                     st.write(llm_response)
# #                 st.session_state.messages.append({"role": "assistant", "content": llm_response})

# #                 # Store in full_responses for consistent display
# #                 st.session_state.full_responses.append({
# #                     "user_query": user_query,
# #                     "text_response": llm_response,
# #                     "data": None,
# #                     "visualization": None,
# #                     "visualization_notes": None
# #                 })

# #                 # Add to chat history for OpenAI (only if history is enabled)
# #                 if st.session_state.show_history:
# #                     st.session_state.chat_history.append({"role": "assistant", "content": llm_response})
# # else:
# #     st.info("Please connect to Snowflake to use the chatbot.")






# # Chat interface
# if st.session_state.initialized:
#     # Display all past messages (except the most recent interaction)
#     for idx, response in enumerate(st.session_state.full_responses[:-1] if st.session_state.full_responses else []):
#         with st.chat_message("user", avatar=user_avatar):
#             st.write(response.get("user_query", ""))
#         with st.chat_message("assistant", avatar=assistant_avatar):
#             st.write(response.get("text_response", ""))
#             if response.get("data") is not None:
#                 st.dataframe(response["data"])
#             if response.get("visualization") is not None:
#                 st.plotly_chart(response["visualization"], use_container_width=True, key=f"history_fig_chart_{idx}")
#                 if response.get("visualization_notes"):
#                     st.caption(response["visualization_notes"])
#     if st.session_state.full_responses:
#         latest = st.session_state.full_responses[-1]
#         with st.chat_message("user", avatar=user_avatar):
#             st.write(latest.get("user_query", ""))
#         with st.chat_message("assistant", avatar=assistant_avatar):
#             st.write(latest.get("text_response", ""))
#             if latest.get("data") is not None:
#                 st.dataframe(latest["data"])
#             if latest.get("visualization") is not None:
#                 st.plotly_chart(latest["visualization"], use_container_width=True, key="latest_fig_chart")
#                 if latest.get("visualization_notes"):
#                     st.caption(latest["visualization_notes"])
#     # Create a separate placeholder for the new query and response
#     new_message_placeholder = st.empty()
#     # User input for new query
#     if user_query := st.chat_input("What would you like to know about the OEE data?"):
#         # Add to display messages
#         st.session_state.messages.append({"role": "user", "content": user_query})
#         if st.session_state.show_history:
#             st.session_state.chat_history.append({"role": "user", "content": user_query})
#         # Use the placeholder to display the new user query and spinner
#         with new_message_placeholder.container():
#             with st.chat_message("user", avatar=user_avatar):
#                 st.write(user_query)
#             # Show spinner while generating response
#             with st.spinner("Generating response..."):
#                 # Get column info for context
#                 column_info = {col: str(st.session_state.df[col].dtype) for col in st.session_state.df.columns}
#                 conversation_history = st.session_state.chat_history if st.session_state.show_history else None
#                 # Process query with RAG
#                 if st.session_state.vector_store and st.session_state.embedding_status == "Completed":
#                     rag_response = process_query_with_rag(
#                         user_query=user_query,
#                         vector_store=st.session_state.vector_store,
#                         table_name=st.session_state.table_name,
#                         schema_name="O3_AI_DB_SCHEMA",
#                         database_name="O3_AI_DB",
#                         column_info=column_info,
#                         conversation_history=conversation_history
#                     )
#                     if "```sql" in rag_response:
#                         sql_query = rag_response.split("```sql")[1].split("```")[0].strip()
#                         result_df = execute_snowflake_query(sql_query)
#                         if result_df is not None and not result_df.empty:
#                             nlp_summary = generate_nlp_summary(user_query, sql_query, result_df)
#                             if st.session_state.debug_mode:
#                                 final_response = f"Based on your question, I generated this SQL query:\n```sql\n{sql_query}\n```\n\n{nlp_summary}\n\nHere are the detailed results:\n"
#                             else:
#                                 final_response = f"{nlp_summary}\n\nHere are the detailed results:\n"
#                             with st.spinner("Creating visualization..."):
#                                 vis_recommendation = determine_visualization_type(user_query, sql_query, result_df)
#                                 log_entry = {
#                                     "user_query": user_query,
#                                     "sql_query": sql_query,
#                                     "vis_recommendation": vis_recommendation,
#                                     "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#                                 }
#                                 with open("vis_logs.txt", "a", encoding="utf-8") as log_file:
#                                     log_file.write(json.dumps(log_entry, indent=2))
#                                     log_file.write("\n\n" + "="*80 + "\n\n")
#                                 fig = create_visualization(result_df, vis_recommendation)
#                             with st.chat_message("assistant", avatar=assistant_avatar):
#                                 st.write(final_response)
#                                 st.dataframe(result_df)
#                                 if fig:
#                                     st.plotly_chart(fig, use_container_width=True, key="main_fig_chart")
#                                     description = vis_recommendation.get("description", "")
#                                     if description:
#                                         st.caption(f"Visualization notes: {description}")
#                             response_content = final_response + "[Results shown in table format and visualization]"
#                             st.session_state.messages.append({"role": "assistant", "content": response_content})
#                             visualization_notes = f"Visualization notes: {vis_recommendation.get('description')}" if fig and vis_recommendation.get("description") else ""
#                             st.session_state.full_responses.append({
#                                 "user_query": user_query,
#                                 "text_response": final_response,
#                                 "data": result_df,
#                                 "visualization": fig,
#                                 "visualization_notes": visualization_notes,
#                                 "sql_query": sql_query if st.session_state.debug_mode else None
#                             })
#                             if st.session_state.show_history:
#                                 st.session_state.chat_history.append({"role": "assistant", "content": response_content})
#                         elif result_df is not None and result_df.empty:
#                             no_data_msg = "I apologize, no results found regarding your query."
#                             with st.chat_message("assistant", avatar=assistant_avatar):
#                                 st.warning(no_data_msg)
#                             st.session_state.messages.append({"role": "assistant", "content": no_data_msg})
#                             st.session_state.full_responses.append({
#                                 "user_query": user_query,
#                                 "text_response": no_data_msg,
#                                 "data": None,
#                                 "visualization_notes": None,
#                                 "sql_query": sql_query if st.session_state.debug_mode else None
#                             })
#                             if st.session_state.show_history:
#                                 st.session_state.chat_history.append({"role": "assistant", "content": no_data_msg})
#                         else:
#                             if st.session_state.debug_mode:
#                                 final_response = f"I generated this SQL query, but there was an error executing it:\n```sql\n{sql_query}\n```"
#                             else:
#                                 final_response = f"I couldn't retrieve the data you asked for. There might be an issue with the query or connection."
#                             with st.chat_message("assistant", avatar=assistant_avatar):
#                                 st.write(final_response)
#                             st.session_state.messages.append({"role": "assistant", "content": final_response})
#                             st.session_state.full_responses.append({
#                                 "user_query": user_query,
#                                 "text_response": final_response,
#                                 "data": None,
#                                 "visualization": None,
#                                 "visualization_notes": None,
#                                 "sql_query": sql_query if st.session_state.debug_mode else None
#                             })
#                             if st.session_state.show_history:
#                                 st.session_state.chat_history.append({"role": "assistant", "content": final_response})
#                     else:
#                         with st.chat_message("assistant", avatar=assistant_avatar):
#                             st.write(rag_response)
#                         st.session_state.messages.append({"role": "assistant", "content": rag_response})
#                         st.session_state.full_responses.append({
#                             "user_query": user_query,
#                             "text_response": rag_response,
#                             "data": None,
#                             "visualization": None,
#                             "visualization_notes": None
#                         })
#                         if st.session_state.show_history:
#                             st.session_state.chat_history.append({"role": "assistant", "content": rag_response})
#                 else:
#                     llm_response = get_llm_response(
#                         user_query=user_query,
#                         table_name=st.session_state.table_name,
#                         schema_name="O3_AI_DB",
#                         database_name="O3_AI_DB_SCHEMA",
#                         column_info=column_info,
#                         conversation_history=conversation_history
#                     )
#                     with st.chat_message("assistant", avatar=assistant_avatar):
#                         st.write(llm_response)
#                     st.session_state.messages.append({"role": "assistant", "content": llm_response})
#                     st.session_state.full_responses.append({
#                         "user_query": user_query,
#                         "text_response": llm_response,
#                         "data": None,
#                         "visualization": None,
#                         "visualization_notes": None
#                     })
#                     if st.session_state.show_history:
#                         st.session_state.chat_history.append({"role": "assistant", "content": llm_response})
#         # Clear the placeholder after the response is generated to avoid re-rendering
#         st.rerun()
# else:
#     st.info("Please connect to Snowflake to use the chatbot.")






















import streamlit as st
import snowflake.connector
import pandas as pd
import numpy as np
import os
from datetime import datetime
import re
import json
from prompts import generate_introduction, get_llm_response, call_openai
from rag_utils import (
    initialize_vector_store,
    process_query_with_rag,
    get_openai_embedding
)
from openai import OpenAI
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# Safe full paths to avatars inside src folder
BASE_DIR = Path(__file__).parent
user_avatar = (BASE_DIR / "user.png").resolve().as_posix()
assistant_avatar = (BASE_DIR / "Assistant.png").resolve().as_posix()


# Set page config
st.set_page_config(
    page_title="O3 Agent",
    page_icon="📊",
    layout="wide"
)

# OpenAI client
# client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
GPT_MODEL = "gpt-4o"  # the newest OpenAI model is "gpt-4o" which was released May 13, 2024.

# Initialize session state variables
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'full_responses' not in st.session_state:
    st.session_state.full_responses = []  # For storing full responses with dataframes and visualizations
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []  # For the OpenAI conversation history
if 'df' not in st.session_state:
    st.session_state.df = None
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'embedding_status' not in st.session_state:
    st.session_state.embedding_status = "Not Started"
if 'show_history' not in st.session_state:
    st.session_state.show_history = True  # Toggle for conversation memory
if 'debug_mode' not in st.session_state:
    st.session_state.debug_mode = False  # Toggle for developer mode
if 'table_name' not in st.session_state:
    st.session_state.table_name = "OEESHIFTWISE_AI"  # Default table name
if 'schema_columns' not in st.session_state:
    st.session_state.schema_columns = []  # To track schema changes

def init_snowflake_connection():
    try:
        conn = snowflake.connector.connect(
            user=st.secrets["snowflake"]["user"],
            password=st.secrets["snowflake"]["password"],
            account=st.secrets["snowflake"]["account"],
            warehouse=st.secrets["snowflake"]["warehouse"],
            # database='O3_AI_DB',
            # schema='O3_AI_DB_SCHEMA'
            database="O3_AI_DB",
            schema="O3_AI_DB_SCHEMA"
           
        )
        return conn
    except Exception as e:
        st.error(f"Error connecting to Snowflake: {e}")
        return None


def execute_snowflake_query(query):
    conn = init_snowflake_connection()
    if conn:
        try:
            cursor = conn.cursor()
            cursor.execute(query)
            result = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            df = pd.DataFrame(result, columns=columns)
            cursor.close()
            return df
        except Exception as e:
            # st.error(f"Error executing query: {e}")
            st.error("There may be some issue with executing query, please rephrase your question or try it later.")

            return None
    return None

def determine_visualization_type(user_query, sql_query, result_df):
    """Determine the appropriate visualization type based on the query and results."""
    try:
        vis_prompt = f"""
        I need to visualize the following SQL query results for the question: "{user_query}"

        The SQL query was:
        ```sql
        {sql_query}
        ```

        The query returned {len(result_df)} rows with the following column names and data types:
        {[(col, str(result_df[col].dtype)) for col in result_df.columns]}

        Based on the user's question and the data returned, determine the most appropriate visualization type.
        Respond with a JSON object with the following structure:
        {{
            "viz_type": "line|bar|scatter|pie|histogram|heatmap|none",
            "x_column": "name of column to use for x-axis or categories",
            "y_column": "name of column to use for y-axis or values",
            "color_column": "name of column to use for color differentiation (optional, can be null)",
            "title": "Suggested title for the visualization",
            "description": "Brief rationale for why this visualization type is appropriate"
        }}

        Suggest a visualization  for the data and query.
        """

        system_prompt = "You are a data visualization expert that chooses appropriate chart types based on query results. Always respond with valid JSON."

        # Set response format to JSON for structured response
        vis_response = client.chat.completions.create(
            model=GPT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": vis_prompt}
            ],
            response_format={"type": "json_object"}
        )
        # st.write(vis_response['usage'])
        vis_recommendation = json.loads(vis_response.choices[0].message.content)
        return vis_recommendation

    except Exception as e:
        # st.warning(f"Could not determine visualization type: {str(e)}")
        st.warning("Could not determine visualization type, please rephrase your question or try again it later.")

        return {"viz_type": "none"}

def create_visualization(result_df, vis_recommendation):
    """Create an appropriate visualization based on the recommendation."""
    try:
        viz_type = vis_recommendation.get("viz_type", "none")
        if viz_type == "none" or len(result_df) == 0:
            return None
        
        # 🌈 Custom color palette
        custom_palette = ["#242bf0", "#7ECF9A"]


        x_col = vis_recommendation.get("x_column")
        y_col = vis_recommendation.get("y_column")
        color_col = vis_recommendation.get("color_column")
        title = vis_recommendation.get("title", "Data Visualization")

        # Check if the recommended columns exist in the dataframe
        available_cols = result_df.columns.tolist()
        if x_col and x_col not in available_cols:
            x_col = available_cols[0] if available_cols else None
        if y_col and y_col not in available_cols:
            y_col = available_cols[1] if len(available_cols) > 1 else available_cols[0] if available_cols else None
        if color_col and color_col not in available_cols:
            color_col = None

        if not x_col or not y_col:
            return None

        # Create the appropriate plot based on visualization type
        if viz_type == "bar":
            # Handle aggregation if needed
            if len(result_df) > 25:  # Too many bars becomes unreadable
                # Try to aggregate data if it makes sense
                result_df = result_df.groupby(x_col, as_index=False)[y_col].agg('sum')

            fig = px.bar(result_df, x=x_col, y=y_col, color=color_col, title=title,color_discrete_sequence=custom_palette)

        elif viz_type == "line":
            # Sort by x if it's a datetime or numeric column
            if pd.api.types.is_datetime64_any_dtype(result_df[x_col]) or pd.api.types.is_numeric_dtype(result_df[x_col]):
                result_df = result_df.sort_values(by=x_col)

            fig = px.line(result_df, x=x_col, y=y_col, color=color_col, title=title, markers=True,color_discrete_sequence=custom_palette)

        elif viz_type == "scatter":
            fig = px.scatter(result_df, x=x_col, y=y_col, color=color_col, title=title,color_discrete_sequence=custom_palette)

        elif viz_type == "pie":
            fig = px.pie(result_df, names=x_col, values=y_col, title=title,color_discrete_sequence=custom_palette)

        elif viz_type == "histogram":
            fig = px.histogram(result_df, x=x_col, title=title,color_discrete_sequence=custom_palette)

        elif viz_type == "heatmap":
            # Create a pivot table for heatmap
            if color_col:
                pivot_df = result_df.pivot_table(values=color_col, index=y_col, columns=x_col, aggfunc='mean')
                fig = px.imshow(pivot_df, title=title)
            else:
                return None
        else:
            return None

        # Style the figure for better appearance
        fig.update_layout(
            template="plotly_dark",
            height=500,
            margin=dict(l=50, r=50, t=80, b=50)
        )

        return fig

    except Exception as e:
        # st.warning(f"Could not create visualization: {str(e)}")
        st.warning("Could not create visualization, try rephrasing your question or try again.")

        return None

def generate_nlp_summary(user_query, sql_query, result_df):
    """Generate a natural language summary of SQL query results."""
    try:
        # with st.spinner("Generating natural language summary..."):
            nlp_summary_prompt = f"""
            I need a natural language summary of the following SQL query results for the question: "{user_query}"

            The SQL query was:
            ```sql
            {sql_query}
            ```

            The query returned {len(result_df)} rows with the following data:
            {result_df.to_string(index=False, max_rows=10)}

            Please provide a 1-2 sentence natural language summary of these results that directly answers the user's question.
            Focus on the key metrics, highest/lowest values, or trends as appropriate.
            Be specific and include the actual values from the data.
            """

            nlp_summary = call_openai(nlp_summary_prompt, "You are a data analyst summarizing SQL query results in plain language.")
            return nlp_summary
    except Exception as e:
        st.warning(f"Could not generate natural language summary: {str(e)}")
        return "I couldn't generate a natural language summary for these results."




# Sidebar for Snowflake credentials and OpenAI API key
with st.sidebar:
    # st.header("Connection Settings")
    api_key = st.secrets.get("OPENAI_API_KEY")
    # if api_key:
    #     st.success("OpenAI API Key is configured")
    # else:
    #     st.error("OpenAI API Key is missing")
    #     st.info("Please add your OpenAI API key to use this application")

    # st.header("Chat Settings")
    # # Toggle for conversation memory
    # st.session_state.show_history = st.checkbox("Enable conversation memory", value=st.session_state.show_history)
    # if st.session_state.show_history:
    #     st.success("Conversation memory is enabled")
    #     st.info("The chatbot will remember previous messages for context")
    # else:
    #     st.info("Conversation memory is disabled")

    # # Developer Mode toggle
    # if 'debug_mode' not in st.session_state:
    #     st.session_state.debug_mode = False
    # st.session_state.debug_mode = st.checkbox("Developer Mode", value=st.session_state.debug_mode)
    # if st.session_state.debug_mode:
    #     st.success("Developer Mode is enabled")
    #     st.info("SQL queries will be shown in responses")
    # else:
    #     st.info("Developer Mode is disabled")
    #     st.info("SQL queries will be hidden in responses")

    # if st.button("Clear Chat History"):
    #     st.session_state.messages = []
    #     st.session_state.chat_history = []
    #     st.session_state.full_responses = []
    #     st.success("Chat history cleared!")
    #     st.rerun()

    st.header("Snowflake Connection")

    # Check Snowflake credentials
    snowflake_creds = st.secrets.get("snowflake")
    if snowflake_creds and all(k in snowflake_creds for k in ["user", "password", "account", "warehouse"]):
        # st.success("Snowflake credentials are configured")

        # Auto-connect to Snowflake if not initialized
        if not st.session_state.initialized:
            # Initialize connection
            with st.spinner("Connecting to Snowflake..."):
                # Verify API key
                if not api_key:
                    st.error("Please provide an OpenAI API key before connecting")
                    st.stop()

                # Now connect to Snowflake
                conn = init_snowflake_connection()
                if conn:
                    st.success("Connected to Snowflake!")
                # Get sample data for the model to understand the schema
                with st.spinner("Fetching sample data..."):
                  sample_query = f"SELECT * FROM {st.session_state.table_name} LIMIT 1000"
                  df = execute_snowflake_query(sample_query)
                  if df is not None:
                    st.session_state.df = df
                    # Save schema information for checking changes later
                    st.session_state.schema_columns = list(df.columns)  # Store column names

                    # Generate introduction about the data
                    # schema_info = {col: str(df[col].dtype) for col in df.columns}
                    # introduction = generate_introduction(schema_info, table_name=st.session_state.table_name)
                    # st.session_state.messages.append({"role": "assistant", "content": introduction})

                    # # Add introduction to full_responses
                    # st.session_state.full_responses.append({
                    #     "user_query": "Hi, can you tell me about the OEE data?",
                    #     "text_response": introduction,
                    #     "data": None,
                    #     "visualization": None,
                    #     "visualization_notes": None
                    # })
                    st.session_state.initialized = True

                    # Initialize vector store with the data
                    progress_placeholder = st.empty()
                    progress_placeholder.info("Creating embeddings and initializing vector store...")
                    st.session_state.embedding_status = "In Progress"
                    st.session_state.vector_store = initialize_vector_store(df)
                    st.session_state.embedding_status = "Completed"
                    # progress_placeholder.success("Embeddings created successfully!")
    else:
        # st.success("Connected to Snowflake!")
        st.info(f"Current table: {st.session_state.table_name}")
        st.info(f"Embedding Status: {st.session_state.embedding_status}")

        if st.button("Disconnect"):
            st.session_state.initialized = False
            st.session_state.messages = []
            st.session_state.chat_history = []  # Clear conversation history as well
            st.session_state.full_responses = []  # Clear full responses with tables and visualizations
            st.session_state.df = None
            st.session_state.vector_store = None
            st.session_state.embedding_status = "Not Started"
            st.rerun()

# Chat interface
if st.session_state.initialized:
    # Display all past messages (except the most recent interaction)
    for idx, response in enumerate(st.session_state.full_responses[:-1] if st.session_state.full_responses else []):
        with st.chat_message("user", avatar=user_avatar):
            st.write(response.get("user_query", ""))
        with st.chat_message("assistant", avatar=assistant_avatar):
            st.write(response.get("text_response", ""))
            if response.get("data") is not None:
                st.dataframe(response["data"])
            if response.get("visualization") is not None:
                st.plotly_chart(response["visualization"], use_container_width=True, key=f"history_fig_chart_{idx}")
                if response.get("visualization_notes"):
                    st.caption(response["visualization_notes"])

    if st.session_state.full_responses:
        latest = st.session_state.full_responses[-1]
        with st.chat_message("user", avatar=user_avatar):
            st.write(latest.get("user_query", ""))
        with st.chat_message("assistant", avatar=assistant_avatar):
            st.write(latest.get("text_response", ""))
            if latest.get("data") is not None:
                st.dataframe(latest["data"])
            if latest.get("visualization") is not None:
                st.plotly_chart(latest["visualization"], use_container_width=True, key="latest_fig_chart")
                if latest.get("visualization_notes"):
                    st.caption(latest["visualization_notes"])                
    # Create a separate placeholder for the new query and response
    new_message_placeholder = st.empty()

    # User input for new query
    if user_query := st.chat_input("What would you like to know about the OEE data?"):
        # Add to display messages
        st.session_state.messages.append({"role": "user", "content": user_query})
        if st.session_state.show_history:
            st.session_state.chat_history.append({"role": "user", "content": user_query})
          

        # Use the placeholder to display the new user query and spinner
        with new_message_placeholder.container():
            with st.chat_message("user", avatar=user_avatar):
                st.write(user_query)

            # Show spinner while generating response
            with st.spinner("Generating response..."):
                # Get column info for context
                column_info = {col: str(st.session_state.df[col].dtype) for col in st.session_state.df.columns}
                conversation_history = st.session_state.chat_history if st.session_state.show_history else None

                # Process query with RAG
                if st.session_state.vector_store and st.session_state.embedding_status == "Completed":
                    rag_response = process_query_with_rag(
                        user_query=user_query,
                        vector_store=st.session_state.vector_store,
                        table_name=st.session_state.table_name,
                        schema_name="O3_AI_DB_SCHEMA",
                        database_name="O3_AI_DB",
                        column_info=column_info,
                        conversation_history=conversation_history
                    )

                    if "```sql" in rag_response:
                        sql_query = rag_response.split("```sql")[1].split("```")[0].strip()
                        result_df = execute_snowflake_query(sql_query)

                        if result_df is not None and not result_df.empty:
                            nlp_summary = generate_nlp_summary(user_query, sql_query, result_df)
                            if st.session_state.debug_mode:
                                final_response = f"Based on your question, I generated this SQL query:\n```sql\n{sql_query}\n```\n\n{nlp_summary}\n\nHere are the detailed results:\n"
                            else:
                                final_response = f"{nlp_summary}\n\nHere are the detailed results:\n"

                            with st.spinner("Almost done... building your chart!"):
                                vis_recommendation = determine_visualization_type(user_query, sql_query, result_df)
                                log_entry = {
                                    "user_query": user_query,
                                    "sql_query": sql_query,
                                    "vis_recommendation": vis_recommendation,
                                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                }
                                with open("vis_logs.txt", "a", encoding="utf-8") as log_file:
                                    log_file.write(json.dumps(log_entry, indent=2))
                                    log_file.write("\n\n" + "="*80 + "\n\n")
                                fig = create_visualization(result_df, vis_recommendation)

                            with st.chat_message("assistant", avatar=assistant_avatar):
                                st.write(final_response)
                                st.dataframe(result_df)
                                if fig:
                                    st.plotly_chart(fig, use_container_width=True, key="main_fig_chart")
                                    description = vis_recommendation.get("description", "")
                                    if description:
                                        st.caption(f"Visualization notes: {description}")

                            response_content = final_response + "[Results shown in table format and visualization]"
                            st.session_state.messages.append({"role": "assistant", "content": response_content})
                            visualization_notes = f"Visualization notes: {vis_recommendation.get('description')}" if fig and vis_recommendation.get("description") else ""
                            st.session_state.full_responses.append({
                                "user_query": user_query,
                                "text_response": final_response,
                                "data": result_df,
                                "visualization": fig,
                                "visualization_notes": visualization_notes,
                                "sql_query": sql_query if st.session_state.debug_mode else None
                            })
                            if st.session_state.show_history:
                                st.session_state.chat_history.append({"role": "assistant", "content": response_content})

                        elif result_df is not None and result_df.empty:
                            no_data_msg = "I apologize, no results found regarding your query."
                            with st.chat_message("assistant", avatar=assistant_avatar):
                                st.warning(no_data_msg)
                            st.session_state.messages.append({"role": "assistant", "content": no_data_msg})
                            st.session_state.full_responses.append({
                                "user_query": user_query,
                                "text_response": no_data_msg,
                                "data": None,
                                "visualization_notes": None,
                                "sql_query": sql_query if st.session_state.debug_mode else None
                            })
                            if st.session_state.show_history:
                                st.session_state.chat_history.append({"role": "assistant", "content": no_data_msg})
                        else:
                            if st.session_state.debug_mode:
                                final_response = f"I generated this SQL query, but there was an error executing it:\n```sql\n{sql_query}\n```"
                            else:
                                final_response = f"I couldn't retrieve the data you asked for. There might be an issue with the query or connection."
                            with st.chat_message("assistant", avatar=assistant_avatar):
                                st.write(final_response)
                            st.session_state.messages.append({"role": "assistant", "content": final_response})
                            st.session_state.full_responses.append({
                                "user_query": user_query,
                                "text_response": final_response,
                                "data": None,
                                "visualization": None,
                                "visualization_notes": None,
                                "sql_query": sql_query if st.session_state.debug_mode else None
                            })
                            if st.session_state.show_history:
                                st.session_state.chat_history.append({"role": "assistant", "content": final_response})
                    else:
                        with st.chat_message("assistant", avatar=assistant_avatar):
                            st.write(rag_response)
                        st.session_state.messages.append({"role": "assistant", "content": rag_response})
                        st.session_state.full_responses.append({
                            "user_query": user_query,
                            "text_response": rag_response,
                            "data": None,
                            "visualization": None,
                            "visualization_notes": None
                        })
                        if st.session_state.show_history:
                            st.session_state.chat_history.append({"role": "assistant", "content": rag_response})
                else:
                    llm_response = get_llm_response(
                        user_query=user_query,
                        table_name=st.session_state.table_name,
                        schema_name="O3_AI_DB",
                        database_name="O3_AI_DB_SCHEMA",
                        column_info=column_info,
                        conversation_history=conversation_history
                    )
                    with st.chat_message("assistant", avatar=assistant_avatar):
                        st.write(llm_response)
                    st.session_state.messages.append({"role": "assistant", "content": llm_response})
                    st.session_state.full_responses.append({
                        "user_query": user_query,
                        "text_response": llm_response,
                        "data": None,
                        "visualization": None,
                        "visualization_notes": None
                    })
                    if st.session_state.show_history:
                        st.session_state.chat_history.append({"role": "assistant", "content": llm_response})

        # Clear the placeholder after the response is generated to avoid re-rendering
        st.rerun()

else:
    st.info("Please connect to Snowflake to use the chatbot.")
