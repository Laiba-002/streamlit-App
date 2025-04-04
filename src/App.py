
import streamlit as st
import snowflake.connector
import pandas as pd
import numpy as np
import os
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

# Set page config
st.set_page_config(
    page_title="OEE Manager",
    page_icon="📊",
    layout="wide"
)

# OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
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

# Snowflake Connection Functions
def init_snowflake_connection():
    try:
        conn = snowflake.connector.connect(
            user=st.session_state.snowflake_user,
            password=st.session_state.snowflake_password,
            account=st.session_state.snowflake_account,
            warehouse=st.session_state.snowflake_warehouse,
            database='O3_DEV_DB',
            schema='O3_DEV_RAW_SCH'
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
            st.error(f"Error executing query: {e}")
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
        
        Only suggest a visualization if it makes sense for the data and query. If visualization is not appropriate, return "viz_type": "none".
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
        
        # Parse the JSON response
        vis_recommendation = json.loads(vis_response.choices[0].message.content)
        return vis_recommendation
        
    except Exception as e:
        st.warning(f"Could not determine visualization type: {str(e)}")
        return {"viz_type": "none"}

def create_visualization(result_df, vis_recommendation):
    """Create an appropriate visualization based on the recommendation."""
    try:
        viz_type = vis_recommendation.get("viz_type", "none")
        if viz_type == "none" or len(result_df) == 0:
            return None
            
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
                
            fig = px.bar(result_df, x=x_col, y=y_col, color=color_col, title=title)
            
        elif viz_type == "line":
            # Sort by x if it's a datetime or numeric column
            if pd.api.types.is_datetime64_any_dtype(result_df[x_col]) or pd.api.types.is_numeric_dtype(result_df[x_col]):
                result_df = result_df.sort_values(by=x_col)
                
            fig = px.line(result_df, x=x_col, y=y_col, color=color_col, title=title, markers=True)
            
        elif viz_type == "scatter":
            fig = px.scatter(result_df, x=x_col, y=y_col, color=color_col, title=title)
            
        elif viz_type == "pie":
            fig = px.pie(result_df, names=x_col, values=y_col, title=title)
            
        elif viz_type == "histogram":
            fig = px.histogram(result_df, x=x_col, title=title)
            
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
            template="plotly_white",
            height=500,
            margin=dict(l=50, r=50, t=80, b=50)
        )
        
        return fig
        
    except Exception as e:
        st.warning(f"Could not create visualization: {str(e)}")
        return None

def generate_nlp_summary(user_query, sql_query, result_df):
    """Generate a natural language summary of SQL query results."""
    try:
        with st.spinner("Generating natural language summary..."):
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

# UI Components
# Title and sidebar
st.title("OEE Manager")

# Sidebar for Snowflake credentials and OpenAI API key
with st.sidebar:
    st.header("Connection Settings")
    
    # Check OpenAI API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        st.success("OpenAI API Key is configured")
    else:
        st.error("OpenAI API Key is missing")
        st.info("Please add your OpenAI API key to use this application")
    
    st.header("Chat Settings")
    # Toggle for conversation memory
    st.session_state.show_history = st.checkbox("Enable conversation memory", value=st.session_state.show_history)
    if st.session_state.show_history:
        st.success("Conversation memory is enabled")
        st.info("The chatbot will remember previous messages for context")
    else:
        st.info("Conversation memory is disabled")
        
    # Developer Mode toggle
    if 'debug_mode' not in st.session_state:
        st.session_state.debug_mode = False
    st.session_state.debug_mode = st.checkbox("Developer Mode", value=st.session_state.debug_mode)
    if st.session_state.debug_mode:
        st.success("Developer Mode is enabled")
        st.info("SQL queries will be shown in responses")
    else:
        st.info("Developer Mode is disabled")
        st.info("SQL queries will be hidden in responses")
    
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.session_state.full_responses = []
        st.success("Chat history cleared!")
        st.rerun()
    
    st.header("Snowflake Connection")
    
    if not st.session_state.initialized:
        st.session_state.snowflake_user = st.text_input("Snowflake Username")
        st.session_state.snowflake_password = st.text_input("Snowflake Password", type="password")
        st.session_state.snowflake_account = st.text_input("Snowflake Account")
        st.session_state.snowflake_warehouse = st.text_input("Snowflake Warehouse")
        
        connect_button = st.button("Connect")
        
        if connect_button:
            # Verify API key
            if not api_key:
                st.error("Please provide an OpenAI API key before connecting")
                st.stop()
                
            # Now connect to Snowflake
            conn = init_snowflake_connection()
            if conn:
                st.success("Connected to Snowflake!")
                # Get sample data for the model to understand the schema
                sample_query = f"SELECT * FROM {st.session_state.table_name} LIMIT 1000"
                df = execute_snowflake_query(sample_query)
                if df is not None:
                    st.session_state.df = df
                    # Save schema information for checking changes later
                    st.session_state.schema_columns = list(df.columns)  # Store column names
                    
                    # Generate introduction about the data
                    schema_info = {col: str(df[col].dtype) for col in df.columns}
                    introduction = generate_introduction(schema_info, table_name=st.session_state.table_name)
                    st.session_state.messages.append({"role": "assistant", "content": introduction})
                    
                    # Add introduction to full_responses
                    st.session_state.full_responses.append({
                        "user_query": "Hi, can you tell me about the OEE data?",
                        "text_response": introduction,
                        "data": None,
                        "visualization": None,
                        "visualization_notes": None
                    })
                    st.session_state.initialized = True
                    
                    # Initialize vector store with the data
                    progress_placeholder = st.empty()
                    progress_placeholder.info("Creating embeddings and initializing vector store...")
                    st.session_state.embedding_status = "In Progress"
                    st.session_state.vector_store = initialize_vector_store(df)
                    st.session_state.embedding_status = "Completed"
                    progress_placeholder.success("Embeddings created successfully!")
    else:
        st.success("Connected to Snowflake!")
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
    # Display chat messages and full responses (with tables and visualizations)
    if len(st.session_state.full_responses) > 0:
        # Display messages with full content (tables and visualizations)
        for idx, response in enumerate(st.session_state.full_responses):
            # Display user message
            with st.chat_message("user"):
                st.write(response.get("user_query", ""))
            
            # Display assistant response with tables and visualizations
            with st.chat_message("assistant"):
                st.write(response.get("text_response", ""))
                
                # Display data table if available
                if response.get("data") is not None:
                    st.dataframe(response["data"])
                
                # Display visualization if available
                if response.get("visualization") is not None:
                    st.plotly_chart(response["visualization"], use_container_width=True)
                    if response.get("visualization_notes"):
                        st.caption(response["visualization_notes"])
    else:
        # Fall back to just displaying text messages if no full responses exist yet
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
    
    # User input
    user_query = st.chat_input("What would you like to know about the OEE data?")
    if user_query:
        # Add to display messages
        st.session_state.messages.append({"role": "user", "content": user_query})
        # Add to chat history for OpenAI (only if history is enabled)
        if st.session_state.show_history:
            st.session_state.chat_history.append({"role": "user", "content": user_query})
        
        with st.chat_message("user"):
            st.write(user_query)
        
        with st.spinner("Generating response..."):
            # Get column info for context
            column_info = {col: str(st.session_state.df[col].dtype) for col in st.session_state.df.columns}
            
            # Prepare conversation history if enabled
            conversation_history = st.session_state.chat_history if st.session_state.show_history else None
            
            # Process query with RAG
            if st.session_state.vector_store and st.session_state.embedding_status == "Completed":
                rag_response = process_query_with_rag(
                    user_query=user_query,
                    vector_store=st.session_state.vector_store,
                    table_name=st.session_state.table_name,
                    schema_name="O3_DEV_RAW_SCH",
                    database_name="O3_DEV_DB",
                    column_info=column_info,
                    conversation_history=conversation_history
                )
                
                # Extract SQL query from response if available
                if "```sql" in rag_response:
                    sql_query = rag_response.split("```sql")[1].split("```")[0].strip()
                    
                    # Execute SQL query
                    result_df = execute_snowflake_query(sql_query)
                    
                    # Format final response
                    if result_df is not None:
                        # Generate natural language summary
                        nlp_summary = generate_nlp_summary(user_query, sql_query, result_df)
                        
                        # Format response based on debug mode
                        if st.session_state.debug_mode:
                            # Show SQL in debug mode
                            final_response = f"Based on your question, I generated this SQL query:\n```sql\n{sql_query}\n```\n\n"
                            # Add the natural language summary
                            final_response += f"{nlp_summary}\n\n"
                            # Add the result
                            final_response += "Here are the detailed results:\n"
                        else:
                            # Hide SQL in regular mode - just show the summary
                            final_response = f"{nlp_summary}\n\n"
                            # Add the result
                            final_response += "Here are the detailed results:\n"
                        
                        # Determine appropriate visualization
                        with st.spinner("Creating visualization..."):
                            vis_recommendation = determine_visualization_type(user_query, sql_query, result_df)
                            fig = create_visualization(result_df, vis_recommendation)
                        
                        with st.chat_message("assistant"):
                            st.write(final_response)
                            
                            # Display the data table
                            st.dataframe(result_df)
                            
                            # Display visualization if available
                            if fig:
                                st.plotly_chart(fig, use_container_width=True)
                                description = vis_recommendation.get("description", "")
                                if description:
                                    st.caption(f"Visualization notes: {description}")
                        
                        # Save message without the dataframe (for OpenAI history)
                        response_content = final_response + "[Results shown in table format and visualization]"
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": response_content
                        })
                        
                        # Store full response with dataframe and visualization for display
                        visualization_notes = ""
                        if fig and vis_recommendation.get("description"):
                            visualization_notes = f"Visualization notes: {vis_recommendation.get('description')}"
                            
                        st.session_state.full_responses.append({
                            "user_query": user_query,
                            "text_response": final_response,
                            "data": result_df,
                            "visualization": fig,
                            "visualization_notes": visualization_notes,
                            "sql_query": sql_query if st.session_state.debug_mode else None
                        })
                        
                        # Add to chat history for OpenAI (only if history is enabled)
                        if st.session_state.show_history:
                            st.session_state.chat_history.append({
                                "role": "assistant", 
                                "content": response_content
                            })
                    else:
                        # Format error response based on debug mode
                        if st.session_state.debug_mode:
                            final_response = f"I generated this SQL query, but there was an error executing it:\n```sql\n{sql_query}\n```"
                        else:
                            final_response = f"I couldn't retrieve the data you asked for. There might be an issue with the query or connection."
                        
                        with st.chat_message("assistant"):
                            st.write(final_response)
                        st.session_state.messages.append({"role": "assistant", "content": final_response})
                        
                        # Store error in full_responses for consistent display
                        st.session_state.full_responses.append({
                            "user_query": user_query,
                            "text_response": final_response,
                            "data": None,
                            "visualization": None,
                            "visualization_notes": None,
                            "sql_query": sql_query if st.session_state.debug_mode else None
                        })
                        
                        # Add to chat history for OpenAI (only if history is enabled)
                        if st.session_state.show_history:
                            st.session_state.chat_history.append({"role": "assistant", "content": final_response})
                else:
                    # If no SQL was generated
                    with st.chat_message("assistant"):
                        st.write(rag_response)
                    st.session_state.messages.append({"role": "assistant", "content": rag_response})
                    
                    # Store in full_responses for consistent display
                    st.session_state.full_responses.append({
                        "user_query": user_query,
                        "text_response": rag_response,
                        "data": None,
                        "visualization": None,
                        "visualization_notes": None
                    })
                    
                    # Add to chat history for OpenAI (only if history is enabled)
                    if st.session_state.show_history:
                        st.session_state.chat_history.append({"role": "assistant", "content": rag_response})
            else:
                # Fallback to regular LLM response if vector store not ready
                llm_response = get_llm_response(
                    user_query=user_query,
                    table_name=st.session_state.table_name,
                    schema_name="O3_DEV_RAW_SCH",
                    database_name="O3_DEV_DB",
                    column_info=column_info,
                    conversation_history=conversation_history
                )
                with st.chat_message("assistant"):
                    st.write(llm_response)
                st.session_state.messages.append({"role": "assistant", "content": llm_response})
                
                # Store in full_responses for consistent display
                st.session_state.full_responses.append({
                    "user_query": user_query,
                    "text_response": llm_response,
                    "data": None,
                    "visualization": None,
                    "visualization_notes": None
                })
                
                # Add to chat history for OpenAI (only if history is enabled)
                if st.session_state.show_history:
                    st.session_state.chat_history.append({"role": "assistant", "content": llm_response})
else:
    st.info("Please connect to Snowflake to use the chatbot.")







