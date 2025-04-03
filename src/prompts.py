# # # # # import os
# # # # # from openai import OpenAI

# # # # # # the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
# # # # # # do not change this unless explicitly requested by the user
# # # # # MODEL_NAME = "gpt-4o"

# # # # # # Initialize OpenAI client
# # # # # # client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# # # # # # client=OpenAI(api_key='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')
# # # # # client=OpenAI(api_key='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')

# # # # # def generate_introduction(schema_info):
# # # # #     """Generate an introduction about the OEE data based on the schema."""
    
# # # # #     prompt = f"""
# # # # #     You are an assistant that specializes in Overall Equipment Effectiveness (OEE) analysis.
# # # # #     You have access to a database table called OEESHIFTWISE with the following columns and data types:
    
# # # # #     {schema_info}
    
# # # # #     Please provide a brief introduction explaining what OEE is and how this data can help users
# # # # #     analyze manufacturing equipment performance. Also suggest a few types of questions the user
# # # # #     could ask about this data.
# # # # #     """
    
# # # # #     try:
# # # # #         response = client.chat.completions.create(
# # # # #             model=MODEL_NAME,
# # # # #             messages=[
# # # # #                 {"role": "system", "content": "You are a helpful assistant that specializes in manufacturing analytics."},
# # # # #                 {"role": "user", "content": prompt}
# # # # #             ],
# # # # #             temperature=0.1,
# # # # #             max_tokens=500
# # # # #         )
# # # # #         return response.choices[0].message.content
# # # # #     except Exception as e:
# # # # #         return f"I encountered an error generating an introduction: {str(e)}. You can ask me questions about OEE data, and I'll do my best to answer them."

# # # # # def get_llm_response(user_query, table_name, schema_name, database_name, column_info):
# # # # #     """Generate a response to a user query about OEE data."""
    
# # # # #     prompt = f"""
# # # # #     You are an assistant that helps users query and analyze OEE (Overall Equipment Effectiveness) data.
    
# # # # #     The user is querying a Snowflake database with the following details:
# # # # #     - Database: {database_name}
# # # # #     - Schema: {schema_name}
# # # # #     - Table: {table_name}
    
# # # # #     The table has these columns with their data types:
# # # # #     {column_info}
    
# # # # #     The user's query is: "{user_query}"
    
# # # # #     If the query is asking for data that can be retrieved with SQL, please:
# # # # #     1. Write a SQL query to get the requested information from the {table_name} table
# # # # #     2. Format your SQL code block with ```sql at the beginning and ``` at the end
# # # # #     3. Make sure to write standard SQL compatible with Snowflake
# # # # #     4. Use proper column names as shown above
# # # # #     5. Keep your SQL query efficient and focused on answering the specific question
    
# # # # #     If the query is not asking for data or cannot be answered with SQL:
# # # # #     1. Provide a helpful explanation about OEE concepts
# # # # #     2. Suggest a reformulation of their question that could be answered with the available data
    
# # # # #     Remember, OEE (Overall Equipment Effectiveness) is a standard metric in manufacturing that measures 
# # # # #     productivity by combining availability, performance, and quality metrics.
# # # # #     """
    
# # # # #     try:
# # # # #         response = client.chat.completions.create(
# # # # #             model=MODEL_NAME,
# # # # #             messages=[
# # # # #                 {"role": "system", "content": "You are a helpful assistant that specializes in SQL queries and OEE analytics."},
# # # # #                 {"role": "user", "content": prompt}
# # # # #             ],
# # # # #             temperature=0.1
# # # # #         )
# # # # #         return response.choices[0].message.content
# # # # #     except Exception as e:
# # # # #         return f"I encountered an error: {str(e)}. Please try rephrasing your question or try again later."







# # import os
# # from openai import OpenAI

# # # Initialize OpenAI client
# # client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
# # # client = OpenAI(api_key='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')
# # GPT_MODEL = "gpt-4o"  # the newest OpenAI model is "gpt-4o" which was released May 13, 2024.

# # def call_openai(prompt, system_prompt="You are a helpful assistant.", conversation_history=None):
# #     """
# #     Call the OpenAI API with the specified prompt and optional conversation history.
    
# #     Args:
# #         prompt: The current user query
# #         system_prompt: The system prompt to set the assistant's behavior
# #         conversation_history: A list of previous messages in the format [{"role": "user|assistant", "content": "message"}]
# #     """
# #     # Prepare messages with system prompt first
# #     messages = [{"role": "system", "content": system_prompt}]
    
# #     # Add conversation history if provided
# #     if conversation_history and len(conversation_history) > 0:
# #         messages.extend(conversation_history)
    
# #     # Add current prompt
# #     messages.append({"role": "user", "content": prompt})
    
# #     try:
# #         response = client.chat.completions.create(
# #             model=GPT_MODEL,
# #             messages=messages
# #         )
# #         return response.choices[0].message.content
# #     except Exception as e:
# #         return f"Error connecting to OpenAI: {str(e)}"

# # def generate_introduction(schema_info):
# #     """Generate an introduction about the OEE data based on the schema."""
    
# #     prompt = f"""
# #     You are an assistant that specializes in Overall Equipment Effectiveness (OEE) analysis.
# #     You have access to a database table called OEESHIFTWISE with the following columns and data types:
    
# #     {schema_info}
    
# #     Please provide a brief introduction explaining what OEE is and how this data can help users
# #     analyze manufacturing equipment performance. Also suggest a few types of questions the user
# #     could ask about this data.
# #     """
    
# #     system_prompt = "You are a helpful assistant that specializes in manufacturing analytics."
    
# #     try:
# #         response = call_openai(prompt, system_prompt)
# #         return response
# #     except Exception as e:
# #         return f"I encountered an error generating an introduction: {str(e)}. You can ask me questions about OEE data, and I'll do my best to answer them."

# # def get_llm_response(user_query, table_name, schema_name, database_name, column_info, conversation_history=None):
# #     """
# #     Generate a response to a user query about OEE data.
    
# #     Args:
# #         user_query: The user's question
# #         table_name: The name of the table in Snowflake
# #         schema_name: The name of the schema in Snowflake
# #         database_name: The name of the database in Snowflake
# #         column_info: Information about the columns in the table
# #         conversation_history: List of previous messages in the format [{"role": "user|assistant", "content": "message"}]
# #     """
    
# #     prompt = f"""
# #     You are an assistant that helps users query and analyze OEE (Overall Equipment Effectiveness) data.
    
# #     The user is querying a Snowflake database with the following details:
# #     - Database: {database_name}
# #     - Schema: {schema_name}
# #     - Table: {table_name}
    
# #     The table has these columns with their data types:
# #     {column_info}
    
# #     The user's current query is: "{user_query}"
    
# #     When responding, consider the conversation context and previous questions if they are provided.
    
# #     If the query is asking for data that can be retrieved with SQL, please:
# #     1. Write a SQL query to get the requested information from the {table_name} table
# #     2. Format your SQL code block with ```sql at the beginning and ``` at the end
# #     3. Make sure to write standard SQL compatible with Snowflake
# #     4. Use proper column names as shown above
# #     5. Keep your SQL query efficient and focused on answering the specific question
    
# #     If the query is not asking for data or cannot be answered with SQL:
# #     1. Provide a helpful explanation about OEE concepts
# #     2. Suggest a reformulation of their question that could be answered with the available data
    
# #     Remember, OEE (Overall Equipment Effectiveness) is a standard metric in manufacturing that measures 
# #     productivity by combining availability, performance, and quality metrics.
# #     """
    
# #     system_prompt = "You are a helpful assistant that specializes in SQL queries and OEE analytics."
    
# #     try:
# #         response = call_openai(prompt, system_prompt, conversation_history)
# #         return response
# #     except Exception as e:
# #         return f"I encountered an error: {str(e)}. Please try rephrasing your question or try again later."



# # # import os
# # # from openai import OpenAI

# # # # Initialize OpenAI client
# # # client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
# # # # client = OpenAI(api_key='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')

# # # GPT_MODEL = "gpt-4o"  # the newest OpenAI model is "gpt-4o" which was released May 13, 2024.

# # # def call_openai(prompt, system_prompt="You are a helpful assistant.", conversation_history=None):
# # #     """
# # #     Call the OpenAI API with the specified prompt and optional conversation history.
    
# # #     Args:
# # #         prompt: The current user query
# # #         system_prompt: The system prompt to set the assistant's behavior
# # #         conversation_history: A list of previous messages in the format [{"role": "user|assistant", "content": "message"}]
# # #     """
# # #     # Prepare messages with system prompt first
# # #     messages = [{"role": "system", "content": system_prompt}]
    
# # #     # Add conversation history if provided
# # #     if conversation_history and len(conversation_history) > 0:
# # #         messages.extend(conversation_history)
    
# # #     # Add current prompt
# # #     messages.append({"role": "user", "content": prompt})
    
# # #     try:
# # #         response = client.chat.completions.create(
# # #             model=GPT_MODEL,
# # #             messages=messages
# # #         )
# # #         return response.choices[0].message.content
# # #     except Exception as e:
# # #         return f"Error connecting to OpenAI: {str(e)}"

# # # def generate_introduction(schema_info, table_name="OEESHIFTWISE_AI"):
# # #     """Generate an introduction about the OEE data based on the schema."""
    
# # #     prompt = f"""
# # #     You are an assistant that specializes in Overall Equipment Effectiveness (OEE) analysis.
# # #     You have access to a database table called {table_name} with the following columns and data types:
    
# # #     {schema_info}
    
# # #     Please provide a brief introduction explaining what OEE is and how this data can help users
# # #     analyze manufacturing equipment performance. Also suggest a few types of questions the user
# # #     could ask about this data.
# # #     """
    
# # #     system_prompt = "You are a helpful assistant that specializes in manufacturing analytics."
    
# # #     try:
# # #         response = call_openai(prompt, system_prompt)
# # #         return response
# # #     except Exception as e:
# # #         return f"I encountered an error generating an introduction: {str(e)}. You can ask me questions about OEE data, and I'll do my best to answer them."

# # # def get_llm_response(user_query, table_name, schema_name, database_name, column_info, conversation_history=None):
# # #     """
# # #     Generate a response to a user query about OEE data.
    
# # #     Args:
# # #         user_query: The user's question
# # #         table_name: The name of the table in Snowflake
# # #         schema_name: The name of the schema in Snowflake
# # #         database_name: The name of the database in Snowflake
# # #         column_info: Information about the columns in the table
# # #         conversation_history: List of previous messages in the format [{"role": "user|assistant", "content": "message"}]
# # #     """
    
# # #     prompt = f"""
# # #     You are an assistant that helps users query and analyze OEE (Overall Equipment Effectiveness) data.
    
# # #     The user is querying a Snowflake database with the following details:
# # #     - Database: {database_name}
# # #     - Schema: {schema_name}
# # #     - Table: {table_name}
    
# # #     The table has these columns with their data types:
# # #     {column_info}
    
# # #     The user's current query is: "{user_query}"
    
# # #     When responding, consider the conversation context and previous questions if they are provided.
    
# # #     If the query is asking for data that can be retrieved with SQL, please:
# # #     1. Write a SQL query to get the requested information from the {table_name} table
# # #     2. Format your SQL code block with ```sql at the beginning and ``` at the end
# # #     3. Make sure to write standard SQL compatible with Snowflake
# # #     4. Use proper column names as shown above
# # #     5. Keep your SQL query efficient and focused on answering the specific question
    
# # #     If the query is not asking for data or cannot be answered with SQL:
# # #     1. Provide a helpful explanation about OEE concepts
# # #     2. Suggest a reformulation of their question that could be answered with the available data
    
# # #     Remember, OEE (Overall Equipment Effectiveness) is a standard metric in manufacturing that measures 
# # #     productivity by combining availability, performance, and quality metrics.
# # #     """
    
# # #     system_prompt = "You are a helpful assistant that specializes in SQL queries and OEE analytics."
    
# # #     try:
# # #         response = call_openai(prompt, system_prompt, conversation_history)
# # #         return response
# # #     except Exception as e:
# # #         return f"I encountered an error: {str(e)}. Please try rephrasing your question or try again later."































































# """
# RAG (Retrieval-Augmented Generation) utilities for enhancing LLM responses with context.
# This module handles embeddings, vector store operations, and RAG processing for the OEE chatbot.
# """
# import os
# import json
# import pickle
# import pandas as pd
# import numpy as np
# import streamlit as st
# from sklearn.metrics.pairwise import cosine_similarity
# from openai import OpenAI

# # Initialize OpenAI client
# client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# # Models
# GPT_MODEL = "gpt-4o"  # the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
# EMBEDDING_MODEL = "text-embedding-3-small"

# # Constants
# EMBEDDINGS_FILE = "embeddings_store.pkl"
# CHUNK_SIZE = 5
# TOP_K_CONTEXTS = 3

# def get_openai_embedding(text):
#     """Get embeddings from the OpenAI API.
    
#     Args:
#         text (str): Text to embed
        
#     Returns:
#         list: The embedding vector or empty list if error
#     """
#     try:
#         response = client.embeddings.create(
#             model=EMBEDDING_MODEL,
#             input=text
#         )
#         return response.data[0].embedding
#     except Exception as e:
#         st.error(f"Error getting embeddings from OpenAI: {str(e)}")
#         return []

# def call_openai(prompt, system_prompt="You are a helpful assistant.", conversation_history=None):
#     """Call the OpenAI API with the specified prompt and optional conversation history.
    
#     Args:
#         prompt (str): The current user query
#         system_prompt (str): The system prompt to set the assistant's behavior
#         conversation_history (list): Previous messages in format [{"role": "user|assistant", "content": "message"}]
        
#     Returns:
#         str: The response from OpenAI
#     """
#     # Prepare messages with system prompt first
#     messages = [{"role": "system", "content": system_prompt}]
    
#     # Add conversation history if provided
#     if conversation_history and len(conversation_history) > 0:
#         messages.extend(conversation_history)
    
#     # Add current prompt
#     messages.append({"role": "user", "content": prompt})
    
#     try:
#         response = client.chat.completions.create(
#             model=GPT_MODEL,
#             messages=messages
#         )
#         return response.choices[0].message.content
#     except Exception as e:
#         return f"Error connecting to OpenAI: {str(e)}"

# def create_document_chunks(df, chunk_size=CHUNK_SIZE):
#     """Create chunks of data from the dataframe for better context.
    
#     Args:
#         df (DataFrame): Pandas DataFrame with the data
#         chunk_size (int): Number of rows per chunk
        
#     Returns:
#         list: List of text chunks representing the data
#     """
#     chunks = []
#     total_rows = len(df)
    
#     # Create chunks by combining rows into meaningful textual descriptions
#     for i in range(0, total_rows, chunk_size):
#         end_idx = min(i + chunk_size, total_rows)
#         chunk_df = df.iloc[i:end_idx]
        
#         # Convert chunk to text representation with column names and values
#         chunk_text = "Data chunk with the following information:\n"
#         for _, row in chunk_df.iterrows():
#             row_text = " | ".join([f"{col}: {row[col]}" for col in df.columns])
#             chunk_text += row_text + "\n"
        
#         chunks.append(chunk_text)
    
#     # Add schema information as a separate chunk
#     schema_chunk = "Table schema information:\n"
#     for col in df.columns:
#         schema_chunk += f"Column: {col}, Type: {df[col].dtype}\n"
#     chunks.append(schema_chunk)
    
#     # Add statistical information for numeric columns
#     for col in df.select_dtypes(include=[np.number]).columns:
#         stats_chunk = f"Statistical information for column {col}:\n"
#         stats_chunk += f"Min: {df[col].min()}, Max: {df[col].max()}, Mean: {df[col].mean()}, Median: {df[col].median()}\n"
#         chunks.append(stats_chunk)
    
#     return chunks

# def save_vector_store(vector_store, file_path=EMBEDDINGS_FILE):
#     """Save the vector store to disk.
    
#     Args:
#         vector_store (dict): The vector store with chunks and embeddings
#         file_path (str): Path to save the pickle file
        
#     Returns:
#         bool: True if saved successfully, False otherwise
#     """
#     try:
#         with open(file_path, 'wb') as file:
#             pickle.dump(vector_store, file)
#         return True
#     except Exception as e:
#         st.warning(f"Could not save embeddings to file: {str(e)}")
#         return False

# def load_vector_store(file_path=EMBEDDINGS_FILE):
#     """Load the vector store from disk.
    
#     Args:
#         file_path (str): Path to the pickle file
        
#     Returns:
#         dict: The loaded vector store or None if error
#     """
#     try:
#         with open(file_path, 'rb') as file:
#             return pickle.load(file)
#     except Exception as e:
#         st.error(f"Error loading embeddings: {str(e)}")
#         return None

# def initialize_vector_store(df):
#     """Initialize the vector store with embeddings from the dataframe.
    
#     Args:
#         df (DataFrame): Pandas DataFrame with the data
        
#     Returns:
#         dict: The vector store with chunks and embeddings
#     """
#     progress_placeholder = st.empty()
    
#     # Check if embeddings file already exists
#     if os.path.exists(EMBEDDINGS_FILE):
#         progress_placeholder.text("Loading existing embeddings...")
#         vector_store = load_vector_store()
#         if vector_store:
#             progress_placeholder.text("Embeddings loaded successfully!")
#             return vector_store
#         progress_placeholder.warning("Failed to load embeddings, creating new ones...")
    
#     # Create document chunks from dataframe
#     progress_placeholder.text("Creating document chunks...")
#     chunks = create_document_chunks(df)
    
#     # Create embeddings for each chunk using OpenAI
#     embeddings = []
#     for i, chunk in enumerate(chunks):
#         progress_placeholder.text(f"Creating embeddings... ({i+1}/{len(chunks)})")
#         embedding = get_openai_embedding(chunk)
#         embeddings.append(embedding)
    
#     # Create vector store
#     vector_store = {
#         "embeddings": embeddings,
#         "chunks": chunks
#     }
    
#     # Save embeddings to file
#     if save_vector_store(vector_store):
#         progress_placeholder.text("Embeddings created and saved successfully!")
#     else:
#         progress_placeholder.warning("Embeddings created but not saved!")
    
#     return vector_store

# def compute_similarity(query_embedding, stored_embedding):
#     """Compute cosine similarity between two embeddings.
    
#     Args:
#         query_embedding (list): The query embedding
#         stored_embedding (list): The stored embedding
        
#     Returns:
#         float: The cosine similarity
#     """
#     if not query_embedding or not stored_embedding or len(stored_embedding) == 0:
#         return 0
        
#     # Convert to numpy arrays
#     emb_array = np.array(stored_embedding).reshape(1, -1)
#     query_array = np.array(query_embedding).reshape(1, -1)
    
#     # Calculate cosine similarity
#     return cosine_similarity(emb_array, query_array)[0][0]

# def retrieve_relevant_contexts(query, vector_store, top_k=TOP_K_CONTEXTS):
#     """Retrieve the most relevant contexts for a query.
    
#     Args:
#         query (str): The user query
#         vector_store (dict): The vector store with chunks and embeddings
#         top_k (int): Number of top contexts to retrieve
        
#     Returns:
#         list: The most relevant contexts
#     """
#     chunks = vector_store["chunks"]
#     stored_embeddings = vector_store["embeddings"]
    
#     # Get query embedding
#     query_embedding = get_openai_embedding(query)
    
#     # Fallback if embeddings are not available
#     if not query_embedding or not stored_embeddings:
#         return chunks[:min(top_k, len(chunks))]
    
#     # Calculate similarities
#     similarities = [compute_similarity(query_embedding, emb) for emb in stored_embeddings]
    
#     # Get indices of top_k most similar chunks
#     if not similarities:
#         return chunks[:min(top_k, len(chunks))]
    
#     indices = np.argsort(similarities)[-top_k:][::-1]
    
#     # Get the relevant chunks
#     relevant_chunks = [chunks[idx] for idx in indices]
    
#     return relevant_chunks

# def process_query_with_rag(user_query, vector_store, table_name, schema_name, database_name, column_info, conversation_history=None):
#     """Process a user query using RAG to provide contextually enhanced answers.
    
#     Args:
#         user_query (str): The user's question
#         vector_store (dict): The vector store containing embeddings and chunks
#         table_name (str): The name of the table in Snowflake
#         schema_name (str): The name of the schema in Snowflake
#         database_name (str): The name of the database in Snowflake
#         column_info (str): Information about the columns in the table
#         conversation_history (list): Previous messages in format [{"role": "user|assistant", "content": "message"}]
        
#     Returns:
#         str: The response from the LLM
#     """
#     # Retrieve relevant contexts
#     relevant_contexts = retrieve_relevant_contexts(user_query, vector_store)
    
#     # Combine contexts into a single string
#     combined_context = "\n".join(relevant_contexts)
    
#     # Create a prompt with the retrieved context
#     prompt = f"""
#     You are an assistant that helps users query and analyze OEE (Overall Equipment Effectiveness) data.
    
#     The user is querying a Snowflake database with the following details:
#     - Database: {database_name}
#     - Schema: {schema_name}
#     - Table: {table_name}
    
#     The table has these columns with their data types:
#     {column_info}
    
#     I've retrieved some relevant context from the database to help you answer:
    
#     {combined_context}
    
#     The user's current query is: "{user_query}"
    
#     When responding, consider the conversation context and previous questions if they are provided.
    
#     If the query is asking for data that can be retrieved with SQL, please:
#     1. Write a SQL query to get the requested information from the {table_name} table
#     2. Format your SQL code block with ```sql at the beginning and ``` at the end
#     3. Make sure to write standard SQL compatible with Snowflake
#     4. Use proper column names as shown above
#     5. Keep your SQL query efficient and focused on answering the specific question
    
#     If the query is not asking for data or cannot be answered with SQL:
#     1. Provide a helpful explanation about OEE concepts
#     2. Suggest a reformulation of their question that could be answered with the available data
    
#     Remember, OEE (Overall Equipment Effectiveness) is a standard metric in manufacturing that measures 
#     productivity by combining availability, performance, and quality metrics.
#     """
    
#     system_prompt = "You are a helpful assistant that specializes in SQL queries and OEE analytics."
    
#     try:
#         response = call_openai(prompt, system_prompt, conversation_history)
#         return response
#     except Exception as e:
#         return f"I encountered an error: {str(e)}. Please try rephrasing your question or try again later."
































# import os
# from openai import OpenAI

# # Initialize OpenAI client
# client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
# GPT_MODEL = "gpt-4o"  # the newest OpenAI model is "gpt-4o" which was released May 13, 2024.

# def call_openai(prompt, system_prompt="You are a helpful assistant.", conversation_history=None):
#     """
#     Call the OpenAI API with the specified prompt and optional conversation history.
    
#     Args:
#         prompt: The current user query
#         system_prompt: The system prompt to set the assistant's behavior
#         conversation_history: A list of previous messages in the format [{"role": "user|assistant", "content": "message"}]
#     """
#     # Prepare messages with system prompt first
#     messages = [{"role": "system", "content": system_prompt}]
    
#     # Add conversation history if provided
#     if conversation_history and len(conversation_history) > 0:
#         messages.extend(conversation_history)
    
#     # Add current prompt
#     messages.append({"role": "user", "content": prompt})
    
#     try:
#         response = client.chat.completions.create(
#             model=GPT_MODEL,
#             messages=messages
#         )
#         return response.choices[0].message.content
#     except Exception as e:
#         return f"Error connecting to OpenAI: {str(e)}"

# def generate_introduction(schema_info, table_name="OEESHIFTWISE_AI"):
#     """Generate an introduction about the OEE data based on the schema."""
    
#     prompt = f"""
#     You are an assistant that specializes in Overall Equipment Effectiveness (OEE) analysis.
#     You have access to a database table called {table_name} with the following columns and data types:
    
#     {schema_info}
    
#     Please provide a brief introduction explaining what OEE is and how this data can help users
#     analyze manufacturing equipment performance. Also suggest a few types of questions the user
#     could ask about this data.
#     """
    
#     system_prompt = "You are a helpful assistant that specializes in manufacturing analytics."
    
#     try:
#         response = call_openai(prompt, system_prompt)
#         return response
#     except Exception as e:
#         return f"I encountered an error generating an introduction: {str(e)}. You can ask me questions about OEE data, and I'll do my best to answer them."

# def get_llm_response(user_query, table_name, schema_name, database_name, column_info, conversation_history=None):
#     """
#     Generate a response to a user query about OEE data.
    
#     Args:
#         user_query: The user's question
#         table_name: The name of the table in Snowflake
#         schema_name: The name of the schema in Snowflake
#         database_name: The name of the database in Snowflake
#         column_info: Information about the columns in the table
#         conversation_history: List of previous messages in the format [{"role": "user|assistant", "content": "message"}]
#     """
    
#     prompt = f"""
#     You are an assistant that helps users query and analyze OEE (Overall Equipment Effectiveness) data.
    
#     The user is querying a Snowflake database with the following details:
#     - Database: {database_name}
#     - Schema: {schema_name}
#     - Table: {table_name}
    
#     The table has these columns with their data types:
#     {column_info}
    
#     The user's current query is: "{user_query}"
    
#     When responding, consider the conversation context and previous questions if they are provided.
    
#     If the query is asking for data that can be retrieved with SQL, please:
#     1. Write a SQL query to get the requested information from the {table_name} table
#     2. Format your SQL code block with ```sql at the beginning and ``` at the end
#     3. Make sure to write standard SQL compatible with Snowflake
#     4. Use proper column names as shown above
#     5. Keep your SQL query efficient and focused on answering the specific question
    
#     If the query is not asking for data or cannot be answered with SQL:
#     1. Provide a helpful explanation about OEE concepts
#     2. Suggest a reformulation of their question that could be answered with the available data
    
#     Remember, OEE (Overall Equipment Effectiveness) is a standard metric in manufacturing that measures 
#     productivity by combining availability, performance, and quality metrics.
#     """
    
#     system_prompt = "You are a helpful assistant that specializes in SQL queries and OEE analytics."
    
#     try:
#         response = call_openai(prompt, system_prompt, conversation_history)
#         return response
#     except Exception as e:
#         return f"I encountered an error: {str(e)}. Please try rephrasing your question or try again later."
























import os
from openai import OpenAI
import streamlit as st

# Initialize OpenAI client
# client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
client = st.secrets["openai"]["api_key"]
GPT_MODEL = "gpt-4o"  # the newest OpenAI model is "gpt-4o" which was released May 13, 2024.

def call_openai(prompt, system_prompt="You are a helpful assistant.", conversation_history=None):
    """
    Call the OpenAI API with the specified prompt and optional conversation history.
    
    Args:
        prompt: The current user query
        system_prompt: The system prompt to set the assistant's behavior
        conversation_history: A list of previous messages in the format [{"role": "user|assistant", "content": "message"}]
    """
    # Prepare messages with system prompt first
    messages = [{"role": "system", "content": system_prompt}]
    
    # Add conversation history if provided
    if conversation_history and len(conversation_history) > 0:
        messages.extend(conversation_history)
    
    # Add current prompt
    messages.append({"role": "user", "content": prompt})
    
    try:
        response = client.chat.completions.create(
            model=GPT_MODEL,
            messages=messages
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error connecting to OpenAI: {str(e)}"

def generate_introduction(schema_info, table_name="OEESHIFTWISE_AI"):
    """Generate an introduction about the OEE data based on the schema."""
    
    prompt = f"""
    You are an assistant that specializes in Overall Equipment Effectiveness (OEE) analysis.
    You have access to a database table called {table_name} with the following columns and data types:
    
    {schema_info}
    
    Please provide a brief introduction explaining what OEE is and how this data can help users
    analyze manufacturing equipment performance. Also suggest a few types of questions the user
    could ask about this data.
    """
    
    system_prompt = "You are a helpful assistant that specializes in manufacturing analytics."
    
    try:
        response = call_openai(prompt, system_prompt)
        return response
    except Exception as e:
        return f"I encountered an error generating an introduction: {str(e)}. You can ask me questions about OEE data, and I'll do my best to answer them."

def get_llm_response(user_query, table_name, schema_name, database_name, column_info, conversation_history=None):
    """
    Generate a response to a user query about OEE data.
    
    Args:
        user_query: The user's question
        table_name: The name of the table in Snowflake
        schema_name: The name of the schema in Snowflake
        database_name: The name of the database in Snowflake
        column_info: Information about the columns in the table
        conversation_history: List of previous messages in the format [{"role": "user|assistant", "content": "message"}]
    """
    
    prompt = f"""
    You are an assistant that helps users query and analyze OEE (Overall Equipment Effectiveness) data.
    
    The user is querying a Snowflake database with the following details:
    - Database: {database_name}
    - Schema: {schema_name}
    - Table: {table_name}
    
    The table has these columns with their data types:
    {column_info}
    
    The user's current query is: "{user_query}"
    
    When responding, consider the conversation context and previous questions if they are provided.
    
    If the query is asking for data that can be retrieved with SQL, please:
    1. Write a SQL query to get the requested information from the {table_name} table
    2. Format your SQL code block with ```sql at the beginning and ``` at the end
    3. Make sure to write standard SQL compatible with Snowflake
    4. Use proper column names as shown above
    5. Keep your SQL query efficient and focused on answering the specific question
    
    If the query is not asking for data or cannot be answered with SQL:
    1. Provide a helpful explanation about OEE concepts
    2. Suggest a reformulation of their question that could be answered with the available data
    
    Remember, OEE (Overall Equipment Effectiveness) is a standard metric in manufacturing that measures 
    productivity by combining availability, performance, and quality metrics.
    """
    
    system_prompt = "You are a helpful assistant that specializes in SQL queries and OEE analytics."
    
    try:
        response = call_openai(prompt, system_prompt, conversation_history)
        return response
    except Exception as e:
        return f"I encountered an error: {str(e)}. Please try rephrasing your question or try again later."