# # # # # import pandas as pd
# # # # # import numpy as np
# # # # # import os
# # # # # from sentence_transformers import SentenceTransformer 
# # # # # import faiss
# # # # # from openai import OpenAI
# # # # # import streamlit as st

# # # # # # the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
# # # # # # do not change this unless explicitly requested by the user
# # # # # MODEL_NAME = "gpt-4o-mini"

# # # # # # Initialize OpenAI client
# # # # # # client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# # # # # # client=OpenAI(api_key='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')
# # # # # client=OpenAI(api_key='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')


# # # # # # Load the sentence transformer model for embeddings
# # # # # @st.cache_resource
# # # # # def get_embedding_model():
# # # # #     return SentenceTransformer('all-MiniLM-L6-v2')

# # # # # def create_document_chunks(df, chunk_size=5):
# # # # #     """Create chunks of data from the dataframe for better context."""
# # # # #     chunks = []
# # # # #     total_rows = len(df)
    
# # # # #     # Create chunks by combining rows into meaningful textual descriptions
# # # # #     for i in range(0, total_rows, chunk_size):
# # # # #         end_idx = min(i + chunk_size, total_rows)
# # # # #         chunk_df = df.iloc[i:end_idx]
        
# # # # #         # Convert chunk to text representation with column names and values
# # # # #         chunk_text = "Data chunk with the following information:\n"
# # # # #         for _, row in chunk_df.iterrows():
# # # # #             row_text = " | ".join([f"{col}: {row[col]}" for col in df.columns])
# # # # #             chunk_text += row_text + "\n"
        
# # # # #         chunks.append(chunk_text)
    
# # # # #     # Add schema information as a separate chunk
# # # # #     schema_chunk = "Table schema information:\n"
# # # # #     for col in df.columns:
# # # # #         schema_chunk += f"Column: {col}, Type: {df[col].dtype}\n"
# # # # #     chunks.append(schema_chunk)
    
# # # # #     # Add statistical information as separate chunks
# # # # #     for col in df.select_dtypes(include=[np.number]).columns:
# # # # #         stats_chunk = f"Statistical information for column {col}:\n"
# # # # #         stats_chunk += f"Min: {df[col].min()}, Max: {df[col].max()}, Mean: {df[col].mean()}, Median: {df[col].median()}\n"
# # # # #         chunks.append(stats_chunk)
    
# # # # #     return chunks

# # # # # def initialize_vector_store(df):
# # # # #     """Initialize the vector store with embeddings from the dataframe."""
# # # # #     model = get_embedding_model()
    
# # # # #     # Create document chunks from dataframe
# # # # #     chunks = create_document_chunks(df)
    
# # # # #     # Create embeddings for each chunk
# # # # #     embeddings = model.encode(chunks)
    
# # # # #     # Normalize embeddings for cosine similarity
# # # # #     faiss.normalize_L2(embeddings)
    
# # # # #     # Create FAISS index
# # # # #     dimension = embeddings.shape[1]
# # # # #     index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity with normalized vectors
# # # # #     index.add(embeddings)
    
# # # # #     return {
# # # # #         "index": index,
# # # # #         "chunks": chunks,
# # # # #         "model": model
# # # # #     }

# # # # # def retrieve_relevant_contexts(query, vector_store, top_k=3):
# # # # #     """Retrieve the most relevant contexts for a query."""
# # # # #     model = vector_store["model"]
# # # # #     index = vector_store["index"]
# # # # #     chunks = vector_store["chunks"]
    
# # # # #     # Create query embedding
# # # # #     query_embedding = model.encode([query])[0]
# # # # #     query_embedding = query_embedding / np.linalg.norm(query_embedding)  # Normalize
# # # # #     query_embedding = query_embedding.reshape(1, -1)  # Reshape for FAISS
    
# # # # #     # Search for similar contexts
# # # # #     scores, indices = index.search(query_embedding, top_k)
    
# # # # #     # Get the relevant chunks
# # # # #     relevant_chunks = [chunks[idx] for idx in indices[0]]
    
# # # # #     return relevant_chunks

# # # # # def process_query_with_rag(user_query, vector_store, table_name, schema_name, database_name, column_info):
# # # # #     """Process a user query using RAG to provide contextually enhanced answers."""
    
# # # # #     # Retrieve relevant contexts
# # # # #     relevant_contexts = retrieve_relevant_contexts(user_query, vector_store)
    
# # # # #     # Combine contexts into a single string
# # # # #     combined_context = "\n".join(relevant_contexts)
    
# # # # #     # Create a prompt with the retrieved context
# # # # #     prompt = f"""
# # # # #     You are an assistant that helps users query and analyze OEE (Overall Equipment Effectiveness) data.
    
# # # # #     The user is querying a Snowflake database with the following details:
# # # # #     - Database: {database_name}
# # # # #     - Schema: {schema_name}
# # # # #     - Table: {table_name}
    
# # # # #     The table has these columns with their data types:
# # # # #     {column_info}
    
# # # # #     I've retrieved some relevant context from the database to help you answer:
    
# # # # #     {combined_context}
    
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











# # # # import pandas as pd
# # # # import numpy as np
# # # # import os
# # # # import json
# # # # import streamlit as st
# # # # from sklearn.metrics.pairwise import cosine_similarity
# # # # from openai import OpenAI

# # # # # Initialize OpenAI client
# # # # # client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
# # # # client = OpenAI(api_key='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')
# # # # GPT_MODEL = "gpt-4o"  # the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
# # # # EMBEDDING_MODEL = "text-embedding-3-small"

# # # # def get_openai_embedding(text):
# # # #     """Get embeddings from the OpenAI API."""
# # # #     try:
# # # #         response = client.embeddings.create(
# # # #             model=EMBEDDING_MODEL,
# # # #             input=text
# # # #         )
# # # #         return response.data[0].embedding
# # # #     except Exception as e:
# # # #         st.error(f"Error getting embeddings from OpenAI: {str(e)}")
# # # #         return []

# # # # def call_openai(prompt, system_prompt="You are a helpful assistant.", conversation_history=None):
# # # #     """
# # # #     Call the OpenAI API with the specified prompt and optional conversation history.
    
# # # #     Args:
# # # #         prompt: The current user query
# # # #         system_prompt: The system prompt to set the assistant's behavior
# # # #         conversation_history: A list of previous messages in the format [{"role": "user|assistant", "content": "message"}]
# # # #     """
# # # #     # Prepare messages with system prompt first
# # # #     messages = [{"role": "system", "content": system_prompt}]
    
# # # #     # Add conversation history if provided
# # # #     if conversation_history and len(conversation_history) > 0:
# # # #         messages.extend(conversation_history)
    
# # # #     # Add current prompt
# # # #     messages.append({"role": "user", "content": prompt})
    
# # # #     try:
# # # #         response = client.chat.completions.create(
# # # #             model=GPT_MODEL,
# # # #             messages=messages
# # # #         )
# # # #         return response.choices[0].message.content
# # # #     except Exception as e:
# # # #         return f"Error connecting to OpenAI: {str(e)}"

# # # # def create_document_chunks(df, chunk_size=5):
# # # #     """Create chunks of data from the dataframe for better context."""
# # # #     chunks = []
# # # #     total_rows = len(df)
    
# # # #     # Create chunks by combining rows into meaningful textual descriptions
# # # #     for i in range(0, total_rows, chunk_size):
# # # #         end_idx = min(i + chunk_size, total_rows)
# # # #         chunk_df = df.iloc[i:end_idx]
        
# # # #         # Convert chunk to text representation with column names and values
# # # #         chunk_text = "Data chunk with the following information:\n"
# # # #         for _, row in chunk_df.iterrows():
# # # #             row_text = " | ".join([f"{col}: {row[col]}" for col in df.columns])
# # # #             chunk_text += row_text + "\n"
        
# # # #         chunks.append(chunk_text)
    
# # # #     # Add schema information as a separate chunk
# # # #     schema_chunk = "Table schema information:\n"
# # # #     for col in df.columns:
# # # #         schema_chunk += f"Column: {col}, Type: {df[col].dtype}\n"
# # # #     chunks.append(schema_chunk)
    
# # # #     # Add statistical information as separate chunks
# # # #     for col in df.select_dtypes(include=[np.number]).columns:
# # # #         stats_chunk = f"Statistical information for column {col}:\n"
# # # #         stats_chunk += f"Min: {df[col].min()}, Max: {df[col].max()}, Mean: {df[col].mean()}, Median: {df[col].median()}\n"
# # # #         chunks.append(stats_chunk)
    
# # # #     return chunks

# # # # @st.cache_resource
# # # # def initialize_vector_store(df):
# # # #     """Initialize the vector store with embeddings from the dataframe."""
# # # #     # Create document chunks from dataframe
# # # #     chunks = create_document_chunks(df)
    
# # # #     # Create embeddings for each chunk using OpenAI
# # # #     embeddings = []
# # # #     # Using a plain progress message instead of nested status
# # # #     progress_placeholder = st.empty()
# # # #     for i, chunk in enumerate(chunks):
# # # #         progress_placeholder.text(f"Creating embeddings... ({i+1}/{len(chunks)})")
# # # #         embedding = get_openai_embedding(chunk)
# # # #         embeddings.append(embedding)
    
# # # #     progress_placeholder.text("Embeddings created successfully!")
    
# # # #     return {
# # # #         "embeddings": embeddings,
# # # #         "chunks": chunks
# # # #     }

# # # # def retrieve_relevant_contexts(query, vector_store, top_k=3):
# # # #     """Retrieve the most relevant contexts for a query."""
# # # #     chunks = vector_store["chunks"]
# # # #     stored_embeddings = vector_store["embeddings"]
    
# # # #     # Get query embedding
# # # #     query_embedding = get_openai_embedding(query)
    
# # # #     if not query_embedding or not stored_embeddings:
# # # #         # Fallback if embeddings fail
# # # #         return chunks[:min(top_k, len(chunks))]
    
# # # #     # Calculate similarities
# # # #     similarities = []
# # # #     for emb in stored_embeddings:
# # # #         if len(emb) == 0:  # Skip empty embeddings
# # # #             similarities.append(0)
# # # #             continue
            
# # # #         # Convert to numpy arrays
# # # #         emb_array = np.array(emb).reshape(1, -1)
# # # #         query_array = np.array(query_embedding).reshape(1, -1)
        
# # # #         # Calculate cosine similarity
# # # #         similarity = cosine_similarity(emb_array, query_array)[0][0]
# # # #         similarities.append(similarity)
    
# # # #     # Get indices of top_k most similar chunks
# # # #     if not similarities:
# # # #         return chunks[:min(top_k, len(chunks))]
    
# # # #     indices = np.argsort(similarities)[-top_k:][::-1]
    
# # # #     # Get the relevant chunks
# # # #     relevant_chunks = [chunks[idx] for idx in indices]
    
# # # #     return relevant_chunks

# # # # def process_query_with_rag(user_query, vector_store, table_name, schema_name, database_name, column_info, conversation_history=None):
# # # #     """
# # # #     Process a user query using RAG to provide contextually enhanced answers.
    
# # # #     Args:
# # # #         user_query: The user's question
# # # #         vector_store: The vector store containing embeddings and chunks
# # # #         table_name: The name of the table in Snowflake
# # # #         schema_name: The name of the schema in Snowflake
# # # #         database_name: The name of the database in Snowflake
# # # #         column_info: Information about the columns in the table
# # # #         conversation_history: List of previous messages in the format [{"role": "user|assistant", "content": "message"}]
# # # #     """
    
# # # #     # Retrieve relevant contexts
# # # #     relevant_contexts = retrieve_relevant_contexts(user_query, vector_store)
    
# # # #     # Combine contexts into a single string
# # # #     combined_context = "\n".join(relevant_contexts)
    
# # # #     # Create a prompt with the retrieved context
# # # #     prompt = f"""
# # # #     You are an assistant that helps users query and analyze OEE (Overall Equipment Effectiveness) data.
    
# # # #     The user is querying a Snowflake database with the following details:
# # # #     - Database: {database_name}
# # # #     - Schema: {schema_name}
# # # #     - Table: {table_name}
    
# # # #     The table has these columns with their data types:
# # # #     {column_info}
    
# # # #     I've retrieved some relevant context from the database to help you answer:
    
# # # #     {combined_context}
    
# # # #     The user's current query is: "{user_query}"
    
# # # #     When responding, consider the conversation context and previous questions if they are provided.
    
# # # #     If the query is asking for data that can be retrieved with SQL, please:
# # # #     1. Write a SQL query to get the requested information from the {table_name} table
# # # #     2. Format your SQL code block with ```sql at the beginning and ``` at the end
# # # #     3. Make sure to write standard SQL compatible with Snowflake
# # # #     4. Use proper column names as shown above
# # # #     5. Keep your SQL query efficient and focused on answering the specific question
    
# # # #     If the query is not asking for data or cannot be answered with SQL:
# # # #     1. Provide a helpful explanation about OEE concepts
# # # #     2. Suggest a reformulation of their question that could be answered with the available data
    
# # # #     Remember, OEE (Overall Equipment Effectiveness) is a standard metric in manufacturing that measures 
# # # #     productivity by combining availability, performance, and quality metrics.
# # # #     """
    
# # # #     system_prompt = "You are a helpful assistant that specializes in SQL queries and OEE analytics."
    
# # # #     try:
# # # #         response = call_openai(prompt, system_prompt, conversation_history)
# # # #         return response
# # # #     except Exception as e:
# # # #         return f"I encountered an error: {str(e)}. Please try rephrasing your question or try again later."




# # import pandas as pd
# # import numpy as np
# # import os
# # import json
# # import pickle
# # import streamlit as st
# # from sklearn.metrics.pairwise import cosine_similarity
# # from openai import OpenAI

# # # Initialize OpenAI client
# # client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
# # # client = OpenAI(api_key='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')
# # GPT_MODEL = "gpt-4o"  # the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
# # EMBEDDING_MODEL = "text-embedding-3-small"

# # def get_openai_embedding(text):
# #     """Get embeddings from the OpenAI API."""
# #     try:
# #         response = client.embeddings.create(
# #             model=EMBEDDING_MODEL,
# #             input=text
# #         )
# #         return response.data[0].embedding
# #     except Exception as e:
# #         st.error(f"Error getting embeddings from OpenAI: {str(e)}")
# #         return []

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

# # def create_document_chunks(df, chunk_size=5):
# #     """Create chunks of data from the dataframe for better context."""
# #     chunks = []
# #     total_rows = len(df)
    
# #     # Create chunks by combining rows into meaningful textual descriptions
# #     for i in range(0, total_rows, chunk_size):
# #         end_idx = min(i + chunk_size, total_rows)
# #         chunk_df = df.iloc[i:end_idx]
        
# #         # Convert chunk to text representation with column names and values
# #         chunk_text = "Data chunk with the following information:\n"
# #         for _, row in chunk_df.iterrows():
# #             row_text = " | ".join([f"{col}: {row[col]}" for col in df.columns])
# #             chunk_text += row_text + "\n"
        
# #         chunks.append(chunk_text)
    
# #     # Add schema information as a separate chunk
# #     schema_chunk = "Table schema information:\n"
# #     for col in df.columns:
# #         schema_chunk += f"Column: {col}, Type: {df[col].dtype}\n"
# #     chunks.append(schema_chunk)
    
# #     # Add statistical information as separate chunks
# #     for col in df.select_dtypes(include=[np.number]).columns:
# #         stats_chunk = f"Statistical information for column {col}:\n"
# #         stats_chunk += f"Min: {df[col].min()}, Max: {df[col].max()}, Mean: {df[col].mean()}, Median: {df[col].median()}\n"
# #         chunks.append(stats_chunk)
    
# #     return chunks

# # # Initialize vector store function

# # def initialize_vector_store(df):
# #     """Initialize the vector store with embeddings from the dataframe."""
# #     # Define the file path where embeddings will be stored
# #     embeddings_file = "embeddings_store.pkl"
    
# #     # Check if embeddings file already exists
# #     if os.path.exists(embeddings_file):
# #         progress_placeholder = st.empty()
# #         progress_placeholder.text("Loading existing embeddings...")
# #         try:
# #             # Load stored embeddings from file
# #             with open(embeddings_file, 'rb') as file:
# #                 vector_store = pickle.load(file)
# #                 progress_placeholder.text("Embeddings loaded successfully!")
# #                 return vector_store
# #         except Exception as e:
# #             progress_placeholder.error(f"Error loading embeddings: {str(e)}. Will create new embeddings.")
    
# #     # Create document chunks from dataframe
# #     chunks = create_document_chunks(df)
    
# #     # Create embeddings for each chunk using OpenAI
# #     embeddings = []
# #     # Using a plain progress message instead of nested status
# #     progress_placeholder = st.empty()
# #     for i, chunk in enumerate(chunks):
# #         progress_placeholder.text(f"Creating embeddings... ({i+1}/{len(chunks)})")
# #         embedding = get_openai_embedding(chunk)
# #         embeddings.append(embedding)
    
# #     # Create vector store
# #     vector_store = {
# #         "embeddings": embeddings,
# #         "chunks": chunks
# #     }
    
# #     # Save embeddings to file
# #     try:
# #         with open(embeddings_file, 'wb') as file:
# #             pickle.dump(vector_store, file)
# #         progress_placeholder.text("Embeddings created and saved successfully!")
# #     except Exception as e:
# #         progress_placeholder.warning(f"Could not save embeddings to file: {str(e)}")
# #         progress_placeholder.text("Embeddings created but not saved!")
    
# #     return vector_store

# # def retrieve_relevant_contexts(query, vector_store, top_k=3):
# #     """Retrieve the most relevant contexts for a query."""
# #     chunks = vector_store["chunks"]
# #     stored_embeddings = vector_store["embeddings"]
    
# #     # Get query embedding
# #     query_embedding = get_openai_embedding(query)
    
# #     if not query_embedding or not stored_embeddings:
# #         # Fallback if embeddings fail
# #         return chunks[:min(top_k, len(chunks))]
    
# #     # Calculate similarities
# #     similarities = []
# #     for emb in stored_embeddings:
# #         if len(emb) == 0:  # Skip empty embeddings
# #             similarities.append(0)
# #             continue
            
# #         # Convert to numpy arrays
# #         emb_array = np.array(emb).reshape(1, -1)
# #         query_array = np.array(query_embedding).reshape(1, -1)
        
# #         # Calculate cosine similarity
# #         similarity = cosine_similarity(emb_array, query_array)[0][0]
# #         similarities.append(similarity)
    
# #     # Get indices of top_k most similar chunks
# #     if not similarities:
# #         return chunks[:min(top_k, len(chunks))]
    
# #     indices = np.argsort(similarities)[-top_k:][::-1]
    
# #     # Get the relevant chunks
# #     relevant_chunks = [chunks[idx] for idx in indices]
    
# #     return relevant_chunks

# # def process_query_with_rag(user_query, vector_store, table_name, schema_name, database_name, column_info, conversation_history=None):
# #     """
# #     Process a user query using RAG to provide contextually enhanced answers.
    
# #     Args:
# #         user_query: The user's question
# #         vector_store: The vector store containing embeddings and chunks
# #         table_name: The name of the table in Snowflake
# #         schema_name: The name of the schema in Snowflake
# #         database_name: The name of the database in Snowflake
# #         column_info: Information about the columns in the table
# #         conversation_history: List of previous messages in the format [{"role": "user|assistant", "content": "message"}]
# #     """
    
# #     # Retrieve relevant contexts
# #     relevant_contexts = retrieve_relevant_contexts(user_query, vector_store)
    
# #     # Combine contexts into a single string
# #     combined_context = "\n".join(relevant_contexts)
    
# #     # Create a prompt with the retrieved context
# #     prompt = f"""
# #     You are an assistant that helps users query and analyze OEE (Overall Equipment Effectiveness) data.
    
# #     The user is querying a Snowflake database with the following details:
# #     - Database: {database_name}
# #     - Schema: {schema_name}
# #     - Table: {table_name}
    
# #     The table has these columns with their data types:
# #     {column_info}
    
# #     I've retrieved some relevant context from the database to help you answer:
    
# #     {combined_context}
    
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









# # import pandas as pd
# # import numpy as np
# # import os
# # import json
# # import pickle
# # import streamlit as st
# # from sklearn.metrics.pairwise import cosine_similarity
# # from openai import OpenAI

# # # Initialize OpenAI client
# # client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
# # # client = OpenAI(api_key='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')

# # GPT_MODEL = "gpt-4o"  # the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
# # EMBEDDING_MODEL = "text-embedding-3-small"

# # def get_openai_embedding(text):
# #     """Get embeddings from the OpenAI API."""
# #     try:
# #         response = client.embeddings.create(
# #             model=EMBEDDING_MODEL,
# #             input=text
# #         )
# #         return response.data[0].embedding
# #     except Exception as e:
# #         st.error(f"Error getting embeddings from OpenAI: {str(e)}")
# #         return []

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

# # def create_document_chunks(df, chunk_size=5):
# #     """Create chunks of data from the dataframe for better context."""
# #     chunks = []
# #     total_rows = len(df)
    
# #     # Create chunks by combining rows into meaningful textual descriptions
# #     for i in range(0, total_rows, chunk_size):
# #         end_idx = min(i + chunk_size, total_rows)
# #         chunk_df = df.iloc[i:end_idx]
        
# #         # Convert chunk to text representation with column names and values
# #         chunk_text = "Data chunk with the following information:\n"
# #         for _, row in chunk_df.iterrows():
# #             row_text = " | ".join([f"{col}: {row[col]}" for col in df.columns])
# #             chunk_text += row_text + "\n"
        
# #         chunks.append(chunk_text)
    
# #     # Add schema information as a separate chunk
# #     schema_chunk = "Table schema information:\n"
# #     for col in df.columns:
# #         schema_chunk += f"Column: {col}, Type: {df[col].dtype}\n"
# #     chunks.append(schema_chunk)
    
# #     # Add statistical information as separate chunks
# #     for col in df.select_dtypes(include=[np.number]).columns:
# #         stats_chunk = f"Statistical information for column {col}:\n"
# #         stats_chunk += f"Min: {df[col].min()}, Max: {df[col].max()}, Mean: {df[col].mean()}, Median: {df[col].median()}\n"
# #         chunks.append(stats_chunk)
    
# #     return chunks

# # # Initialize vector store function

# # def initialize_vector_store(df, table_name=None):
# #     """Initialize the vector store with embeddings from the dataframe."""
# #     # Define the file path where embeddings will be stored
# #     embeddings_file = "embeddings_store.pkl"
# #     metadata_file = "embeddings_metadata.json"
    
# #     # Flag to track if we need to regenerate embeddings
# #     regenerate_embeddings = False
    
# #     # Current data information for checking changes
# #     current_columns = list(df.columns)
# #     current_table = table_name or "OEESHIFTWISE"  # Default if not provided
    
# #     # Check if metadata exists to detect schema changes
# #     if os.path.exists(metadata_file):
# #         try:
# #             with open(metadata_file, 'r') as file:
# #                 metadata = json.load(file)
                
# #             # Check if schema or table has changed
# #             if metadata.get("table_name") != current_table:
# #                 st.warning(f"Table name has changed from {metadata.get('table_name')} to {current_table}. Will regenerate embeddings.")
# #                 regenerate_embeddings = True
# #             elif set(metadata.get("columns", [])) != set(current_columns):
# #                 st.warning("Table schema has changed. Will regenerate embeddings.")
# #                 regenerate_embeddings = True
# #         except Exception as e:
# #             st.warning(f"Could not read metadata file: {str(e)}. Will check embeddings file.")
    
# #     # Check if embeddings file already exists
# #     if os.path.exists(embeddings_file) and not regenerate_embeddings:
# #         progress_placeholder = st.empty()
# #         progress_placeholder.text("Loading existing embeddings...")
# #         try:
# #             # Load stored embeddings from file
# #             with open(embeddings_file, 'rb') as file:
# #                 vector_store = pickle.load(file)
# #                 progress_placeholder.text("Embeddings loaded successfully!")
# #                 return vector_store
# #         except Exception as e:
# #             progress_placeholder.error(f"Error loading embeddings: {str(e)}. Will create new embeddings.")
    
# #     # Create document chunks from dataframe
# #     chunks = create_document_chunks(df)
    
# #     # Create embeddings for each chunk using OpenAI
# #     embeddings = []
# #     # Using a plain progress message instead of nested status
# #     progress_placeholder = st.empty()
# #     for i, chunk in enumerate(chunks):
# #         progress_placeholder.text(f"Creating embeddings... ({i+1}/{len(chunks)})")
# #         embedding = get_openai_embedding(chunk)
# #         embeddings.append(embedding)
    
# #     # Create vector store
# #     vector_store = {
# #         "embeddings": embeddings,
# #         "chunks": chunks
# #     }
    
# #     # Save embeddings to file
# #     try:
# #         with open(embeddings_file, 'wb') as file:
# #             pickle.dump(vector_store, file)
        
# #         # Save metadata to track schema changes
# #         metadata = {
# #             "table_name": current_table,
# #             "columns": current_columns,
# #             "created_at": str(pd.Timestamp.now()),
# #             "num_chunks": len(chunks),
# #             "num_embeddings": len(embeddings)
# #         }
# #         with open(metadata_file, 'w') as file:
# #             json.dump(metadata, file, indent=2)
            
# #         progress_placeholder.text("Embeddings created and saved successfully!")
# #     except Exception as e:
# #         progress_placeholder.warning(f"Could not save embeddings to file: {str(e)}")
# #         progress_placeholder.text("Embeddings created but not saved!")
    
# #     return vector_store

# # def retrieve_relevant_contexts(query, vector_store, top_k=3):
# #     """Retrieve the most relevant contexts for a query."""
# #     chunks = vector_store["chunks"]
# #     stored_embeddings = vector_store["embeddings"]
    
# #     # Get query embedding
# #     query_embedding = get_openai_embedding(query)
    
# #     if not query_embedding or not stored_embeddings:
# #         # Fallback if embeddings fail
# #         return chunks[:min(top_k, len(chunks))]
    
# #     # Calculate similarities
# #     similarities = []
# #     for emb in stored_embeddings:
# #         if len(emb) == 0:  # Skip empty embeddings
# #             similarities.append(0)
# #             continue
            
# #         # Convert to numpy arrays
# #         emb_array = np.array(emb).reshape(1, -1)
# #         query_array = np.array(query_embedding).reshape(1, -1)
        
# #         # Calculate cosine similarity
# #         similarity = cosine_similarity(emb_array, query_array)[0][0]
# #         similarities.append(similarity)
    
# #     # Get indices of top_k most similar chunks
# #     if not similarities:
# #         return chunks[:min(top_k, len(chunks))]
    
# #     indices = np.argsort(similarities)[-top_k:][::-1]
    
# #     # Get the relevant chunks
# #     relevant_chunks = [chunks[idx] for idx in indices]
    
# #     return relevant_chunks

# # def process_query_with_rag(user_query, vector_store, table_name, schema_name, database_name, column_info, conversation_history=None):
# #     """
# #     Process a user query using RAG to provide contextually enhanced answers.
    
# #     Args:
# #         user_query: The user's question
# #         vector_store: The vector store containing embeddings and chunks
# #         table_name: The name of the table in Snowflake
# #         schema_name: The name of the schema in Snowflake
# #         database_name: The name of the database in Snowflake
# #         column_info: Information about the columns in the table
# #         conversation_history: List of previous messages in the format [{"role": "user|assistant", "content": "message"}]
# #     """
    
# #     # Retrieve relevant contexts
# #     relevant_contexts = retrieve_relevant_contexts(user_query, vector_store)
    
# #     # Combine contexts into a single string
# #     combined_context = "\n".join(relevant_contexts)
    
# #     # Create a prompt with the retrieved context
# #     prompt = f"""
# #     You are an assistant that helps users query and analyze OEE (Overall Equipment Effectiveness) data.
    
# #     The user is querying a Snowflake database with the following details:
# #     - Database: {database_name}
# #     - Schema: {schema_name}
# #     - Table: {table_name}
    
# #     The table has these columns with their data types:
# #     {column_info}
    
# #     I've retrieved some relevant context from the database to help you answer:
    
# #     {combined_context}
    
# #     The user's current query is: "{user_query}"
    
# #     When responding, consider the conversation context and previous questions if they are provided.
    
# #     If the query is asking for data that can be retrieved with SQL, please:
# # If the query is about data stored in the database, use the provided context to answer in **plain language**.  
# # Explain the insights from the retrieved information concisely, without explicitly showing SQL queries.  
    
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







































































# import pandas as pd
# import numpy as np
# import os
# import json
# import pickle
# import streamlit as st
# from sklearn.metrics.pairwise import cosine_similarity
# from openai import OpenAI

# # Initialize OpenAI client
# # client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
# client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
# GPT_MODEL = "gpt-4o"  # the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
# EMBEDDING_MODEL = "text-embedding-3-small"

# def get_openai_embedding(text):
#     """Get embeddings from the OpenAI API."""
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

# def create_document_chunks(df, chunk_size=5):
#     """Create chunks of data from the dataframe for better context."""
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
    
#     # Add statistical information as separate chunks
#     for col in df.select_dtypes(include=[np.number]).columns:
#         stats_chunk = f"Statistical information for column {col}:\n"
#         stats_chunk += f"Min: {df[col].min()}, Max: {df[col].max()}, Mean: {df[col].mean()}, Median: {df[col].median()}\n"
#         chunks.append(stats_chunk)
    
#     return chunks

# # Initialize vector store function

# def initialize_vector_store(df, table_name=None):
#     """Initialize the vector store with embeddings from the dataframe."""
#     # Define the file path where embeddings will be stored
#     embeddings_file = "embeddings_store.pkl"
#     metadata_file = "embeddings_metadata.json"
    
#     # Flag to track if we need to regenerate embeddings
#     regenerate_embeddings = False
    
#     # Current data information for checking changes
#     current_columns = list(df.columns)
#     current_table = table_name or "OEESHIFTWISE"  # Default if not provided
    
#     # Check if metadata exists to detect schema changes
#     if os.path.exists(metadata_file):
#         try:
#             with open(metadata_file, 'r') as file:
#                 metadata = json.load(file)
                
#             # Check if schema or table has changed
#             if metadata.get("table_name") != current_table:
#                 st.warning(f"Table name has changed from {metadata.get('table_name')} to {current_table}. Will regenerate embeddings.")
#                 regenerate_embeddings = True
#             elif set(metadata.get("columns", [])) != set(current_columns):
#                 st.warning("Table schema has changed. Will regenerate embeddings.")
#                 regenerate_embeddings = True
#         except Exception as e:
#             st.warning(f"Could not read metadata file: {str(e)}. Will check embeddings file.")
    
#     # Check if embeddings file already exists
#     if os.path.exists(embeddings_file) and not regenerate_embeddings:
#         progress_placeholder = st.empty()
#         progress_placeholder.text("Loading existing embeddings...")
#         try:
#             # Load stored embeddings from file
#             with open(embeddings_file, 'rb') as file:
#                 vector_store = pickle.load(file)
#                 progress_placeholder.text("Embeddings loaded successfully!")
#                 return vector_store
#         except Exception as e:
#             progress_placeholder.error(f"Error loading embeddings: {str(e)}. Will create new embeddings.")
    
#     # Create document chunks from dataframe
#     chunks = create_document_chunks(df)
    
#     # Create embeddings for each chunk using OpenAI
#     embeddings = []
#     # Using a plain progress message instead of nested status
#     progress_placeholder = st.empty()
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
#     try:
#         with open(embeddings_file, 'wb') as file:
#             pickle.dump(vector_store, file)
        
#         # Save metadata to track schema changes
#         metadata = {
#             "table_name": current_table,
#             "columns": current_columns,
#             "created_at": str(pd.Timestamp.now()),
#             "num_chunks": len(chunks),
#             "num_embeddings": len(embeddings)
#         }
#         with open(metadata_file, 'w') as file:
#             json.dump(metadata, file, indent=2)
            
#         progress_placeholder.text("Embeddings created and saved successfully!")
#     except Exception as e:
#         progress_placeholder.warning(f"Could not save embeddings to file: {str(e)}")
#         progress_placeholder.text("Embeddings created but not saved!")
    
#     return vector_store

# def retrieve_relevant_contexts(query, vector_store, top_k=3):
#     """Retrieve the most relevant contexts for a query."""
#     chunks = vector_store["chunks"]
#     stored_embeddings = vector_store["embeddings"]
    
#     # Get query embedding
#     query_embedding = get_openai_embedding(query)
    
#     if not query_embedding or not stored_embeddings:
#         # Fallback if embeddings fail
#         return chunks[:min(top_k, len(chunks))]
    
#     # Calculate similarities
#     similarities = []
#     for emb in stored_embeddings:
#         if len(emb) == 0:  # Skip empty embeddings
#             similarities.append(0)
#             continue
            
#         # Convert to numpy arrays
#         emb_array = np.array(emb).reshape(1, -1)
#         query_array = np.array(query_embedding).reshape(1, -1)
        
#         # Calculate cosine similarity
#         similarity = cosine_similarity(emb_array, query_array)[0][0]
#         similarities.append(similarity)
    
#     # Get indices of top_k most similar chunks
#     if not similarities:
#         return chunks[:min(top_k, len(chunks))]
    
#     indices = np.argsort(similarities)[-top_k:][::-1]
    
#     # Get the relevant chunks
#     relevant_chunks = [chunks[idx] for idx in indices]
    
#     return relevant_chunks

# def process_query_with_rag(user_query, vector_store, table_name, schema_name, database_name, column_info, conversation_history=None):
#     """
#     Process a user query using RAG to provide contextually enhanced answers.
    
#     Args:
#         user_query: The user's question
#         vector_store: The vector store containing embeddings and chunks
#         table_name: The name of the table in Snowflake
#         schema_name: The name of the schema in Snowflake
#         database_name: The name of the database in Snowflake
#         column_info: Information about the columns in the table
#         conversation_history: List of previous messages in the format [{"role": "user|assistant", "content": "message"}]
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



























































import pandas as pd
import numpy as np
import os
import json
import pickle
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI

# Initialize OpenAI client
# client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
GPT_MODEL = "gpt-4o"  # the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
EMBEDDING_MODEL = "text-embedding-3-small"

def get_openai_embedding(text):
    """Get embeddings from the OpenAI API."""
    try:
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        st.error(f"Error getting embeddings from OpenAI: {str(e)}")
        return []

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

def create_document_chunks(df, chunk_size=5):
    """Create chunks of data from the dataframe for better context."""
    chunks = []
    total_rows = len(df)

    # Create chunks by combining rows into meaningful textual descriptions
    for i in range(0, total_rows, chunk_size):
        end_idx = min(i + chunk_size, total_rows)
        chunk_df = df.iloc[i:end_idx]

        # Convert chunk to text representation with column names and values
        chunk_text = "Data chunk with the following information:\n"
        for _, row in chunk_df.iterrows():
            row_text = " | ".join([f"{col}: {row[col]}" for col in df.columns])
            chunk_text += row_text + "\n"

        chunks.append(chunk_text)

    # Add schema information as a separate chunk
    schema_chunk = "Table schema information:\n"
    for col in df.columns:
        schema_chunk += f"Column: {col}, Type: {df[col].dtype}\n"
    chunks.append(schema_chunk)

    # Add statistical information as separate chunks
    for col in df.select_dtypes(include=[np.number]).columns:
        stats_chunk = f"Statistical information for column {col}:\n"
        stats_chunk += f"Min: {df[col].min()}, Max: {df[col].max()}, Mean: {df[col].mean()}, Median: {df[col].median()}\n"
        chunks.append(stats_chunk)

    return chunks

# Initialize vector store function

def initialize_vector_store(df, table_name=None):
    """Initialize the vector store with embeddings from the dataframe."""
    # Define the file path where embeddings will be stored
    embeddings_file = "embeddings_store1.pkl"
    metadata_file = "embeddings_metadata.json"

    # Flag to track if we need to regenerate embeddings
    regenerate_embeddings = False

    # Current data information for checking changes
    current_columns = list(df.columns)
    current_table = table_name or "OEESHIFTWISE"  # Default if not provided

    # Check if metadata exists to detect schema changes
    if os.path.exists(metadata_file):
        try:
            with open(metadata_file, 'r') as file:
                metadata = json.load(file)

            # Check if schema or table has changed
            if metadata.get("table_name") != current_table:
                st.warning(f"Table name has changed from {metadata.get('table_name')} to {current_table}. Will regenerate embeddings.")
                regenerate_embeddings = True
            elif set(metadata.get("columns", [])) != set(current_columns):
                st.warning("Table schema has changed. Will regenerate embeddings.")
                regenerate_embeddings = True
        except Exception as e:
            st.warning(f"Could not read metadata file: {str(e)}. Will check embeddings file.")

    # Check if embeddings file already exists
    if os.path.exists(embeddings_file) and not regenerate_embeddings:
        progress_placeholder = st.empty()
        progress_placeholder.text("Loading existing embeddings...")
        try:
            # Load stored embeddings from file
            with open(embeddings_file, 'rb') as file:
                vector_store = pickle.load(file)
                progress_placeholder.text("Embeddings loaded successfully!")
                return vector_store
        except Exception as e:
            progress_placeholder.error(f"Error loading embeddings: {str(e)}. Will create new embeddings.")

    # Create document chunks from dataframe
    chunks = create_document_chunks(df)

    # Create embeddings for each chunk using OpenAI
    embeddings = []
    # Using a plain progress message instead of nested status
    progress_placeholder = st.empty()
    for i, chunk in enumerate(chunks):
        progress_placeholder.text(f"Creating embeddings... ({i+1}/{len(chunks)})")
        embedding = get_openai_embedding(chunk)
        embeddings.append(embedding)

    # Create vector store
    vector_store = {
        "embeddings": embeddings,
        "chunks": chunks
    }

    # Save embeddings to file
    try:
        with open(embeddings_file, 'wb') as file:
            pickle.dump(vector_store, file)

        # Save metadata to track schema changes
        metadata = {
            "table_name": current_table,
            "columns": current_columns,
            "created_at": str(pd.Timestamp.now()),
            "num_chunks": len(chunks),
            "num_embeddings": len(embeddings)
        }
        with open(metadata_file, 'w') as file:
            json.dump(metadata, file, indent=2)

        progress_placeholder.text("Embeddings created and saved successfully!")
    except Exception as e:
        progress_placeholder.warning(f"Could not save embeddings to file: {str(e)}")
        progress_placeholder.text("Embeddings created but not saved!")

    return vector_store

def retrieve_relevant_contexts(query, vector_store, top_k=3):
    """Retrieve the most relevant contexts for a query."""
    chunks = vector_store["chunks"]
    stored_embeddings = vector_store["embeddings"]

    # Get query embedding
    query_embedding = get_openai_embedding(query)

    if not query_embedding or not stored_embeddings:
        # Fallback if embeddings fail
        return chunks[:min(top_k, len(chunks))]

    # Calculate similarities
    similarities = []
    for emb in stored_embeddings:
        if len(emb) == 0:  # Skip empty embeddings
            similarities.append(0)
            continue

        # Convert to numpy arrays
        emb_array = np.array(emb).reshape(1, -1)
        query_array = np.array(query_embedding).reshape(1, -1)

        # Calculate cosine similarity
        similarity = cosine_similarity(emb_array, query_array)[0][0]
        similarities.append(similarity)

    # Get indices of top_k most similar chunks
    if not similarities:
        return chunks[:min(top_k, len(chunks))]

    indices = np.argsort(similarities)[-top_k:][::-1]

    # Get the relevant chunks
    relevant_chunks = [chunks[idx] for idx in indices]

    return relevant_chunks

def process_query_with_rag(user_query, vector_store, table_name, schema_name, database_name, column_info, conversation_history=None):
    """
    Process a user query using RAG to provide contextually enhanced answers.

    Args:
        user_query: The user's question
        vector_store: The vector store containing embeddings and chunks
        table_name: The name of the table in Snowflake
        schema_name: The name of the schema in Snowflake
        database_name: The name of the database in Snowflake
        column_info: Information about the columns in the table
        conversation_history: List of previous messages in the format [{"role": "user|assistant", "content": "message"}]
    """

    # Retrieve relevant contexts
    relevant_contexts = retrieve_relevant_contexts(user_query, vector_store)

    # Combine contexts into a single string
    combined_context = "\n".join(relevant_contexts)

    # Create a prompt with the retrieved context
    prompt = f"""
    You are an assistant that helps users query and analyze OEE (Overall Equipment Effectiveness) data.

    The user is querying a Snowflake database with the following details:
    - Database: {database_name}
    - Schema: {schema_name}
    - Table: {table_name}

    The table has these columns with their data types:
    {column_info}

    I've retrieved some relevant context from the database to help you answer:

    {combined_context}

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