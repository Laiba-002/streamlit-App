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
    
    When presenting results (especially tables), always:
    ✅ Replace technical column names with clean, human-readable labels (e.g., replace 'MAX_PERFORMANCE_RATE' with 'Max Performance Rate')
    ✅ Use proper casing and spacing to make results understandable to non-technical users
    ✅ Maintain clear formatting for tables, but ensure column headers are friendly
    
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
