import pandas as pd
import numpy as np
import json
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
import snowflake.connector

# Initialize OpenAI client
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
GPT_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"
BATCH_SIZE = 1000  # Number of records to insert in each batch

# Get OpenAI embedding
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

# Create a function to call OpenAI API with the prompt and conversation history
def call_openai(prompt, system_prompt="You are a helpful assistant.", conversation_history=None):
    """
    Call the OpenAI API with the specified prompt and optional conversation history.

    Args:
        prompt: The current user query
        system_prompt: The system prompt to set the assistant's behavior
        conversation_history: A list of previous messages in the format [{"role": "user|assistant", "content": "message"}]
    """
    messages = [{"role": "system", "content": system_prompt}]
    if conversation_history and len(conversation_history) > 0:
        messages.extend(conversation_history)
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

    for i in range(0, total_rows, chunk_size):
        end_idx = min(i + chunk_size, total_rows)
        chunk_df = df.iloc[i:end_idx]
        chunk_text = "Data chunk with the following information:\n"
        for _, row in chunk_df.iterrows():
            row_text = " | ".join([f"{col}: {row[col]}" for col in df.columns])
            chunk_text += row_text + "\n"
        chunks.append(chunk_text)

    schema_chunk = "Table schema information:\n"
    for col in df.columns:
        schema_chunk += f"Column: {col}, Type: {df[col].dtype}\n"
    chunks.append(schema_chunk)

    for col in df.select_dtypes(include=[np.number]).columns:
        stats_chunk = f"Statistical information for column {col}:\n"
        stats_chunk += f"Min: {df[col].min()}, Max: {df[col].max()}, Mean: {df[col].mean()}, Median: {df[col].median()}\n"
        chunks.append(stats_chunk)

    return chunks

def init_snowflake_connection():
    """Initialize Snowflake connection."""
    try:
        conn = snowflake.connector.connect(
            user=st.secrets["snowflake"]["user"],
            password=st.secrets["snowflake"]["password"],
            account=st.secrets["snowflake"]["account"],
            warehouse=st.secrets["snowflake"]["warehouse"],
            database="O3_AI_DB",
            schema="O3_AI_DB_SCHEMA"
        )
        return conn
    except Exception as e:
        st.error(f"Error connecting to Snowflake: {e}")
        return None

def initialize_vector_store(df, table_name=None):
    """Initialize the vector store with embeddings from Snowflake using batch processing."""
    table_name = table_name or "OEESHIFTWISE_AI"
    conn = init_snowflake_connection()
    if not conn:
        st.error("Failed to connect to Snowflake. Cannot initialize vector store.")
        return None

    cursor = conn.cursor()
    try:
        # Check if embeddings exist for the table
        cursor.execute(f"""
            SELECT COUNT(*) 
            FROM O3_AI_EMBEDDINGS 
            WHERE TABLE_NAME = %s
        """, (table_name,))
        count = cursor.fetchone()[0]

        if count > 0:
            # Load existing embeddings from Snowflake
            cursor.execute(f"""
                SELECT RECORD_ID, CHUNK_TEXT, EMBEDDING 
                FROM O3_AI_EMBEDDINGS 
                WHERE TABLE_NAME = %s
            """, (table_name,))
            results = cursor.fetchall()
            chunks = []
            embeddings = []
            for row in results:
                chunks.append(json.loads(row[1]) if row[1] else "")
                embeddings.append(json.loads(row[2]) if row[2] else [])
            st.success("Loaded embeddings from Snowflake successfully!")
            return {"embeddings": embeddings, "chunks": chunks}

        # Generate new embeddings using batch processing for OpenAI API
        chunks = create_document_chunks(df)
        embeddings = []
        progress_placeholder = st.empty()
        
        # Prepare batch request for OpenAI embeddings
        progress_placeholder.text("Creating embeddings in batch...")
        try:
            response = client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=chunks
            )
            batch_embeddings = [item.embedding for item in response.data]
            
            # Prepare data for batch insertion into Snowflake
            batch_data = []
            for i, (chunk, embedding) in enumerate(zip(chunks, batch_embeddings)):
                chunk_json = json.dumps(chunk)
                embedding_json = json.dumps(embedding)
                batch_data.append((f"{table_name}_{i}", table_name, chunk_json, embedding_json))
                embeddings.append(embedding)

            # Insert embeddings into Snowflake in batches
            for i in range(0, len(batch_data), BATCH_SIZE):
                batch_chunk = batch_data[i:i + BATCH_SIZE]
                cursor.executemany("""
                    INSERT INTO O3_AI_EMBEDDINGS (RECORD_ID, TABLE_NAME, CHUNK_TEXT, EMBEDDING)
                    VALUES (%s, %s, %s, %s)
                """, batch_chunk)
                progress_placeholder.text(f"Inserted {min(i + BATCH_SIZE, len(batch_data))} of {len(batch_data)} embeddings into Snowflake...")

            conn.commit()
            st.success("Embeddings created and stored in Snowflake successfully!")
            return {"embeddings": embeddings, "chunks": chunks}
        
        except Exception as e:
            st.warning(f"Failed to generate batch embeddings: {str(e)}. Falling back to default behavior.")
            return None

    except Exception as e:
        st.error(f"Error initializing vector store with Snowflake: {str(e)}")
        return None
    finally:
        cursor.close()
        conn.close()

def retrieve_relevant_contexts(query, vector_store, top_k=3):
    """Retrieve the most relevant contexts for a query."""
    chunks = vector_store["chunks"]
    stored_embeddings = vector_store["embeddings"]
    query_embedding = get_openai_embedding(query)

    if not query_embedding or not stored_embeddings:
        return chunks[:min(top_k, len(chunks))]

    similarities = []
    for emb in stored_embeddings:
        if len(emb) == 0:
            similarities.append(0)
            continue
        emb_array = np.array(emb).reshape(1, -1)
        query_array = np.array(query_embedding).reshape(1, -1)
        similarity = cosine_similarity(emb_array, query_array)[0][0]
        similarities.append(similarity)

    if not similarities:
        return chunks[:min(top_k, len(chunks))]

    indices = np.argsort(similarities)[-top_k:][::-1]
    relevant_chunks = [chunks[idx] for idx in indices]
    return relevant_chunks

def process_query_with_rag(user_query, vector_store, table_name, schema_name, database_name, column_info, conversation_history=None):
    """
    Process a user query using RAG to provide contextually enhanced answers.
    """
    relevant_contexts = retrieve_relevant_contexts(user_query, vector_store)
    combined_context = "\n".join(relevant_contexts)

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
    6. In case of sql query did not generate it don't explicitly show error message, instead show user friendly response for example: "My bad, please rephrase your question or try again later."

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

