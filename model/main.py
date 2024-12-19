from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
import openai
import os
import numpy as np
import pandas as pd
import psycopg2 # type: ignore
import pymongo # type: ignore
from sqlalchemy import create_engine



def connectDB(db_type, db_config):
    try:
        if db_type== "postgress":
            connection=psycopg2.connect(
                host=db_config.get("host"),
                port=db_config.get("port", 5432),
                database=db_config.get("dbname"),
                user=db_config.get("user"),
                password=db_config.get("password")
            )
            return connection

        elif db_type == "mongodb":
            client = pymongo.MongoClient(db_config.get("uri"))
            return client[db_config.get("dbname")]
            print("Connected the MongoDB")

        elif db_type == "generic":
            engine = create_engine(db_config.get("connection_string"))
            connection = engine.connect()
            return connection
            print("Connected using SQLAlchemy")

    except Exception as e :
        print(f"error connecting to the DB: {e}")       

# pass in the connection to the DBdef getAllTables(conn, schema='public'):
def getAllTables(conn, schema='public'):
    try:
        with conn.cursor() as cursor:  # Fixed cursor usage
            query = """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = %s;
            """
            cursor.execute(query, (schema,))
            tables = cursor.fetchall()
            return [table[0] for table in tables]
    except Exception as e:
        print(f"Error getting all tables: {e}")
        return None  # Return None on failure

def getColumns(conn, table_name):
    try:
        with conn.cursor() as cursor:  
            query = """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = %s;
            """
            cursor.execute(query, (table_name,))
            columns = cursor.fetchall()
            return [column[0] for column in columns] 
    except Exception as e:
        print(f"Error getting columns for table {table_name}: {e}")
        return None  

def getDBSchema(conn, schema='public'):
    try:
        tables = getAllTables(conn, schema)
        if tables is None:
            return None  # Return None if no tables found

        db_schema = {}
        for table in tables:
            columns = getColumns(conn, table)
            if columns is None:
                print(f"Skipping table {table} due to an error.")
                continue
            db_schema[table] = columns

        return db_schema
    except Exception as e:
        print(f"Error in fetching tables/columns: {e}")
        return None 


# Load SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

#Compute embeddings for all column values in a table.
def compute_embeddings(conn,table):
    column_value_embeddings = {}
    schema = getDBSchema(conn, "public")
    table_data = pd.read_sql(f"SELECT * FROM {table}", conn)
    if schema and table in schema:
        columns = schema[table]
    for column_name in columns:
        column_values = table_data[column_name].astype(str).tolist()
        embeddings = model.encode(column_values)
        column_value_embeddings[column_name] = {
            "values": column_values,
            "embeddings": embeddings
        }
    return column_value_embeddings

#Index column values using KNN
def knn_indexing(column_value_embeddings):
    knn_indices = {}
    for column_name, data in column_value_embeddings.items():
        knn_model = NearestNeighbors(n_neighbors=5, metric='cosine')
        knn_model.fit(data["embeddings"])
        knn_indices[column_name] = {
            "model": knn_model,
            "values": data["values"],
            "embeddings": data["embeddings"]
        }
    return knn_indices

def find_similar_values(query, knn_indices):
    query_embedding = model.encode([query])
    relations = {}
    for column_name, knn_data in knn_indices.items():
        distances, indices = knn_data["model"].kneighbors(query_embedding)

        similar_values = [
            (knn_data["values"][idx], 1 - distances[0][i])  
            for i, idx in enumerate(indices[0])
        ]
        relations[column_name] = similar_values
    
    return relations


def generate_insights_with_llm(query, relations):
    relation_summary = ""
    for column_name, values in relations.items():
        value_scores = ", ".join([f"'{val[0]}' (similarity: {val[1]:.2f})" for val in values])
        relation_summary += f"Column: {column_name}, Matches: {value_scores}\n"

    prompt = f"""
    User Query: "{query}"
    Relevant matches based on the KNN analysis provided from the embeddings:
    {relation_summary}
    Generate a human-readable explanation of how the query relates to these matches and why these columns are relevant. 
    Give a proper, reliable and technically correct response from the retrieved information.
    """

    openai.api_key = "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"

    response = openai.ChatCompletion.create(
        model="gpt-4", 
        messages=[
            {"role": "system", "content": "You are a genius that can discover relations between vectors and their meanings. Give me what relations the embeddings have with the query. Dont just read from the table"},
            {"role": "user", "content": prompt}
        ],
        max_tokens=500,
        temperature=0.8
    )
    
    return response['choices'][0]['message']['content'].strip() 


def process_query_with_insights(query, table, conn):
    column_value_embeddings = compute_embeddings(conn,table)
    knn_indices = knn_indexing(column_value_embeddings)
    relations = find_similar_values(query, knn_indices)
    insights = generate_insights_with_llm(query, relations)
    return insights

def main():
    db_config = {
        "host": "aws-0-ap-southeast-1.pooler.supabase.com",
        "port": "6543",
        "dbname": "postgres",
        "user": "postgres.egkmupciopviyycuuuym",
        "password": "O2fETdNwU9r4nmzy"
    }

#  Test event for the pipeline    
    connection = connectDB("postgress", db_config)
    query = "Find stock-profiles having MRF as one of their stocks and provide their names"
    table= "stakeholders"
    relations, insights = process_query_with_insights(query,table, connection)
    print(relations)


if __name__ == "__main__":
    main()