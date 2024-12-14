from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
import google.generativeai as genai
from model.connectDB import connectDB, getDBSchema
import os
import numpy as np

# Load SentenceTransformer model
model = SentenceTransformer('bert-base-nli-mean-tokens')

#Compute embeddings for all column values in a table.
def compute_embeddings(table):
    column_value_embeddings = {}
    for column_name in table.columns:
        column_values = table[column_name].astype(str).tolist()
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

 # gemini api_key= AIzaSyBbloMkukzcbAr7NnYzOiKuIE8jPrzB9BQ 
genai.configure(api_key=os.getenv("GOOGLE_GENERATIVE_AI_KEY"))
gemini_model = genai.GenerativeModel('gemini-1.5-pro-latest')


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
    
    response = gemini_model.generate_text(prompt, max_output_tokens=300)
    return response.result 

def process_query_with_insights(query, table):
    column_value_embeddings = compute_embeddings(table)
    knn_indices = knn_indexing(column_value_embeddings)
    relations = find_similar_values(query, knn_indices)
    insights = generate_insights_with_llm(query, relations)
    return relations, insights


def main():
    db_config = {
        "host": "aws-0-ap-southeast-1.pooler.supabase.com",
        "port": "6543",
        "dbname": "postgres",
        "user": "postgres.egkmupciopviyycuuuym",
        "password": "O2fETdNwU9r4nmzy"
    }
        
    connection = connectDB("postgress", db_config)
    query = "Find stock-profiles having MRF as one of their stocks and provide their names"
    relations, insights = process_query_with_insights(query, connection)
    print(relations)
    print(insights)


if __name__ == "__main__":
    main()
    