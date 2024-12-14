# **QueryMatch: Smart Query Processing and Insight Generation**

**QueryMatch** is a robust and intelligent system designed to analyze user queries, retrieve relevant information from databases, and generate comprehensive insights. It integrates advanced database connectivity, state-of-the-art language models, and clustering techniques to deliver accurate and meaningful responses.

By leveraging **BERT embeddings** and **K-Nearest Neighbors (KNN) indexing**, QueryMatch ensures a seamless and insightful experience for users querying their data.

---

## **Features**

- **Universal Database Connectivity**: Effortlessly connect to databases, including PostgreSQL, MongoDB, and other SQL-compliant databases.
- **Automated Schema Exploration**: Extract tables, columns, and database structures programmatically for easy navigation and processing.
- **Intelligent Query Matching**: Utilize **BERT embeddings** for finding semantically relevant matches in database entries.
- **Efficient Clustering**: Group similar data points using KNN and threshold-based clustering for improved search accuracy.
- **Insightful Responses**: Generate human-readable, technically accurate insights with Google’s **Gemini AI model**.

---

## **How It Works**

QueryMatch operates in two stages, combining database management with AI-powered query processing:

### **1. Database Management (`connectDB.py`)**
This script handles the connection to various databases and retrieves schema details.

#### Key Functions:
1. **`connectDB(db_type, db_config)`**:  
   Establishes a connection to the specified database type (PostgreSQL, MongoDB, or SQLAlchemy-supported databases).  

2. **`getAllTables(conn, schema='public')`**:  
   Retrieves all table names within a specified schema.

3. **`getColumns(conn, table_name)`**:  
   Fetches column names for a specific table.

4. **`getDBSchema(conn, schema='public')`**:  
   Constructs a comprehensive schema map of all tables and their columns.

#### Example Workflow:
```python
db_config = {
    "host": "your-database-host",
    "port": 5432,
    "dbname": "your-database-name",
    "user": "your-username",
    "password": "your-password"
}
connection = connectDB("postgress", db_config)
schema = getDBSchema(connection)
print("Database Schema:", schema)
