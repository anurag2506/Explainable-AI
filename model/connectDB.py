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
            print("Connected the Postgress")

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

# pass in the connection to the DB
def getAllTables(conn, schema='public'):
    try:
        with conn.cursor as cursor:
            query = """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = %s;
            """
            cursor.execute(query, (schema,))
            tables=cursor.fetchall()
            return [table[0] for table in tables]
    except Exception as e:
        print(f"error getting all tables: {e}")

def getColumns(conn, table_name):
    try:
        with conn.cursor() as cursor:
            query = f"""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = %s;
            """
            cursor.execute(query, (table_name,))
            columns = cursor.fetchall()
            return [column[0] for column in columns]
    except Exception as e:
        print(f"Error getting columns for table {table_name}: {e}")

def getDBSchema(conn, schema='public'):
    try:
        tables=getAllTables(conn, schema)
        db_schema= {}

        for table in tables:
            columns=getColumns(conn,table)
            db_schema[table]=columns

        return db_schema
    
    except Exception as e:
        print(f"Error in fetching tables/columns :{e}")

try:
    db_config = {
            "host": "aws-0-ap-southeast-1.pooler.supabase.com",
            "port": "6543",
            "dbname": "postgres",
            "user": "postgres.egkmupciopviyycuuuym",
            "password": "O2fETdNwU9r4nmzy"
        }
            
    connection = connectDB("postgress", db_config)

    print("DB connected successfully")

except Exception as e:
    print(e)



