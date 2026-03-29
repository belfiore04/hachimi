import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

def init_db():
    # Connect to default database to create new one
    try:
        conn = psycopg2.connect(
            user="postgres",
            password="123456",
            host="localhost"
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()

        # Create database if not exists
        db_name = "cat_recognition"
        cursor.execute(f"SELECT 1 FROM pg_catalog.pg_database WHERE datname = '{db_name}'")
        exists = cursor.fetchone()
        if not exists:
            print(f"Creating database {db_name}...")
            cursor.execute(f"CREATE DATABASE {db_name}")
        else:
            print(f"Database {db_name} already exists.")
        
        cursor.close()
        conn.close()

        # Connect to the specific database
        conn = psycopg2.connect(
            dbname=db_name,
            user="postgres",
            password="123456",
            host="localhost"
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()

        # Enable vector extension
        print("Enabling vector extension...")
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")

        # Create table
        print("Creating table cat_gallery...")
        create_table_query = """
        CREATE TABLE IF NOT EXISTS cat_gallery (
            id SERIAL PRIMARY KEY,
            cat_name TEXT,
            image_path TEXT,
            embedding vector(512)
        );
        """
        cursor.execute(create_table_query)
        
        print("Database initialization complete.")
        cursor.close()
        conn.close()
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    init_db()
