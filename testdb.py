import os
from sqlalchemy import create_engine
from dotenv import load_dotenv
import streamlit as st

# Load environment variables
load_dotenv()

# Get database connection details from the .env file
db_host = os.getenv("DB_HOST")
db_port = os.getenv("DB_PORT")
db_name = os.getenv("DB_NAME")
db_user = os.getenv("DB_USER")
db_password = os.getenv("DB_PASSWORD")

# Create the database connection URL
db_url = f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"

# Test the database connection
try:
    engine = create_engine(db_url)
    connection = engine.connect()
    st.write("Connected to the database successfully!")
    connection.close()
except Exception as e:
    st.write(f"Failed to connect to the database: {e}")
