from sqlalchemy import create_engine

# Replace these values with your actual database credentials
user = "postgres"
password = "6004Thakur"
host = "34.131.216.70"
port = "5432"
dbname = "postgres"

DATABASE_URL = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"

engine = create_engine(DATABASE_URL)

# Now you can use the engine to interact with your database
connection = engine.connect()
print("Database connected successfully")
