import os
from dotenv import load_dotenv
import sqlalchemy as db
import streamlit as st

# Load .env file
load_dotenv()

# Get the database URL
DATABASE_URL = os.getenv("DATABASE_URL")

# Set up the connection to the PostgreSQL database
engine = db.create_engine(DATABASE_URL)
connection = engine.connect()
metadata = db.MetaData()

# Example table creation (only do this once)
example_table = db.Table('example_table', metadata,
                         db.Column('id', db.Integer, primary_key=True),
                         db.Column('shop_name', db.String(50)),
                         db.Column('delivery_status', db.String(50)),
                         db.Column('amount_paid', db.Float),
                         db.Column('outstanding_balance', db.Float)
                         )

# Create the table in the database (uncomment to run once)
# metadata.create_all(engine)

st.title("Delivery Optimization App with PostgreSQL")

st.write("Connected to the database!")

# Example data insertion (this should be connected to your delivery form logic)
def insert_data(shop_name, delivery_status, amount_paid, outstanding_balance):
    query = db.insert(example_table).values(shop_name=shop_name, delivery_status=delivery_status, amount_paid=amount_paid, outstanding_balance=outstanding_balance)
    connection.execute(query)

# Example form to insert data
with st.form("delivery_form"):
    shop_name = st.text_input("Shop Name")
    delivery_status = st.text_input("Delivery Status")
    amount_paid = st.number_input("Amount Paid", min_value=0.0)
    outstanding_balance = st.number_input("Outstanding Balance", min_value=0.0)
    submitted = st.form_submit_button("Submit")
    if submitted:
        insert_data(shop_name, delivery_status, amount_paid, outstanding_balance)
        st.success("Data inserted successfully!")

