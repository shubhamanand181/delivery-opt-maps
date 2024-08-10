import os
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import streamlit as st
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Database connection string
DATABASE_URL = os.getenv("DATABASE_URL")

# Create the database engine
engine = create_engine(DATABASE_URL)

# Create a session
Session = sessionmaker(bind=engine)
session = Session()

# Define the base class for declarative models
Base = declarative_base()

# Define the DeliveryForm model
class DeliveryForm(Base):
    __tablename__ = 'delivery_forms'
    id = Column(Integer, primary_key=True)
    shop_name = Column(String, nullable=False)
    payment_made = Column(Float, nullable=False)
    previous_due = Column(Float, nullable=False)
    updated_balance = Column(Float, nullable=False)
    return_note = Column(String)

# Create the table in the database
Base.metadata.create_all(engine)

# Streamlit UI
st.title("Delivery Form Management")

# Form to input delivery details
with st.form(key='delivery_form'):
    shop_name = st.text_input('Shop Name')
    payment_made = st.number_input('Payment Made', min_value=0.0, format="%.2f")
    previous_due = st.number_input('Previous Due', min_value=0.0, format="%.2f")
    updated_balance = st.number_input('Updated Balance', min_value=0.0, format="%.2f")
    return_note = st.text_area('Return Note', '')

    submit_button = st.form_submit_button(label='Submit')

if submit_button:
    # Create a new delivery form entry
    new_form = DeliveryForm(
        shop_name=shop_name,
        payment_made=payment_made,
        previous_due=previous_due,
        updated_balance=updated_balance,
        return_note=return_note
    )

    # Add to session and commit to the database
    session.add(new_form)
    session.commit()

    st.success("Delivery form submitted successfully!")
