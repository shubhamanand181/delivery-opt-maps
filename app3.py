import os
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from geopy.distance import great_circle
from ortools.linear_solver import pywraplp
from dotenv import load_dotenv
import streamlit as st
import urllib.parse

# Load .env file
load_dotenv()

# Get the Google Maps API key
google_maps_api_key = os.getenv("GOOGLE_MAPS_API_KEY")

# Upload and read Excel file
st.title("Delivery Optimization App with Google Maps Integration")

uploaded_file = st.file_uploader("Choose an Excel file", type="xlsx")
if uploaded_file:
    df_locations = pd.read_excel(uploaded_file)  # Ensure openpyxl is in requirements.txt
    
    # Display the column names to verify
    st.write("Column Names:", df_locations.columns)

    # Ensure column names are as expected
    expected_columns = ['Party', 'Latitude', 'Longitude', 'Weight (KG)']
    if all(col in df_locations.columns for col in expected_columns):
        st.write("All expected columns are present.")
    else:
        st.write("One or more expected columns are missing. Please check the column names in the Excel file.")
        st.stop()

    # Remove rows with NaN values in Latitude or Longitude
    df_locations.dropna(subset=['Latitude', 'Longitude'], inplace=True)

    # Categorize weights
    def categorize_weights(df):
        D_a = df[(df['Weight (KG)'] > 0) & (df['Weight (KG)'] <= 2)]
        D_b = df[(df['Weight (KG)'] > 2) & (df['Weight (KG)'] <= 10)]
        D_c = df[(df['Weight (KG)'] > 10) & (df['Weight (KG)'] <= 200)]
        return D_a, D_b, D_c

    D_a, D_b, D_c = categorize_weights(df_locations)

    # Load optimization (Keep the existing code for load optimization as is)

    # Route generation (Keep the existing code for route generation as is)

    # Function to render the map with shop names
    def render_map(df, map_name):
        latitudes = df['Latitude'].tolist()
        longitudes = df['Longitude'].tolist()
        names = df['Party'].tolist()

        markers = '|'.join(f"{lat},{lon}%7Clabel:{urllib.parse.quote(name)}" for lat, lon, name in zip(latitudes, longitudes, names))
        return f"https://www.google.com/maps/dir/?api=1&origin={latitudes[0]},{longitudes[0]}&destination={latitudes[-1]},{longitudes[-1]}&travelmode=driving&waypoints=" + markers

    def render_cluster_maps(df_locations):
        if 'vehicle_assignments' not in st.session_state:
            st.write("Please optimize the load first.")
            return

        vehicle_assignments = st.session_state.vehicle_assignments
        vehicle_routes, summary_df = generate_routes(vehicle_assignments, df_locations)

        for vehicle, routes in vehicle_routes.items():
            for idx, route_df in enumerate(routes):
                route_name = f"{vehicle} Cluster {idx}"
                link = render_map(route_df, route_name)
                st.write(f"[{route_name}]({link})")

        st.write("Summary of Clusters:")
        st.table(summary_df)

        def generate_excel(vehicle_routes, summary_df):
            file_path = 'optimized_routes.xlsx'
            with pd.ExcelWriter(file_path, engine='xlsxwriter') as writer:
                for vehicle, routes in vehicle_routes.items():
                    for idx, route_df in enumerate(routes):
                        route_df.to_excel(writer, sheet_name=f'{vehicle}_Cluster_{idx}', index=False)
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
            with open(file_path, "rb") as f:
                st.download_button(
                    label="Download Excel file",
                    data=f,
                    file_name="optimized_routes.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

        generate_excel(vehicle_routes, summary_df)

    if st.button("Generate Routes"):
        render_cluster_maps(df_locations)
