import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
from dotenv import load_dotenv
import os

# Load .env file
load_dotenv()

# Get the Google Maps API key
google_maps_api_key = os.getenv("GOOGLE_MAPS_API_KEY")

# Upload and read Excel file
st.title("Google Maps with Custom Markers")

uploaded_file = st.file_uploader("Choose an Excel file", type="xlsx")
if uploaded_file:
    df_locations = pd.read_excel(uploaded_file)
    st.write("Column Names:", df_locations.columns)

    expected_columns = ['Party', 'Latitude', 'Longitude']
    if all(col in df_locations.columns for col in expected_columns):
        st.write("All expected columns are present.")
    else:
        st.write("One or more expected columns are missing. Please check the column names in the Excel file.")
        st.stop()

    df_locations.dropna(subset=['Latitude', 'Longitude'], inplace=True)

    def generate_custom_map(df):
        markers = []
        for _, row in df.iterrows():
            marker = f'''
            new google.maps.Marker({{
                position: {{lat: {row['Latitude']}, lng: {row['Longitude']}}},
                map: map,
                title: "{row['Party']}"
            }});
            '''
            markers.append(marker)
        return f'''
        <html>
        <head>
        <script src="https://maps.googleapis.com/maps/api/js?key={google_maps_api_key}&callback=initMap" async defer></script>
        <script>
        function initMap() {{
            var map = new google.maps.Map(document.getElementById('map'), {{
                zoom: 10,
                center: {{lat: {df['Latitude'].mean()}, lng: {df['Longitude'].mean()}}}
            }});

            {''.join(markers)}
        }}
        </script>
        </head>
        <body>
        <div id="map" style="height: 500px; width: 100%;"></div>
        </body>
        </html>
        '''

    custom_map_html = generate_custom_map(df_locations)
    components.html(custom_map_html, height=600)
