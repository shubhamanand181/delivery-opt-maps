import os
import pandas as pd
from dotenv import load_dotenv
import streamlit as st
import streamlit.components.v1 as components

# Load .env file
load_dotenv()

# Get the Google Maps API key
google_maps_api_key = os.getenv("GOOGLE_MAPS_API_KEY")

# Function to create the HTML code for Google Maps with custom markers
def create_map_html(df, api_key):
    markers = []
    for index, row in df.iterrows():
        markers.append(f"""
            var marker = new google.maps.Marker({{
                position: {{lat: {row['Latitude']}, lng: {row['Longitude']}}},
                map: map,
                title: '{row['Party']}'
            }});
            var infoWindow = new google.maps.InfoWindow({{
                content: '<b>{row['Party']}</b><br>Lat: {row['Latitude']}<br>Lng: {row['Longitude']}<br><a href="https://www.google.com/maps/dir/?api=1&destination={row['Latitude']},{row['Longitude']}" target="_blank">Navigate</a>'
            }});
            marker.addListener('click', function() {{
                infoWindow.open(map, marker);
            }});
        """)
    markers_js = "\n".join(markers)
    
    html_code = f"""
    <!DOCTYPE html>
    <html>
      <head>
        <title>Simple Map</title>
        <script src="https://maps.googleapis.com/maps/api/js?key={api_key}&callback=initMap" async defer></script>
        <script>
          function initMap() {{
            var map = new google.maps.Map(document.getElementById('map'), {{
              center: {{lat: {df['Latitude'].mean()}, lng: {df['Longitude'].mean()}}},
              zoom: 12
            }});
            {markers_js}
          }}
        </script>
      </head>
      <body>
        <div id="map" style="height: 500px; width: 100%;"></div>
      </body>
    </html>
    """
    return html_code

# Streamlit UI
st.title("Map with Custom Markers")

uploaded_file = st.file_uploader("Choose an Excel file", type="xlsx")
if uploaded_file:
    df_locations = pd.read_excel(uploaded_file)
    
    # Ensure the dataframe has the required columns
    if 'Party' in df_locations.columns and 'Latitude' in df_locations.columns and 'Longitude' in df_locations.columns:
        # Clean the dataframe: Remove rows with missing or invalid coordinates
        df_locations = df_locations.dropna(subset=['Latitude', 'Longitude'])
        df_locations = df_locations[df_locations['Latitude'].apply(lambda x: pd.notnull(x))]
        df_locations = df_locations[df_locations['Longitude'].apply(lambda x: pd.notnull(x))]
        
        st.write("Map with Custom Markers:")
        map_html = create_map_html(df_locations, google_maps_api_key)
        components.html(map_html, height=500)
    else:
        st.write("Please ensure your Excel file has the columns: 'Party', 'Latitude', 'Longitude'")
