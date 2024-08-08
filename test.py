import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
from dotenv import load_dotenv
import os

# Load .env file
load_dotenv()

# Get the Google Maps API key
google_maps_api_key = os.getenv("GOOGLE_MAPS_API_KEY")

# Basic HTML to display the map
html_code = f"""
<!DOCTYPE html>
<html>
  <head>
    <title>Simple Map</title>
    <script src="https://maps.googleapis.com/maps/api/js?key={google_maps_api_key}&callback=initMap" async defer></script>
    <script>
      function initMap() {{
        var map = new google.maps.Map(document.getElementById('map'), {{
          center: {{lat: -34.397, lng: 150.644}},
          zoom: 8
        }});
      }}
    </script>
  </head>
  <body>
    <div id="map" style="height: 500px; width: 100%;"></div>
  </body>
</html>
"""

# Display the map in Streamlit
st.title("Google Maps Integration Test")
components.html(html_code, height=600)
