import os
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from geopy.distance import great_circle
from ortools.linear_solver import pywraplp
from dotenv import load_dotenv
import streamlit as st
import streamlit.components.v1 as components
import folium
from folium.plugins import MarkerCluster
from io import BytesIO

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

    # Load optimization
    cost_v1 = st.number_input("Enter cost for V1:", value=62.8156)
    cost_v2 = st.number_input("Enter cost for V2:", value=33.0)
    cost_v3 = st.number_input("Enter cost for V3:", value=29.0536)
    v1_capacity = st.number_input("Enter capacity for V1:", value=64)
    v2_capacity = st.number_input("Enter capacity for V2:", value=66)
    v3_capacity = st.number_input("Enter capacity for V3:", value=72)

    scenario = st.selectbox(
        "Select a scenario:",
        ("Scenario 1: V1, V2, V3", "Scenario 2: V1, V2", "Scenario 3: V1, V3")
    )

    def optimize_load(D_a_count, D_b_count, D_c_count, cost_v1, cost_v2, cost_v3, v1_capacity, v2_capacity, v3_capacity, scenario):
        solver = pywraplp.Solver.CreateSolver('SCIP')
        if not solver:
            return None

        # Variables
        V1 = solver.IntVar(0, solver.infinity(), 'V1')
        V2 = solver.IntVar(0, solver.infinity(), 'V2')
        V3 = solver.IntVar(0, solver.infinity(), 'V3')

        A1 = solver.NumVar(0, solver.infinity(), 'A1')
        B1 = solver.NumVar(0, solver.infinity(), 'B1')
        C1 = solver.NumVar(0, solver.infinity(), 'C1')
        A2 = solver.NumVar(0, solver.infinity(), 'A2')
        B2 = solver.NumVar(0, solver.infinity(), 'B2')
        A3 = solver.NumVar(0, solver.infinity(), 'A3')

        # Constraints
        solver.Add(A1 + A2 + A3 == D_a_count)
        solver.Add(B1 + B2 == D_b_count)
        solver.Add(C1 == D_c_count)

        if scenario == "Scenario 1: V1, V2, V3":
            solver.Add(v1_capacity * V1 >= C1 + B1 + A1)
            solver.Add(v2_capacity * V2 >= B2 + A2)
            solver.Add(v3_capacity * V3 >= A3)
            solver.Add(C1 == D_c_count)
            solver.Add(B1 <= v1_capacity * V1 - C1)
            solver.Add(B2 == D_b_count - B1)
            solver.Add(A1 <= v1_capacity * V1 - C1 - B1)
            solver.Add(A2 <= v2_capacity * V2 - B2)
            solver.Add(A3 == D_a_count - A1 - A2)
        elif scenario == "Scenario 2: V1, V2":
            solver.Add(v1_capacity * V1 >= C1 + B1 + A1)
            solver.Add(v2_capacity * V2 >= B2 + A2)
            solver.Add(C1 == D_c_count)
            solver.Add(B1 <= v1_capacity * V1 - C1)
            solver.Add(B2 == D_b_count - B1)
            solver.Add(A1 <= v1_capacity * V1 - C1 - B1)
            solver.Add(A2 <= v2_capacity * V2 - B2)
            solver.Add(V3 == 0)  # Ensure V3 is not used
            solver.Add(A3 == 0)  # Ensure A3 is not used
        elif scenario == "Scenario 3: V1, V3":
            solver.Add(v1_capacity * V1 >= C1 + B1 + A1)
            solver.Add(v3_capacity * V3 >= A3)
            solver.Add(C1 == D_c_count)
            solver.Add(B1 <= v1_capacity * V1 - C1)
            solver.Add(A1 <= v1_capacity * V1 - C1 - B1)
            solver.Add(A3 == D_a_count - A1)
            solver.Add(V2 == 0)  # Ensure V2 is not used
            solver.Add(B2 == 0)  # Ensure B2 is not used
            solver.Add(A2 == 0)  # Ensure A2 is not used

        # Objective
        solver.Minimize(cost_v1 * V1 + cost_v2 * V2 + cost_v3 * V3)

        status = solver.Solve()

        if status == pywraplp.Solver.OPTIMAL:
            return {
                "Status": "Optimal",
                "V1": V1.solution_value(),
                "V2": V2.solution_value(),
                "V3": V3.solution_value(),
                "Total Cost": solver.Objective().Value(),
                "Deliveries assigned to V1": C1.solution_value() + B1.solution_value() + A1.solution_value(),
                "Deliveries assigned to V2": B2.solution_value() + A2.solution_value(),
                "Deliveries assigned to V3": A3.solution_value()
            }
        else:
            return {
                "Status": "Not Optimal",
                "Result": {
                    "V1": V1.solution_value(),
                    "V2": V2.solution_value(),
                    "V3": V3.solution_value(),
                    "Total Cost": solver.Objective().Value(),
                    "Deliveries assigned to V1": C1.solution_value() + B1.solution_value() + A1.solution_value(),
                    "Deliveries assigned to V2": B2.solution_value() + A2.solution_value(),
                    "Deliveries assigned to V3": A3.solution_value()
                }
            }

    if st.button("Optimize Load"):
        result = optimize_load(len(D_a), len(D_b), len(D_c), cost_v1, cost_v2, cost_v3, v1_capacity, v2_capacity, v3_capacity, scenario)
        st.write("Load Optimization Results:")
        st.write(f"Status: {result['Status']}")
        
        if result['Status'] == "Optimal":
            st.write(f"V1: {result['V1']}")
            st.write(f"V2: {result['V2']}")
            st.write(f"V3: {result['V3']}")
            st.write(f"Total Cost: {result['Total Cost']}")
            st.write(f"Deliveries assigned to V1: {result['Deliveries assigned to V1']}")
            st.write(f"Deliveries assigned to V2: {result['Deliveries assigned to V2']}")
            st.write(f"Deliveries assigned to V3: {result['Deliveries assigned to V3']}")
        else:
            st.write("Optimization did not reach optimal status. Here are the partial results:")
            st.write(result["Result"])

        vehicle_assignments = {
            "V1": D_c.index.tolist() + D_b.index[:int(result['Deliveries assigned to V1'] - len(D_c))].tolist() + D_a.index[:int(result['Deliveries assigned to V1'] - len(D_c) - len(D_b.index[:int(result['Deliveries assigned to V1'] - len(D_c))]))].tolist(),
            "V2": D_b.index[int(result['Deliveries assigned to V1'] - len(D_c) - len(D_b.index[:int(result['Deliveries assigned to V1'] - len(D_c))])):int(result['Deliveries assigned to V1'] - len(D_c))].tolist() + D_a.index[int(result['Deliveries assigned to V1'] - len(D_c) - len(D_b.index[:int(result['Deliveries assigned to V1'] - len(D_c))])):].tolist(),
            "V3": D_a.index[int(result['Deliveries assigned to V1'] - len(D_c) - len(D_b.index[:int(result['Deliveries assigned to V1'] - len(D_c))])) + int(result['Deliveries assigned to V2']):].tolist()
        }
        st.write("Vehicle Assignments:")
        st.write(vehicle_assignments)

    # Clustering
    def cluster_locations(df):
        coords = df[['Latitude', 'Longitude']].values
        clustering = DBSCAN(eps=0.5, min_samples=2, metric='haversine').fit(np.radians(coords))
        df['Cluster'] = clustering.labels_
        return df

    df_locations = cluster_locations(df_locations)

    # Generate map for each cluster
    def create_map(df, cluster_id):
        map_html = '''
        <html>
        <head>
        <script src="https://maps.googleapis.com/maps/api/js?key={}&callback=initMap" async defer></script>
        <script>
        var map;
        function initMap() {{
            map = new google.maps.Map(document.getElementById('map'), {{
                center: {{lat: {}, lng: {}}},
                zoom: 14
            }});

            var markers = {};
            for (var i = 0; i < markers.length; i++) {{
                var marker = new google.maps.Marker({{
                    position: new google.maps.LatLng(markers[i].lat, markers[i].lng),
                    map: map,
                    title: markers[i].name
                }});
            }}
        }}
        </script>
        </head>
        <body>
        <div id="map" style="height: 500px; width: 100%;"></div>
        </body>
        </html>
        '''.format(
            google_maps_api_key,
            df['Latitude'].mean(),
            df['Longitude'].mean(),
            df[['Latitude', 'Longitude', 'Party']].apply(lambda x: {'lat': x['Latitude'], 'lng': x['Longitude'], 'name': x['Party']}, axis=1).to_list()
        )

        return map_html

    # Create and display maps for each cluster
    if st.button("Generate Maps"):
        for cluster_id in df_locations['Cluster'].unique():
            cluster_df = df_locations[df_locations['Cluster'] == cluster_id]
            cluster_map_html = create_map(cluster_df, cluster_id)
            st.write(f"Cluster {cluster_id} Map:")
            components.html(cluster_map_html, height=500)

        st.write("Maps have been generated for each cluster.")

    # Excel download for each cluster
    def download_excel(df, cluster_id):
        buffer = BytesIO()
        df[df['Cluster'] == cluster_id].to_excel(buffer, index=False)
        buffer.seek(0)
        return buffer

    if st.button("Download Excel Files"):
        for cluster_id in df_locations['Cluster'].unique():
            cluster_df = df_locations[df_locations['Cluster'] == cluster_id]
            buffer = download_excel(df_locations, cluster_id)
            st.download_button(
                label=f"Download Cluster {cluster_id} Excel",
                data=buffer,
                file_name=f"cluster_{cluster_id}_data.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
