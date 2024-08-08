import os
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from geopy.distance import great_circle
from ortools.linear_solver import pywraplp
from dotenv import load_dotenv
import streamlit as st
import streamlit.components.v1 as components

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
            "V2": D_b.index[int(result['Deliveries assigned to V1'] - len(D_c)):].tolist() + D_a.index[int(result['Deliveries assigned to V1'] - len(D_c) - len(D_b.index[:int(result['Deliveries assigned to V1'] - len(D_c))])):int(result['Deliveries assigned to V1'] - len(D_c) - len(D_b.index[:int(result['Deliveries assigned to V1'] - len(D_c))]) + result['Deliveries assigned to V2'] - len(D_b.index[int(result['Deliveries assigned to V1'] - len(D_c)):]))].tolist(),
            "V3": D_a.index[int(result['Deliveries assigned to V1'] - len(D_c) - len(D_b.index[:int(result['Deliveries assigned to V1'] - len(D_c))]) + result['Deliveries assigned to V2'] - len(D_b.index[int(result['Deliveries assigned to V1'] - len(D_c)):])):].tolist()
        }

        st.session_state.vehicle_assignments = vehicle_assignments
        st.write("Vehicle Assignments:", vehicle_assignments)

    def calculate_distance_matrix(df):
        distance_matrix = np.zeros((len(df), len(df)))
        for i, (lat1, lon1) in enumerate(zip(df['Latitude'], df['Longitude'])):
            for j, (lat2, lon2) in enumerate(zip(df['Latitude'], df['Longitude'])):
                if i != j:
                    distance_matrix[i, j] = great_circle((lat1, lon1), (lat2, lon2)).kilometers
        return distance_matrix

    def generate_routes(vehicle_assignments, df_locations):
        vehicle_routes = {}
        summary_data = []

        for vehicle, assignments in vehicle_assignments.items():
            df_vehicle = df_locations.loc[assignments]

            if df_vehicle.empty:
                st.write(f"No assignments for {vehicle}")
                continue

            distance_matrix = calculate_distance_matrix(df_vehicle)
            if np.isnan(distance_matrix).any() or np.isinf(distance_matrix).any():
                st.write(f"Invalid values in distance matrix for {vehicle}")
                continue

            db = DBSCAN(eps=0.5, min_samples=1, metric='precomputed')
            db.fit(distance_matrix)

            labels = db.labels_
            df_vehicle['Cluster'] = labels

            for cluster in set(labels):
                cluster_df = df_vehicle[df_vehicle['Cluster'] == cluster]
                if cluster_df.empty:
                    continue
                centroid = cluster_df[['Latitude', 'Longitude']].mean().values
                total_distance = cluster_df.apply(lambda row: great_circle(centroid, (row['Latitude'], row['Longitude'])).kilometers, axis=1).sum()

                route_name = f"{vehicle} Cluster {cluster}"
                route_df = cluster_df.copy()
                route_df['Distance'] = total_distance

                if vehicle not in vehicle_routes:
                    vehicle_routes[vehicle] = []

                vehicle_routes[vehicle].append(route_df)
                summary_data.append({
                    'Vehicle': vehicle,
                    'Cluster': cluster,
                    'Centroid Latitude': centroid[0],
                    'Centroid Longitude': centroid[1],
                    'Number of Shops': len(cluster_df),
                    'Total Distance': total_distance
                })

        summary_df = pd.DataFrame(summary_data)
        return vehicle_routes, summary_df

    def render_map(df, map_name):
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
            <title>Map - {map_name}</title>
            <script src="https://maps.googleapis.com/maps/api/js?key={google_maps_api_key}&callback=initMap" async defer></script>
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
