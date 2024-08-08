import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from geopy.distance import great_circle
from ortools.linear_solver import pywraplp
from dotenv import load_dotenv
import os

# Load .env file
load_dotenv()

# Get the Google Maps API key
google_maps_api_key = os.getenv("GOOGLE_MAPS_API_KEY")

# Initialize session state for delivered shops
if 'delivered_shops' not in st.session_state:
    st.session_state.delivered_shops = []

# Function to create the HTML code for Google Maps with custom markers and navigation links
def create_map_html(df, api_key, delivered_shops):
    markers = []
    for index, row in df.iterrows():
        if row['Party'] not in delivered_shops:
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
st.title("Delivery Optimization App with Google Maps Integration")

uploaded_file = st.file_uploader("Choose an Excel file", type="xlsx")
if uploaded_file:
    df_locations = pd.read_excel(uploaded_file)
    
    # Ensure the dataframe has the required columns
    if 'Party' in df_locations.columns and 'Latitude' in df_locations.columns and 'Longitude' in df_locations.columns:
        # Clean the dataframe: Remove rows with missing or invalid coordinates
        df_locations = df_locations.dropna(subset=['Latitude', 'Longitude'])
        df_locations = df_locations[df_locations['Latitude'].apply(lambda x: np.isfinite(x))]
        df_locations = df_locations[df_locations['Longitude'].apply(lambda x: np.isfinite(x))]

        # Load optimization
        st.subheader("Load Optimization")
        cost_v1 = st.number_input("Enter cost for V1:", value=62.8156)
        cost_v2 = st.number_input("Enter cost for V2:", value=33.0)
        cost_v3 = st.number_input("Enter cost for V3:", value=29.0536)
        v1_capacity = st.number_input("Enter capacity for V1:", value=64)
        v2_capacity = st.number_input("Enter capacity for V2:", value=66)
        v3_capacity = st.number_input("Enter capacity for V3:", value=72)
        
        def optimize_load(D_a_count, D_b_count, D_c_count, cost_v1, cost_v2, cost_v3, v1_capacity, v2_capacity, v3_capacity):
            solver = pywraplp.Solver.CreateSolver('SCIP')
            if not solver:
                return None

            V1 = solver.IntVar(0, solver.infinity(), 'V1')
            V2 = solver.IntVar(0, solver.infinity(), 'V2')
            V3 = solver.IntVar(0, solver.infinity(), 'V3')

            A1 = solver.NumVar(0, solver.infinity(), 'A1')
            B1 = solver.NumVar(0, solver.infinity(), 'B1')
            C1 = solver.NumVar(0, solver.infinity(), 'C1')
            A2 = solver.NumVar(0, solver.infinity(), 'A2')
            B2 = solver.NumVar(0, solver.infinity(), 'B2')
            A3 = solver.NumVar(0, solver.infinity(), 'A3')

            solver.Add(A1 + A2 + A3 == D_a_count)
            solver.Add(B1 + B2 == D_b_count)
            solver.Add(C1 == D_c_count)

            solver.Add(v1_capacity * V1 >= C1 + B1 + A1)
            solver.Add(v2_capacity * V2 >= B2 + A2)
            solver.Add(v3_capacity * V3 >= A3)
            solver.Add(C1 == D_c_count)
            solver.Add(B1 <= v1_capacity * V1 - C1)
            solver.Add(B2 == D_b_count - B1)
            solver.Add(A1 <= v1_capacity * V1 - C1 - B1)
            solver.Add(A2 <= v2_capacity * V2 - B2)
            solver.Add(A3 == D_a_count - A1 - A2)

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
            D_a_count = len(df_locations[df_locations['Weight (KG)'] <= 2])
            D_b_count = len(df_locations[(df_locations['Weight (KG)'] > 2) & (df_locations['Weight (KG)'] <= 10)])
            D_c_count = len(df_locations[df_locations['Weight (KG)'] > 10])
            result = optimize_load(D_a_count, D_b_count, D_c_count, cost_v1, cost_v2, cost_v3, v1_capacity, v2_capacity, v3_capacity)
            st.write("Load Optimization Results:")
            st.write(result)
            st.session_state.load_optimization_result = result

            vehicle_assignments = {
                "V1": df_locations[df_locations['Weight (KG)'] > 10].index.tolist(),
                "V2": df_locations[(df_locations['Weight (KG)'] > 2) & (df_locations['Weight (KG)'] <= 10)].index.tolist(),
                "V3": df_locations[df_locations['Weight (KG)'] <= 2].index.tolist()
            }

            st.session_state.vehicle_assignments = vehicle_assignments
            st.write("Vehicle Assignments:", vehicle_assignments)
        
        # Generate the HTML for the map
        html_code = create_map_html(df_locations, google_maps_api_key, st.session_state.delivered_shops)
        
        # Display the map in Streamlit
        components.html(html_code, height=600)

        # Proof of Delivery Form
        st.subheader("Proof of Delivery Form")
        selected_shop = st.selectbox("Select Shop", df_locations['Party'])
        payment_made = st.number_input("Payment Made", min_value=0.0, format="%.2f")
        previous_due = st.number_input("Previous Due", min_value=0.0, format="%.2f")
        updated_balance = st.number_input("Updated Balance", min_value=0.0, format="%.2f")
        return_note = st.text_area("Return Note")

        if st.button("Submit Delivery"):
            st.session_state.delivered_shops.append(selected_shop)
            st.success(f"Delivery details for {selected_shop} recorded.")
            st.write("Delivery Details:")
            st.write(f"Payment Made: {payment_made}")
            st.write(f"Previous Due: {previous_due}")
            st.write(f"Updated Balance: {updated_balance}")
            st.write(f"Return Note: {return_note}")
        
        # Function to generate routes and summary
        def generate_routes(vehicle_assignments, df_locations):
            vehicle_routes = {}
            summary_data = []

            for vehicle, assignments in vehicle_assignments.items():
                df_vehicle = df_locations.loc[assignments]

                if df_vehicle.empty:
                    st.write(f"No assignments for {vehicle}")
                    continue

                distance_matrix = np.zeros((len(df_vehicle), len(df_vehicle)))
                for i, (lat1, lon1) in enumerate(zip(df_vehicle['Latitude'], df_vehicle['Longitude'])):
                    for j, (lat2, lon2) in enumerate(zip(df_vehicle['Latitude'], df_vehicle['Longitude'])):
                        if i != j:
                            distance_matrix[i, j] = great_circle((lat1, lon1), (lat2, lon2)).kilometers
                
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

        def render_cluster_maps(df_locations):
            if 'vehicle_assignments' not in st.session_state:
                st.write("Please optimize the load first.")
                return

            vehicle_assignments = st.session_state.vehicle_assignments
            vehicle_routes, summary_df = generate_routes(vehicle_assignments, df_locations)

            for vehicle, routes in vehicle_routes.items():
                for idx, route_df in enumerate(routes):
                    route_name = f"{vehicle} Cluster {idx}"
                    link = f"https://www.google.com/maps/dir/?api=1&origin={route_df.iloc[0]['Latitude']},{route_df.iloc[0]['Longitude']}&destination={route_df.iloc[-1]['Latitude']},{route_df.iloc[-1]['Longitude']}&travelmode=driving&waypoints=" + '|'.join(f"{lat},{lon}" for lat, lon in zip(route_df['Latitude'], route_df['Longitude']))
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
        
        if st.button("Update Routes"):
            render_cluster_maps(df_locations)
    else:
        st.error("The uploaded file does not have the required columns: 'Party', 'Latitude', 'Longitude'")

