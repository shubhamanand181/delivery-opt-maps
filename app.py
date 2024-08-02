import os
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from geopy.distance import great_circle
import pulp
import folium
from streamlit_folium import st_folium
import streamlit as st
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get Google Maps API key from environment variables
google_maps_api_key = os.getenv("GOOGLE_MAPS_API_KEY")

# Streamlit app setup
st.title("Delivery Optimization App with Google Maps Integration")

# Upload the file
uploaded_file = st.file_uploader("Choose an Excel file", type="xlsx")

if uploaded_file is not None:
    # Load data from the uploaded file
    df_locations = pd.read_excel(uploaded_file)
    
    # Display the first few rows of the DataFrame
    st.write("Uploaded Data:")
    st.write(df_locations.head())

    # Ensure column names are as expected
    expected_columns = ['Party', 'Latitude', 'Longitude', 'Weight (KG)']
    if all(col in df_locations.columns for col in expected_columns):
        st.write("All expected columns are present.")
    else:
        st.write("One or more expected columns are missing. Please check the column names in the Excel file.")
    
    # Remove rows with NaN values in Latitude or Longitude
    df_locations.dropna(subset=['Latitude', 'Longitude'], inplace=True)
    
    # Categorize weights
    def categorize_weights(df):
        D_a = df[(df['Weight (KG)'] > 0) & (df['Weight (KG)'] <= 2)]
        D_b = df[(df['Weight (KG)'] > 2) & (df['Weight (KG)'] <= 10)]
        D_c = df[(df['Weight (KG)'] > 10) & (df['Weight (KG)'] <= 200)]
        return D_a, D_b, D_c

    D_a, D_b, D_c = categorize_weights(df_locations)

    # Print the count of each category
    st.write(f"Type A Deliveries (0-2 kg): {len(D_a)}")
    st.write(f"Type B Deliveries (2-10 kg): {len(D_b)}")
    st.write(f"Type C Deliveries (10-200 kg): {len(D_c)}")

    # Load optimization function
    def optimize_load(D_a_count, D_b_count, D_c_count, cost_v1, cost_v2, cost_v3, v1_capacity, v2_capacity, v3_capacity, scenario):
        if scenario == "Scenario 1: V1, V2, V3":
            lp_problem = pulp.LpProblem("Delivery_Cost_Minimization", pulp.LpMinimize)
            V1 = pulp.LpVariable('V1', lowBound=0, cat='Integer')
            V2 = pulp.LpVariable('V2', lowBound=0, cat='Integer')
            V3 = pulp.LpVariable('V3', lowBound=0, cat='Integer')

            A1 = pulp.LpVariable('A1', lowBound=0, cat='Continuous')
            B1 = pulp.LpVariable('B1', lowBound=0, cat='Continuous')
            C1 = pulp.LpVariable('C1', lowBound=0, cat='Continuous')
            A2 = pulp.LpVariable('A2', lowBound=0, cat='Continuous')
            B2 = pulp.LpVariable('B2', lowBound=0, cat='Continuous')
            A3 = pulp.LpVariable('A3', lowBound=0, cat='Continuous')

            lp_problem += cost_v1 * V1 + cost_v2 * V2 + cost_v3 * V3, "Total Cost"
            lp_problem += A1 + A2 + A3 == D_a_count, "Total_Deliveries_A_Constraint"
            lp_problem += B1 + B2 == D_b_count, "Total_Deliveries_B_Constraint"
            lp_problem += C1 == D_c_count, "Total_Deliveries_C_Constraint"
            lp_problem += v1_capacity * V1 >= C1 + B1 + A1, "V1_Capacity_Constraint"
            lp_problem += v2_capacity * V2 >= B2 + A2, "V2_Capacity_Constraint"
            lp_problem += v3_capacity * V3 >= A3, "V3_Capacity_Constraint"
            lp_problem += C1 == D_c_count, "Assign_C_To_V1"
            lp_problem += B1 <= v1_capacity * V1 - C1, "Assign_B_To_V1"
            lp_problem += B2 == D_b_count - B1, "Assign_Remaining_B_To_V2"
            lp_problem += A1 <= v1_capacity * V1 - C1 - B1, "Assign_A_To_V1"
            lp_problem += A2 <= v2_capacity * V2 - B2, "Assign_A_To_V2"
            lp_problem += A3 == D_a_count - A1 - A2, "Assign_Remaining_A_To_V3"
            lp_problem.solve()

            return {
                "Status": pulp.LpStatus[lp_problem.status],
                "V1": pulp.value(V1),
                "V2": pulp.value(V2),
                "V3": pulp.value(V3),
                "Total Cost": pulp.value(lp_problem.objective),
                "Deliveries assigned to V1": pulp.value(C1 + B1 + A1),
                "Deliveries assigned to V2": pulp.value(B2 + A2),
                "Deliveries assigned to V3": pulp.value(A3)
            }
        # Add similar cases for other scenarios if needed
        return None

    # Scenario selection and cost input
    scenario = st.selectbox("Select Scenario", ["Scenario 1: V1, V2, V3"])
    cost_v1 = st.number_input("Cost for V1", value=62.8156)
    cost_v2 = st.number_input("Cost for V2", value=33.0)
    cost_v3 = st.number_input("Cost for V3", value=29.0536)
    v1_capacity = st.number_input("Capacity for V1", value=64)
    v2_capacity = st.number_input("Capacity for V2", value=66)
    v3_capacity = st.number_input("Capacity for V3", value=72)

    if st.button("Optimize Load"):
        result = optimize_load(len(D_a), len(D_b), len(D_c), cost_v1, cost_v2, cost_v3, v1_capacity, v2_capacity, v3_capacity, scenario)
        st.write("Load Optimization Results:")
        st.write(f"Status: {result['Status']}")
        st.write(f"V1: {result['V1']}")
        st.write(f"V2: {result['V2']}")
        st.write(f"V3: {result['V3']}")
        st.write(f"Total Cost: {result['Total Cost']}")
        st.write(f"Deliveries assigned to V1: {result['Deliveries assigned to V1']}")
        st.write(f"Deliveries assigned to V2: {result['Deliveries assigned to V2']}")
        st.write(f"Deliveries assigned to V3: {result['Deliveries assigned to V3']}")

        # Assign deliveries to vehicles
        vehicle_assignments = {
            "V1": D_c.index.tolist() + D_b.index[:int(result['Deliveries assigned to V1'] - len(D_c))].tolist() + D_a.index[:int(result['Deliveries assigned to V1'] - len(D_c) - len(D_b.index[:int(result['Deliveries assigned to V1'] - len(D_c))]))].tolist(),
            "V2": D_b.index[int(result['Deliveries assigned to V1'] - len(D_c)):].tolist() + D_a.index[int(result['Deliveries assigned to V1'] - len(D_c) - len(D_b.index[:int(result['Deliveries assigned to V1'] - len(D_c))])):int(result['Deliveries assigned to V1'] - len(D_c) - len(D_b.index[:int(result['Deliveries assigned to V1'] - len(D_c))]) + result['Deliveries assigned to V2'] - len(D_b.index[int(result['Deliveries assigned to V1'] - len(D_c)):]))].tolist(),
            "V3": D_a.index[int(result['Deliveries assigned to V1'] - len(D_c) - len(D_b.index[:int(result['Deliveries assigned to V1'] - len(D_c))]) + result['Deliveries assigned to V2'] - len(D_b.index[int(result['Deliveries assigned to V1'] - len(D_c)):])):].tolist()
        }

        st.write("Vehicle Assignments:")
        st.write(vehicle_assignments)

        # Save vehicle assignments in the session state for route optimization
        st.session_state['vehicle_assignments'] = vehicle_assignments
        st.session_state['df_locations'] = df_locations

# Route Optimization
if 'vehicle_assignments' in st.session_state and 'df_locations' in st.session_state:
    vehicle_assignments = st.session_state['vehicle_assignments']
    df_locations = st.session_state['df_locations']

    if st.button("Generate Routes"):
        # Route optimization function
        def calculate_distance_matrix(df):
            num_locations = len(df)
            distance_matrix = np.zeros((num_locations, num_locations))

            for i in range(num_locations):
                for j in range(num_locations):
                    if i != j:
                        coords_1 = (df.iloc[i]['Latitude'], df.iloc[i]['Longitude'])
                        coords_2 = (df.iloc[j]['Latitude'], df.iloc[j]['Longitude'])
                        distance_matrix[i][j] = great_circle(coords_1, coords_2).meters
            return distance_matrix

        def nearest_neighbor(distance_matrix):
            num_locations = len(distance_matrix)
            visited = [False] * num_locations
            route = [0]
            visited[0] = True
            total_distance = 0

            for _ in range(num_locations - 1):
                last_index = route[-1]
                next_index = None
                min_distance = float('inf')

                for j in range(num_locations):
                    if not visited[j] and distance_matrix[last_index][j] < min_distance:
                        next_index = j
                        min_distance = distance_matrix[last_index][j]

                route.append(next_index)
                visited[next_index] = True
                total_distance += min_distance

            # Return to start point
            total_distance += distance_matrix[route[-1]][route[0]]
            route.append(0)

            return route, total_distance

        def generate_routes(vehicle_assignments, df_locations):
            vehicle_routes = {}
            summary_data = []

            for vehicle, indices in vehicle_assignments.items():
                df_vehicle = df_locations.loc[indices]
                distance_matrix = calculate_distance_matrix(df_vehicle)
                db = DBSCAN(eps=500, min_samples=1, metric='precomputed')
                db.fit(distance_matrix)

                df_vehicle['Cluster'] = db.labels_
                vehicle_routes[vehicle] = []

                for cluster_id in df_vehicle['Cluster'].unique():
                    cluster_df = df_vehicle[df_vehicle['Cluster'] == cluster_id]
                    distance_matrix_cluster = calculate_distance_matrix(cluster_df)
                    route, total_distance = nearest_neighbor(distance_matrix_cluster)
                    vehicle_routes[vehicle].append((cluster_id, route, total_distance))

                    summary_data.append({
                        'Vehicle': vehicle,
                        'Cluster': cluster_id,
                        'Num Shops': len(cluster_df),
                        'Total Distance': total_distance / 1000,  # in kilometers
                        'Latitude': cluster_df['Latitude'].mean(),
                        'Longitude': cluster_df['Longitude'].mean()
                    })

            summary_df = pd.DataFrame(summary_data)
            return vehicle_routes, summary_df

        vehicle_routes, summary_df = generate_routes(vehicle_assignments, df_locations)

        # Display the summary table
        st.write("Summary Table")
        st.write(summary_df)

        # Generate Excel with routes and summary
        def generate_excel(vehicle_routes, summary_df):
            file_path = '/mnt/data/optimized_routes.xlsx'
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            with pd.ExcelWriter(file_path, engine='xlsxwriter') as writer:
                for vehicle, routes in vehicle_routes.items():
                    for cluster_id, route, total_distance in routes:
                        cluster_df = df_locations.loc[route[:-1]]
                        cluster_df['Sequence'] = range(1, len(cluster_df) + 1)
                        cluster_df.to_excel(writer, sheet_name=f'{vehicle}_Cluster_{cluster_id}', index=False)

                summary_df.to_excel(writer, sheet_name='Summary', index=False)

            return file_path

        file_path = generate_excel(vehicle_routes, summary_df)

        # Download link for Excel file
        st.write(f"[Download Optimized Routes]({file_path})")

        # Render maps for each cluster
        def render_map(df, name="Map"):
            center_lat = df['Latitude'].mean()
            center_lon = df['Longitude'].mean()
            m = folium.Map(location=[center_lat, center_lon], zoom_start=13)

            for _, row in df.iterrows():
                folium.Marker(
                    location=[row['Latitude'], row['Longitude']],
                    popup=row['Party'],
                ).add_to(m)

            st_folium(m, width=700, height=500)

        st.write("Cluster Maps")
        for vehicle, routes in vehicle_routes.items():
            st.write(f"Vehicle: {vehicle}")
            for cluster_id, route, total_distance in routes:
                cluster_df = df_locations.loc[route[:-1]]
                st.write(f"Cluster {cluster_id} Route")
                render_map(cluster_df)

        # Option to choose starting point for vehicle-specific maps
        st.write("Vehicle Specific Maps with Custom Starting Points")
        vehicle_choice = st.selectbox("Select Vehicle", list(vehicle_routes.keys()))
        if vehicle_choice:
            route_choice = st.selectbox("Select Route", list(range(len(vehicle_routes[vehicle_choice]))))
            starting_point = st.selectbox("Select Starting Point", vehicle_routes[vehicle_choice][route_choice][1][:-1])
            if st.button("Generate Vehicle Map with Custom Start"):
                route, total_distance = nearest_neighbor(calculate_distance_matrix(df_locations.loc[vehicle_routes[vehicle_choice][route_choice][1][:-1]]))
                custom_start_df = df_locations.loc[route]
                render_map(custom_start_df)

