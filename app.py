import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from geopy.distance import great_circle
import streamlit as st
import os
import pulp
import requests

# Load environment variables
google_maps_api_key = st.secrets["GOOGLE_MAPS_API_KEY"]

# Title for the Streamlit App
st.title("Delivery Optimization and Route Generation")

# File uploader for user to upload the delivery data
uploaded_file = st.file_uploader("Upload your delivery data Excel file", type=["xlsx"])

# Main function to handle the operations
if uploaded_file:
    df_locations = pd.read_excel(uploaded_file, engine='openpyxl')

    # Display the column names to verify
    st.write("Column Names:", df_locations.columns)

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

    # Define the load optimization function
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

    # Input fields for optimization parameters
    cost_v1 = st.number_input("Cost for Vehicle 1", value=62.8156)
    cost_v2 = st.number_input("Cost for Vehicle 2", value=33.0)
    cost_v3 = st.number_input("Cost for Vehicle 3", value=29.0536)
    v1_capacity = st.number_input("Capacity for Vehicle 1", value=64)
    v2_capacity = st.number_input("Capacity for Vehicle 2", value=66)
    v3_capacity = st.number_input("Capacity for Vehicle 3", value=72)
     scenario = st.selectbox(
        "Select Load Optimization Scenario",
        ["Scenario 1: V1, V2, V3", "Scenario 2: V1, V2", "Scenario 3: V1, V3"]
    )

    # Optimize load
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

    st.write("Vehicle Assignments:", vehicle_assignments)

    # Function to calculate distance matrix
    def calculate_distance_matrix(df):
        num_locations = len(df)
        distance_matrix = np.zeros((num_locations, num_locations))

        for i in range(num_locations):
            for j in range(num_locations):
                try:
                    coords_1 = (df.loc[i, 'Latitude'], df.loc[i, 'Longitude'])
                    coords_2 = (df.loc[j, 'Latitude'], df.loc[j, 'Longitude'])
                    distance_matrix[i][j] = great_circle(coords_1, coords_2).meters
                except Exception as e:
                    print(f"Error calculating distance between locations {i} and {j}: {e}")
                    distance_matrix[i][j] = float('inf')  # Assign a large value in case of error
        return distance_matrix

    # Function to generate Google Maps link for a given route
    def generate_google_maps_link(locations):
        base_url = "https://www.google.com/maps/dir/?api=1"
        origin = f"{locations[0]['Latitude']},{locations[0]['Longitude']}"
        destination = f"{locations[-1]['Latitude']},{locations[-1]['Longitude']}"
        waypoints = "|".join([f"{loc['Latitude']},{loc['Longitude']}" for loc in locations[1:-1]])

        link = f"{base_url}&origin={origin}&destination={destination}&waypoints={waypoints}&travelmode=driving"
        return link

    # Function to generate routes for each vehicle
    def generate_routes(vehicle_assignments, df):
        vehicle_routes = {}
        cluster_summary = []

        for vehicle, indices in vehicle_assignments.items():
            df_vehicle = df.loc[indices].reset_index(drop=True)
            distance_matrix = calculate_distance_matrix(df_vehicle)

            db = DBSCAN(eps=500, min_samples=1, metric='precomputed')
            db.fit(distance_matrix)

            df_vehicle['Cluster'] = db.labels_

            vehicle_routes[vehicle] = []
            for cluster_label in df_vehicle['Cluster'].unique():
                cluster_df = df_vehicle[df_vehicle['Cluster'] == cluster_label].reset_index(drop=True)
                cluster_coords = cluster_df[['Latitude', 'Longitude']].to_dict('records')

                # Generate Google Maps link
                google_maps_link = generate_google_maps_link(cluster_coords)
                vehicle_routes[vehicle].append({
                    'Cluster': cluster_label,
                    'Link': google_maps_link,
                    'Details': cluster_df
                })

                # Summary for clusters
                cluster_summary.append({
                    'Vehicle': vehicle,
                    'Cluster': cluster_label,
                    'Num_Shops': len(cluster_df),
                    'Total_Distance': sum(distance_matrix.flatten())  # Approximate total distance
                })

        return vehicle_routes, pd.DataFrame(cluster_summary)

    # Generate routes for vehicles
    vehicle_routes, cluster_summary = generate_routes(vehicle_assignments, df_locations)

    # Display Google Maps links for each vehicle and cluster
    for vehicle, routes in vehicle_routes.items():
        for route in routes:
            st.write(f"Vehicle {vehicle}, Cluster {route['Cluster']}: {route['Link']}")

    # Display cluster summary
    st.write("Cluster Summary:")
    st.write(cluster_summary)

    # Generate vehicle-specific maps
    for vehicle, routes in vehicle_routes.items():
        all_coords = []
        for route in routes:
            all_coords.extend(route['Details'][['Latitude', 'Longitude']].to_dict('records'))
        vehicle_map_link = generate_google_maps_link(all_coords)
        st.write(f"Vehicle {vehicle} Overall Route: {vehicle_map_link}")
