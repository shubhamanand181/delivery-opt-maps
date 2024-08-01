import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from geopy.distance import great_circle
import pulp
import os
import openpyxl

# Set the title of the Streamlit app
st.title("Delivery Optimization App with Google Maps")

# Function to calculate the distance matrix
def calculate_distance_matrix(df):
    num_locations = len(df)
    distance_matrix = np.zeros((num_locations, num_locations))

    for i in range(num_locations):
        for j in range(num_locations):
            if i != j:
                try:
                    coords_1 = (float(df.iloc[i]['Latitude']), float(df.iloc[i]['Longitude']))
                    coords_2 = (float(df.iloc[j]['Latitude']), float(df.iloc[j]['Longitude']))
                    distance_matrix[i][j] = great_circle(coords_1, coords_2).meters
                except ValueError as e:
                    print(f"Invalid coordinates at index {i} or {j}: {e}")
                    distance_matrix[i][j] = np.inf  # Assign a large value to indicate invalid distance
            else:
                distance_matrix[i][j] = 0
    return distance_matrix

# Function to optimize load
def optimize_load(D_a, D_b, D_c, cost_v1, cost_v2, cost_v3, v1_capacity, v2_capacity, v3_capacity, scenario):
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
    lp_problem += A1 + A2 + A3 == D_a, "Total_Deliveries_A_Constraint"
    lp_problem += B1 + B2 == D_b, "Total_Deliveries_B_Constraint"
    lp_problem += C1 == D_c, "Total_Deliveries_C_Constraint"
    lp_problem += v1_capacity * V1 >= C1 + B1 + A1, "V1_Capacity_Constraint"
    lp_problem += v2_capacity * V2 >= B2 + A2, "V2_Capacity_Constraint"
    lp_problem += v3_capacity * V3 >= A3, "V3_Capacity_Constraint"
    lp_problem += C1 == D_c, "Assign_C_To_V1"
    lp_problem += B1 <= v1_capacity * V1 - C1, "Assign_B_To_V1"
    lp_problem += B2 == D_b - B1, "Assign_Remaining_B_To_V2"
    lp_problem += A1 <= v1_capacity * V1 - C1 - B1, "Assign_A_To_V1"
    lp_problem += A2 <= v2_capacity * V2 - B2, "Assign_A_To_V2"
    lp_problem += A3 == D_a - A1 - A2, "Assign_Remaining_A_To_V3"
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

# Function to generate routes
def generate_routes(vehicle_assignments, df_locations):
    vehicle_routes = {}
    summary_data = []

    for vehicle, indices in vehicle_assignments.items():
        df_vehicle = df_locations.loc[indices].reset_index(drop=True)
        distance_matrix = calculate_distance_matrix(df_vehicle)

        epsilon = 500  # meters
        db = DBSCAN(eps=epsilon, min_samples=1, metric='precomputed')
        db.fit(distance_matrix)

        df_vehicle['Cluster'] = db.labels_

        clusters = df_vehicle['Cluster'].unique()
        cluster_routes = {}

        for cluster_id in clusters:
            cluster_df = df_vehicle[df_vehicle['Cluster'] == cluster_id]
            centroid_lat = cluster_df['Latitude'].mean()
            centroid_lon = cluster_df['Longitude'].mean()
            cluster_distance_matrix = calculate_distance_matrix(cluster_df)
            route, total_distance = nearest_neighbor(cluster_distance_matrix)
            mapped_route = cluster_df.index[route]

            cluster_routes[cluster_id] = {
                "route": mapped_route,
                "total_distance": total_distance / 1000
            }

            summary_data.append({
                "Cluster": cluster_id,
                "Latitude": centroid_lat,
                "Longitude": centroid_lon,
                "Vehicle": vehicle,
                "Number of Shops": len(cluster_df),
                "Total Distance (km)": total_distance / 1000
            })

        vehicle_routes[vehicle] = cluster_routes

    summary_df = pd.DataFrame(summary_data)
    return vehicle_routes, summary_df

# Function to render maps
def render_map(df):
    st.map(df[['Latitude', 'Longitude']])

# Function to render cluster maps
def render_cluster_maps(df_locations):
    st.subheader("Cluster Maps")
    clusters = df_locations['Cluster'].unique()
    for cluster_id in clusters:
        st.write(f"Cluster {cluster_id} Route:")
        cluster_df = df_locations[df_locations['Cluster'] == cluster_id]
        render_map(cluster_df)

# Function to generate Excel file
def generate_excel(vehicle_routes, summary_df):
    file_path = '/mnt/data/optimized_routes.xlsx'
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with pd.ExcelWriter(file_path, engine='xlsxwriter') as writer:
        for vehicle, clusters in vehicle_routes.items():
            for cluster_id, data in clusters.items():
                df_cluster = df_locations.loc[data['route']]
                df_cluster.to_excel(writer, sheet_name=f'{vehicle}_Cluster_{cluster_id}', index=False)

        summary_df.to_excel(writer, sheet_name='Summary', index=False)

    st.success("Routes optimized and saved to Excel file.")
    st.download_button(label="Download Excel file", data=open(file_path, 'rb').read(), file_name="optimized_routes.xlsx")

# Load data from the uploaded file
uploaded_file = st.file_uploader("Choose an Excel file", type="xlsx")
if uploaded_file is not None:
    df_locations = pd.read_excel(uploaded_file)

    # Display the column names to verify
    st.write("Column Names:", df_locations.columns)

    # Ensure column names are as expected
    expected_columns = ['Party', 'Latitude', 'Longitude', 'Weight (KG)']
    if all(col in df_locations.columns for col in expected_columns):
        st.write("All expected columns are present.")
    else:
        st.error("One or more expected columns are missing. Please check the column names in the Excel file.")

    # Remove rows with NaN values in Latitude or Longitude
    df_locations.dropna(subset=['Latitude', 'Longitude'], inplace=True)

    # Categorize weights
    D_a = df_locations[(df_locations['Weight (KG)'] > 0) & (df_locations['Weight (KG)'] <= 2)]
    D_b = df_locations[(df_locations['Weight (KG)'] > 2) & (df_locations['Weight (KG)'] <= 10)]
    D_c = df_locations[(df_locations['Weight (KG)'] > 10) & (df_locations['Weight (KG)'] <= 200)]

    # Select scenario and vehicle costs
    scenario = st.selectbox("Select Scenario", ["Scenario 1: V1, V2, V3", "Scenario 2: V1, V2", "Scenario 3: V1, V3"])
    cost_v1 = st.number_input("Cost of V1", value=62.8156)
    cost_v2 = st.number_input("Cost of V2", value=33.0)
    cost_v3 = st.number_input("Cost of V3", value=29.0536)
    v1_capacity = st.number_input("Capacity of V1", value=64)
    v2_capacity = st.number_input("Capacity of V2", value=66)
    v3_capacity = st.number_input("Capacity of V3", value=72)

    if st.button("Optimize Load"):
        result = optimize_load(len(D_a), len(D_b), len(D_c), cost_v1, cost_v2, cost_v3, v1_capacity, v2_capacity, v3_capacity, scenario)
        st.write(f"Status: {result['Status']}")
        st.write(f"V1: {result['V1']}")
        st.write(f"V2: {result['V2']}")
        st.write(f"V3: {result['V3']}")
        st.write(f"Total Cost: {result['Total Cost']}")
        st.write(f"Deliveries assigned to V1: {result['Deliveries assigned to V1']}")
        st.write(f"Deliveries assigned to V2: {result['Deliveries assigned to V2']}")
        st.write(f"Deliveries assigned to V3: {result['Deliveries assigned to V3']}")

        vehicle_assignments = {
            "V1": D_c.index.tolist() + D_b.index[:int(result['Deliveries assigned to V1'] - len(D_c))].tolist() + D_a.index[:int(result['Deliveries assigned to V1'] - len(D_c) - len(D_b.index[:int(result['Deliveries assigned to V1'] - len(D_c))]))].tolist(),
            "V2": D_b.index[int(result['Deliveries assigned to V1'] - len(D_c)):].tolist() + D_a.index[int(result['Deliveries assigned to V1'] - len(D_c) - len(D_b.index[:int(result['Deliveries assigned to V1'] - len(D_c))])):int(result['Deliveries assigned to V1'] - len(D_c) - len(D_b.index[:int(result['Deliveries assigned to V1'] - len(D_c))]) + result['Deliveries assigned to V2'] - len(D_b.index[int(result['Deliveries assigned to V1'] - len(D_c)):]))].tolist(),
            "V3": D_a.index[int(result['Deliveries assigned to V1'] - len(D_c) - len(D_b.index[:int(result['Deliveries assigned to V1'] - len(D_c))]) + result['Deliveries assigned to V2'] - len(D_b.index[int(result['Deliveries assigned to V1'] - len(D_c)):])):].tolist()
        }

        st.write("Vehicle Assignments:", vehicle_assignments)

        if st.button("Generate Routes"):
            vehicle_routes, summary_df = generate_routes(vehicle_assignments, df_locations)
            st.write("Routes Generated")
            st.dataframe(summary_df)

            generate_excel(vehicle_routes, summary_df)
            render_cluster_maps(df_locations)
