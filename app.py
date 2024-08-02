import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from geopy.distance import great_circle
import pulp
import streamlit as st
import requests
from io import BytesIO
import os

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Define the Google Maps API key
google_maps_api_key = os.getenv("GOOGLE_MAPS_API_KEY")

# Streamlit app title
st.title("Delivery Optimization and Route Planning")

# Upload the delivery data file
uploaded_file = st.file_uploader("Choose an Excel file with delivery data", type="xlsx")

if uploaded_file is not None:
    # Load data from the provided file
    df_locations = pd.read_excel(uploaded_file)

    # Display the column names to verify
    st.write("Column Names:", df_locations.columns)

    # Check the first few rows of the DataFrame
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

    # Select the scenario
    scenarios = ["Scenario 1: V1, V2, V3", "Scenario 2: V1, V2", "Scenario 3: V1, V3"]
    selected_scenario = st.selectbox("Select Scenario", scenarios)

    # Define the load optimization function
    def optimize_load(D_a_count, D_b_count, D_c_count, cost_v1, cost_v2, cost_v3, v1_capacity, v2_capacity, v3_capacity, scenario):
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

        if scenario == "Scenario 1: V1, V2, V3":
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
        elif scenario == "Scenario 2: V1, V2":
            lp_problem += cost_v1 * V1 + cost_v2 * V2, "Total Cost"
            lp_problem += A1 + A2 == D_a_count, "Total_Deliveries_A_Constraint"
            lp_problem += B1 + B2 == D_b_count, "Total_Deliveries_B_Constraint"
            lp_problem += C1 == D_c_count, "Total_Deliveries_C_Constraint"
            lp_problem += v1_capacity * V1 >= C1 + B1 + A1, "V1_Capacity_Constraint"
            lp_problem += v2_capacity * V2 >= B2 + A2, "V2_Capacity_Constraint"
            lp_problem += C1 == D_c_count, "Assign_C_To_V1"
            lp_problem += B1 <= v1_capacity * V1 - C1, "Assign_B_To_V1"
            lp_problem += B2 == D_b_count - B1, "Assign_Remaining_B_To_V2"
            lp_problem += A1 <= v1_capacity * V1 - C1 - B1, "Assign_A_To_V1"
            lp_problem += A2 == D_a_count - A1, "Assign_Remaining_A_To_V2"
        elif scenario == "Scenario 3: V1, V3":
            lp_problem += cost_v1 * V1 + cost_v3 * V3, "Total Cost"
            lp_problem += A1 + A3 == D_a_count, "Total_Deliveries_A_Constraint"
            lp_problem += B1 == D_b_count, "Total_Deliveries_B_Constraint"
            lp_problem += C1 == D_c_count, "Total_Deliveries_C_Constraint"
            lp_problem += v1_capacity * V1 >= C1 + B1 + A1, "V1_Capacity_Constraint"
            lp_problem += v3_capacity * V3 >= A3, "V3_Capacity_Constraint"
            lp_problem += C1 == D_c_count, "Assign_C_To_V1"
            lp_problem += B1 <= v1_capacity * V1 - C1, "Assign_B_To_V1"
            lp_problem += A1 <= v1_capacity * V1 - C1 - B1, "Assign_A_To_V1"
            lp_problem += A3 == D_a_count - A1, "Assign_Remaining_A_To_V3"
        lp_problem.solve()

        return {
            "Status": pulp.LpStatus[lp_problem.status],
            "V1": pulp.value(V1),
            "V2": pulp.value(V2) if scenario != "Scenario 3: V1, V3" else None,
            "V3": pulp.value(V3) if scenario != "Scenario 2: V1, V2" else None,
            "Total Cost": pulp.value(lp_problem.objective),
            "Deliveries assigned to V1": pulp.value(C1 + B1 + A1),
            "Deliveries assigned to V2": pulp.value(B2 + A2) if scenario != "Scenario 3: V1, V3" else None,
            "Deliveries assigned to V3": pulp.value(A3) if scenario != "Scenario 2: V1, V2" else None
        }

    # Define button for load optimization
    if st.button("Optimize Load"):
        # Example Load Optimization with extracted data
        cost_v1 = 62.8156
        cost_v2 = 33.0
        cost_v3 = 29.0536
        v1_capacity = 64
        v2_capacity = 66
        v3_capacity = 72

        # Optimize load
        result = optimize_load(len(D_a), len(D_b), len(D_c), cost_v1, cost_v2, cost_v3, v1_capacity, v2_capacity, v3_capacity, selected_scenario)
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
            "V2": D_b.index[int(result['Deliveries assigned to V1'] - len(D_c)):].tolist() + D_a.index[int(result['Deliveries assigned to V1'] - len(D_c) - len(D_b.index[:int(result['Deliveries assigned to V1'] - len(D_c))])):int(result['Deliveries assigned to V1'] - len(D_c) - len(D_b.index[:int(result['Deliveries assigned to V1'] - len(D_c))]) + result['Deliveries assigned to V2'] - len(D_b.index[int(result['Deliveries assigned to V1'] - len(D_c)):]))].tolist() if selected_scenario != "Scenario 3: V1, V3" else [],
            "V3": D_a.index[int(result['Deliveries assigned to V1'] - len(D_c) - len(D_b.index[:int(result['Deliveries assigned to V1'] - len(D_c))]) + result['Deliveries assigned to V2'] - len(D_b.index[int(result['Deliveries assigned to V1'] - len(D_c)):])):].tolist() if selected_scenario != "Scenario 2: V1, V2" else []
        }

        st.write("Vehicle Assignments:", vehicle_assignments)

    # Define function to calculate distance matrix
    def calculate_distance_matrix(df):
        coords = df[['Latitude', 'Longitude']].values
        distance_matrix = np.zeros((len(coords), len(coords)))
        for i, coord1 in enumerate(coords):
            for j, coord2 in enumerate(coords):
                distance_matrix[i][j] = great_circle(coord1, coord2).kilometers
        return distance_matrix

    # Define function to generate routes
    def generate_routes(vehicle_assignments, df_locations):
        vehicle_routes = {}
        summary_data = []

        for vehicle, indices in vehicle_assignments.items():
            if not indices:
                continue
            df_vehicle = df_locations.loc[indices]
            distance_matrix = calculate_distance_matrix(df_vehicle)

            # Perform DBSCAN clustering
            db = DBSCAN(eps=0.5, min_samples=1, metric='precomputed')
            db.fit(distance_matrix)
            labels = db.labels_
            df_vehicle['Cluster'] = labels

            # Store the routes and summary data
            for cluster in np.unique(labels):
                cluster_df = df_vehicle[df_vehicle['Cluster'] == cluster]
                vehicle_routes[f'{vehicle}_Cluster_{cluster}'] = cluster_df
                summary_data.append({
                    'Cluster': f'{vehicle}_Cluster_{cluster}',
                    'Vehicle': vehicle,
                    'Num_Shops': len(cluster_df),
                    'Total_Distance': cluster_df['Distance'].sum(),
                    'Centroid_Lat': cluster_df['Latitude'].mean(),
                    'Centroid_Lon': cluster_df['Longitude'].mean()
                })

        summary_df = pd.DataFrame(summary_data)
        return vehicle_routes, summary_df

    # Generate maps and provide download link
    def generate_excel(vehicle_routes, summary_df):
        file_path = '/mnt/data/optimized_routes.xlsx'
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with pd.ExcelWriter(file_path, engine='xlsxwriter') as writer:
            for route_name, route_df in vehicle_routes.items():
                route_df.to_excel(writer, sheet_name=route_name, index=False)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
        return file_path

    # Generate cluster maps
    def render_cluster_maps(df_locations):
        st.write("Generating routes and cluster maps...")
        vehicle_routes, summary_df = generate_routes(vehicle_assignments, df_locations)

        for route_name, route_df in vehicle_routes.items():
            render_map(route_df, route_name)
            st.write(f"Map saved for {route_name}")

        # Display summary
        st.write("Cluster Summary")
        st.dataframe(summary_df)

        # Generate Excel file and provide download link
        excel_file_path = generate_excel(vehicle_routes, summary_df)
        st.write(f"[Download the optimized routes and summary]({excel_file_path})")

    # Define function to render a map for a cluster
    def render_map(df, title):
        gmap = gmplot.GoogleMapPlotter(df['Latitude'].mean(), df['Longitude'].mean(), 13, apikey=google_maps_api_key)
        gmap.scatter(df['Latitude'], df['Longitude'], '#FF0000', size=40, marker=False)
        for i, row in df.iterrows():
            gmap.text(row['Latitude'], row['Longitude'], row['Party'])
        file_path = f'/mnt/data/{title}.html'
        gmap.draw(file_path)
        st.write(f"[View {title} Map](file_path)")

    # Define button for route generation
    if st.button("Generate Routes"):
        render_cluster_maps(df_locations)
