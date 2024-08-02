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

# Scenario options for load optimization
scenario_options = ["Scenario 1: V1, V2, V3", "Scenario 2: V1, V2", "Scenario 3: V1, V3"]
selected_scenario = st.selectbox("Select a scenario for load optimization", scenario_options)

# If a file is uploaded, process it
if uploaded_file is not None:
    # Load the Excel file
    df_locations = pd.read_excel(uploaded_file)

    # Display the column names to verify
    st.write("Column Names:", df_locations.columns.tolist())

    # Check the first few rows of the DataFrame
    st.write("First few rows of the data:", df_locations.head())

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
        elif scenario == "Scenario 2: V1, V2":
            lp_problem = pulp.LpProblem("Delivery_Cost_Minimization", pulp.LpMinimize)
            V1 = pulp.LpVariable('V1', lowBound=0, cat='Integer')
            V2 = pulp.LpVariable('V2', lowBound=0, cat='Integer')

            A1 = pulp.LpVariable('A1', lowBound=0, cat='Continuous')
            B1 = pulp.LpVariable('B1', lowBound=0, cat='Continuous')
            C1 = pulp.LpVariable('C1', lowBound=0, cat='Continuous')
            A2 = pulp.LpVariable('A2', lowBound=0, cat='Continuous')
            B2 = pulp.LpVariable('B2', lowBound=0, cat='Continuous')

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
            lp_problem += A2 <= v2_capacity * V2 - B2, "Assign_A_To_V2"
            lp_problem.solve()

            return {
                "Status": pulp.LpStatus[lp_problem.status],
                "V1": pulp.value(V1),
                "V2": pulp.value(V2),
                "Total Cost": pulp.value(lp_problem.objective),
                "Deliveries assigned to V1": pulp.value(C1 + B1 + A1),
                "Deliveries assigned to V2": pulp.value(B2 + A2),
            }
        elif scenario == "Scenario 3: V1, V3":
            lp_problem = pulp.LpProblem("Delivery_Cost_Minimization", pulp.LpMinimize)
            V1 = pulp.LpVariable('V1', lowBound=0, cat='Integer')
            V3 = pulp.LpVariable('V3', lowBound=0, cat='Integer')

            A1 = pulp.LpVariable('A1', lowBound=0, cat='Continuous')
            B1 = pulp.LpVariable('B1', lowBound=0, cat='Continuous')
            C1 = pulp.LpVariable('C1', lowBound=0, cat='Continuous')
            A3 = pulp.LpVariable('A3', lowBound=0, cat='Continuous')

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
                "V3": pulp.value(V3),
                "Total Cost": pulp.value(lp_problem.objective),
                "Deliveries assigned to V1": pulp.value(C1 + B1 + A1),
                "Deliveries assigned to V3": pulp.value(A3),
            }

    # Define the load optimization button and its functionality
    if st.button("Optimize Load"):
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
        st.write(f"V1: {result.get('V1', 'N/A')}")
        st.write(f"V2: {result.get('V2', 'N/A')}")
        st.write(f"V3: {result.get('V3', 'N/A')}")
        st.write(f"Total Cost: {result['Total Cost']}")
        st.write(f"Deliveries assigned to V1: {result.get('Deliveries assigned to V1', 'N/A')}")
        st.write(f"Deliveries assigned to V2: {result.get('Deliveries assigned to V2', 'N/A')}")
        st.write(f"Deliveries assigned to V3: {result.get('Deliveries assigned to V3', 'N/A')}")

    # Ensure vehicle assignments are based on the selected scenario
    def assign_deliveries_to_vehicles(result):
        if selected_scenario == "Scenario 1: V1, V2, V3":
            return {
                "V1": D_c.index.tolist() + D_b.index[:int(result['Deliveries assigned to V1'] - len(D_c))].tolist() + D_a.index[:int(result['Deliveries assigned to V1'] - len(D_c) - len(D_b.index[:int(result['Deliveries assigned to V1'] - len(D_c))]))].tolist(),
                "V2": D_b.index[int(result['Deliveries assigned to V1'] - len(D_c)):].tolist() + D_a.index[int(result['Deliveries assigned to V1'] - len(D_c) - len(D_b.index[:int(result['Deliveries assigned to V1'] - len(D_c))])):int(result['Deliveries assigned to V1'] - len(D_c) - len(D_b.index[:int(result['Deliveries assigned to V1'] - len(D_c))]) + result['Deliveries assigned to V2'] - len(D_b.index[int(result['Deliveries assigned to V1'] - len(D_c)):]))].tolist(),
                "V3": D_a.index[int(result['Deliveries assigned to V1'] - len(D_c) - len(D_b.index[:int(result['Deliveries assigned to V1'] - len(D_c))]) + result['Deliveries assigned to V2'] - len(D_b.index[int(result['Deliveries assigned to V1'] - len(D_c)):])):].tolist()
            }
        elif selected_scenario == "Scenario 2: V1, V2":
            return {
                "V1": D_c.index.tolist() + D_b.index[:int(result['Deliveries assigned to V1'] - len(D_c))].tolist() + D_a.index[:int(result['Deliveries assigned to V1'] - len(D_c) - len(D_b.index[:int(result['Deliveries assigned to V1'] - len(D_c))]))].tolist(),
                "V2": D_b.index[int(result['Deliveries assigned to V1'] - len(D_c)):].tolist() + D_a.index[int(result['Deliveries assigned to V1'] - len(D_c) - len(D_b.index[:int(result['Deliveries assigned to V1'] - len(D_c))])):].tolist()
            }
        elif selected_scenario == "Scenario 3: V1, V3":
            return {
                "V1": D_c.index.tolist() + D_b.index[:int(result['Deliveries assigned to V1'] - len(D_c))].tolist() + D_a.index[:int(result['Deliveries assigned to V1'] - len(D_c) - len(D_b.index[:int(result['Deliveries assigned to V1'] - len(D_c))]))].tolist(),
                "V3": D_a.index[int(result['Deliveries assigned to V1'] - len(D_c) - len(D_b.index[:int(result['Deliveries assigned to V1'] - len(D_c))])):].tolist()
            }

    # Assign deliveries to vehicles based on the optimization result
    if 'result' in locals():
        vehicle_assignments = assign_deliveries_to_vehicles(result)

        st.write(f"Vehicle Assignments: {vehicle_assignments}")

        # Button to generate routes
        if st.button("Generate Routes"):
            st.write("Generating Routes...")  # Debug statement

            # Function to calculate the distance matrix
            def calculate_distance_matrix(df):
                coords = df[['Latitude', 'Longitude']].values
                return np.array([[great_circle(c1, c2).kilometers for c2 in coords] for c1 in coords])

            # Function to generate routes for each vehicle
            def generate_routes(vehicle_assignments, df_locations):
                vehicle_routes = {}
                summary_data = []

                for vehicle, indices in vehicle_assignments.items():
                    st.write(f"Processing {vehicle}...")  # Debug statement
                    df_vehicle = df_locations.loc[indices].reset_index(drop=True)
                    distance_matrix = calculate_distance_matrix(df_vehicle)

                    db = DBSCAN(eps=0.5, min_samples=1, metric='precomputed')
                    db.fit(distance_matrix)
                    labels = db.labels_
                    df_vehicle['Cluster'] = labels

                    vehicle_routes[vehicle] = {}
                    for cluster in np.unique(labels):
                        cluster_df = df_vehicle[df_vehicle['Cluster'] == cluster]
                        vehicle_routes[vehicle][f'Cluster {cluster}'] = cluster_df

                        # Save the summary data
                        centroid = cluster_df[['Latitude', 'Longitude']].mean().to_list()
                        summary_data.append({
                            'Vehicle': vehicle,
                            'Cluster': cluster,
                            'Centroid Latitude': centroid[0],
                            'Centroid Longitude': centroid[1],
                            'Number of Shops': len(cluster_df),
                            'Total Distance': cluster_df['Weight (KG)'].sum()  # Assuming 'Total Distance' means sum of weights in this context
                        })

                        # Render map for each cluster
                        render_map(cluster_df, vehicle, cluster)

                summary_df = pd.DataFrame(summary_data)
                return vehicle_routes, summary_df

            # Function to render a map
            def render_map(df, vehicle, cluster):
                map_html = f"{vehicle}_Cluster_{cluster}.html"
                map_url = f"https://www.google.com/maps/dir/?api=1&destination={df.iloc[0]['Latitude']},{df.iloc[0]['Longitude']}&waypoints="

                for i in range(1, len(df)):
                    map_url += f"{df.iloc[i]['Latitude']},{df.iloc[i]['Longitude']}|"

                map_url = map_url.strip("|")
                df['Google Maps URL'] = map_url

                st.write(f"Map saved for {vehicle} Cluster {cluster}")
                st.map(df[['Latitude', 'Longitude']])
                st.write(f"Google Maps URL: [Open in Google Maps]({map_url})")

            # Generate routes and summary
            vehicle_routes, summary_df = generate_routes(vehicle_assignments, df_locations)

            # Display the summary DataFrame
            st.write("Summary of Clusters:")
            st.dataframe(summary_df)

            # Function to generate Excel file with routes and summary
            def generate_excel(vehicle_routes, summary_df):
                file_path = "/mnt/data/optimized_routes.xlsx"
                with pd.ExcelWriter(file_path, engine='xlsxwriter') as writer:
                    for vehicle, clusters in vehicle_routes.items():
                        for cluster, df_cluster in clusters.items():
                            df_cluster.to_excel(writer, sheet_name=f"{vehicle}_{cluster}", index=False)

                    summary_df.to_excel(writer, sheet_name='Summary', index=False)

                return file_path

            # Generate Excel file with routes and summary
            excel_file_path = generate_excel(vehicle_routes, summary_df)

            # Provide download link for the Excel file
            st.write(f"[Download the optimized routes and summary]({excel_file_path})")
