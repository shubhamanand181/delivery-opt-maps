import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from geopy.distance import great_circle
import pulp
import gmplot
import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

google_maps_api_key = os.getenv("GOOGLE_MAPS_API_KEY")
gmplot.GoogleMapPlotter.google_api_key = google_maps_api_key

st.title("Delivery Optimization and Route Mapping")

# Load data from the provided file
uploaded_file = st.file_uploader("Choose an Excel file", type=["xlsx"])
if uploaded_file:
    df_locations = pd.read_excel(uploaded_file)

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
    cost_v1 = st.number_input('Cost for Vehicle Type 1', value=62.8156)
    cost_v2 = st.number_input('Cost for Vehicle Type 2', value=33.0)
    cost_v3 = st.number_input('Cost for Vehicle Type 3', value=29.0536)
    v1_capacity = st.number_input('Capacity for Vehicle Type 1', value=64)
    v2_capacity = st.number_input('Capacity for Vehicle Type 2', value=66)
    v3_capacity = st.number_input('Capacity for Vehicle Type 3', value=72)
    scenario = st.selectbox(
        'Choose Scenario',
        ("Scenario 1: V1, V2, V3", "Scenario 2: V1, V2", "Scenario 3: V1, V3")
    )

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

        lp_problem += cost_v1 * V1 + cost_v2 * V2 + cost_v3 * V3, "Total Cost"
        lp_problem += A1 + A2 + A3 == D_a_count, "Total_Deliveries_A_Constraint"
        lp_problem += B1 + B2 == D_b_count, "Total_Deliveries_B_Constraint"
        lp_problem += C1 == D_c_count, "Total_Deliveries_C_Constraint"

        if scenario == "Scenario 1: V1, V2, V3":
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
            lp_problem += v1_capacity * V1 >= C1 + B1 + A1, "V1_Capacity_Constraint"
            lp_problem += v2_capacity * V2 >= B2 + A2, "V2_Capacity_Constraint"
            lp_problem += C1 == D_c_count, "Assign_C_To_V1"
            lp_problem += B1 <= v1_capacity * V1 - C1, "Assign_B_To_V1"
            lp_problem += B2 == D_b_count - B1, "Assign_Remaining_B_To_V2"
            lp_problem += A1 <= v1_capacity * V1 - C1 - B1, "Assign_A_To_V1"
            lp_problem += A2 <= v2_capacity * V2 - B2, "Assign_A_To_V2"
        elif scenario == "Scenario 3: V1, V3":
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
            "V2": pulp.value(V2),
            "V3": pulp.value(V3),
            "Total Cost": pulp.value(lp_problem.objective),
            "Deliveries assigned to V1": pulp.value(C1 + B1 + A1),
            "Deliveries assigned to V2": pulp.value(B2 + A2),
            "Deliveries assigned to V3": pulp.value(A3)
        }

    if st.button('Optimize Load'):
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

        st.session_state['vehicle_assignments'] = vehicle_assignments

    def calculate_distance_matrix(df):
        num_locations = df.shape[0]
        distance_matrix = np.zeros((num_locations, num_locations))
        for i in range(num_locations):
            for j in range(i + 1, num_locations):
                distance = great_circle((df.iloc[i]['Latitude'], df.iloc[i]['Longitude']),
                                        (df.iloc[j]['Latitude'], df.iloc[j]['Longitude'])).kilometers
                distance_matrix[i, j] = distance
                distance_matrix[j, i] = distance
        return distance_matrix

    def generate_routes(vehicle_assignments, df):
        routes = {}
        summary_data = []

        for vehicle, assignments in vehicle_assignments.items():
            df_vehicle = df.loc[assignments].reset_index(drop=True)
            distance_matrix = calculate_distance_matrix(df_vehicle)

            db = DBSCAN(eps=0.5, min_samples=1, metric='precomputed')
            db.fit(distance_matrix)

            df_vehicle['Cluster'] = db.labels_

            for cluster in df_vehicle['Cluster'].unique():
                cluster_df = df_vehicle[df_vehicle['Cluster'] == cluster]
                cluster_df['Distance'] = 0.0  # Placeholder for distance calculation
                cluster_df.reset_index(drop=True, inplace=True)

                centroid = (cluster_df['Latitude'].mean(), cluster_df['Longitude'].mean())
                total_distance = 0
                for i in range(len(cluster_df) - 1):
                    total_distance += great_circle(
                        (cluster_df.iloc[i]['Latitude'], cluster_df.iloc[i]['Longitude']),
                        (cluster_df.iloc[i + 1]['Latitude'], cluster_df.iloc[i + 1]['Longitude'])
                    ).kilometers
                    cluster_df.at[i + 1, 'Distance'] = total_distance

                summary_data.append({
                    'Vehicle': vehicle,
                    'Cluster': cluster,
                    'Centroid Latitude': centroid[0],
                    'Centroid Longitude': centroid[1],
                    'Number of Shops': len(cluster_df),
                    'Total Distance': total_distance
                })

                route_name = f"{vehicle}_Cluster_{cluster}"
                routes[route_name] = cluster_df

        summary_df = pd.DataFrame(summary_data)
        return routes, summary_df

    def render_map(df, route_name):
        if df.empty:
            return

        gmap = gmplot.GoogleMapPlotter(df['Latitude'].mean(), df['Longitude'].mean(), 13)
        gmap.scatter(df['Latitude'], df['Longitude'], '#FF0000', size=40, marker=False)
        gmap.plot(df['Latitude'], df['Longitude'], 'cornflowerblue', edge_width=2.5)
        
        file_path = f"maps/{route_name}.html"
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        gmap.draw(file_path)
        st.write(f"Map for {route_name} saved at {file_path}")

    def render_cluster_maps(df):
        if 'vehicle_assignments' not in st.session_state:
            st.write("Please optimize the load first.")
            return

        vehicle_assignments = st.session_state['vehicle_assignments']
        vehicle_routes, summary_df = generate_routes(vehicle_assignments, df)
        st.write("Vehicle Routes Summary", summary_df)

        for route_name, route_df in vehicle_routes.items():
            render_map(route_df, route_name)

    if st.button('Generate Routes'):
        render_cluster_maps(df_locations)

    # Function to generate Google Maps link
    def generate_gmaps_link(df):
        if df.empty:
            return ""

        base_url = "https://www.google.com/maps/dir/"
        waypoints = "/".join([f"{lat},{lon}" for lat, lon in zip(df['Latitude'], df['Longitude'])])
        return f"{base_url}{waypoints}"

    # Generate links for routes
    if 'vehicle_assignments' in st.session_state:
        vehicle_assignments = st.session_state['vehicle_assignments']
        vehicle_routes, summary_df = generate_routes(vehicle_assignments, df_locations)

        for route_name, route_df in vehicle_routes.items():
            gmaps_link = generate_gmaps_link(route_df)
            st.write(f"Google Maps Link for {route_name}: [Link]({gmaps_link})")

    # Function to generate and download Excel file
    def generate_excel(vehicle_routes, summary_df):
        file_path = '/mnt/data/optimized_routes.xlsx'
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with pd.ExcelWriter(file_path, engine='xlsxwriter') as writer:
            for route_name, route_df in vehicle_routes.items():
                route_df.to_excel(writer, sheet_name=route_name[:31], index=False)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)

        return file_path

    if st.button('Download Excel'):
        if 'vehicle_assignments' in st.session_state:
            vehicle_assignments = st.session_state['vehicle_assignments']
            vehicle_routes, summary_df = generate_routes(vehicle_assignments, df_locations)
            file_path = generate_excel(vehicle_routes, summary_df)
            st.write(f"[Download Excel]({file_path})")
