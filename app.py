import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler

# 1. Clean Page Setup
st.set_page_config(page_title="Unsupervised Learning: Customer Segmentation", layout="wide")
st.title("Unsupervised Learning: Customer Segmentation")
st.markdown("Dimensionality reduction and distance-based clustering analysis to discover hidden mathematical groupings in retail data.")

# Load Data Safely
try:
    customer_df = pd.read_csv("segmented_customers_list.csv")
    
    # Enforce clear identification of CustomerID column safely checking the first item
    if "CustomerID" not in customer_df.columns:
        first_col = str(customer_df.columns[0])
        if first_col.startswith("Unnamed") or first_col == "index":
            customer_df = customer_df.rename(columns={customer_df.columns[0]: "CustomerID"})
        else:
            customer_df = customer_df.reset_index().rename(columns={"index": "CustomerID"})
            
    final_report = pd.read_csv("customer_segmentation_summary.csv", index_col=0)
except FileNotFoundError:
    st.error("Missing data files. Please run your notebook export script first.")
    st.stop()

# 2. Sidebar Filters
st.sidebar.header("Segment Targeting Filters")
all_personas = ["All Personas"] + list(final_report.index)
selected_persona = st.sidebar.selectbox("Select Customer Target Group:", all_personas)

# Range filters
min_spend, max_spend = st.sidebar.slider(
    "Total Spend Range ($):",
    int(customer_df["Total_Spent"].min()),
    int(customer_df["Total_Spent"].max()),
    (int(customer_df["Total_Spent"].min()), int(customer_df["Total_Spent"].max()))
)

# Apply Filters
filtered_df = customer_df.copy()
if selected_persona != "All Personas":
    filtered_df = filtered_df[filtered_df["Customer_Persona"] == selected_persona]
filtered_df = filtered_df[(filtered_df["Total_Spent"] >= min_spend) & (filtered_df["Total_Spent"] <= max_spend)]

# 3. Target Group Performance Indicators
st.header("Target Group Analytics")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Filtered Customers", f"{len(filtered_df):,}")
with col2:
    st.metric("Average Spending", f"${filtered_df['Total_Spent'].mean():,.2f}")
with col3:
    st.metric("Average Cart Volume", f"{filtered_df['Total_Items_Bought'].mean():.2f} units")
with col4:
    # CSV Download Button
    csv_data = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Export Target List CSV",
        data=csv_data,
        file_name="targeted_customer_cohort.csv",
        mime="text/csv",
    )

st.markdown("---")

# 4. Dimensionality Reduction & Clustering Visualizations
col_left, col_right = st.columns(2)
with col_left:
    st.subheader("3D Customer Clusters (PCA Space)")
    fig_3d = px.scatter_3d(
        customer_df, x="PC1", y="PC2", z="PC3",
        color="Customer_Persona", opacity=0.6,
        title="Full Spatial Customer Universe"
    )
    st.plotly_chart(fig_3d, use_container_width=True)

with col_right:
    st.subheader("Metric Strengths Comparison")
    profile_data = final_report.drop(columns=["Customer_Count"], errors="ignore")
    scaler = MinMaxScaler()
    norm_df = pd.DataFrame(scaler.fit_transform(profile_data), columns=profile_data.columns, index=profile_data.index).reset_index()
    melted_df = norm_df.melt(id_vars="Customer_Persona", var_name="Metric", value_name="Scaled Value")
    
    fig_bar = px.bar(
        melted_df, x="Metric", y="Scaled Value", color="Customer_Persona",
        barmode="group", title="Relative Behavioral Values"
    )
    st.plotly_chart(fig_bar, use_container_width=True)

st.markdown("---")

# 5. Customer Profile Lookup
st.header("Individual Customer Dossier Lookup")
lookup_id = st.text_input("Enter Customer ID (e.g., C10002):", "").strip()

if lookup_id:
    # Safely convert column values to strings for strict matching
    customer_df["CustomerID"] = customer_df["CustomerID"].astype(str)
    user_record = customer_df[customer_df["CustomerID"].str.upper() == lookup_id.upper()]
    
    if not user_record.empty:
        st.success(f"Record found for customer {lookup_id}!")
        
        # Extract singular items using .iloc[0] to prevent numpy array crashes
        persona_val = user_record['Customer_Persona'].iloc[0]
        spend_val = float(user_record['Total_Spent'].iloc[0])
        recency_val = int(user_record['Recency'].iloc[0])
        
        # Display profile metrics cleanly
        c1, c2, c3 = st.columns(3)
        c1.markdown(f"**Assigned Persona:** {persona_val}")
        c2.markdown(f"**Total Lifetime Value:** ${spend_val:,.2f}")
        c3.markdown(f"**Days Since Last Purchase:** {recency_val} days")
    else:
        st.warning("Customer ID not found in database.")

st.markdown("---")
st.header("Filtered Customer Records Directory")
st.dataframe(filtered_df, use_container_width=True)
