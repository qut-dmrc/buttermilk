import json

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# Dashboard Configuration
st.set_page_config(
    page_title="Detailed Performance Analysis Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

@st.cache_resource
def init_bm():
    """Initializes the Buttermilk instance."""
    from buttermilk.utils import cli
    
    # Simple one-liner initialization using the CLI utility
    # Path points to the buttermilk conf directory from this example location
    bm = cli.init(
        job="detailed_performance_analysis",
        name="detailed_performance_dashboard", 
        path="../../buttermilk/conf"
    )
    return bm

bm = init_bm()

# --- Title and Introduction ---
st.title("🔬 Detailed Performance Analysis Dashboard")
st.markdown("""
**Individual Agent Performance & Synth Lift Analysis**

This dashboard provides comprehensive analysis of JUDGE and SYNTHESISER agent performance, 
with detailed synth lift metrics showing the value added by the synthesis workflow.
""")

# --- Data Loading Functions ---
@st.cache_data
def load_performance_data():
    """Load performance analysis data from BigQuery."""
    
    # Individual agent performance
    sql_individual = """
    SELECT * FROM `prosocial-443205.bmdev.individual_agent_performance`
    ORDER BY agent_role, criteria, agent_model, avg_accuracy DESC
    """
    individual_performance = bm.run_query(sql_individual)
    
    # Synth lift analysis  
    sql_lift = """
    SELECT * FROM `prosocial-443205.bmdev.synth_lift_analysis`
    ORDER BY absolute_accuracy_lift DESC
    """
    synth_lift = bm.run_query(sql_lift)
    
    # Performance matrix
    sql_matrix = """
    SELECT * FROM `prosocial-443205.bmdev.performance_matrix`
    ORDER BY dimension_type, performance_rank
    """
    performance_matrix = bm.run_query(sql_matrix)
    
    # Dashboard summary
    sql_summary = """
    SELECT * FROM `prosocial-443205.bmdev.performance_dashboard_summary`
    ORDER BY summary_type, display_value_1 DESC
    """
    dashboard_summary = bm.run_query(sql_summary)
    
    return individual_performance, synth_lift, performance_matrix, dashboard_summary

# Load data
try:
    individual_perf_df, synth_lift_df, matrix_df, summary_df = load_performance_data()
    data_loaded = True
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# --- Sidebar Filters ---
st.sidebar.header("🔧 Dashboard Filters")

# Get available filter options
available_criteria = sorted(individual_perf_df["criteria"].unique()) if not individual_perf_df.empty else []
available_models = sorted(individual_perf_df["agent_model"].unique()) if not individual_perf_df.empty else []
available_experiments = sorted(individual_perf_df["experiment_name"].dropna().unique()) if not individual_perf_df.empty else []

# Filter controls
selected_criteria = st.sidebar.multiselect(
    "Select Criteria",
    available_criteria,
    default=available_criteria
)

selected_models = st.sidebar.multiselect(
    "Select Models", 
    available_models,
    default=available_models
)

selected_experiments = st.sidebar.multiselect(
    "Select Experiments",
    available_experiments,
    default=available_experiments
)

# Data quality filter
min_data_quality = st.sidebar.selectbox(
    "Minimum Data Quality",
    ["ALL", "LIMITED", "SUFFICIENT"],
    index=1
)

# Apply filters
filtered_individual = individual_perf_df.copy()
if selected_criteria:
    filtered_individual = filtered_individual[filtered_individual["criteria"].isin(selected_criteria)]
if selected_models:
    filtered_individual = filtered_individual[filtered_individual["agent_model"].isin(selected_models)]
if selected_experiments:
    filtered_individual = filtered_individual[filtered_individual["experiment_name"].isin(selected_experiments)]
if min_data_quality != "ALL":
    quality_levels = ["SUFFICIENT"] if min_data_quality == "SUFFICIENT" else ["SUFFICIENT", "LIMITED"]
    filtered_individual = filtered_individual[filtered_individual["data_sufficiency"].isin(quality_levels)]

# Filter synth lift data accordingly
filtered_lift = synth_lift_df.copy()
if selected_criteria:
    filtered_lift = filtered_lift[filtered_lift["criteria"].isin(selected_criteria)]
if selected_models:
    filtered_lift = filtered_lift[filtered_lift["agent_model"].isin(selected_models)]
if selected_experiments:
    filtered_lift = filtered_lift[filtered_lift["experiment_name"].isin(selected_experiments)]

st.sidebar.markdown(f"**Showing {len(filtered_individual)} agent performance records**")
st.sidebar.markdown(f"**Showing {len(filtered_lift)} lift comparisons**")

# === PAGE 1: EXECUTIVE SUMMARY ===
st.header("📊 Executive Summary")

if not summary_df.empty:
    # Extract key metrics from summary
    overall_metrics = summary_df[summary_df["summary_type"] == "OVERALL_METRICS"]
    lift_metrics = summary_df[summary_df["summary_type"] == "JUDGE_VS_SYNTH"]
    lift_detail = summary_df[summary_df["summary_type"] == "SYNTH_LIFT_DETAIL"]
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if not overall_metrics.empty:
            total_sessions = int(overall_metrics.iloc[0]["display_value_1"])
            st.metric("Total Sessions Analyzed", f"{total_sessions:,}")
    
    with col2:
        if not overall_metrics.empty:
            avg_accuracy = float(overall_metrics.iloc[0]["display_value_2"])
            st.metric("Overall Average Accuracy", f"{avg_accuracy:.1%}")
    
    with col3:
        if not lift_metrics.empty:
            accuracy_lift = float(lift_metrics.iloc[0]["display_value_1"])
            st.metric("Average Synth Accuracy Lift", f"{accuracy_lift:+.3f}")
    
    with col4:
        if not lift_detail.empty:
            positive_lift_pct = float(lift_detail.iloc[0]["display_value_1"])
            st.metric("Sessions with Positive Lift", f"{positive_lift_pct:.1f}%")

# === PAGE 2: INDIVIDUAL AGENT ANALYSIS ===
st.header("🤖 Individual Agent Performance Analysis")

if not filtered_individual.empty:
    # Performance comparison: JUDGE vs SYNTH
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("JUDGE Agent Performance")
        judge_data = filtered_individual[filtered_individual["agent_role"] == "JUDGE"]
        
        if not judge_data.empty:
            # Create performance distribution plot
            fig_judge = px.histogram(
                judge_data,
                x="avg_accuracy",
                color="agent_model",
                title="JUDGE Accuracy Distribution",
                labels={"avg_accuracy": "Accuracy", "count": "Number of Sessions"},
                nbins=20
            )
            fig_judge.update_layout(height=300)
            st.plotly_chart(fig_judge, use_container_width=True)
            
            # Top performers
            top_judge = judge_data.nlargest(5, "avg_accuracy")[["agent_model", "criteria", "avg_accuracy", "prediction_count"]]
            st.dataframe(top_judge, use_container_width=True)
    
    with col2:
        st.subheader("SYNTHESISER Agent Performance")
        synth_data = filtered_individual[filtered_individual["agent_role"] == "SYNTHESISER"]
        
        if not synth_data.empty:
            # Create performance distribution plot
            fig_synth = px.histogram(
                synth_data,
                x="avg_accuracy",
                color="agent_model",
                title="SYNTHESISER Accuracy Distribution",
                labels={"avg_accuracy": "Accuracy", "count": "Number of Sessions"},
                nbins=20
            )
            fig_synth.update_layout(height=300)
            st.plotly_chart(fig_synth, use_container_width=True)
            
            # Top performers
            top_synth = synth_data.nlargest(5, "avg_accuracy")[["agent_model", "criteria", "avg_accuracy", "prediction_count"]]
            st.dataframe(top_synth, use_container_width=True)

# === PAGE 3: SYNTH LIFT ANALYSIS ===
st.header("🚀 Synth Lift Analysis")

if not filtered_lift.empty:
    # Lift distribution visualization
    col1, col2 = st.columns(2)
    
    with col1:
        # Lift histogram
        fig_lift_dist = px.histogram(
            filtered_lift,
            x="absolute_accuracy_lift",
            title="Distribution of Synth Accuracy Lift",
            labels={"absolute_accuracy_lift": "Absolute Accuracy Lift", "count": "Number of Sessions"},
            nbins=15
        )
        fig_lift_dist.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="No Lift")
        st.plotly_chart(fig_lift_dist, use_container_width=True)
    
    with col2:
        # Lift by model and criteria
        if len(filtered_lift) > 1:
            fig_lift_scatter = px.scatter(
                filtered_lift,
                x="judge_avg_accuracy",
                y="synth_avg_accuracy",
                color="agent_model",
                size="judge_total_predictions",
                hover_data=["criteria", "absolute_accuracy_lift"],
                title="JUDGE vs SYNTH Accuracy Comparison",
                labels={"judge_avg_accuracy": "JUDGE Accuracy", "synth_avg_accuracy": "SYNTH Accuracy"}
            )
            # Add diagonal line for parity
            fig_lift_scatter.add_shape(
                type="line",
                x0=0, y0=0, x1=1, y1=1,
                line=dict(dash="dash", color="gray"),
            )
            st.plotly_chart(fig_lift_scatter, use_container_width=True)
    
    # Detailed lift analysis table
    st.subheader("Detailed Lift Analysis")
    lift_display = filtered_lift[[
        "agent_model", "criteria", "experiment_name",
        "judge_avg_accuracy", "synth_avg_accuracy", 
        "absolute_accuracy_lift", "relative_accuracy_lift_pct",
        "lift_category", "effect_size", "synth_value_assessment"
    ]].round(3)
    
    # Color-code by lift category
    def highlight_lift(row):
        if row["lift_category"] in ["HIGH_POSITIVE_LIFT", "MODERATE_POSITIVE_LIFT"]:
            return ["background-color: lightgreen"] * len(row)
        elif row["lift_category"] == "NEUTRAL_LIFT":
            return ["background-color: lightyellow"] * len(row)
        else:
            return ["background-color: lightcoral"] * len(row)
    
    st.dataframe(
        lift_display.style.apply(highlight_lift, axis=1),
        use_container_width=True
    )

# === PAGE 4: PERFORMANCE MATRIX ===
st.header("📈 Multi-Dimensional Performance Matrix")

if not matrix_df.empty:
    # Dimension type selector
    dimension_types = matrix_df["dimension_type"].unique()
    selected_dimension = st.selectbox("Select Performance Dimension", dimension_types)
    
    matrix_filtered = matrix_df[matrix_df["dimension_type"] == selected_dimension]
    
    if not matrix_filtered.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            # Performance heatmap
            if selected_dimension == "Model_Agent":
                # Create pivot table for heatmap
                heatmap_data = matrix_filtered.pivot_table(
                    index="dimension_1", 
                    columns="dimension_2", 
                    values="avg_accuracy", 
                    fill_value=0
                )
                
                fig_heatmap = px.imshow(
                    heatmap_data.values,
                    x=heatmap_data.columns,
                    y=heatmap_data.index,
                    title=f"Performance Heatmap: {selected_dimension}",
                    labels={"color": "Accuracy"},
                    color_continuous_scale="Viridis"
                )
                st.plotly_chart(fig_heatmap, use_container_width=True)
        
        with col2:
            # Performance ranking
            ranking_data = matrix_filtered.sort_values("performance_rank").head(10)
            
            fig_ranking = px.bar(
                ranking_data,
                x="performance_rank",
                y="primary_dimension_label",
                color="avg_accuracy",
                title=f"Top 10 Performers: {selected_dimension}",
                labels={"performance_rank": "Rank", "avg_accuracy": "Accuracy"},
                orientation="h",
                color_continuous_scale="RdYlGn"
            )
            st.plotly_chart(fig_ranking, use_container_width=True)
        
        # Detailed matrix table
        st.subheader(f"Detailed {selected_dimension} Performance Matrix")
        matrix_display = matrix_filtered[[
            "primary_dimension_label", "secondary_dimension_label",
            "avg_accuracy", "session_count", "performance_grade",
            "consistency_grade", "relative_performance"
        ]].round(3)
        st.dataframe(matrix_display, use_container_width=True)

# === PAGE 5: INSIGHTS & RECOMMENDATIONS ===
st.header("💡 Key Insights & Recommendations")

if not filtered_lift.empty and not filtered_individual.empty:
    insights_col1, insights_col2 = st.columns(2)
    
    with insights_col1:
        st.subheader("🏆 Top Insights")
        
        # Calculate key insights
        best_lift = filtered_lift.loc[filtered_lift["absolute_accuracy_lift"].idxmax()]
        worst_lift = filtered_lift.loc[filtered_lift["absolute_accuracy_lift"].idxmin()]
        
        best_judge = filtered_individual[filtered_individual["agent_role"] == "JUDGE"].loc[
            filtered_individual[filtered_individual["agent_role"] == "JUDGE"]["avg_accuracy"].idxmax()
        ] if not filtered_individual[filtered_individual["agent_role"] == "JUDGE"].empty else None
        
        best_synth = filtered_individual[filtered_individual["agent_role"] == "SYNTHESISER"].loc[
            filtered_individual[filtered_individual["agent_role"] == "SYNTHESISER"]["avg_accuracy"].idxmax()
        ] if not filtered_individual[filtered_individual["agent_role"] == "SYNTHESISER"].empty else None
        
        st.info(f"""
        **🚀 Best Synth Lift:** {best_lift['agent_model']} on {best_lift['criteria']} 
        (+{best_lift['absolute_accuracy_lift']:.3f} accuracy improvement)
        
        **⚠️ Worst Synth Performance:** {worst_lift['agent_model']} on {worst_lift['criteria']} 
        ({worst_lift['absolute_accuracy_lift']:+.3f} accuracy change)
        """)
        
        if best_judge is not None:
            st.success(f"""
            **🎯 Best JUDGE Performance:** {best_judge['agent_model']} on {best_judge['criteria']} 
            ({best_judge['avg_accuracy']:.1%} accuracy)
            """)
        
        if best_synth is not None:
            st.success(f"""
            **🧠 Best SYNTH Performance:** {best_synth['agent_model']} on {best_synth['criteria']} 
            ({best_synth['avg_accuracy']:.1%} accuracy)
            """)
    
    with insights_col2:
        st.subheader("📋 Recommendations")
        
        # Calculate recommendation metrics
        positive_lift_count = len(filtered_lift[filtered_lift["absolute_accuracy_lift"] > 0])
        total_lift_count = len(filtered_lift)
        positive_lift_rate = positive_lift_count / total_lift_count if total_lift_count > 0 else 0
        
        high_value_sessions = len(filtered_lift[filtered_lift["synth_value_assessment"] == "HIGH_VALUE"])
        
        if positive_lift_rate > 0.7:
            st.success("✅ **Synthesis workflow is highly beneficial** - use SYNTH for most tasks")
        elif positive_lift_rate > 0.5:
            st.warning("⚖️ **Mixed results** - use SYNTH selectively based on model/criteria combination")
        else:
            st.error("❌ **Synthesis shows limited benefit** - consider direct JUDGE approach")
        
        st.info(f"""
        **Synthesis Statistics:**
        - {positive_lift_rate:.1%} of sessions show positive lift
        - {high_value_sessions} sessions show high value from synthesis
        - Average lift: {filtered_lift['absolute_accuracy_lift'].mean():+.3f}
        """)

# === FOOTER ===
st.markdown("---")
st.markdown("*Dashboard powered by DBT models with session-based agent relationship analysis*")

<function_calls>
<invoke name="TodoWrite">
<parameter name="todos">[{"content": "Create individual_agent_performance DBT model", "status": "completed", "activeForm": "Creating individual_agent_performance DBT model"}, {"content": "Create synth_lift_analysis DBT model", "status": "completed", "activeForm": "Creating synth_lift_analysis DBT model"}, {"content": "Create performance_matrix DBT model", "status": "completed", "activeForm": "Creating performance_matrix DBT model"}, {"content": "Create performance_dashboard_summary DBT model", "status": "completed", "activeForm": "Creating performance_dashboard_summary DBT model"}, {"content": "Build and test all new DBT models", "status": "completed", "activeForm": "Building and testing all new DBT models"}, {"content": "Create enhanced Streamlit dashboard", "status": "completed", "activeForm": "Creating enhanced Streamlit dashboard"}]