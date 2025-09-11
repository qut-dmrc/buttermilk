import asyncio

import pandas as pd
import plotly.express as px
import streamlit as st
from hydra import compose, initialize

from buttermilk import set_bm
from buttermilk._core.config_bootstrap import ConfigurationBootstrapper


@st.cache_resource
def init_bm():
    """Initializes the Buttermilk instance."""
    # Load configuration
    with initialize(version_base=None, config_path="../../buttermilk/conf"):
        cfg = compose(config_name="config")

    # Create bootstrapper
    bootstrapper = ConfigurationBootstrapper(config=cfg)

    # Step 1: Bootstrap full context (ExecutionContext + Infrastructure)
    execution_context, infrastructure = asyncio.run(bootstrapper.bootstrap_full_context())

    # Step 2: Bootstrap session context using existing infrastructure
    bm = asyncio.run(
        bootstrapper.bootstrap_session_context(
            name="streamlit_dashboard",
            job="tja_template_analysis",
            infrastructure=infrastructure,  # Use existing infrastructure
        )
    )

    # Step 3: Set as global singleton (if needed)
    set_bm(bm)
    return bm


bm = init_bm()


# --- Page Configuration ---
st.set_page_config(
    page_title="TJA Template Analysis Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Title and Introduction ---
st.title("TJA Template A/B Testing Analysis")
st.markdown(
    """
This dashboard provides an analysis of the TJA template A/B testing experiment.
It visualizes data from DBT models to compare template performance, analyze data completeness,
and measure statistical significance.
"""
)


# --- Data Loading ---
@st.cache_data
def load_data():
    """
    Loads data from BigQuery using the buttermilk instance.
    """
    sql_perf = """SELECT * FROM `prosocial-443205.bmdev.template_performance_comparison`"""
    template_performance_comparison = bm.run_query(sql_perf)

    sql_completeness = """SELECT * FROM `prosocial-443205.bmdev.int_experiment_completeness`"""
    int_experiment_completeness = bm.run_query(sql_completeness)

    return template_performance_comparison, int_experiment_completeness


template_performance_df, completeness_df = load_data()


# --- Main Dashboard ---

st.header("1. Overall Template Performance")

# Calculate overall accuracy
if not template_performance_df.empty:
    overall_accuracy = template_performance_df[template_performance_df["data_sufficiency"] == "SUFFICIENT"]
    if not overall_accuracy.empty:
        overall_accuracy = (
            overall_accuracy.groupby("template_label")
            .agg(
                overall_accuracy=pd.NamedAgg(column="avg_accuracy", aggfunc="mean"),
                accuracy_variance=pd.NamedAgg(column="avg_accuracy", aggfunc="std"),
                total_tests=pd.NamedAgg(column="total_predictions", aggfunc="sum"),
            )
            .reset_index()
        )

        fig_overall = px.bar(
            overall_accuracy,
            x="template_label",
            y="overall_accuracy",
            error_y="accuracy_variance",
            color="template_label",
            labels={"template_label": "Template", "overall_accuracy": "Overall Accuracy"},
            title="Overall Template Performance (Sufficient Data)",
        )
        fig_overall.update_layout(showlegend=False)
        st.plotly_chart(fig_overall, use_container_width=True)
    else:
        st.warning("No sufficient data available for Overall Template Performance.")
else:
    st.warning("No data available for Overall Template Performance.")

st.header("2. Data Completeness & Gap Detection")
st.write("*(Charts to be implemented as per guide)*")
# Placeholder for Coverage Matrix Heatmap
# Placeholder for Data Sufficiency Bar Chart

st.header("3. Template Comparison Charts")
st.write("*(Charts to be implemented as per guide)*")
# Placeholder for Template Performance by Model
# Placeholder for Template Performance by Criteria

st.header("4. Lift Measurement & Statistical Validation")
st.write("*(Charts to be implemented as per guide)*")
# Placeholder for Lift Magnitude Scatter Plot
# Placeholder for Statistical Significance Forest Plot

st.header("5. Multi-Dimensional Analysis")
st.write("*(Charts to be implemented as per guide)*")
# Placeholder for Performance Heatmap Matrix
# Placeholder for Radar Chart Comparison

st.header("6. Actionable Insights")
st.write("*(Charts to be implemented as per guide)*")
# Placeholder for Top Performing Configurations
# Placeholder for Biggest Improvement Opportunities

# --- Sidebar ---
st.sidebar.header("Filters")
st.sidebar.write("*(Filters to be implemented)*")
