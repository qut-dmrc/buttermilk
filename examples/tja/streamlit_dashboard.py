import asyncio

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from hydra import compose, initialize
from plotly.subplots import make_subplots

from buttermilk import set_bm
from buttermilk._core.config_bootstrap import ConfigurationBootstrapper
from streamlit_config import DashboardConfig

# Constants
MIN_TEMPLATES_FOR_COMPARISON = 2


@st.cache_resource
def init_bm():
    """Initializes the Buttermilk instance."""
    # Load configuration
    with initialize(version_base=None, config_path="../../buttermilk/conf"):
        cfg = compose(config_name="config")

    # Create bootstrapper
    bootstrapper = ConfigurationBootstrapper(config=cfg)

    # Step 1: Bootstrap full context (ExecutionContext + Infrastructure)
    _, infrastructure = asyncio.run(bootstrapper.bootstrap_full_context())

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
config = DashboardConfig(bm)


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
    
    sql_multidimensional = """SELECT * FROM `prosocial-443205.bmdev.multidimensional_analysis`"""
    multidimensional_analysis = bm.run_query(sql_multidimensional)
    
    # Enhanced judge_scores query with data quality indicators
    sql_judge_scores = """
    SELECT *,
           -- Add data freshness indicator
           TIMESTAMP_DIFF(CURRENT_TIMESTAMP(), timestamp, HOUR) as hours_since_run
    FROM `prosocial-443205.bmdev.judge_scores`
    WHERE score_quality IS NOT NULL  -- Only include records with quality assessment
    """
    judge_scores = bm.run_query(sql_judge_scores)

    return template_performance_comparison, int_experiment_completeness, multidimensional_analysis, judge_scores

@st.cache_data
def load_dbt_quality_data():
    """
    Loads DBT data quality test results and configuration.
    """
    # Load data sufficiency results
    sql_sufficiency = """
    SELECT * FROM (
        SELECT 
            judge_criteria,
            judge_model,
            insufficiency_reason,
            records_evaluated,
            score_coverage,
            avg_runs_per_record
        FROM `prosocial-443205.bmdev.assert_data_sufficiency`
        WHERE insufficiency_reason IS NOT NULL
    )
    """
    data_sufficiency_issues = bm.run_query(sql_sufficiency)
    
    # Load experiment integrity results  
    sql_integrity = """
    SELECT * FROM `prosocial-443205.bmdev.assert_experiment_integrity`
    """
    experiment_integrity_issues = bm.run_query(sql_integrity)
    
    # Load pipeline completeness results
    sql_pipeline = """
    SELECT * FROM `prosocial-443205.bmdev.assert_complete_pipeline`
    """
    pipeline_completeness_issues = bm.run_query(sql_pipeline)
    
    return data_sufficiency_issues, experiment_integrity_issues, pipeline_completeness_issues


template_performance_df, completeness_df, multidimensional_df, judge_scores_df = load_data()

# Load DBT quality data
try:
    sufficiency_issues_df, integrity_issues_df, pipeline_issues_df = load_dbt_quality_data()
except Exception as e:
    st.warning(f"Could not load DBT quality data: {e}")
    sufficiency_issues_df = pd.DataFrame()
    integrity_issues_df = pd.DataFrame()
    pipeline_issues_df = pd.DataFrame()


# --- Sidebar Filters ---
st.sidebar.header("Dashboard Filters")

# Get available filter options
available_criteria = sorted(template_performance_df["criteria"].unique()) if not template_performance_df.empty else []
available_models = sorted(template_performance_df["model"].unique()) if not template_performance_df.empty else []
available_roles = sorted(template_performance_df["agent_role"].unique()) if not template_performance_df.empty else []

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

selected_roles = st.sidebar.multiselect(
    "Select Agent Roles",
    available_roles,
    default=available_roles
)

data_sufficiency_filter = st.sidebar.selectbox(
    "Data Sufficiency",
    ["All", "SUFFICIENT", "INSUFFICIENT"],
    index=1  # Default to SUFFICIENT
)

# Add score quality filter
score_quality_filter = st.sidebar.multiselect(
    "Score Quality",
    ["valid", "duplicate_resolved", "missing_score"],
    default=["valid", "duplicate_resolved"]  # Exclude missing scores by default
)

# Data freshness filter
max_hours_old = st.sidebar.slider(
    "Max Hours Since Run",
    min_value=1,
    max_value=168,  # 1 week
    value=72,  # 3 days default
    help="Filter out data older than this many hours"
)

# Apply filters
filtered_df = template_performance_df.copy()
if selected_criteria:
    filtered_df = filtered_df[filtered_df["criteria"].isin(selected_criteria)]
if selected_models:
    filtered_df = filtered_df[filtered_df["model"].isin(selected_models)]
if selected_roles:
    filtered_df = filtered_df[filtered_df["agent_role"].isin(selected_roles)]
if data_sufficiency_filter != "All":
    filtered_df = filtered_df[filtered_df["data_sufficiency"] == data_sufficiency_filter]

# Apply score quality and freshness filters to judge_scores_df
filtered_judge_scores = judge_scores_df.copy()
if score_quality_filter:
    filtered_judge_scores = filtered_judge_scores[filtered_judge_scores["score_quality"].isin(score_quality_filter)]
if max_hours_old:
    filtered_judge_scores = filtered_judge_scores[filtered_judge_scores["hours_since_run"] <= max_hours_old]

st.sidebar.markdown(f"**Showing {len(filtered_df)} template records**")
st.sidebar.markdown(f"**Showing {len(filtered_judge_scores)} judge scores**")

# --- Main Dashboard ---

# === 1. Overall Template Performance ===
st.header("📊 Overall Template Performance")

if not filtered_df.empty:
    # Calculate overall accuracy with proper aggregation
    overall_accuracy = (
        filtered_df.groupby("template_label")
        .agg(
            overall_accuracy=pd.NamedAgg(column="avg_accuracy", aggfunc="mean"),
            accuracy_variance=pd.NamedAgg(column="avg_accuracy", aggfunc="std"),
            total_tests=pd.NamedAgg(column="total_predictions", aggfunc="sum"),
        )
        .reset_index()
    )
    
    # Handle NaN in variance
    overall_accuracy["accuracy_variance"] = overall_accuracy["accuracy_variance"].fillna(0)

    fig_overall = px.bar(
        overall_accuracy,
        x="template_label",
        y="overall_accuracy",
        error_y="accuracy_variance",
        color="template_label",
        text="total_tests",
        labels={"template_label": "Template", "overall_accuracy": "Overall Accuracy"},
        title="Overall Template Performance with Error Bars (±1 SD)",
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    fig_overall.update_traces(texttemplate="%{text} tests", textposition="outside")
    fig_overall.update_layout(showlegend=False, yaxis=dict(range=[0, 1]))
    st.plotly_chart(fig_overall, use_container_width=True)
    
    # Summary statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Experiments", len(filtered_df))
    with col2:
        st.metric("Unique Templates", filtered_df["template_label"].nunique())
    with col3:
        st.metric("Total Predictions", int(filtered_df["total_predictions"].sum()))
else:
    st.warning("No data available for the selected filters.")

# === 2. Data Completeness & Gap Detection ===
st.header("🔍 Data Completeness & Gap Detection")

if not completeness_df.empty:
    # Apply similar filters to completeness data
    filtered_completeness = completeness_df.copy()
    if selected_criteria:
        filtered_completeness = filtered_completeness[filtered_completeness["criteria"].isin(selected_criteria)]
    if selected_models:
        filtered_completeness = filtered_completeness[filtered_completeness["model"].isin(selected_models)]
    if selected_roles:
        filtered_completeness = filtered_completeness[filtered_completeness["agent_role"].isin(selected_roles)]
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Coverage Matrix Heatmap
        if not filtered_completeness.empty:
            pivot_coverage = filtered_completeness.pivot_table(
                index="criteria",
                columns=["model", "agent_role"],
                values="actual_predictions",
                fill_value=0
            )
            
            fig_heatmap = px.imshow(
                pivot_coverage.values,
                labels=dict(x="Model × Role", y="Criteria", color="Predictions"),
                x=[f"{col[0]} - {col[1]}" for col in pivot_coverage.columns],
                y=pivot_coverage.index,
                title="Coverage Matrix: Predictions per Model-Role-Criteria",
                color_continuous_scale="Viridis"
            )
            fig_heatmap.update_xaxes(tickangle=45)
            st.plotly_chart(fig_heatmap, use_container_width=True)
    
    with col2:
        # Data Sufficiency Bar Chart
        if not filtered_df.empty:
            sufficiency_data = (
                filtered_df.groupby(["criteria", "agent_role", "data_sufficiency"])
                .agg(total_predictions=("total_predictions", "sum"))
                .reset_index()
            )
            sufficiency_data["dimension"] = sufficiency_data["criteria"] + " - " + sufficiency_data["agent_role"]
            
            fig_sufficiency = px.bar(
                sufficiency_data,
                x="dimension",
                y="total_predictions",
                color="data_sufficiency",
                title="Data Sufficiency by Dimension",
                color_discrete_map={"SUFFICIENT": "green", "INSUFFICIENT": "red"}
            )
            # Get DBT configuration values
            min_runs_threshold = config.min_stochastic_runs
            fig_sufficiency.add_hline(y=min_runs_threshold, line_dash="dash",
                                    annotation_text=f"Minimum threshold ({min_runs_threshold})",
                                    annotation_position="bottom right")
            fig_sufficiency.update_xaxes(tickangle=45)
            st.plotly_chart(fig_sufficiency, use_container_width=True)
else:
    st.warning("No completeness data available.")

# === 3. Template Comparison Charts ===
st.header("⚖️ Template Comparison Analysis")

if not filtered_df.empty:
    col1, col2 = st.columns(2)
    
    with col1:
        # Template Performance by Model
        model_performance = (
            filtered_df.groupby(["model", "template_label"])
            .agg(
                accuracy=("avg_accuracy", "mean"),
                prediction_count=("total_predictions", "mean")
            )
            .reset_index()
        )
        
        fig_by_model = px.bar(
            model_performance,
            x="model",
            y="accuracy",
            color="template_label",
            text="prediction_count",
            title="Template Performance by Model",
            labels={"accuracy": "Average Accuracy", "model": "Language Model"},
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig_by_model.update_traces(texttemplate="%{text:.0f}", textposition="inside")
        fig_by_model.update_layout(yaxis=dict(range=[0, 1]))
        st.plotly_chart(fig_by_model, use_container_width=True)
    
    with col2:
        # Template Performance by Criteria
        criteria_performance = (
            filtered_df.groupby(["criteria", "template_label"])
            .agg(
                accuracy=("avg_accuracy", "mean"),
                model_count=("model", "nunique")
            )
            .reset_index()
        )
        
        fig_by_criteria = px.bar(
            criteria_performance,
            x="criteria",
            y="accuracy",
            color="template_label",
            text="model_count",
            title="Template Performance by Criteria",
            labels={"accuracy": "Average Accuracy", "criteria": "Evaluation Criteria"},
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig_by_criteria.update_traces(texttemplate="%{text} models", textposition="inside")
        fig_by_criteria.update_xaxes(tickangle=45)
        fig_by_criteria.update_layout(yaxis=dict(range=[0, 1]))
        st.plotly_chart(fig_by_criteria, use_container_width=True)

# === 4. Lift Measurement & Statistical Validation ===
st.header("📈 Lift Measurement & Statistical Validation")

if not filtered_df.empty:
    # Filter for records with lift data
    lift_data = filtered_df[filtered_df["absolute_lift"].notna()]
    
    if not lift_data.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            # Lift Magnitude Scatter Plot
            fig_lift_scatter = px.scatter(
                lift_data,
                x="absolute_lift",
                y="relative_lift_percent",
                color="criteria",
                symbol="agent_role",
                size="statistical_confidence",
                size_max=15,
                title="Lift Magnitude Analysis",
                labels={
                    "absolute_lift": "Absolute Lift (Accuracy Difference)",
                    "relative_lift_percent": "Relative Lift (%)",
                    "criteria": "Evaluation Criteria"
                },
                hover_data=["model", "effect_size"]
            )
            fig_lift_scatter.add_vline(x=0, line_dash="dash", annotation_text="No difference")
            fig_lift_scatter.add_hline(y=0, line_dash="dash")
            st.plotly_chart(fig_lift_scatter, use_container_width=True)
        
        with col2:
            # Statistical Significance Summary
            if not multidimensional_df.empty:
                # Filter multidimensional data
                filtered_multi = multidimensional_df[
                    multidimensional_df["statistical_confidence"].isin(["HIGH_CONFIDENCE", "MEDIUM_CONFIDENCE"])
                ].copy()
                
                if not filtered_multi.empty:
                    # Sort by lift magnitude
                    filtered_multi["abs_lift"] = abs(filtered_multi["relative_lift_percent"])
                    filtered_multi = filtered_multi.sort_values("abs_lift", ascending=True)
                    
                    fig_forest = px.bar(
                        filtered_multi.tail(10),  # Top 10 by magnitude
                        x="relative_lift_percent",
                        y="dimension_value",
                        color="statistical_confidence",
                        title="Top 10 Statistically Significant Lifts",
                        labels={
                            "relative_lift_percent": "Relative Lift (%)",
                            "dimension_value": "Dimension"
                        },
                        color_discrete_map={
                            "HIGH_CONFIDENCE": "darkgreen",
                            "MEDIUM_CONFIDENCE": "lightgreen"
                        }
                    )
                    fig_forest.add_vline(x=0, line_dash="dash", annotation_text="No effect")
                    st.plotly_chart(fig_forest, use_container_width=True)
                else:
                    st.info("No statistically significant results in multidimensional analysis.")
            else:
                st.info("Multidimensional analysis data not available.")
    else:
        st.info("No lift data available for the selected filters.")

# === 5. Multi-Dimensional Analysis ===
st.header("🎯 Multi-Dimensional Analysis")

if not filtered_df.empty:
    col1, col2 = st.columns(2)
    
    with col1:
        # Performance Heatmap Matrix
        sufficient_data = filtered_df[filtered_df["data_sufficiency"] == "SUFFICIENT"]
        
        if not sufficient_data.empty:
            # Create separate heatmaps for each template
            templates = sufficient_data["template_label"].unique()
            
            if len(templates) >= MIN_TEMPLATES_FOR_COMPARISON:
                # Create subplots for template comparison
                fig_heatmap_multi = make_subplots(
                    rows=1, cols=len(templates),
                    subplot_titles=templates,
                    shared_yaxes=True
                )
                
                for i, template in enumerate(templates):
                    template_data = sufficient_data[sufficient_data["template_label"] == template]
                    pivot_perf = template_data.pivot_table(
                        index="model",
                        columns="criteria",
                        values="avg_accuracy",
                        fill_value=0
                    )
                    
                    heatmap = go.Heatmap(
                        z=pivot_perf.values,
                        x=pivot_perf.columns,
                        y=pivot_perf.index,
                        colorscale="Viridis",
                        showscale=(i == len(templates) - 1)  # Show scale only on last subplot
                    )
                    
                    fig_heatmap_multi.add_trace(heatmap, row=1, col=i + 1)
                
                fig_heatmap_multi.update_layout(
                    title="Performance Heatmap by Template",
                    height=400
                )
                st.plotly_chart(fig_heatmap_multi, use_container_width=True)
            else:
                st.info("Need at least 2 templates for comparison heatmap.")
    
    with col2:
        # Radar Chart Comparison
        if not sufficient_data.empty:
            # Prepare data for radar chart
            radar_data = (
                sufficient_data.groupby(["template_label", "model", "criteria"])
                .agg(avg_accuracy=("avg_accuracy", "mean"))
                .reset_index()
            )
            radar_data["dimension"] = radar_data["model"] + " - " + radar_data["criteria"]
            
            # Create radar chart
            fig_radar = go.Figure()
            
            for template in radar_data["template_label"].unique():
                template_radar = radar_data[radar_data["template_label"] == template]
                
                fig_radar.add_trace(go.Scatterpolar(
                    r=template_radar["avg_accuracy"],
                    theta=template_radar["dimension"],
                    fill="toself",
                    name=template,
                    line_color=px.colors.qualitative.Set2[list(radar_data["template_label"].unique()).index(template)]
                ))
            
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )
                ),
                title="Template Performance Radar Chart",
                showlegend=True
            )
            st.plotly_chart(fig_radar, use_container_width=True)
        else:
            st.info("No sufficient data for radar chart.")

# === 6. Data Quality Dashboard ===
st.header("🔍 Data Quality Dashboard")

# Show data quality alerts first
if not sufficiency_issues_df.empty or not integrity_issues_df.empty or not pipeline_issues_df.empty:
    st.warning("⚠️ Data quality issues detected. Review the tabs below.")
    
    tab1, tab2, tab3 = st.tabs(["Data Sufficiency", "Experiment Integrity", "Pipeline Completeness"])
    
    with tab1:
        if not sufficiency_issues_df.empty:
            st.subheader("🔻 Data Sufficiency Issues")
            st.dataframe(sufficiency_issues_df, use_container_width=True)
            
            # Visualize sufficiency issues
            if "score_coverage" in sufficiency_issues_df.columns:
                fig_coverage = px.bar(
                    sufficiency_issues_df,
                    x="judge_criteria",
                    y="score_coverage",
                    color="judge_model",
                    title="Score Coverage by Criteria and Model",
                    labels={"score_coverage": "Score Coverage Rate"}
                )
                fig_coverage.add_hline(y=config.min_coverage_threshold, line_dash="dash",
                                      annotation_text=f"Target: {config.min_coverage_threshold:.0%}")
                st.plotly_chart(fig_coverage, use_container_width=True)
        else:
            st.success("✅ No data sufficiency issues detected.")
    
    with tab2:
        if not integrity_issues_df.empty:
            st.subheader("⚠️ Experiment Integrity Issues")
            st.dataframe(integrity_issues_df, use_container_width=True)
            
            # Visualize integrity issues by type
            if "issue_type" in integrity_issues_df.columns:
                issue_counts = integrity_issues_df["issue_type"].value_counts()
                fig_integrity = px.pie(
                    values=issue_counts.values,
                    names=issue_counts.index,
                    title="Distribution of Integrity Issues"
                )
                st.plotly_chart(fig_integrity, use_container_width=True)
        else:
            st.success("✅ No experiment integrity issues detected.")
    
    with tab3:
        if not pipeline_issues_df.empty:
            st.subheader("🔧 Pipeline Completeness Issues")
            st.dataframe(pipeline_issues_df, use_container_width=True)
            
            # Show pipeline stage completion rates
            if "details" in pipeline_issues_df.columns:
                st.markdown("**Common Pipeline Issues:**")
                issue_summary = pipeline_issues_df["description"].value_counts().head(5)
                for issue, count in issue_summary.items():
                    st.markdown(f"- {issue}: {count} sessions")
        else:
            st.success("✅ No pipeline completeness issues detected.")
else:
    st.success("✅ All data quality checks passed!")

# Score Quality Distribution
if not filtered_judge_scores.empty and "score_quality" in filtered_judge_scores.columns:
    st.subheader("📊 Score Quality Distribution")
    
    quality_counts = filtered_judge_scores["score_quality"].value_counts()
    col1, col2 = st.columns(2)
    
    with col1:
        fig_quality = px.pie(
            values=quality_counts.values,
            names=quality_counts.index,
            title="Distribution of Score Quality",
            color_discrete_map={
                "valid": "green",
                "duplicate_resolved": "orange",
                "missing_score": "red"
            }
        )
        st.plotly_chart(fig_quality, use_container_width=True)
    
    with col2:
        st.subheader("Quality Metrics")
        total_scores = len(filtered_judge_scores)
        valid_scores = len(filtered_judge_scores[filtered_judge_scores["score_quality"] == "valid"])
        coverage_rate = valid_scores / total_scores if total_scores > 0 else 0
        
        st.metric("Total Predictions", total_scores)
        st.metric("Valid Score Rate", f"{coverage_rate:.1%}")
        st.metric("Data Freshness", f"{filtered_judge_scores['hours_since_run'].median():.1f}h median age")

# === 7. Actionable Insights ===
st.header("💡 Actionable Insights")

if not filtered_df.empty:
    col1, col2 = st.columns(2)
    
    with col1:
        # Top Performing Configurations
        st.subheader("🏆 Top Performing Configurations")
        
        sufficient_configs = filtered_df[filtered_df["data_sufficiency"] == "SUFFICIENT"].copy()
        if not sufficient_configs.empty:
            sufficient_configs["configuration"] = (
                sufficient_configs["template_label"] + " - " +
                sufficient_configs["model"] + " - " +
                sufficient_configs["criteria"]
            )
            
            top_configs = sufficient_configs.nlargest(10, "avg_accuracy")[[
                "configuration", "avg_accuracy", "total_predictions", "confidence_level"
            ]]
            
            fig_top_configs = px.bar(
                top_configs,
                y="configuration",
                x="avg_accuracy",
                color="confidence_level",
                text="total_predictions",
                title="Top 10 Configurations by Accuracy",
                orientation="h",
                color_discrete_map={
                    "HIGH": "darkgreen",
                    "MEDIUM": "orange",
                    "LOW": "red"
                }
            )
            fig_top_configs.update_traces(texttemplate="%{text} tests", textposition="inside")
            fig_top_configs.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig_top_configs, use_container_width=True)
        else:
            st.info("No sufficient data for top configurations.")
    
    with col2:
        # Biggest Improvement Opportunities
        st.subheader("📊 Improvement Opportunities")
        
        if not multidimensional_df.empty:
            improvements = multidimensional_df[
                (multidimensional_df["recommendation"].str.contains("RECOMMEND_B", na=False)) &
                (multidimensional_df["statistical_confidence"] != "LOW_CONFIDENCE")
            ].copy()
            
            if not improvements.empty:
                improvements = improvements.nlargest(10, "relative_lift_percent")[[
                    "dimension_value", "relative_lift_percent", "recommendation", "statistical_confidence"
                ]]
                
                fig_improvements = px.bar(
                    improvements,
                    y="dimension_value",
                    x="relative_lift_percent",
                    color="recommendation",
                    title="Top 10 Improvement Opportunities",
                    orientation="h",
                    labels={
                        "relative_lift_percent": "Improvement (%)",
                        "dimension_value": "Dimension"
                    }
                )
                fig_improvements.update_layout(yaxis={"categoryorder": "total ascending"})
                st.plotly_chart(fig_improvements, use_container_width=True)
            else:
                st.info("No improvement opportunities found.")
        else:
            st.info("Multidimensional analysis data not available for improvements.")

# === 7. Data Quality Indicators ===
st.header("🔍 Data Quality Indicators")

if not filtered_df.empty:
    # Variance Analysis Box Plot
    sufficient_variance = filtered_df[filtered_df["data_sufficiency"] == "SUFFICIENT"].copy()
    
    if not sufficient_variance.empty:
        sufficient_variance["template_criteria"] = (
            sufficient_variance["template_label"] + " × " + sufficient_variance["criteria"]
        )
        
        fig_variance = px.box(
            sufficient_variance,
            x="template_criteria",
            y="avg_accuracy",
            color="template_label",
            title="Accuracy Distribution by Template and Criteria",
            labels={
                "avg_accuracy": "Accuracy Score",
                "template_criteria": "Template × Criteria"
            }
        )
        fig_variance.update_xaxes(tickangle=45)
        fig_variance.update_layout(height=500)
        st.plotly_chart(fig_variance, use_container_width=True)
        
        # Summary statistics table
        st.subheader("📊 Summary Statistics")
        summary_stats = sufficient_variance.groupby("template_label").agg({
            "avg_accuracy": ["mean", "std", "min", "max", "count"],
            "total_predictions": "sum"
        }).round(3)
        
        # Flatten column names
        summary_stats.columns = ["_".join(col).strip() for col in summary_stats.columns]
        st.dataframe(summary_stats, use_container_width=True)
    else:
        st.info("No sufficient data for variance analysis.")
else:
    st.warning("No data available for the selected filters.")

# === Footer ===
st.markdown("---")
st.markdown(
    """
    **Dashboard Guide**: This dashboard implements the TJA Template Analysis Dashboard Guide recommendations.
    Use the sidebar filters to focus on specific experiments, models, or criteria.
    
    **Key Metrics Tracked**:
    - Coverage Completeness: Percentage of experiment combinations with ≥10 runs
    - Template Lift: Overall accuracy improvement between templates
    - Statistical Confidence: Reliability of performance comparisons
    - Model Consistency: Variance in performance across different models
    - Criteria Effectiveness: Which guidelines show the largest template differences
    """
)
