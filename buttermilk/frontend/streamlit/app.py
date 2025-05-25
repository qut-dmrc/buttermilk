# Importing packages: streamlit for the frontend, requests to make the api calls
import json

import httpx
import hydra
import streamlit as st
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, Field

from buttermilk._core.constants import CONFIG_PATH
from buttermilk._core.dmrc import BM, set_bm
from buttermilk._core.types import RunRequest

"""Streamlit dashboard for exploring Buttermilk chat run outputs.

This dashboard provides interactive analysis of chat sessions, agent performance,
and conversation patterns using direct database access with pandas.
"""

from datetime import datetime

import pandas as pd
import plotly.express as px

from buttermilk import get_bm
from buttermilk._core.log import logger


@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_chat_data(days_back: int = 30, limit: int = 1000) -> pd.DataFrame:
    """Load chat data from the database with caching."""
    try:
        sql = f"""
        SELECT *
        FROM `prosocial-443205.testing.judge_scored`
        WHERE timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {days_back} DAY)
        ORDER BY timestamp DESC
        LIMIT {limit}
        """
        bm = get_bm()  # Get the Buttermilk instance
        df = bm.run_query(sql, return_df=True)

        # Data preprocessing
        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df["score"] = pd.to_numeric(df["score"], errors="coerce")
            df["duration_ms"] = pd.to_numeric(df["duration_ms"], errors="coerce")
            df["hour"] = df["timestamp"].dt.hour
            df["date"] = df["timestamp"].dt.date

        return df
    except Exception as e:
        logger.error(f"Error loading chat data: {e}")
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=300)
def load_session_detail(session_id: str) -> pd.DataFrame:
    """Load detailed conversation data for a specific session."""
    try:
        sql = f"""
        SELECT *
        FROM `dmrc-analysis.toxicity.flow`
        WHERE session_id = '{session_id}'
        ORDER BY timestamp ASC
        """
        bm = get_bm()  # Get the Buttermilk instance
        df = bm.run_query(sql, return_df=True)
        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df["score"] = pd.to_numeric(df["score"], errors="coerce")

        return df
    except Exception as e:
        logger.error(f"Error loading session detail: {e}")
        st.error(f"Error loading session detail: {e}")
        return pd.DataFrame()


class MakeCalls(BaseModel):
    url: str = Field(
        default="http://localhost:8080",
        description="URL of the server. Default value is set to local host: http://localhost:8080",
    )
    _headers: str = {"Content-Type": "application/json"}

    def flow_info(self, flow: str) -> dict:
        model_info_url = self.url + f"api/flow_info/{flow}"
        models = httpx.get(url=model_info_url)
        return json.loads(models.text)

    def run_flow(
        self,
        run_request: RunRequest,
    ) -> str:
        inference_enpoint = self.url + f"api/{run_request.flow}"

        result = httpx.post(
            url=inference_enpoint,
            headers=self._headers,
            data=run_request.model_dump(),
        )
        return json.loads(result.text)


def render_sidebar_filters(df: pd.DataFrame) -> dict:
    """Render sidebar filters and return selected values."""
    st.sidebar.header("🔍 Filters")

    filters = {}

    if not df.empty:
        # Date range filter
        min_date = df["date"].min()
        max_date = df["date"].max()

        filters["date_range"] = st.sidebar.date_input(
            "📅 Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
        )

        # Session filter
        sessions = sorted(df["session_id"].unique())
        filters["sessions"] = st.sidebar.multiselect(
            "💬 Sessions",
            options=sessions,
            default=[],
            help="Select specific sessions to analyze",
        )

        # Agent filter
        agents = sorted(df["agent_name"].unique())
        filters["agents"] = st.sidebar.multiselect(
            "🤖 Agents",
            options=agents,
            default=agents,
            help="Filter by agent types",
        )

        # Flow filter
        flows = sorted(df["flow_name"].unique())
        filters["flows"] = st.sidebar.multiselect(
            "🔄 Flows",
            options=flows,
            default=flows,
            help="Filter by flow types",
        )

        # Score range filter
        if df["score"].notna().any():
            score_min, score_max = float(df["score"].min()), float(df["score"].max())
            filters["score_range"] = st.sidebar.slider(
                "📊 Score Range",
                min_value=score_min,
                max_value=score_max,
                value=(score_min, score_max),
                step=0.1,
            )

    return filters


def apply_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    """Apply selected filters to the dataframe."""
    if df.empty:
        return df

    filtered_df = df.copy()

    # Apply date filter
    if "date_range" in filters and len(filters["date_range"]) == 2:
        start_date, end_date = filters["date_range"]
        filtered_df = filtered_df[(filtered_df["date"] >= start_date) & (filtered_df["date"] <= end_date)]

    # Apply session filter
    if filters.get("sessions"):
        filtered_df = filtered_df[filtered_df["session_id"].isin(filters["sessions"])]

    # Apply agent filter
    if filters.get("agents"):
        filtered_df = filtered_df[filtered_df["agent_name"].isin(filters["agents"])]

    # Apply flow filter
    if filters.get("flows"):
        filtered_df = filtered_df[filtered_df["flow_name"].isin(filters["flows"])]

    # Apply score filter
    if "score_range" in filters and filtered_df["score"].notna().any():
        score_min, score_max = filters["score_range"]
        filtered_df = filtered_df[(filtered_df["score"] >= score_min) & (filtered_df["score"] <= score_max)]

    return filtered_df


def render_overview_metrics(df: pd.DataFrame):
    """Render overview metrics cards."""
    if df.empty:
        st.warning("No data available for the selected filters.")
        return

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="🗨️ Total Messages",
            value=f"{len(df):,}",
            delta=None,
        )

    with col2:
        unique_sessions = df["session_id"].nunique()
        st.metric(
            label="👥 Active Sessions",
            value=f"{unique_sessions:,}",
            delta=None,
        )

    with col3:
        avg_score = df["score"].mean() if df["score"].notna().any() else 0
        st.metric(
            label="📊 Avg Score",
            value=f"{avg_score:.2f}",
            delta=None,
        )

    with col4:
        avg_duration = df["duration_ms"].mean() if df["duration_ms"].notna().any() else 0
        st.metric(
            label="⏱️ Avg Duration (ms)",
            value=f"{avg_duration:.0f}",
            delta=None,
        )


def render_activity_charts(df: pd.DataFrame):
    """Render activity and trend charts."""
    if df.empty:
        return

    st.subheader("📈 Activity Analysis")

    col1, col2 = st.columns(2)

    with col1:
        # Messages over time
        daily_activity = df.groupby("date").size().reset_index()
        daily_activity.columns = ["date", "messages"]

        fig_activity = px.line(
            daily_activity,
            x="date",
            y="messages",
            title="Messages Over Time",
            labels={"messages": "Number of Messages", "date": "Date"},
        )
        fig_activity.update_layout(showlegend=False)
        st.plotly_chart(fig_activity, use_container_width=True)

    with col2:
        # Activity by hour
        hourly_activity = df.groupby("hour").size().reset_index()
        hourly_activity.columns = ["hour", "messages"]

        fig_hourly = px.bar(
            hourly_activity,
            x="hour",
            y="messages",
            title="Activity by Hour of Day",
            labels={"messages": "Number of Messages", "hour": "Hour"},
        )
        st.plotly_chart(fig_hourly, use_container_width=True)


def render_agent_analysis(df: pd.DataFrame):
    """Render agent performance analysis."""
    if df.empty:
        return

    st.subheader("🤖 Agent Analysis")

    col1, col2 = st.columns(2)

    with col1:
        # Agent message counts
        agent_counts = df["agent_name"].value_counts().reset_index()
        agent_counts.columns = ["agent", "messages"]

        fig_agents = px.pie(
            agent_counts,
            values="messages",
            names="agent",
            title="Messages by Agent",
        )
        st.plotly_chart(fig_agents, use_container_width=True)

    with col2:
        # Agent performance (score distribution)
        if df["score"].notna().any():
            fig_scores = px.box(
                df,
                x="agent_name",
                y="score",
                title="Score Distribution by Agent",
            )
            fig_scores.update_xaxes(tickangle=45)
            st.plotly_chart(fig_scores, use_container_width=True)
        else:
            st.info("No score data available for agent performance analysis.")


def render_session_explorer(df: pd.DataFrame):
    """Render session exploration interface."""
    st.subheader("💬 Session Explorer")

    if df.empty:
        st.warning("No sessions available.")
        return

    # Session selector
    sessions = sorted(df["session_id"].unique(), key=lambda x: df[df["session_id"] == x]["timestamp"].max(), reverse=True)

    selected_session = st.selectbox(
        "Select a session to explore:",
        options=sessions,
        format_func=lambda x: f"{x} ({df[df['session_id'] == x]['timestamp'].max().strftime('%Y-%m-%d %H:%M')})",
    )

    if selected_session:
        session_data = load_session_detail(selected_session)

        if not session_data.empty:
            # Session timeline
            fig_timeline = px.scatter(
                session_data,
                x="timestamp",
                y="agent_name",
                color="score",
                size="duration_ms",
                title=f"Session Timeline: {selected_session}",
                labels={"timestamp": "Time", "agent_name": "Agent"},
            )
            st.plotly_chart(fig_timeline, use_container_width=True)

            # Conversation view
            st.subheader("💭 Conversation")

            for _, row in session_data.iterrows():
                timestamp = row["timestamp"].strftime("%H:%M:%S")
                agent = row["agent_name"]
                content = row["content"] if row["content"] else "No content"
                user_input = row["user_input"] if row["user_input"] else None
                score = row["score"] if pd.notna(row["score"]) else "N/A"

                # User message (if present)
                if user_input:
                    st.markdown(
                        f"""
                    <div class="chat-message user-message">
                        <strong>👤 User ({timestamp}):</strong><br>
                        {user_input}
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )

                # Agent response
                st.markdown(
                    f"""
                <div class="chat-message agent-message">
                    <strong>🤖 {agent} ({timestamp}) - Score: {score}:</strong><br>
                    {content}
                </div>
                """,
                    unsafe_allow_html=True,
                )


def render_data_explorer(df: pd.DataFrame):
    """Render raw data exploration interface."""
    st.subheader("🔍 Data Explorer")

    if df.empty:
        st.warning("No data available.")
        return

    # Data overview
    col1, col2 = st.columns(2)

    with col1:
        st.write("**Dataset Shape:**", df.shape)
        st.write("**Date Range:**", f"{df['date'].min()} to {df['date'].max()}")

    with col2:
        st.write("**Unique Sessions:**", df["session_id"].nunique())
        st.write("**Unique Agents:**", df["agent_name"].nunique())

    # Column information
    with st.expander("📋 Column Information"):
        col_info = pd.DataFrame(
            {
                "Column": df.columns,
                "Type": df.dtypes,
                "Non-Null Count": df.count(),
                "Null Count": df.isnull().sum(),
            },
        )
        st.dataframe(col_info)

    # Sample data
    st.subheader("📊 Sample Data")

    # Display options
    col1, col2, col3 = st.columns(3)
    with col1:
        show_rows = st.number_input("Rows to display:", min_value=5, max_value=100, value=10)
    with col2:
        sort_by = st.selectbox("Sort by:", options=df.columns, index=list(df.columns).index("timestamp"))
    with col3:
        ascending = st.checkbox("Ascending", value=False)

    # Display data
    display_df = df.sort_values(sort_by, ascending=ascending).head(show_rows)
    st.dataframe(display_df, use_container_width=True)

    # Download option
    csv = df.to_csv(index=False)
    st.download_button(
        label="📥 Download filtered data as CSV",
        data=csv,
        file_name=f"buttermilk_chat_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
    )


@hydra.main(version_base="1.3", config_path=CONFIG_PATH, config_name="config")
def main(conf: DictConfig) -> None:
    """Main Streamlit application."""
    resolved_cfg_dict = OmegaConf.to_container(conf, resolve=True, throw_on_missing=True)
    bm_instance = BM(**resolved_cfg_dict["bm"])  # type: ignore # Assuming dict matches BM fields
    set_bm(bm_instance)  # Set the Buttermilk instance using the singleton pattern

    # Custom CSS for better styling
    st.markdown(
        """
    <style>
        .metric-card {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 4px solid #1f77b4;
        }
        .chat-message {
            padding: 0.5rem;
            margin: 0.25rem 0;
            border-radius: 0.5rem;
            background-color: #f8f9fa;
        }
        .agent-message {
            background-color: #e3f2fd;
            border-left: 3px solid #2196f3;
        }
        .user-message {
            background-color: #f3e5f5;
            border-left: 3px solid #9c27b0;
        }
    </style>
    """,
        unsafe_allow_html=True,
    )

    st.title("🧈 Buttermilk Chat Analysis Dashboard")
    st.markdown("Explore and analyze your chat run outputs with interactive visualizations.")

    # Load data
    with st.spinner("Loading chat data..."):
        df = load_chat_data(days_back=30, limit=1000)

    if df.empty:
        st.error("No chat data found. Please check your database connection and query.")
        return

    # Sidebar filters
    filters = render_sidebar_filters(df)

    # Apply filters
    filtered_df = apply_filters(df, filters)

    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🤖 Agent Analysis", "💬 Sessions", "🔍 Data Explorer"])

    with tab1:
        render_overview_metrics(filtered_df)
        st.divider()
        render_activity_charts(filtered_df)

    with tab2:
        render_agent_analysis(filtered_df)

    with tab3:
        render_session_explorer(filtered_df)

    with tab4:
        render_data_explorer(filtered_df)

    # Footer
    st.divider()
    st.markdown(
        """
    <div style='text-align: center; color: #666; font-size: 0.8em;'>
        🧈 Buttermilk Chat Analysis Dashboard | 
        Last updated: """
        + datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        + """
    </div>
    """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
