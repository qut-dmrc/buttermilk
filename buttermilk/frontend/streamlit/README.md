# Buttermilk Streamlit Dashboard

Interactive dashboard for exploring and analyzing Buttermilk chat run outputs using pandas and Plotly visualizations.

## Setup

1. **Install UI dependencies**:
   ```bash
   uv sync --group ui
   ```

2. **Ensure database access**: Make sure your Buttermilk configuration has access to your BigQuery database.

## Running the Dashboard

### Option 1: Using the launch script
```bash
uv run python buttermilk/frontend/streamlit/run_dashboard.py
```

### Option 2: Direct Streamlit command
```bash
uv run streamlit run buttermilk/frontend/streamlit/app.py
```

### Option 3: From a Jupyter notebook
```python
import subprocess
subprocess.run(["uv", "run", "streamlit", "run", "buttermilk/frontend/streamlit/app.py"])
```

The dashboard will open in your browser at `http://localhost:8501`

## Features

### 📊 Overview Tab
- **Metrics Cards**: Total messages, active sessions, average scores, duration
- **Activity Charts**: Messages over time, activity by hour of day
- **Interactive Filters**: Date range, sessions, agents, flows, score range

### 🤖 Agent Analysis Tab
- **Message Distribution**: Pie chart showing messages by agent
- **Performance Analysis**: Box plots of score distributions by agent
- **Agent Comparison**: Side-by-side performance metrics

### 💬 Sessions Tab
- **Session Explorer**: Browse individual conversation sessions
- **Timeline Visualization**: Scatter plot of session activity over time
- **Conversation View**: Formatted chat history with timestamps and scores

### 🔍 Data Explorer Tab
- **Raw Data Access**: Browse and sort the underlying data
- **Column Information**: Data types, null counts, and statistics
- **CSV Export**: Download filtered data for external analysis

## Data Source

The dashboard queries your BigQuery database using the existing Buttermilk configuration:

```sql
SELECT 
    session_id,
    timestamp,
    flow_name,
    agent_name,
    JSON_EXTRACT(outputs, '$.content') as content,
    JSON_EXTRACT(outputs, '$.score') as score,
    JSON_EXTRACT(outputs, '$.metadata') as metadata,
    JSON_EXTRACT(agent_info, '$.model') as model,
    status,
    duration_ms
FROM `dmrc-analysis.toxicity.flow`
WHERE timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 30 DAY)
AND flow_name LIKE '%chat%'
ORDER BY timestamp DESC
LIMIT 1000
```

## Customization

### Modifying the SQL Query
Edit the `load_chat_data()` function in `app.py` to change:
- Date range (currently 30 days)
- Row limit (currently 1000)
- Filtering criteria
- Additional columns

### Adding New Visualizations
Create new functions following the pattern:
```python
def render_my_analysis(df: pd.DataFrame):
    """Render custom analysis."""
    st.subheader("My Analysis")
    
    # Your analysis code here
    fig = px.scatter(df, x='timestamp', y='score')
    st.plotly_chart(fig, use_container_width=True)
```

Then add to the main tabs in `main()`.

### Performance Optimization
- Adjust cache TTL in `@st.cache_data(ttl=300)` decorators
- Modify row limits for large datasets
- Add additional filters to reduce data volume

## Jupyter Integration

You can also use the analysis functions directly in Jupyter notebooks:

```python
from buttermilk.frontend.streamlit.app import load_chat_data, render_overview_metrics
import streamlit as st

# Load data
df = load_chat_data(days_back=7, limit=500)

# Use in notebook context (without Streamlit UI)
print(f"Loaded {len(df)} records")
print(f"Date range: {df['date'].min()} to {df['date'].max()}")
```

## Troubleshooting

### Database Connection Issues
- Verify your Google Cloud credentials are set up
- Check that the BigQuery dataset/table exists
- Ensure the Buttermilk `bm` instance is properly configured

### Missing Data
- Check the SQL query matches your table schema
- Verify flow names contain 'chat'
- Adjust date range if data is older

### Performance Issues
- Reduce the row limit in `load_chat_data()`
- Increase cache TTL for slower-changing data
- Filter data more aggressively before visualization

## Dependencies

The dashboard requires these additional packages (installed with `uv sync --group ui`):
- `streamlit>=1.28.0` - Web dashboard framework
- `plotly>=5.17.0` - Interactive visualizations  
- `altair>=5.0.0` - Alternative plotting library

Existing Buttermilk dependencies used:
- `pandas` - Data manipulation
- `buttermilk` - Database access via `bm.run_query()`