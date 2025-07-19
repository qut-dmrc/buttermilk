
from buttermilk._core.log import logger  # noqa

from buttermilk import buttermilk as bm  # Global Buttermilk instance

# Get recent Jobs from BQ
def get_recent_runs(max_n=50):
    """Get recent flow runs from BigQuery.

    Args:
        max_n (int, optional): The maximum number of runs to return. Defaults to 50.

    Returns:
        pandas.DataFrame: A DataFrame containing the recent flow runs.
    """
    sql = """SELECT * FROM `dmrc-analysis.toxicity.flow`
    WHERE TIMESTAMP_TRUNC(timestamp, DAY) >= TIMESTAMP(DATE_SUB(CURRENT_DATE(), INTERVAL 30 DAY))
    ORDER BY TIMESTAMP DESC"""

    rows = bm.run_query(sql, return_df=True)
    return rows
