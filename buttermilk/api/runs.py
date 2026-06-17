from typing import Any

from buttermilk._core.log import logger  # noqa

from buttermilk import bm


# Get recent Jobs from BQ
def get_recent_runs(max_n: int = 50) -> Any:
    sql = """SELECT * FROM `dmrc-analysis.toxicity.flow`
    WHERE TIMESTAMP_TRUNC(timestamp, DAY) >= TIMESTAMP(DATE_SUB(CURRENT_DATE(), INTERVAL 30 DAY))
    ORDER BY TIMESTAMP DESC"""

    rows = bm.run_query(sql, return_df=True)
    return rows
