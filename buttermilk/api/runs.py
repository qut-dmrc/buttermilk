import pydantic
import pandas as pd
from buttermilk import BM
from buttermilk.runner._runner_types import Job

bm = BM()
logger = bm.logger

# Get recent Jobs from BQ
def get_recent_runs(max_n=50):
    sql = f"""SELECT * FROM `dmrc-analysis.toxicity.step_extracted` 
    WHERE TIMESTAMP_TRUNC(timestamp, DAY) >= TIMESTAMP(DATE_SUB(CURRENT_DATE(), INTERVAL 30 DAY)) 
    ORDER BY TIMESTAMP DESC"""

    rows = bm.run_query(sql, df=False)

    results = [ Job(**row) for row in rows]

    return results

