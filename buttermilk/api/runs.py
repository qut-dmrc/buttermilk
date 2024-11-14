from buttermilk import BM


# Get recent Jobs from BQ
def get_recent_runs(max_n=50):
    bm = BM()
    sql = """SELECT * FROM `dmrc-analysis.toxicity.flow` 
    WHERE TIMESTAMP_TRUNC(timestamp, DAY) >= TIMESTAMP(DATE_SUB(CURRENT_DATE(), INTERVAL 30 DAY)) 
    ORDER BY TIMESTAMP DESC"""

    rows = bm.run_query(sql, df=True)
    return rows
