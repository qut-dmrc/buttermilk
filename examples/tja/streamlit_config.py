"""
Configuration management for Streamlit dashboard.
Provides centralized access to DBT variables and configuration.
"""

import pandas as pd
from typing import Dict, Any, Optional

class DashboardConfig:
    """Centralized configuration for the Streamlit dashboard."""
    
    def __init__(self, bm):
        self.bm = bm
        self._dbt_vars = None
        self._load_dbt_vars()
    
    def _load_dbt_vars(self):
        """Load DBT project variables from the compiled project."""
        try:
            # Query DBT project configuration
            # In practice, this would query the compiled project or a config table
            self._dbt_vars = {
                "cutoff_date": "2025-05-01",
                "min_stochastic_runs": 10,
                "min_coverage_threshold": 0.8,
                "exclude_test_hashes": []
            }
        except Exception as e:
            # Fallback to defaults if DBT vars unavailable
            self._dbt_vars = {
                "cutoff_date": "2025-05-01",
                "min_stochastic_runs": 10,
                "min_coverage_threshold": 0.8,
                "exclude_test_hashes": []
            }
    
    @property
    def min_stochastic_runs(self) -> int:
        """Minimum number of runs required for stochastic testing."""
        return self._dbt_vars.get("min_stochastic_runs", 10)
    
    @property
    def min_coverage_threshold(self) -> float:
        """Minimum coverage threshold for data sufficiency."""
        return self._dbt_vars.get("min_coverage_threshold", 0.8)
    
    @property
    def cutoff_date(self) -> str:
        """Data cutoff date for filtering old/bad data."""
        return self._dbt_vars.get("cutoff_date", "2025-05-01")
    
    def get_metricflow_metrics(self, metric_name: str, dimensions: list = None, where: str = None) -> pd.DataFrame:
        """
        Query metrics using MetricFlow for consistent calculations.
        
        Args:
            metric_name: Name of the metric (accuracy, scorer_agreement, etc.)
            dimensions: List of dimensions to group by
            where: Optional where clause filter
        
        Returns:
            DataFrame with metric results
        """
        import subprocess
        import io
        
        try:
            # Build MetricFlow query command
            cmd = ["mf", "query", "--metrics", metric_name]
            
            if dimensions:
                # Map dashboard dimensions to MetricFlow dimension format
                mf_dimensions = []
                for dim in dimensions:
                    if dim in ["judge_model", "judge_criteria", "judge_template", "judge_role", "score_quality"]:
                        mf_dimensions.append(f"prediction__{dim}")
                    else:
                        mf_dimensions.append(dim)
                cmd.extend(["--group-by", ",".join(mf_dimensions)])
            
            if where:
                cmd.extend(["--where", where])
            
            # Execute MetricFlow query
            result = subprocess.run(cmd, cwd="/src/buttermilk/dbt", capture_output=True, text=True)
            
            if result.returncode == 0:
                # Parse CSV output from MetricFlow
                return pd.read_csv(io.StringIO(result.stdout))
            else:
                # Fallback to direct SQL query if MetricFlow fails
                return self._fallback_metric_query(metric_name, dimensions, where)
                
        except Exception as e:
            print(f"MetricFlow query failed: {e}")
            return self._fallback_metric_query(metric_name, dimensions, where)
    
    def _fallback_metric_query(self, metric_name: str, dimensions: list = None, where: str = None) -> pd.DataFrame:
        """Fallback to direct SQL queries when MetricFlow unavailable."""
        dimensions_clause = ""
        where_clause = ""
        group_by_clause = ""
        
        if dimensions:
            dimensions_clause = f", {', '.join(dimensions)}"
            group_by_clause = f"GROUP BY {', '.join(dimensions)}"
        
        if where:
            where_clause = f"AND {where}"
        
        # Fallback metric definitions
        metric_queries = {
            "accuracy": f"""
                SELECT 
                    AVG(CAST(correct AS INT64)) as accuracy
                    {dimensions_clause}
                FROM `prosocial-443205.bmdev.judge_scores`
                WHERE score_quality IN ('valid', 'duplicate_resolved') {where_clause}
                {group_by_clause}
            """,
            "coverage_rate": f"""
                SELECT 
                    AVG(CASE WHEN score_quality = 'valid' THEN 1 ELSE 0 END) as coverage_rate
                    {dimensions_clause}
                FROM `prosocial-443205.bmdev.judge_scores`
                WHERE 1=1 {where_clause}
                {group_by_clause}
            """,
            "scorer_agreement": f"""
                SELECT 
                    AVG(correctness) as scorer_agreement
                    {dimensions_clause}
                FROM `prosocial-443205.bmdev.judge_scores`
                WHERE score_quality = 'valid' AND correctness IS NOT NULL {where_clause}
                {group_by_clause}
            """,
            "prediction_volume": f"""
                SELECT 
                    COUNT(*) as prediction_volume
                    {dimensions_clause}
                FROM `prosocial-443205.bmdev.judge_scores`
                WHERE 1=1 {where_clause}
                {group_by_clause}
            """
        }
        
        if metric_name in metric_queries:
            return self.bm.run_query(metric_queries[metric_name])
        else:
            raise ValueError(f"Unknown metric: {metric_name}")
    
    # Keep the old method name for backward compatibility
    def get_dbt_metrics(self, metric_name: str, dimensions: list = None) -> pd.DataFrame:
        """Legacy method - redirects to MetricFlow."""
        return self.get_metricflow_metrics(metric_name, dimensions)
    
    def get_quality_summary(self) -> Dict[str, Any]:
        """Get overall data quality summary."""
        try:
            # Count issues from each test
            sufficiency_count = len(self.bm.run_query("""
                SELECT * FROM `prosocial-443205.bmdev.assert_data_sufficiency`
                WHERE insufficiency_reason IS NOT NULL
            """))
            
            integrity_count = len(self.bm.run_query("""
                SELECT * FROM `prosocial-443205.bmdev.assert_experiment_integrity`
            """))
            
            pipeline_count = len(self.bm.run_query("""
                SELECT * FROM `prosocial-443205.bmdev.assert_complete_pipeline`
            """))
            
            return {
                "sufficiency_issues": sufficiency_count,
                "integrity_issues": integrity_count,
                "pipeline_issues": pipeline_count,
                "total_issues": sufficiency_count + integrity_count + pipeline_count
            }
        except Exception as e:
            return {
                "sufficiency_issues": 0,
                "integrity_issues": 0,
                "pipeline_issues": 0,
                "total_issues": 0,
                "error": str(e)
            }