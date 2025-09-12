{{
  config(
    materialized='table',
    description='Time spine model for MetricFlow - provides date dimension'
  )
}}

-- Generate a time spine from 2025-01-01 to 2030-12-31 at daily granularity
{{ dbt_utils.date_spine(
    datepart="day",
    start_date="cast('2025-01-01' as date)",
    end_date="cast('2030-12-31' as date)"
   )
}}