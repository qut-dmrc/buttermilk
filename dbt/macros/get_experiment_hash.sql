{% macro get_experiment_hash(model_col, criteria_col, template_col, record_col) %}
  {# 
    Generate consistent experiment hash for grouping identical setups
    
    Args:
      model_col: Column containing the model name
      criteria_col: Column containing the criteria
      template_col: Column containing the template hash
      record_col: Column containing the record ID
    
    Returns:
      SQL expression that creates a stable hash for experiment grouping
  #}
  
  TO_HEX(
    SHA256(
      CONCAT(
        COALESCE({{ model_col }}, ''),
        '|',
        COALESCE({{ criteria_col }}, ''),
        '|', 
        COALESCE({{ template_col }}, ''),
        '|',
        COALESCE({{ record_col }}, '')
      )
    )
  )
  
{% endmacro %}