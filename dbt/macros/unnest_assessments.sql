{% macro unnest_assessments(assessments_col, prefix='assessment') %}
  {# 
    Safely unnest and extract assessment data from JSON arrays
    
    Args:
      assessments_col: Column containing JSON array of assessments
      prefix: Prefix for generated column names (default: 'assessment')
    
    Returns:
      SQL expressions for unnesting assessment data
  #}
  
  ARRAY_AGG(
    {{ extract_json_safely(prefix, '$.correct', 'null', 'BOOLEAN') }} 
    IGNORE NULLS
  ) AS {{ prefix }}_correct,
  
  ARRAY_AGG(
    {{ extract_json_safely(prefix, '$.feedback', "''") }} 
    IGNORE NULLS
  ) AS {{ prefix }}_feedback,
  
  ARRAY_AGG(
    {{ extract_json_safely(prefix, '$.confidence', 'null', 'FLOAT64') }} 
    IGNORE NULLS
  ) AS {{ prefix }}_confidence
  
{% endmacro %}