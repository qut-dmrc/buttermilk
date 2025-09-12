{% macro extract_json_safely(column, path, default_value="null", cast_as=none) %}
  {# 
    Safely extract JSON values with fallback handling
    
    Args:
      column: The JSON column to extract from
      path: JSON path (e.g., '$.field.subfield')
      default_value: Value to return if extraction fails (default: null)
      cast_as: Optional SQL type to cast result to (e.g., 'BOOLEAN', 'FLOAT64')
    
    Returns:
      SQL expression for safe JSON extraction
  #}
  
  {% set extract_expr %}
    COALESCE(
      JSON_VALUE({{ column }}, '{{ path }}'),
      {{ default_value }}
    )
  {% endset %}
  
  {% if cast_as %}
    SAFE_CAST({{ extract_expr }} AS {{ cast_as }})
  {% else %}
    {{ extract_expr }}
  {% endif %}
  
{% endmacro %}