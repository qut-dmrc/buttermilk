{% macro create_dataset(dataset_name) %}
  {{ log("Creating dataset " ~ dataset_name, info=True) }}
  {% set sql %}
    CREATE SCHEMA IF NOT EXISTS {{ dataset_name }};
  {% endset %}
  {% do run_query(sql) %}
  {{ log("Successfully created dataset " ~ dataset_name, info=True) }}
{% endmacro %}
