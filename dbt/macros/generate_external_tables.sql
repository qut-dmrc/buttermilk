{% macro create_tja_records_external_table() %}
CREATE OR REPLACE EXTERNAL TABLE `prosocial-443205.testing.records_tja`
OPTIONS (
  format = 'PARQUET',
  uris = ['gs://prosocial-dev/data/tja_train-20250901.parquet']
);
/* Later, we might automate to a native table if necessary (maybe not?): */
/*
CREATE OR REPLACE TABLE `your_project.your_dataset.processed_table` AS
SELECT * FROM `your_project.your_dataset.external_parquet_table`
WHERE date_column >= CURRENT_DATE() - 30  -- example filter
*/

/* we could also use: 
bq load \
  --source_format=PARQUET \
  --replace \
  --autodetect \
  prosocial-443205.testing.records_tja \
  gs://prosocial-dev/data/tja_train-20250728T0202Z.parquet */
{% endmacro %}