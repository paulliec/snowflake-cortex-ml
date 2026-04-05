-- Run these in a Snowflake worksheet as the same user/role the pipeline uses.
-- Confirms database access, schema context, and Cortex functions this project needs.

USE DATABASE ATTRITION_ML;
USE SCHEMA GOLD;

-- 1) Session context (should match your .env)
SELECT CURRENT_ACCOUNT() AS account, CURRENT_USER() AS user, CURRENT_ROLE() AS role,
       CURRENT_DATABASE() AS db, CURRENT_SCHEMA() AS schema, CURRENT_WAREHOUSE() AS wh;

-- 2) Cortex: sentiment (Bronze → Silver uses SNOWFLAKE.CORTEX.SENTIMENT)
SELECT SNOWFLAKE.CORTEX.SENTIMENT('voluntary resignation for family reasons') AS sentiment_sample;

-- 3) Optional: confirm you can create an ML model in this schema (may require CREATE ML MODEL / OWNERSHIP on schema)
-- If this fails with privilege errors, ask an admin to grant the needed ML privileges on SCHEMA GOLD.
-- CREATE OR REPLACE SNOWFLAKE.ML.CLASSIFICATION _PROBE_MODEL(
--     INPUT_DATA => SYSTEM$REFERENCE('TABLE', 'GOLD.EMPLOYEE_ML_READY'),
--     TARGET_COLNAME => 'CHURNED',
--     CONFIG_OBJECT => { 'ON_ERROR': 'SKIP' }
-- );
-- DROP ML MODEL IF EXISTS _PROBE_MODEL;
