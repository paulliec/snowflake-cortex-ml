-- Attrition ML: database and medallion schema setup
-- Run once to initialize Snowflake objects

CREATE DATABASE IF NOT EXISTS ATTRITION_ML;
USE DATABASE ATTRITION_ML;

-- ============================================================
-- BRONZE — raw CSV load, everything VARCHAR
-- ============================================================
CREATE SCHEMA IF NOT EXISTS BRONZE;

CREATE OR REPLACE TABLE BRONZE.EMPLOYEE_RAW (
    EMPLOYEE_ID         VARCHAR,
    EMPLOYEE_NAME       VARCHAR,
    ROLE                VARCHAR,
    DEPARTMENT          VARCHAR,
    REGION              VARCHAR,
    TENURE_YEARS        VARCHAR,
    SALARY              VARCHAR,
    PERFORMANCE_SCORE   VARCHAR,
    MANAGER_RATING      VARCHAR,
    PROMOTIONS_LAST_3_YEARS VARCHAR,
    FLIGHT_HOURS_YTD    VARCHAR,
    OVERTIME_HOURS_MONTHLY VARCHAR,
    DAYS_SINCE_LAST_RAISE VARCHAR,
    TEAM_SIZE           VARCHAR,
    REMOTE_ELIGIBLE     VARCHAR,
    HIRE_DATE           VARCHAR,
    TERMINATION_DATE    VARCHAR,
    CHURNED             VARCHAR,
    EXIT_SURVEY_TEXT    VARCHAR,
    EXIT_SURVEY_SENTIMENT VARCHAR,
    LOADED_AT           TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
);

-- ============================================================
-- SILVER — typed, cleaned, sentiment-enriched
-- ============================================================
CREATE SCHEMA IF NOT EXISTS SILVER;

CREATE OR REPLACE TABLE SILVER.EMPLOYEE_CLEANSED (
    EMPLOYEE_ID             VARCHAR(20),
    EMPLOYEE_NAME           VARCHAR(200),
    ROLE                    VARCHAR(50),
    DEPARTMENT              VARCHAR(100),
    REGION                  VARCHAR(50),
    TENURE_YEARS            FLOAT,
    SALARY                  NUMBER(10,0),
    PERFORMANCE_SCORE       NUMBER(1,0),
    MANAGER_RATING          NUMBER(1,0),
    PROMOTIONS_LAST_3_YEARS NUMBER(1,0),
    FLIGHT_HOURS_YTD        NUMBER(6,0),
    OVERTIME_HOURS_MONTHLY  FLOAT,
    DAYS_SINCE_LAST_RAISE   NUMBER(5,0),
    TEAM_SIZE               NUMBER(3,0),
    REMOTE_ELIGIBLE         VARCHAR(1),
    HIRE_DATE               DATE,
    TERMINATION_DATE        DATE,
    CHURNED                 VARCHAR(1),
    EXIT_SURVEY_TEXT        VARCHAR(5000),
    EXIT_SURVEY_SENTIMENT   FLOAT,
    CHURN_RISK_SCORE        FLOAT,
    LOADED_AT               TIMESTAMP_NTZ
);

-- ============================================================
-- GOLD — ML-ready feature table
-- ============================================================
CREATE SCHEMA IF NOT EXISTS GOLD;

CREATE OR REPLACE TABLE GOLD.EMPLOYEE_ML_READY (
    EMPLOYEE_ID             VARCHAR(20),
    ROLE                    VARCHAR(50),
    REGION                  VARCHAR(50),
    TENURE_YEARS            FLOAT,
    SALARY                  NUMBER(10,0),
    PERFORMANCE_SCORE       FLOAT,
    MANAGER_RATING          FLOAT,
    PROMOTIONS_LAST_3_YEARS NUMBER(1,0),
    OVERTIME_HOURS_MONTHLY  FLOAT,
    DAYS_SINCE_LAST_RAISE   NUMBER(5,0),
    TEAM_SIZE               NUMBER(3,0),
    REMOTE_ELIGIBLE         NUMBER(1,0),
    EXIT_SURVEY_SENTIMENT   FLOAT,
    CHURNED                 NUMBER(1,0),
    SPLIT                   VARCHAR(5)
);
