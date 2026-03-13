-- =============================================================================
-- 03_income_and_yield_analysis.sql
-- Income generation and yield analysis
--
-- Views created:
--   v_weighted_avg_yield          — portfolio and per-account weighted average yield
--   v_yield_tier_distribution     — holdings bucketed into yield tiers
--   v_account_income_positioning  — income vs. growth orientation per account
--   v_projected_income_by_account — annual income totals and portfolio share
--
-- Sentinel values (cost_basis = 999999.99) are excluded from all calculations.
-- Holdings with NULL or zero market value are excluded from yield weighting.
-- =============================================================================


-- =============================================================================
-- VIEW 1: Weighted Average Yield
-- Computes SUM(market_value * est_yield) / SUM(market_value) — a market-value-
-- weighted average that correctly reflects the yield a dollar invested earns,
-- unlike a simple mean which treats a $500 and $500,000 position equally.
--
-- Uses GROUPING SETS to return both the per-account breakdown and the
-- portfolio-level total in a single query pass:
--   GROUPING SETS ((account, account_type), ())
--   The () empty set produces the portfolio-total row.
-- =============================================================================

CREATE OR REPLACE VIEW v_weighted_avg_yield AS
WITH clean_holdings AS (
    SELECT
        "Account Type"                                          AS account_type,
        "Account"                                               AS account,
        "Market Value"::NUMERIC                                 AS market_value,
        "Est. Annual Income"::NUMERIC                           AS est_annual_income,
        "Est. Yield"::NUMERIC                                   AS est_yield
    FROM brokerage_statement
    WHERE "Market Value" IS NOT NULL
      AND "Market Value"::NUMERIC > 0
      AND "Est. Yield" IS NOT NULL
      AND "Est. Yield"::NUMERIC > 0
)
SELECT
    COALESCE(account, '— PORTFOLIO TOTAL —')                   AS account,
    COALESCE(account_type, '')                                  AS account_type,
    COUNT(*)                                                    AS yielding_holdings,
    ROUND(SUM(market_value), 2)                                 AS total_market_value,
    ROUND(SUM(est_annual_income), 2)                            AS total_est_income,
    -- Weighted average yield: each holding's yield weighted by its market value
    ROUND(
        SUM(market_value * est_yield) / NULLIF(SUM(market_value), 0),
    4)                                                          AS weighted_avg_yield_pct
FROM clean_holdings
GROUP BY GROUPING SETS ((account, account_type), ())
ORDER BY
    account = '— PORTFOLIO TOTAL —',   -- total row last
    total_market_value DESC;


-- =============================================================================
-- VIEW 2: Yield Tier Distribution
-- Buckets every holding into one of four yield tiers using CASE WHEN:
--   No Yield  — est_yield IS NULL or = 0
--   Low       — 0% < yield <= 2%
--   Mid       — 2% < yield <= 4%
--   High      — yield > 4%
-- Reports count and total market value per tier, giving the analyst a fast
-- read on whether the portfolio is skewed toward high-income or growth names.
-- =============================================================================

CREATE OR REPLACE VIEW v_yield_tier_distribution AS
WITH clean_holdings AS (
    SELECT
        "Symbol/CUSIP"                                          AS symbol,
        "Name"                                                  AS name,
        "Asset Class"                                           AS asset_class,
        "Market Value"::NUMERIC                                 AS market_value,
        "Est. Annual Income"::NUMERIC                           AS est_annual_income,
        "Est. Yield"::NUMERIC                                   AS est_yield,
        "Account Type"                                          AS account_type,
        "Account"                                               AS account
    FROM brokerage_statement
    WHERE "Market Value" IS NOT NULL
),
tiered AS (
    SELECT
        *,
        CASE
            WHEN est_yield IS NULL OR est_yield = 0  THEN 'No Yield'
            WHEN est_yield <= 2                       THEN 'Low  (0–2%)'
            WHEN est_yield <= 4                       THEN 'Mid  (2–4%)'
            ELSE                                           'High (4%+)'
        END                                                     AS yield_tier,
        CASE
            WHEN est_yield IS NULL OR est_yield = 0  THEN 1
            WHEN est_yield <= 2                       THEN 2
            WHEN est_yield <= 4                       THEN 3
            ELSE                                           4
        END                                                     AS tier_sort
    FROM clean_holdings
)
SELECT
    yield_tier,
    COUNT(*)                                                    AS holding_count,
    ROUND(SUM(market_value), 2)                                 AS total_market_value,
    ROUND(
        100.0 * SUM(market_value) / SUM(SUM(market_value)) OVER (),
    2)                                                          AS pct_of_portfolio,
    ROUND(SUM(est_annual_income), 2)                            AS total_est_income,
    -- List the symbols in each tier for quick reference
    STRING_AGG(symbol, ', ' ORDER BY market_value DESC)         AS symbols
FROM tiered
GROUP BY yield_tier, tier_sort
ORDER BY tier_sort;


-- =============================================================================
-- VIEW 3: Account Income vs. Growth Positioning
-- Classifies each account as Income-Oriented, Growth-Oriented, or Balanced
-- based on what fraction of its holdings carry a positive Est. Annual Income.
-- Uses a CTE to pre-aggregate per-holding flags before the outer query
-- computes the ratio and applies the CASE WHEN classification.
--
-- Thresholds:
--   >= 60% of holdings yield income  → Income-Oriented
--   <= 30% of holdings yield income  → Growth-Oriented
--   between 30% and 60%              → Balanced
-- =============================================================================

CREATE OR REPLACE VIEW v_account_income_positioning AS
WITH holding_flags AS (
    -- Flag each holding: does it produce income?
    SELECT
        "Account"                                               AS account,
        "Account Type"                                          AS account_type,
        "Market Value"::NUMERIC                                 AS market_value,
        "Est. Annual Income"::NUMERIC                           AS est_annual_income,
        CASE
            WHEN "Est. Annual Income" IS NOT NULL
             AND "Est. Annual Income"::NUMERIC > 0              THEN 1
            ELSE                                                     0
        END                                                     AS is_income_bearing
    FROM brokerage_statement
    WHERE "Market Value" IS NOT NULL
),
account_aggregates AS (
    -- Summarise at account level
    SELECT
        account,
        account_type,
        COUNT(*)                                                AS total_holdings,
        SUM(is_income_bearing)                                  AS income_bearing_count,
        ROUND(SUM(market_value), 2)                             AS total_market_value,
        ROUND(SUM(est_annual_income), 2)                        AS total_est_income,
        ROUND(
            100.0 * SUM(is_income_bearing) / NULLIF(COUNT(*), 0),
        1)                                                      AS income_bearing_pct
    FROM holding_flags
    GROUP BY account, account_type
)
SELECT
    account,
    account_type,
    total_holdings,
    income_bearing_count,
    income_bearing_pct,
    total_market_value,
    total_est_income,
    CASE
        WHEN income_bearing_pct >= 60  THEN 'Income-Oriented'
        WHEN income_bearing_pct <= 30  THEN 'Growth-Oriented'
        ELSE                                'Balanced'
    END                                                         AS account_orientation
FROM account_aggregates
ORDER BY total_market_value DESC;


-- =============================================================================
-- VIEW 4: Projected Annual Income by Account
-- Totals Est. Annual Income per account and computes each account's share of
-- the portfolio's total projected income using SUM() OVER () as a window
-- function. Only holdings with a positive Est. Annual Income are included.
-- =============================================================================

CREATE OR REPLACE VIEW v_projected_income_by_account AS
WITH clean_holdings AS (
    SELECT
        "Account Type"                                          AS account_type,
        "Account"                                               AS account,
        "Symbol/CUSIP"                                          AS symbol,
        "Name"                                                  AS name,
        "Asset Class"                                           AS asset_class,
        "Market Value"::NUMERIC                                 AS market_value,
        "Est. Annual Income"::NUMERIC                           AS est_annual_income,
        "Est. Yield"::NUMERIC                                   AS est_yield
    FROM brokerage_statement
    WHERE "Est. Annual Income" IS NOT NULL
      AND "Est. Annual Income"::NUMERIC > 0
),
account_income AS (
    SELECT
        account_type,
        account,
        COUNT(*)                                                AS income_holding_count,
        ROUND(SUM(market_value), 2)                             AS total_market_value,
        ROUND(SUM(est_annual_income), 2)                        AS total_est_income,
        ROUND(
            SUM(market_value * est_yield) / NULLIF(SUM(market_value), 0),
        4)                                                      AS weighted_avg_yield
    FROM clean_holdings
    GROUP BY account_type, account
)
SELECT
    account,
    account_type,
    income_holding_count,
    total_market_value,
    total_est_income,
    weighted_avg_yield                                          AS weighted_avg_yield_pct,
    -- Each account's share of the portfolio's total projected income
    ROUND(
        100.0 * total_est_income / SUM(total_est_income) OVER (),
    2)                                                          AS pct_of_total_income
FROM account_income
ORDER BY total_est_income DESC;
