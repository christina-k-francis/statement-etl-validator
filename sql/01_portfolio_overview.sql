-- 01_portfolio_overview.sql
-- Portfolio composition and allocation baseline
--
-- Views created:
--   v_asset_class_allocation      — market value and % share by asset class
--   v_account_summary             — cost basis, market value, gain/loss per account
--   v_income_generating_holdings  — income-bearing holdings ranked by contribution
--   v_cost_basis_quality          — data quality flag for missing/sentinel cost basis
--
-- Sentinel values (cost_basis = 999999.99) are excluded from all calculations.
-- -----------------------------------------------------------------------------
-- Shared CTE used across multiple views in this script:
--   clean_holdings — strips sentinel cost basis values and casts columns

-- =============================================================================
-- VIEW 1: Asset Class Allocation
-- Shows each asset class's total market value and its percentage share of the
-- full portfolio. 
CREATE OR REPLACE VIEW v_asset_class_allocation AS
WITH clean_holdings AS (
    SELECT
        "Symbol/CUSIP"                                          AS symbol,
        "Name"                                                  AS name,
        "Asset Class"                                           AS asset_class,
        "Quantity"::NUMERIC                                     AS quantity,
        CASE WHEN "Cost Basis"::NUMERIC = 999999.99
             THEN NULL
             ELSE "Cost Basis"::NUMERIC
        END                                                     AS cost_basis,
        "Market Value"::NUMERIC                                 AS market_value,
        CASE WHEN "Unrealized Gain/Loss"::NUMERIC = 999999.99
             THEN NULL
             ELSE "Unrealized Gain/Loss"::NUMERIC
        END                                                     AS unrealized_gl,
        "Est. Annual Income"::NUMERIC                           AS est_annual_income,
        "Est. Yield"::NUMERIC                                   AS est_yield,
        "Account Type"                                          AS account_type,
        "Account"                                               AS account
    FROM brokerage_statement
)
SELECT
    asset_class,
    COUNT(*)                                                    AS holding_count,
    ROUND(SUM(market_value)::NUMERIC, 2)                        AS total_market_value,
    ROUND(
        100.0 * SUM(market_value) / SUM(SUM(market_value)) OVER (),
    2)                                                          AS pct_of_portfolio,
    ROUND(SUM(cost_basis)::NUMERIC, 2)                          AS total_cost_basis,
    ROUND(SUM(unrealized_gl)::NUMERIC, 2)                       AS total_unrealized_gl
FROM clean_holdings
GROUP BY asset_class
ORDER BY total_market_value DESC;


-- =============================================================================
-- VIEW 2: Account Summary with Rollup
-- Summarises cost basis, market value, and unrealized gain/loss per account,
-- with a ROLLUP grand-total row at the bottom (account = NULL in that row).
CREATE OR REPLACE VIEW v_account_summary AS
WITH clean_holdings AS (
    SELECT
        "Symbol/CUSIP"                                          AS symbol,
        "Asset Class"                                           AS asset_class,
        CASE WHEN "Cost Basis"::NUMERIC = 999999.99
             THEN NULL
             ELSE "Cost Basis"::NUMERIC
        END                                                     AS cost_basis,
        "Market Value"::NUMERIC                                 AS market_value,
        CASE WHEN "Unrealized Gain/Loss"::NUMERIC = 999999.99
             THEN NULL
             ELSE "Unrealized Gain/Loss"::NUMERIC
        END                                                     AS unrealized_gl,
        "Account Type"                                          AS account_type,
        "Account"                                               AS account
    FROM brokerage_statement
)
SELECT
    COALESCE(account, '— TOTAL PORTFOLIO —')                   AS account,
    COALESCE(account_type, '')                                  AS account_type,
    COUNT(*)                                                    AS holding_count,
    ROUND(SUM(cost_basis)::NUMERIC, 2)                          AS total_cost_basis,
    ROUND(SUM(market_value)::NUMERIC, 2)                        AS total_market_value,
    ROUND(SUM(unrealized_gl)::NUMERIC, 2)                       AS total_unrealized_gl,
    -- Gain/loss as % of cost basis — only meaningful when cost basis is known
    ROUND(
        100.0 * SUM(unrealized_gl) / NULLIF(SUM(cost_basis), 0),
    2)                                                          AS return_on_cost_pct
FROM clean_holdings
GROUP BY ROLLUP(account, account_type)
-- Suppress intermediate ROLLUP subtotals (account only, no account_type)
HAVING NOT (account IS NOT NULL AND account_type IS NULL)
ORDER BY
    account IS NULL,        -- grand total row last
    total_market_value DESC;


-- =============================================================================
-- VIEW 3: Income-Generating Holdings Ranked by Contribution
-- Filters to holdings with a known Est. Annual Income > 0, then ranks each
-- within the full portfolio by income contribution.
CREATE OR REPLACE VIEW v_income_generating_holdings AS
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
    WHERE "Est. Annual Income" IS NOT NULL
      AND "Est. Annual Income"::NUMERIC > 0
)
SELECT
    symbol,
    name,
    asset_class,
    account_type,
    account,
    ROUND(market_value, 2)                                      AS market_value,
    ROUND(est_annual_income, 2)                                 AS est_annual_income,
    ROUND(est_yield, 4)                                         AS est_yield_pct,
    -- Rank by income contribution descending across the full portfolio
    RANK() OVER (ORDER BY est_annual_income DESC)               AS income_rank,
    -- Income as % of total portfolio income
    ROUND(
        100.0 * est_annual_income / SUM(est_annual_income) OVER (),
    2)                                                          AS pct_of_total_income
FROM clean_holdings
ORDER BY income_rank;


-- =============================================================================
-- VIEW 4: Cost Basis Data Quality Flag
-- Categorises every holding's cost basis as Known, Missing, or Sentinel
-- (999999.99 placeholder values inserted by some brokerages). Sentinel values
-- distort gain/loss totals and must be identified before trusting any P&L figure.
CREATE OR REPLACE VIEW v_cost_basis_quality AS
SELECT
    "Symbol/CUSIP"                                              AS symbol,
    "Name"                                                      AS name,
    "Asset Class"                                               AS asset_class,
    "Account"                                                   AS account,
    "Cost Basis"                                                AS raw_cost_basis,
    CASE
        WHEN "Cost Basis" IS NULL                               THEN 'Missing'
        WHEN "Cost Basis"::NUMERIC = 999999.99                  THEN 'Sentinel'
        ELSE                                                         'Known'
    END                                                         AS cost_basis_status,
    "Market Value"::NUMERIC                                     AS market_value
FROM brokerage_statement
ORDER BY
    cost_basis_status,
    "Asset Class",
    "Market Value"::NUMERIC DESC;
