-- 02_gain_loss_analysis.sql
-- Unrealized P&L deep-dive and tax-treatment analysis
--
-- Views created:
--   v_return_on_cost_by_holding   — % return per holding, ranked within asset class
--   v_gain_loss_by_tax_treatment  — gains vs. losses by account type (tax exposure)
--   v_short_positions             — isolated analysis of negative-quantity holdings
--   v_gain_loss_quartiles         — NTILE(4) bucketing of holdings by return %
--
-- Sentinel values (cost_basis = 999999.99) are excluded from all calculations.


-- =============================================================================
-- VIEW 1: Return on Cost Basis by Holding
-- Computes (market_value - cost_basis) / cost_basis as return_pct for every
-- holding with a valid, non-sentinel cost basis. 
-- Uses RANK() OVER (PARTITION BY asset_class ...) to rank each holding within 
-- its asset class by return.
CREATE OR REPLACE VIEW v_return_on_cost_by_holding AS
WITH clean_holdings AS (
    SELECT
        "Symbol/CUSIP"                                          AS symbol,
        "Name"                                                  AS name,
        "Asset Class"                                           AS asset_class,
        "Quantity"::NUMERIC                                     AS quantity,
        "Cost Basis"::NUMERIC                                   AS cost_basis,
        "Market Value"::NUMERIC                                 AS market_value,
        "Unrealized Gain/Loss"::NUMERIC                         AS unrealized_gl,
        "Account Type"                                          AS account_type,
        "Account"                                               AS account
    FROM brokerage_statement
    -- Exclude missing and sentinel cost basis — return % is meaningless without it
    WHERE "Cost Basis" IS NOT NULL
      AND "Cost Basis"::NUMERIC != 999999.99
      AND "Cost Basis"::NUMERIC != 0
      -- Exclude short positions (negative quantity) — handled in v_short_positions
      AND "Quantity"::NUMERIC > 0
),
with_return AS (
    SELECT
        *,
        ROUND(
            100.0 * (market_value - cost_basis) / cost_basis,
        2)                                                      AS return_pct
    FROM clean_holdings
)
SELECT
    symbol,
    name,
    asset_class,
    account_type,
    account,
    ROUND(quantity, 4)                                          AS quantity,
    ROUND(cost_basis, 2)                                        AS cost_basis,
    ROUND(market_value, 2)                                      AS market_value,
    ROUND(unrealized_gl, 2)                                     AS unrealized_gl,
    return_pct,
    -- Rank within asset class: 1 = best performer in that class
    RANK() OVER (
        PARTITION BY asset_class
        ORDER BY return_pct DESC
    )                                                           AS rank_within_class,
    -- Rank across entire portfolio
    RANK() OVER (
        ORDER BY return_pct DESC
    )                                                           AS rank_overall
FROM with_return
ORDER BY asset_class, rank_within_class;


-- =============================================================================
-- VIEW 2: Gain/Loss by Tax Treatment
-- Separates total unrealized gains from total unrealized losses, grouped by
-- Account Type (Taxable / Tax-Deferred / Tax-Exempt).
CREATE OR REPLACE VIEW v_gain_loss_by_tax_treatment AS
WITH clean_holdings AS (
    SELECT
        "Asset Class"                                           AS asset_class,
        "Account Type"                                          AS account_type,
        "Market Value"::NUMERIC                                 AS market_value,
        CASE WHEN "Unrealized Gain/Loss"::NUMERIC = 999999.99
             THEN NULL
             ELSE "Unrealized Gain/Loss"::NUMERIC
        END                                                     AS unrealized_gl,
        CASE WHEN "Cost Basis"::NUMERIC = 999999.99
             THEN NULL
             ELSE "Cost Basis"::NUMERIC
        END                                                     AS cost_basis
    FROM brokerage_statement
    WHERE "Unrealized Gain/Loss" IS NOT NULL
)
SELECT
    account_type,
    asset_class,
    COUNT(*)                                                    AS holding_count,
    ROUND(SUM(market_value), 2)                                 AS total_market_value,
    -- Split gains and losses into separate columns before summing
    ROUND(SUM(CASE WHEN unrealized_gl > 0
                   THEN unrealized_gl ELSE 0 END), 2)           AS total_gains,
    ROUND(SUM(CASE WHEN unrealized_gl < 0
                   THEN unrealized_gl ELSE 0 END), 2)           AS total_losses,
    ROUND(SUM(unrealized_gl), 2)                                AS net_unrealized_gl,
    -- Net return as % of total cost basis for this group
    ROUND(
        100.0 * SUM(unrealized_gl) / NULLIF(SUM(cost_basis), 0),
    2)                                                          AS net_return_pct,
    -- Count of winning vs. losing positions
    COUNT(CASE WHEN unrealized_gl > 0 THEN 1 END)               AS winning_positions,
    COUNT(CASE WHEN unrealized_gl < 0 THEN 1 END)               AS losing_positions
FROM clean_holdings
GROUP BY account_type, asset_class
ORDER BY account_type, net_unrealized_gl DESC;


-- =============================================================================
-- VIEW 3: Short Positions Analysis
-- Isolates holdings where Quantity < 0 (short positions). Short positions have
-- inverted gain/loss logic — a negative market value moving further negative
-- is a gain. 
CREATE OR REPLACE VIEW v_short_positions AS
WITH short_holdings AS (
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
        "Account Type"                                          AS account_type,
        "Account"                                               AS account
    FROM brokerage_statement
    WHERE "Quantity"::NUMERIC < 0
),
with_metrics AS (
    SELECT
        *,
        -- For short positions: gain if unrealized_gl > 0 (covered below cost)
        CASE
            WHEN unrealized_gl > 0 THEN 'Gain'
            WHEN unrealized_gl < 0 THEN 'Loss'
            ELSE                        'Flat'
        END                                                     AS position_status,
        -- Return % on short: gain / abs(cost_basis)
        CASE WHEN cost_basis IS NOT NULL AND cost_basis != 0
             THEN ROUND(100.0 * unrealized_gl / ABS(cost_basis), 2)
        END                                                     AS return_pct
    FROM short_holdings
)
SELECT
    symbol,
    name,
    asset_class,
    account_type,
    account,
    ROUND(quantity, 4)                                          AS quantity,
    ROUND(cost_basis, 2)                                        AS cost_basis,
    ROUND(market_value, 2)                                      AS market_value,
    ROUND(unrealized_gl, 2)                                     AS unrealized_gl,
    return_pct,
    position_status
FROM with_metrics
ORDER BY unrealized_gl DESC;


-- =============================================================================
-- VIEW 4: Gain/Loss Quartile Distribution
-- Uses NTILE(4) to bucket all long holdings with a known cost basis into
-- quartiles by return percentage — Q1 = bottom 25% performers,
-- Q4 = top 25% performers. 
CREATE OR REPLACE VIEW v_gain_loss_quartiles AS
WITH clean_holdings AS (
    SELECT
        "Symbol/CUSIP"                                          AS symbol,
        "Name"                                                  AS name,
        "Asset Class"                                           AS asset_class,
        "Quantity"::NUMERIC                                     AS quantity,
        "Cost Basis"::NUMERIC                                   AS cost_basis,
        "Market Value"::NUMERIC                                 AS market_value,
        "Unrealized Gain/Loss"::NUMERIC                         AS unrealized_gl,
        "Account Type"                                          AS account_type,
        "Account"                                               AS account
    FROM brokerage_statement
    WHERE "Cost Basis" IS NOT NULL
      AND "Cost Basis"::NUMERIC != 999999.99
      AND "Cost Basis"::NUMERIC != 0
      AND "Quantity"::NUMERIC > 0
),
with_return AS (
    SELECT
        *,
        ROUND(100.0 * (market_value - cost_basis) / cost_basis, 2) AS return_pct,
        NTILE(4) OVER (ORDER BY
            (market_value - cost_basis) / cost_basis
        )                                                       AS quartile
    FROM clean_holdings
)
SELECT
    symbol,
    name,
    asset_class,
    account_type,
    account,
    ROUND(cost_basis, 2)                                        AS cost_basis,
    ROUND(market_value, 2)                                      AS market_value,
    ROUND(unrealized_gl, 2)                                     AS unrealized_gl,
    return_pct,
    quartile,
    CASE quartile
        WHEN 1 THEN 'Q1 — Bottom 25% (Worst Performers)'
        WHEN 2 THEN 'Q2 — Below Median'
        WHEN 3 THEN 'Q3 — Above Median'
        WHEN 4 THEN 'Q4 — Top 25% (Best Performers)'
    END                                                         AS quartile_label
FROM with_return
ORDER BY quartile, return_pct DESC;
