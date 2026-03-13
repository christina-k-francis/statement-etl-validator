"""
Helpful FXs for Postgres connection and uploads via the financial data pipeline.
"""

import os
import glob
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy_utils import database_exists, create_database


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

def get_engine(user: str, pwd: str, host: str, port: str, db: str):
    """
    Build and return a SQLAlchemy engine for the given Postgres database.
    If the target database does not yet exist, it is created automatically.

    Args:
        user: Postgres username  (from YAML postgres.user)
        pwd:  Postgres password  (from env var PSQL_PWD)
        host: Postgres host      (from YAML postgres.host)
        port: Postgres port      (from YAML postgres.port)
        db:   Target database name (from YAML postgres.db_name)

    Returns:
        SQLAlchemy Engine instance.
    """
    url = f"postgresql://{user}:{pwd}@{host}:{port}/{db}"

    if not database_exists(url):
        create_database(url)
        print(f"  Database '{db}' did not exist — created it.")

    engine = create_engine(url, pool_size=50, echo=False)
    return engine


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------

def upload_holdings(
    holdings_df: pd.DataFrame,
    engine,
    table_name: str,
    if_exists: str = "replace",
) -> None:
    """
    Upload a holdings DataFrame to a Postgres table.

    Args:
        holdings_df: Validated holdings DataFrame produced by the pipeline.
        engine:      SQLAlchemy engine from get_engine().
        table_name:  Target table name in Postgres (from YAML postgres.table_name).
        if_exists:   Pandas if_exists behaviour — 'replace' (default) drops and
                     recreates the table on every run, ensuring idempotency.

    Returns:
        None
    """
    holdings_df.to_sql(
        name=table_name,
        con=engine,
        if_exists=if_exists,
        index=False,
    )
    print(f"  Uploaded {len(holdings_df)} rows → table '{table_name}'")


# ---------------------------------------------------------------------------
# Views
# ---------------------------------------------------------------------------

def create_views(engine, sql_dir: str) -> None:
    """
    Execute all SQL view scripts in sql_dir against the connected database.

    Each .sql file is read in filename order (01_, 02_, 03_, ...) and executed
    in a single transaction. All CREATE OR REPLACE VIEW statements persist in
    Postgres and are immediately queryable after this call.

    Args:
        engine:  SQLAlchemy engine from get_engine().
        sql_dir: Path to the directory containing the .sql view scripts.
                 Expected files:
                   01_portfolio_overview.sql
                   02_gain_loss_analysis.sql
                   03_income_and_yield_analysis.sql

    Returns:
        None
    """
    sql_files = sorted(glob.glob(os.path.join(sql_dir, "*.sql")))

    if not sql_files:
        print(f"  Warning: no .sql files found in '{sql_dir}'. No views created.")
        return

    with engine.begin() as conn:
        for sql_path in sql_files:
            script_name = os.path.basename(sql_path)
            with open(sql_path, "r") as f:
                sql = f.read()
            conn.execute(text(sql))
            print(f"  Views created from: {script_name}")