"""
Postgres connection and upload utilities for the financial data pipeline.

Provides:
    get_engine()        — build a SQLAlchemy engine, creating the DB if absent
    upload_holdings()   — upload a holdings DataFrame to a Postgres table
"""

import os
import pandas as pd
from sqlalchemy import create_engine
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
