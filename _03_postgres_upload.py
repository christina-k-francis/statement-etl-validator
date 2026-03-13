"""
Executable Python script that uploads the final validated holdings CSV
to a Postgres database, consuming the outputs of _02_llm_validation.py.

Can be run standalone:
    python _03_postgres_upload.py fidelity.yaml
"""

import os
import sys
import yaml
import pandas as pd
from prefect import flow

import utils.psql_tools as psql


def load_config(config_path: str) -> dict:
    """Load and return the full YAML config dict from the given path."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


@flow(name="postgres-upload-flow", flow_run_name="postgres-upload: {cfg[brokerage]}")
def postgres_upload_flow(cfg: dict) -> None:
    """
    Description:
        Reads the validated holdings CSV produced by _02_llm_validation.py
        and uploads it to the configured Postgres database and table.

    Input:
        cfg: Full YAML config dict. Expected keys under 'postgres':
               db_name    — target Postgres database name
               table_name — target table name within that database
               host       — Postgres host        (default: 'localhost')
               port       — Postgres port        (default: '5432')
               user       — Postgres username    (default: 'postgres')
             Expected keys under 'parsing':
               primary_dir      — directory containing the holdings CSV
               holdings_filename — filename of the validated holdings CSV
             Environment variable:
               PSQL_PWD — Postgres password (required)

    Output:
        None
    """
    pcfg   = cfg["parsing"]
    pscfg  = cfg["postgres"]

    # --- Resolve holdings CSV path (same source as _02_llm_validation.py) ---
    holdings_file = os.path.join(pcfg["primary_dir"], pcfg["holdings_filename"])

    # --- Postgres connection params ---
    db_name    = pscfg["db_name"]
    sql_dir    = pscfg["sql_dir"]
    table_name = pscfg["table_name"]
    host       = pscfg.get("host", "localhost")
    port       = str(pscfg.get("port", "5432"))
    user       = pscfg.get("user", "postgres")

    pwd = os.getenv("PSQL_PWD")
    if not pwd:
        raise RuntimeError("Environment variable PSQL_PWD is not set.")

    # --- Load holdings ---
    if not os.path.exists(holdings_file):
        raise FileNotFoundError(
            f"Holdings CSV not found at '{holdings_file}'. "
            "Ensure _02_llm_validation.py has completed successfully."
        )
    holdings_df = pd.read_csv(holdings_file)
    print(f"  Loaded holdings CSV: {holdings_file} ({len(holdings_df)} rows)")

    # --- Connect and upload ---
    print(f"  Connecting to Postgres — host: {host}, port: {port}, "
          f"db: {db_name}, user: {user}")
    engine = psql.get_engine(user=user, pwd=pwd, host=host, port=port, db=db_name)

    psql.upload_holdings(
        holdings_df=holdings_df,
        engine=engine,
        table_name=table_name,
        if_exists="replace",
    )

    print(f"\nPOSTGRES UPLOAD COMPLETE!")
    print(f"  Source file:  {holdings_file}")
    print(f"  Database:     {db_name}")
    print(f"  Table:        {table_name}")
    print(f"  Rows written: {len(holdings_df)}")

    # --- Executing SQL analysis scripts ---
    psql.create_views(engine, sql_dir)


# Standalone execution: python _03_postgres_upload.py <config.yaml>
if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python _03_postgres_upload.py <config.yaml>")

    cfg = load_config(sys.argv[1])
    postgres_upload_flow(cfg)
