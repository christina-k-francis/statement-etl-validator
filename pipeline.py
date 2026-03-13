"""
Orchestrated pipeline entry point for the financial data extraction and holdings CSV parsing.

Runs all stages end-to-end from a single YAML config file:
    Stage 1 - _00_extract_tables:  PDF → numbered CSVs
    Stage 2 - _01_parse_holdings:  CSVs → holdings.csv + initial_issue_report.txt
    Stage 3 - _02_llm_validation:  LLM-assisted self-correcting validation
    Stage 4 - _03_postgres_upload: Validated holdings.csv → Postgres

Usage:
    python pipeline.py <config.yaml>

Example:
    python pipeline.py fidelity.yaml
"""

import sys
import yaml
from prefect import flow

from _00_extract_tables   import extraction_flow
from _01_parse_holdings   import parsing_flow
from _02_llm_validation   import self_correcting_llm_validation_flow
from _03_postgres_upload  import postgres_upload_flow

def load_config(config_path: str) -> dict:
    """Load and return the full YAML config dict from the given path."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


@flow(name="full-pipeline", flow_run_name="full-pipeline: {cfg[brokerage]}")
def full_pipeline_flow(cfg: dict) -> None:
    """
    Description:
        Primary financial statement pipeline flow:
        extraction_flow  →  parsing_flow  →  self_correcting_llm_validation_flow  →  postgres_upload_flow

        Failed_page_ranges from extraction_flow are forwarded directly into parsing_flow
        for documentation in the initial issue report.

    Input:
        cfg: Full YAML config dict loaded from a brokerage config file.
             Load with load_config().
    """
    lcfg = cfg['llm_validation']

    # Stage 1: Extract tables from the PDF into numbered CSVs
    extraction_results = extraction_flow(cfg)
    print('--' * 50)
    print('\n')

    # Stage 2: Parse holdings from CSVs, forwarding any extraction failures
    parsing_flow(
        cfg=cfg,
        failed_page_ranges=extraction_results.get('failed_page_ranges', []),
    )
    print('--' * 50)
    print('\n')

    # Stage 3: Conduct self-correcting validation of extracted/parsed holdings data
    self_correcting_llm_validation_flow(cfg)
    print('--' * 50)
    print('\n')

    # Stage 4: Upload validated holdings CSV to Postgres + Executing Analysis scripts
    postgres_upload_flow(cfg)
    print('--' * 50)
    print('\n')

    print(f"See the validation ledger at ({lcfg['ledger_filepath']}) to review all changes")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Requires 2 arguments. Example: python pipeline.py <config.yaml>")

    cfg = load_config(sys.argv[1])
    full_pipeline_flow(cfg)