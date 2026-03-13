"""
Executable python script that parses holdings data from CSVs
sourced from financial statement PDFs.

Can be run standalone:
    python _01_parse_holdings.py Full_Pipeline/fidelity.yaml
"""

import os
import sys
import yaml
import utils.parse_tools as pt
from prefect import flow
import logging


def load_config(config_path: str) -> dict:
    """Load and return the full YAML config dict from the given path."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


@flow(name="parsing-flow", flow_run_name="parsing: {cfg[brokerage]}")
def parsing_flow(
    cfg: dict,
    failed_page_ranges: list[str] | None = None,
) -> tuple:
    """
    Description:
        Main FX for parsing holdings data from CSVs into a structured DataFrame with 
        account, asset class, and tax treatment enrichment.

    Input:
        cfg:                Full YAML config dict.
        failed_page_ranges: Optional list of page-range strings from extraction_flow
                            that were never successfully processed. These are surfaced
                            as extraction_failures in the initial issue report.

    Output:
        Tuple of (final_df, issue_report).
    """
    pcfg            = cfg['parsing']
    csv_dir         = pcfg['csv_dir']
    csv_prefix      = pcfg['csv_prefix']
    primary_dir     = pcfg['primary_dir']
    output_file     = os.path.join(primary_dir, pcfg['holdings_filename'])
    initial_issue_file = os.path.join(primary_dir, pcfg['initial_issue_filename'])

    # silencing noisy prefect info logs
    logger = logging.getLogger("prefect")
    logger.setLevel(logging.WARNING)

    # Initialise initial issue report and surface any upstream extraction failures
    issue_report = pt.IssueReport()
    if failed_page_ranges:
        for page_range in failed_page_ranges:
            issue_report.add_issue(
                'extraction_failures',
                f"PDF pages {page_range} were never successfully extracted "
                f"(truncated at batch_size=1). Holdings on these pages are absent."
            )

    # Run parsing tasks (defined in parse_tools) in sequence
    tables = pt.load_csv_tables(csv_dir, csv_prefix)

    # Attempt to extract account info from the CSV tables first.
    # If no structured account summary table exists (acc_types comes back empty),
    # fall back to a single Gemini call against the source PDF.
    acc_types, acc_nums = pt.extract_account_info(tables)

    if not acc_types:
        pdf_file = cfg.get("llm_validation", {}).get("pdf_file")
        api_key  = os.environ.get("GEMINI_API_KEY")
        if pdf_file and api_key:
            print("  No account info found in CSVs — attempting LLM extraction from PDF...")
            llm_acc_types, llm_acc_nums = pt.llm_extract_account_info(
                pdf_path=pdf_file,
                api_key=api_key,
            )
            acc_types, acc_nums = pt.extract_account_info(
                tables,
                llm_acc_types=llm_acc_types,
                llm_acc_nums=llm_acc_nums,
            )
        else:
            if not pdf_file:
                print("  Warning: no pdf_file configured under llm_validation — "
                      "skipping LLM account extraction.")
            if not api_key:
                print("  Warning: GEMINI_API_KEY not set — "
                      "skipping LLM account extraction.")

    records  = pt.extract_holdings_records(tables, acc_types, acc_nums, issue_report)
    final_df = pt.build_holdings_dataframe(records)

    issue_report.print_report()

    pt.save_holdings_outputs(final_df, issue_report, output_file, initial_issue_file)

    return final_df, issue_report


# Standalone execution: python 01_parse_holdings.py <config.yaml>
if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Requires 2 arguments. Example: python 01_parse_holdings.py <config.yaml>")

    cfg = load_config(sys.argv[1])
    parsing_flow(cfg)