"""
Executable python script for extracting tabular data from financial statement PDFs,
and loading them into numbered CSV files, with flexibility across all brokerage/finance company formats.

Can be run standalone:
    python _00_extract_tables.py Full_Pipeline/fidelity.yaml
"""

import os
import sys
import shutil
from pathlib import Path
import json
import yaml
import utils.extraction_tools as et
from prefect import flow
import logging


def load_config(config_path: str) -> dict:
    """Load and return the full YAML config dict from the given path."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


@flow(name="extraction-flow", flow_run_name="extraction: {cfg[brokerage]}")
def extraction_flow(cfg: dict) -> dict:
    """
    Description:
        Main FX containing the full PDF-to-CSV table extraction workflow:
        -> Loading PDF/ZIP pages as images
        -> Sending pages to Gemini in auto-tuned batches
        -> Parsing each batch response into structured tables
        -> Merging tables across batches (cross-batch + sub-section merging)
        -> Deduplicating whole-table duplicates
        -> Saving each table as a numbered CSV

    Input:
        cfg: Full YAML config dict (both 'extraction' and 'brokerage' keys used).
             Load with load_config() or pass the dict directly from pipeline.py.

    Output:
        dict summarizing extraction results, including 'failed_page_ranges' for
        downstream consumption by parsing_flow.
    """
    ecfg       = cfg['extraction']
    file_path  = ecfg['file_path']
    output_dir = ecfg['output_dir']
    prefix     = ecfg['csv_prefix']
    # Wipe any pre-existing output directory to remove stale CSVs
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # silencing noisy prefect info logs
    logger = logging.getLogger("prefect")
    logger.setLevel(logging.WARNING)

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable not set.")

    # Apply image/batch config from YAML to extraction_tools module globals
    et.PDF_DPI            = ecfg.get('pdf_dpi',            et.PDF_DPI)
    et.MAX_IMAGE_DIMENSION= ecfg.get('max_image_dimension', et.MAX_IMAGE_DIMENSION)
    et.PNG_COMPRESSION    = ecfg.get('png_compression',     et.PNG_COMPRESSION)
    et.BATCH_SIZE         = ecfg.get('batch_size',          et.BATCH_SIZE)
    et.BATCH_DELAY_SECONDS= ecfg.get('batch_delay_seconds', et.BATCH_DELAY_SECONDS)
    et.MAX_RETRIES        = ecfg.get('max_retries',         et.MAX_RETRIES)
    et.RETRY_DELAY_SECONDS= ecfg.get('retry_delay_seconds', et.RETRY_DELAY_SECONDS)
    et.GEMINI_API_URL     = ecfg.get('gemini_api_url',      et.GEMINI_API_URL)

    # 1. Load all page images
    pages = et.load_pages(file_path)

    # 2. Send pages to Gemini in batches (with auto-tuning truncation retry)
    batch_result = et.call_gemini_api_batched(pages, api_key)

    # 3. Parse tables from each successful batch response
    all_tables = []
    for batch_idx, gemini_response in enumerate(batch_result.responses):
        batch_tables = et.parse_tables_from_response(gemini_response.text)
        print(f"  Batch {batch_idx + 1}: parsed {len(batch_tables)} table(s)")
        all_tables.extend(batch_tables)

    # 4. Merge consecutive tables with identical headers
    tables = et.merge_tables_with_shared_headers(all_tables)
    print(f"Total tables after merging: {len(tables)}")

    # 5. Deduplicate whole-table duplicates from overlapping batches
    tables = et.deduplicate_tables(tables)
    print(f"Total tables after deduplication: {len(tables)}")

    # 6. Save each table as a CSV
    output_files = []
    for i, table in enumerate(tables, start=1):
        filename = f"{prefix}_table_{i}.csv"
        filepath = os.path.join(output_dir, filename)
        et.save_table_to_csv(table, filepath)
        output_files.append(filepath)

        row_count = len(table)
        col_count = max(len(r) for r in table)
        print(f"  Saved: {filename} ({row_count} rows x {col_count} cols)")

    # 7. Build and save extraction manifest
    results = {
        'source_file':        file_path,
        'total_pages':        len(pages),
        'total_batches':      len(batch_result.responses),
        'total_tables':       len(tables),
        'output_files':       output_files,
        # Page ranges that were never successfully extracted are passed to the final initial_issue_report
        'failed_page_ranges': batch_result.failed_page_ranges,
    }

    manifest_path = os.path.join(output_dir, f"{prefix}_manifest.json")
    with open(manifest_path, 'w') as f:
        json.dump(results, f, indent=2)

    print("EXTRACTION COMPLETE!")
    print(f"  Pages sent:     {results['total_pages']}")
    print(f"  Batches:        {results['total_batches']}")
    print(f"  Tables found:   {results['total_tables']}")
    print(f"  Output dir:     {output_dir}")
    if results['failed_page_ranges']:
        logger.error(
            f"  WARNING: {len(results['failed_page_ranges'])} page range(s) "
            f"were never successfully extracted: {results['failed_page_ranges']}"
        )

    return results


# Standalone execution: python _00_extract_tables.py <config.yaml>
if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Requires 2 arguments. Example: python 00_extract_tables.py <config.yaml>")

    cfg = load_config(sys.argv[1])
    extraction_flow(cfg)