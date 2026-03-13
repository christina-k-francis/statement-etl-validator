"""
Executable Python script that performs LLM-based self-correcting validation of
parsed holdings data, consuming the outputs of _01_parse_holdings.py.

This is a self-correcting iterative data validation process.

Can be run standalone:
    python _02_llm_validate.py Fidelity_Investments/fidelity.yaml
"""

import os
import sys
import yaml
from prefect import flow
import utils.llm_validation_tools as lv
import utils.parse_tools as pt
import logging


def load_config(config_path: str) -> dict:
    """Load and return the full YAML config dict from the given path."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

@flow(name="self-correcting-llm-validation-flow", flow_run_name="self-correcting-llm-validation: {cfg[brokerage]}")
def self_correcting_llm_validation_flow(cfg: dict) -> lv.LLMValidationReport:
    """
    Description:
        Main FX for executing LLM-assisted self-correcting validation of holdings data.
            1. Loads in initial issue report and holdings CSV produced by _01_parse_holdings.py
            2. Sends statement pdf + missing_symbol, missing_data, negative_values to Gemini (one call per category).
            3. Sends missing_account issues to Gemini via a dedicated verifier that returns
               account_type and account_number for patching both the 'Account' and 'Account Type' columns.
            4. Sends total_mismatches through the two-step mismatch verifier
             (Step 1: read PDF total; Step 2: per-holding spot-check).
             Both steps always run. Verdict assigned by Python.
          5. Patches any correctable verdicts into the holdings DataFrame.
          6. Saves the updated holdings.csv and refreshes initial_issue_report.txt.
          7. Repeats until zero correctable verdicts remain or iteration cap is hit.

    Input:
        cfg: Full YAML config dict. Expected keys under 'llm_validation':
               pdf_file                — path to original brokerage PDF
               llm_validation_filename — e.g. "2_llm_initial_issue_report.txt"
               ledger_filepath         — e.g. "3_validation_ledger.txt"
               max_correction_iterations  - (optional) iteration cap, default 3
             Expected keys under 'parsing':
               primary_dir         — output directory
               initial_issue_filename — e.g. "initial_issue_report.txt"
               holdings_filename   — e.g. "holdings.csv"
               csv_dir             - source CSV directory (for mismatch spot-check)
               csv_prefix          - source CSV filename prefix (e.g. "fidelity_table_")
             Environment variable:
               GEMINI_API_KEY — Gemini API key

    Output:
        Populated LLMValidationReport instance.
    """
    pcfg        = cfg["parsing"]
    lvcfg       = cfg["llm_validation"]
    primary_dir = pcfg["primary_dir"]

    initial_issue_file = os.path.join(primary_dir, pcfg["initial_issue_filename"])
    holdings_file   = os.path.join(primary_dir, pcfg["holdings_filename"])
    llm_output_file = lvcfg["llm_validation_filename"]
    ledger_file     = lvcfg["ledger_filepath"]
    pdf_file        = lvcfg["pdf_file"]
    max_iterations  = int(lvcfg.get("max_correction_iterations", 3))

    # silencing noisy prefect info logs
    logger = logging.getLogger("prefect")
    logger.setLevel(logging.WARNING)

    # Source CSV config - needed to determine which holdings belong to each flagged table
    csv_dir    = pcfg["csv_dir"]
    csv_prefix = pcfg["csv_prefix"]

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable not set.")

    # 1. Load in all input files (no resolved set yet — first load is just for the PDF path)
    _, _, pdf_path = lv.load_validation_inputs(
        initial_issue_file=initial_issue_file,
        holdings_file=holdings_file,
        pdf_file=pdf_file,
    )
    # save statement pdf url
    file_uri = lv.upload_pdf(pdf_path=pdf_path, api_key=api_key)

    # Ledger persists every verdict and correction across all iterations to disk.
    # Written after each iteration so the record survives a mid-run crash.
    ledger = lv.ValidationLedger(ledger_file)

    report: lv.LLMValidationReport | None = None
    all_verdicts: dict[str, list[dict]]   = {}

    # Accumulates issue strings that have received a terminal verdict across
    # all iterations. Passed into load_validation_inputs each iteration so
    # resolved issues are silently excluded without modifying any file on disk.
    resolved_issues: frozenset[str] = frozenset()

    for iteration in range(1, max_iterations + 1):
        print(f"\nLLM VALIDATION - ITERATION {iteration} of {max_iterations}")

        # Load inputs fresh each iteration (latest holdings.csv + exclusion filter)
        issues_by_category, holdings_df, _ = lv.load_validation_inputs(
            initial_issue_file=initial_issue_file,
            holdings_file=holdings_file,
            pdf_file=pdf_file,
            resolved_issues=resolved_issues,
        )

        # Log number of issues queued vs. already resolved
        total_to_evaluate = sum(len(v) for v in issues_by_category.values())
        print(f"\n{total_to_evaluate} issue(s) queued for LLM verification "
              f"({len(resolved_issues)} already resolved and excluded).")

        if total_to_evaluate == 0:
            print("All issues resolved. Exiting loop.")
            report = lv.build_llm_report({})
            break

        # One Gemini call per issue category
        all_verdicts = {}

        # Evaluating Issues in the "standard" Categories (missing symbol/data, abnormal negative values, etc.)
        for category in ("missing_symbol", "missing_data", "negative_values"):
            issues = issues_by_category.get(category, [])
            if not issues:
                all_verdicts[category] = []
                continue
            all_verdicts[category] = lv.run_category_verification(
                category=category,
                issues=issues,
                holdings_df=holdings_df,
                file_uri=file_uri,
                api_key=api_key,
            )

        # Evaluating missing account issues — dedicated verifier that returns
        # account_type and account_number for patching both Account columns.
        missing_account_issues = issues_by_category.get("missing_account", [])
        if missing_account_issues:
            all_verdicts["missing_account"] = lv.run_missing_account_verification(
                issues=missing_account_issues,
                holdings_df=holdings_df,
                file_uri=file_uri,
                api_key=api_key,
            )
        else:
            all_verdicts["missing_account"] = []

        # Evaluating Issues where column-level calc total != summary row total
        mismatch_issues = issues_by_category.get("total_mismatches", [])
        if mismatch_issues:
            all_verdicts["total_mismatches"] = lv.run_mismatch_verification(
                issues=mismatch_issues,
                holdings_df=holdings_df,
                file_uri=file_uri,
                api_key=api_key,
                csv_dir=csv_dir,
                csv_prefix=csv_prefix,
            )
        else:
            all_verdicts["total_mismatches"] = []

        # Assemble and print this iteration's report
        report = lv.build_llm_report(all_verdicts)
        report.print_report()

        # Count correctable verdicts: FALSE POSITIVE, RESOLVED, and VALUES INCORRECT with corrections
        false_positives = sum(
            1 for category, verdicts in all_verdicts.items()
            for v in verdicts
            if (category == "missing_account" and
                v.get("verdict") == lv.ACCOUNT_VERDICT_RESOLVED) or
               (category not in ("total_mismatches", "missing_account") and
                v.get("verdict") == lv.VERDICT_FALSE_POSITIVE) or
               (category == "total_mismatches" and
                v.get("verdict") == lv.MISMATCH_VERDICT_VALUES_INCORRECT and
                v.get("phase1_corrections"))
        )

        if false_positives == 0:
            print("Zero correctable values detected. Validation complete.")
            ledger.append_iteration(iteration, all_verdicts, applied_log=[])
            ledger.save()
            break

        print(f"\n{false_positives} correctable value(s) detected - "
              f"applying corrections to holdings.csv...")

        # Patch corrected values into the holdings DataFrame
        corrected_df, correction_count, applied_log = lv.apply_corrections_to_holdings(
            holdings_df=holdings_df,
            all_verdicts=all_verdicts,
        )
        print(f"  {correction_count} cell(s) updated in holdings DataFrame.")

        # Save the corrected holdings.csv
        corrected_df.to_csv(holdings_file, index=False)
        print(f"  holdings.csv saved: {holdings_file}")

        # Re-validate the patched DataFrame -> fresh initial_issue_report.txt
        fresh_report = pt.revalidate_holdings(corrected_df, csv_dir, csv_prefix)
        fresh_report.save_to_file(initial_issue_file)
        print(f"  initial_issue_report.txt refreshed: {initial_issue_file}")

        # Record this iteration in the ledger and flush to disk immediately
        ledger.append_iteration(iteration, all_verdicts, applied_log)
        ledger.save()

        # Accumulate terminal verdicts so they are excluded from future iterations.
        # VALUES INCORRECT is only marked resolved when correction_count > 0 —
        # if the name-match failed and nothing was patched, it stays in rotation.
        newly_resolved = lv.collect_resolved_issues(all_verdicts, correction_count)
        resolved_issues = resolved_issues | newly_resolved
        if newly_resolved:
            print(f"  {len(newly_resolved)} issue(s) marked resolved and excluded "
                  f"from future iterations.")

        if iteration == max_iterations:
            print(f"\nIteration cap ({max_iterations}) reached. "
                  f"Stopping with {false_positives} correctable verdict(s) remaining.")

    # Save the final LLM report from the last completed iteration
    if report is None:
        report = lv.build_llm_report({})

    report.save_to_file(llm_output_file)
    print(f"\nFinal LLM validation report saved to: {llm_output_file}")

    return report

# Standalone execution: python _02_llm_validate.py <config.yaml>
if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python _02_llm_validate.py <config.yaml>")

    cfg = load_config(sys.argv[1])
    self_correcting_llm_validation_flow(cfg)