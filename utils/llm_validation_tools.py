"""
Helpful FXs and classes for conducting LLM-based second-pass validation of parsed holdings data
"""

import os
import re
import sys
import json
import time
from io import StringIO

import pandas as pd
import requests
from prefect import task


# LLM API variables
_GEMINI_MODEL   = "gemini-2.5-flash"
_GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
_FILE_API_URL   = "https://generativelanguage.googleapis.com/upload/v1beta/files"
_FILE_GET_URL   = "https://generativelanguage.googleapis.com/v1beta/files/{file_id}"

# Categories evaluated by the lLM
_LLM_CATEGORIES = ["negative_values", "missing_symbol", "missing_data", "total_mismatches", "missing_account"]

# Validation Verdict Types
VERDICT_CONFIRMED      = "CONFIRMED"
VERDICT_FALSE_POSITIVE = "FALSE POSITIVE"
VERDICT_NEEDS_REVIEW   = "NEEDS REVIEW"
_VALID_VERDICTS        = {VERDICT_CONFIRMED, VERDICT_FALSE_POSITIVE, VERDICT_NEEDS_REVIEW}

# Missing account verdict types
ACCOUNT_VERDICT_RESOLVED     = "RESOLVED"      # account info found and populated
ACCOUNT_VERDICT_CONFIRMED    = "CONFIRMED"     # genuinely no account info in PDF
ACCOUNT_VERDICT_NEEDS_REVIEW = "NEEDS REVIEW"  # ambiguous or security not found
_VALID_ACCOUNT_VERDICTS = {
    ACCOUNT_VERDICT_RESOLVED,
    ACCOUNT_VERDICT_CONFIRMED,
    ACCOUNT_VERDICT_NEEDS_REVIEW,
}

# Mismatch verdict types
MISMATCH_VERDICT_VALUES_INCORRECT   = "VALUES INCORRECT"        # ≥1 holding value wrong; patch and re-run
MISMATCH_VERDICT_CONFIRMED_TRUE     = "CONFIRMED TRUE MISMATCH" # holdings correct; calculated sum genuinely differs
MISMATCH_VERDICT_BAD_TOTAL_PARSE    = "BAD TOTAL PARSE"         # holdings correct; expected value was mis-parsed
MISMATCH_VERDICT_NEEDS_REVIEW       = "NEEDS REVIEW"            # step 1 or step 2 parse failure
_VALID_MISMATCH_VERDICTS = {
    MISMATCH_VERDICT_VALUES_INCORRECT,
    MISMATCH_VERDICT_CONFIRMED_TRUE,
    MISMATCH_VERDICT_BAD_TOTAL_PARSE,
    MISMATCH_VERDICT_NEEDS_REVIEW,
}

# prompts for each issue category
_PROMPTS = {
    "missing_symbol": """
You are a financial data quality auditor. You have been given:
  1. A brokerage statement PDF.
  2. A list of securities that a parser flagged as MISSING SYMBOL/CUSIP.
  3. The corresponding rows from the parsed holdings CSV.

Your task: for each flagged security, inspect the PDF and determine whether
a ticker symbol, CUSIP, or ISIN truly does not appear anywhere in the statement
for that security, or whether the parser missed it.

FLAGGED ISSUES:
{issues_block}

RELEVANT HOLDINGS CSV ROWS:
{holdings_block}

Respond ONLY with a valid JSON array — no markdown fences, no preamble.
Each element must have exactly these keys:
  "issue"            : the exact issue string from the list above
  "verdict"          : one of "CONFIRMED", "FALSE POSITIVE", or "NEEDS REVIEW"
  "reason"           : one concise sentence explaining what you found in the PDF
  "corrected_symbol" : if verdict is "FALSE POSITIVE", the exact ticker/CUSIP/ISIN
                       string found in the PDF (e.g. "GOOGL" or "38141G104");
                       otherwise null

CONFIRMED      = the PDF genuinely has no symbol/CUSIP for this security
FALSE POSITIVE = the PDF does show a symbol/CUSIP that the parser missed
NEEDS REVIEW   = the PDF is ambiguous or the security cannot be located
""",

    "missing_data": """
You are a financial data quality auditor. You have been given:
  1. A brokerage statement PDF.
  2. A list of securities that a parser flagged as MISSING COST BASIS.
  3. The corresponding rows from the parsed holdings CSV.

Your task: for each flagged security, inspect the PDF and determine whether
the cost basis is genuinely absent (e.g. shown as "N/A", "unknown", "--", or
simply blank), or whether the parser failed to capture a value that is present.

FLAGGED ISSUES:
{issues_block}

RELEVANT HOLDINGS CSV ROWS:
{holdings_block}

Respond ONLY with a valid JSON array — no markdown fences, no preamble.
Each element must have exactly these keys:
  "issue"                : the exact issue string from the list above
  "verdict"              : one of "CONFIRMED", "FALSE POSITIVE", or "NEEDS REVIEW"
  "reason"               : one concise sentence explaining what you found in the PDF
  "corrected_cost_basis" : if verdict is "FALSE POSITIVE", the numeric cost basis
                           value found in the PDF as a plain number (e.g. 1234.56);
                           otherwise null

CONFIRMED      = the PDF genuinely shows no cost basis (N/A, unknown, blank, etc.)
FALSE POSITIVE = the PDF shows a numeric cost basis that the parser missed
NEEDS REVIEW   = the PDF is ambiguous or the security cannot be located
""",

    "negative_values": """
You are a financial data quality auditor. You have been given:
  1. A brokerage statement PDF.
  2. A list of securities that a parser flagged as having NEGATIVE VALUES
     (negative quantity, cost basis, or market value).
  3. The corresponding rows from the parsed holdings CSV.

Your task: for each flagged security, inspect the PDF and determine whether
the negative value is genuinely present in the source document (e.g. a short
position or a transfer-out entry), or whether it is a parser error (e.g.
a sign was incorrectly applied).

FLAGGED ISSUES:
{issues_block}

RELEVANT HOLDINGS CSV ROWS:
{holdings_block}

Respond ONLY with a valid JSON array — no markdown fences, no preamble.
Each element must have exactly these keys:
  "issue"             : the exact issue string from the list above
  "verdict"           : one of "CONFIRMED", "FALSE POSITIVE", or "NEEDS REVIEW"
  "reason"            : one concise sentence explaining what you found in the PDF
  "corrected_value"   : if verdict is "FALSE POSITIVE", the correct positive numeric
                        value found in the PDF as a plain number (e.g. 14510.99);
                        otherwise null

CONFIRMED      = the PDF genuinely shows a negative value for this field
FALSE POSITIVE = the PDF shows a positive value; the parser introduced the sign error
NEEDS REVIEW   = the PDF is ambiguous or the security cannot be located
""",

    "missing_account": """
You are a financial data quality auditor. You have been given:
  1. A brokerage statement PDF.
  2. A list of securities that a parser flagged as MISSING OR UNKNOWN ACCOUNT.
  3. The corresponding rows from the parsed holdings CSV.

Your task: inspect the PDF and determine which brokerage account each flagged
security belongs to. Look everywhere — page headers, account summary tables,
holdings section headings, and any table that associates securities with account
numbers or account names.

FLAGGED ISSUES:
{issues_block}

RELEVANT HOLDINGS CSV ROWS:
{holdings_block}

Respond ONLY with a valid JSON array — no markdown fences, no preamble.
Each element must have exactly these keys:
  "issue"           : the exact issue string from the list above
  "verdict"         : one of "RESOLVED", "CONFIRMED", or "NEEDS REVIEW"
  "reason"          : one concise sentence explaining what you found in the PDF
  "account_type"    : if verdict is "RESOLVED", the account type label exactly as
                      it appears in the PDF (e.g. "Individual TOD", "Traditional IRA");
                      otherwise null
  "account_number"  : if verdict is "RESOLVED", the account number exactly as it
                      appears in the PDF (e.g. "X12-345678"); otherwise null

RESOLVED      = account info was found in the PDF; account_type and account_number
                are populated with the correct values
CONFIRMED     = the security genuinely has no account info in the PDF (rare)
NEEDS REVIEW  = the PDF is ambiguous or the security cannot be located
""",

    # Step 1: read the true PDF total — narrow single-value lookup, no math
    "total_mismatches_read_total": """
You are a financial data quality auditor. You have been given:
  1. A brokerage statement PDF.
  2. A mismatch issue identifying a specific table number and field.

MISMATCH ISSUE:
{issue}

MISMATCHED FIELD: {field}
TABLE NUMBER:     {table_num}

Your task: locate the TOTAL for the identified table and field in the PDF and
report the exact numeric value printed there. Follow these rules exactly:

  - The table will have one or more SUB-SECTION total rows (e.g.
    "Total Corporate Bonds", "Total Municipal Bonds", "Total Other Bonds").
    These are the rows you must use.
  - SUM the sub-section total rows for the field to produce the table total.
    Do NOT read or report any higher-level aggregate rows such as
    "Total Holdings", "Total Portfolio", "Total Bonds", or any row that
    appears immediately after another total row — these aggregate across
    multiple tables and will not match the individual table's holdings.
  - Do NOT calculate or derive values from individual holding rows —
    only read and sum the sub-section total rows.
  - If the table has only one sub-section total row, report that single value.

Respond ONLY with a valid JSON object — no markdown fences, no preamble.
The object must have exactly these keys:
  "field"             : the mismatched field name
  "table_num"         : the table number as an integer
  "pdf_total"         : the sum of sub-section total rows for this field,
                        as a plain number (e.g. 1652.50), or null if not found
  "subsection_totals" : list of the individual sub-section row labels and values
                        used to produce pdf_total, e.g.
                        [{{"label": "Total Corporate Bonds", "value": 750.00}}, ...]
  "reason"            : one concise sentence describing which sub-section rows
                        were found and on which page, or why they could not be found
""",

    # Step 2: spot-check every individual holding — anchored comparison
    "total_mismatches_spotcheck": """
You are a financial data quality auditor. You have been given:
  1. A brokerage statement PDF.
  2. A mismatch issue identifying a specific table and field.
  3. The parsed CSV rows for every holding in that table, with their
     current parsed value for the mismatched field provided as an anchor.

MISMATCH ISSUE:
{issue}

MISMATCHED FIELD: {field}

HOLDINGS IN THIS TABLE (parsed CSV — use these as your comparison anchor):
{holdings_block}

Your task: for EACH holding row listed above, locate that security in the PDF
and confirm whether the "{field}" value in the parsed CSV matches what is
printed in the PDF. You are checking for extraction errors.

Respond ONLY with a valid JSON array — no markdown fences, no preamble.
One element per holding row. Each element must have exactly these keys:
  "name"            : the security name from the holdings block
  "parsed_value"    : the value from the parsed CSV as a number, or null
  "pdf_value"       : the value printed in the PDF as a plain number, or null
  "match"           : true if parsed_value matches pdf_value within normal
                      rounding (e.g. within $1.00); false otherwise
  "corrected_value" : if match is false, the correct numeric value from the
                      PDF as a plain number; otherwise null
""",
}


### FXs for interacting with LLM API ###
def _call_gemini(payload: dict, api_key: str, timeout: int = 120) -> str:
    """
    Description:
        Send a single content generation request to the Gemini REST API and
        return the raw text of the first candidate's response.

        Mirrors the request structure used in extraction_tools.py.

    Input:
        payload: Full request body dict (contents, generationConfig, etc.)
        api_key: Gemini API key string.
        timeout: Request timeout in seconds.

    Output:
        Raw text string from the model, or empty string on failure.
    """
    url      = f"{_GEMINI_API_URL}?key={api_key}"
    response = requests.post(
        url,
        json=payload,
        headers={"Content-Type": "application/json"},
        timeout=timeout,
    )
    response.raise_for_status()

    result     = response.json()
    candidates = result.get("candidates", [])
    if not candidates:
        return ""

    text_parts = [
        p["text"]
        for p in candidates[0].get("content", {}).get("parts", [])
        if "text" in p
    ]
    return "\n".join(text_parts).strip()


def _strip_fences(text: str) -> str:
    """Strip markdown code fences the model may have added despite instructions."""
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _base_payload(file_uri: str, prompt: str) -> dict:
    """Build the standard Gemini payload referencing an uploaded PDF URI."""
    return {
        "contents": [
            {
                "parts": [
                    {"fileData": {"mimeType": "application/pdf", "fileUri": file_uri}},
                    {"text": prompt},
                ]
            }
        ],
        "generationConfig": {
            "temperature":     0.0,
            "maxOutputTokens": 4096,
            "topP":            1.0,
            "topK":            1,
        },
    }

### Validation FXs ###
@task(name="load-llm-validation-inputs")
def load_validation_inputs(
    initial_issue_file: str,
    holdings_file: str,
    pdf_file: str,
    resolved_issues: frozenset[str] | None = None,
) -> tuple[dict, pd.DataFrame, str]:
    """
    Description:
        Load the three inputs needed for LLM validation:
          - initial_issue_report.txt  → parsed into a category → [issues] dict
          - holdings.csv           → loaded as a DataFrame
          - brokerage PDF path     → returned as-is for File API upload

        Any issue string present in resolved_issues is silently excluded from
        issues_by_category before returning. This allows the self-correcting
        loop in _02_llm_validation.py to prevent re-submission of issues that
        have already received a terminal verdict, without modifying any file
        on disk.

        Terminal verdicts that populate resolved_issues:
          Standard categories  — CONFIRMED, FALSE POSITIVE
          total_mismatches     — CONFIRMED TRUE MISMATCH, BAD TOTAL PARSE,
                                 and VALUES INCORRECT once its corrections
                                 have been successfully applied

    Input:
        initial_issue_file:  Path to the initial issue report file.
        holdings_file:    Path to the parsed holdings CSV.
        pdf_file:         Path to the original brokerage statement PDF.
        resolved_issues:  Optional frozenset of issue strings to exclude.
                          Built up across iterations in the flow; None on
                          the first iteration.

    Output:
        Tuple of (issues_by_category, holdings_df, pdf_path).
        issues_by_category keys: "missing_symbol", "missing_data",
        "negative_values", "total_mismatches".
    """
    issues_by_category: dict[str, list[str]] = {cat: [] for cat in _LLM_CATEGORIES}
    resolved = resolved_issues or frozenset()

    try:
        with open(initial_issue_file, "r", encoding="utf-8") as f:
            content = f.read()
    except UnicodeDecodeError:
        with open(initial_issue_file, "r", encoding="windows-1252") as f:
            content = f.read()

    header_to_category = {
        "NEGATIVE VALUES":        "negative_values",
        "MISSING SYMBOL/CUSIP":   "missing_symbol",
        "MISSING FINANCIAL DATA": "missing_data",
        "TOTAL MISMATCHES":       "total_mismatches",
    }

    current_category = None
    for line in content.splitlines():
        stripped = line.strip()

        matched_header = None
        for header, cat in header_to_category.items():
            if stripped.upper().startswith(header):
                matched_header = cat
                break

        if matched_header:
            current_category = matched_header
            continue

        if current_category and stripped.startswith("- "):
            issue_text = stripped[2:].strip()
            if issue_text not in resolved:
                issues_by_category[current_category].append(issue_text)

    holdings_df = pd.read_csv(holdings_file)

    return issues_by_category, holdings_df, pdf_file

@task(name="upload-pdf-to-gemini")
def upload_pdf(pdf_path: str, api_key: str) -> str:
    """
    Description:
        FX uploads the brokerage PDF to the Gemini File API once, and return the
        file URI for reuse across all category-level API calls.

        Uploaded files persist for 48 hours on Google's servers.

    Input:
        pdf_path: Local path to the brokerage statement PDF.
        api_key:  Gemini API key string.

    Output:
        file_uri: The File API URI string (e.g. "files/abc123") for use in
                  subsequent content parts.
    """
    print(f"Uploading PDF to Gemini File API: {pdf_path}")

    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

    # Multipart upload to the REST File API
    upload_response = requests.post(
        _FILE_API_URL,
        params={"key": api_key},
        headers={"X-Goog-Upload-Protocol": "multipart"},
        files={
            "metadata": (None, '{"file": {"mimeType": "application/pdf"}}', "application/json"),
            "file":     (None, pdf_bytes, "application/pdf"),
        },
    )
    upload_response.raise_for_status()
    file_info = upload_response.json().get("file", {})
    file_name = file_info.get("name", "")
    file_uri  = file_info.get("uri", "")

    if not file_name:
        raise RuntimeError(f"File API upload failed — no file name returned: {upload_response.text[:500]}")

    # Extract bare file ID (e.g. "files/abc123" → "abc123") for the GET endpoint
    file_id = file_name.split("/")[-1]

    # Poll until the file reaches ACTIVE state
    for _ in range(30):
        status_response = requests.get(
            _FILE_GET_URL.format(file_id=file_id),
            params={"key": api_key},
        )
        status_response.raise_for_status()
        state = status_response.json().get("state", "")
        if state == "ACTIVE":
            print(f"PDF upload complete. File URI: {file_uri}")
            return file_uri
        time.sleep(2)

    raise RuntimeError(
        f"Gemini File API: PDF '{pdf_path}' did not reach ACTIVE state in time."
    )

def _extract_security_names(issues: list[str]) -> list[str]:
    """
    Description:
        FX extracts security name substrings from issue message strings so we can
        filter holdings_df to only the relevant rows.

    Issue strings look like:
      "Missing Symbol/CUSIP for ADI NET LEASE INC & GROWTH LP XIX..."
      "Negative Quantity = -100.0 for ENSTAR GROUP LIMITED COM STK USD 1.00"
      "Missing Cost Basis for FDIC INSURED DEPOSIT AT FIFTH THIRD BANK..."
    """
    names = []
    for issue in issues:
        # Text after " for " is the security name
        match = re.search(r'\bfor\s+(.+)$', issue, re.IGNORECASE)
        if match:
            names.append(match.group(1).strip())
    return names


def _build_holdings_block(issues: list[str], holdings_df: pd.DataFrame) -> str:
    """
    Return a compact CSV string of only the holdings rows relevant to the
    given issues, to keep prompt size minimal.
    """
    security_names = _extract_security_names(issues)
    if not security_names:
        return holdings_df.to_csv(index=False)

    # Match rows whose Name column contains any of the extracted name fragments
    mask = pd.Series([False] * len(holdings_df), index=holdings_df.index)
    for name_fragment in security_names:
        # Use first ~40 chars of the name for a robust partial match
        fragment = name_fragment[:40].strip()
        mask |= holdings_df["Name"].str.contains(
            re.escape(fragment), case=False, na=False
        )

    matched = holdings_df[mask]
    if matched.empty:
        # Fallback: send full CSV so the LLM has context
        return holdings_df.to_csv(index=False)
    return matched.to_csv(index=False)

def _parse_mismatch_issue(issue: str) -> tuple[int | None, str | None]:
    """
    Parse a total_mismatches issue string into (table_num, field).

    Issue format:
        "Table 18: Est. Annual Income mismatch — Expected: $1,477.50, ..."

    Output:
        (18, "Est. Annual Income") or (None, None) if parsing fails.
    """
    match = re.match(r'Table\s+(\d+):\s+(.+?)\s+mismatch', issue, re.IGNORECASE)
    if match:
        return int(match.group(1)), match.group(2).strip()
    return None, None

def _load_table_holdings(
    table_num: int,
    field: str,
    csv_dir: str,
    csv_prefix: str,
) -> pd.DataFrame | None:
    """
    Description:
        Load the source CSV for a specific table number and return only the
        non-Total rows with their Name and the specified field column.

        Used to supply per-holding values to the spot-check prompt without
        requiring a Source Table column in holdings.csv.

    Input:
        table_num:  Integer table number matching the CSV filename.
        field:      Human-readable field name from the mismatch issue string
                    (e.g. "Est. Annual Income"). Matched fuzzily against columns.
        csv_dir:    Directory containing the numbered source CSVs.
        csv_prefix: Filename prefix, e.g. "fidelity_table_" -> fidelity_table_18.csv

    Output:
        DataFrame with columns ['Description', matched_column], non-Total rows only.
        Returns None if the file does not exist or the column cannot be matched.
    """
    csv_path = os.path.join(csv_dir, f"{csv_prefix}{table_num}.csv")
    if not os.path.exists(csv_path):
        print(f"  Warning: source CSV not found: {csv_path}")
        return None

    try:
        # Header is on row index 1 (row 0 is the positional index row)
        df = pd.read_csv(csv_path, header=1)
    except Exception as exc:
        print(f"  Warning: could not read {csv_path}: {exc}")
        return None

    if "Description" not in df.columns:
        return None

    # Fuzzy-match the field name against available columns
    field_lower = field.lower()
    matched_col = None
    for col in df.columns:
        if field_lower in col.lower() or col.lower() in field_lower:
            matched_col = col
            break

    if matched_col is None:
        # Broader fallback: any column that shares a meaningful word
        field_words = set(re.findall(r'[a-z]+', field_lower)) - {"the", "a", "an", "of"}
        for col in df.columns:
            col_words = set(re.findall(r'[a-z]+', col.lower()))
            if field_words & col_words:
                matched_col = col
                break

    if matched_col is None:
        print(f"  Warning: could not match field '{field}' in columns: {list(df.columns)}")
        return None

    # Filter out Total/summary rows
    non_total = df[
        df["Description"].notna() &
        ~df["Description"].str.match(r'^\s*Total', case=False, na=False)
    ][["Description", matched_col]].copy()

    non_total = non_total.rename(columns={"Description": "Name", matched_col: field})
    return non_total

@task(name="run-category-verification", task_run_name="verify: {category}")
def run_category_verification(
    category: str,
    issues: list[str],
    holdings_df: pd.DataFrame,
    file_uri: str,
    api_key: str,
) -> list[dict]:
    """
    Description:
        FX for making one Gemini API call per issue per category. Provides the
        model with the uploaded PDF (via File API URI), the category-specific
        prompt, and the relevant subset of the holdings CSV.

    Input:
        category:    One of the keys in _LLM_CATEGORIES.
        issues:      List of issue strings for this category.
        holdings_df: Full parsed holdings DataFrame.
        file_uri:    Gemini File API URI from upload_pdf().
        api_key:     Gemini API key string.

    Output:
        List of verdict dicts, each with keys: "issue", "verdict", "reason".
        On parse failure, each dict gets verdict NEEDS_REVIEW with an
        explanatory reason.
    """
    if not issues:
        return []

    issues_block   = "\n".join(f"- {issue}" for issue in issues)
    holdings_block = _build_holdings_block(issues, holdings_df)

    prompt = _PROMPTS[category].format(
        issues_block=issues_block,
        holdings_block=holdings_block,
    )

    print(f"  Sending {len(issues)} '{category}' issue(s) to Gemini...")

    # Build payload — reference the uploaded PDF via its File API URI
    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "fileData": {
                            "mimeType": "application/pdf",
                            "fileUri":  file_uri,
                        }
                    },
                    {"text": prompt},
                ]
            }
        ],
        "generationConfig": {
            "temperature":    0.0,
            "maxOutputTokens": 4096,
            "topP":           1.0,
            "topK":           1,
        },
    }

    url = f"{_GEMINI_API_URL}?key={api_key}"
    response = requests.post(
        url,
        json=payload,
        headers={"Content-Type": "application/json"},
        timeout=120,
    )
    response.raise_for_status()

    result     = response.json()
    candidates = result.get("candidates", [])
    if not candidates:
        print(f"  Warning: No candidates returned for category '{category}'.")
        return [
            {
                "issue":   issue,
                "verdict": VERDICT_NEEDS_REVIEW,
                "reason":  "Gemini returned no candidates.",
            }
            for issue in issues
        ]

    text_parts = [
        p["text"]
        for p in candidates[0].get("content", {}).get("parts", [])
        if "text" in p
    ]
    raw_text = "\n".join(text_parts).strip()

    # Strip markdown fences if the model wrapped output despite instructions
    raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text)
    raw_text = re.sub(r"\s*```$", "", raw_text)

    try:
        verdicts = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        print(f"  Warning: Could not parse JSON for category '{category}': {exc}")
        verdicts = [
            {
                "issue":   issue,
                "verdict": VERDICT_NEEDS_REVIEW,
                "reason":  f"LLM response could not be parsed as JSON: {exc}",
            }
            for issue in issues
        ]
        return verdicts

    # Sanitise verdicts: ensure required keys and valid verdict strings
    sanitised = []
    for item in verdicts:
        verdict = item.get("verdict", VERDICT_NEEDS_REVIEW).strip().upper()
        if verdict not in _VALID_VERDICTS:
            verdict = VERDICT_NEEDS_REVIEW
        sanitised.append(
            {
                "issue":   item.get("issue", "Unknown issue"),
                "verdict": verdict,
                "reason":  item.get("reason", "No reason provided."),
            }
        )

    return sanitised


@task(name="run-missing-account-verification", task_run_name="verify: missing_account")
def run_missing_account_verification(
    issues: list[str],
    holdings_df: pd.DataFrame,
    file_uri: str,
    api_key: str,
) -> list[dict]:
    """
    Description:
        Sends all MISSING ACCOUNT issues to Gemini in a single API call,
        asking the model to locate each security in the PDF and return the
        correct account type and account number.

        Unlike the standard run_category_verification path, verdicts here use
        RESOLVED / CONFIRMED / NEEDS REVIEW rather than FALSE POSITIVE, and
        carry two extra payload fields (account_type, account_number) that
        apply_corrections_to_holdings uses to patch both the 'Account' and
        'Account Type' columns of holdings.csv.

    Input:
        issues:      List of "Missing or Unknown Account for <name>" issue strings.
        holdings_df: Full parsed holdings DataFrame.
        file_uri:    Gemini File API URI from upload_pdf().
        api_key:     Gemini API key string.

    Output:
        List of verdict dicts, each with keys:
          "issue"          — original issue string
          "verdict"        — RESOLVED, CONFIRMED, or NEEDS REVIEW
          "reason"         — one-sentence explanation
          "account_type"   — str if RESOLVED, else None
          "account_number" — str if RESOLVED, else None
    """
    if not issues:
        return []

    issues_block   = "\n".join(f"- {issue}" for issue in issues)
    holdings_block = _build_holdings_block(issues, holdings_df)

    prompt = _PROMPTS["missing_account"].format(
        issues_block=issues_block,
        holdings_block=holdings_block,
    )

    print(f"  Sending {len(issues)} 'missing_account' issue(s) to Gemini...")

    raw_text = _call_gemini(_base_payload(file_uri, prompt), api_key)
    raw_text = _strip_fences(raw_text)

    try:
        verdicts = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        print(f"  Warning: Could not parse JSON for category 'missing_account': {exc}")
        return [
            {
                "issue":          issue,
                "verdict":        ACCOUNT_VERDICT_NEEDS_REVIEW,
                "reason":         f"LLM response could not be parsed as JSON: {exc}",
                "account_type":   None,
                "account_number": None,
            }
            for issue in issues
        ]

    sanitised = []
    for item in verdicts:
        verdict = item.get("verdict", ACCOUNT_VERDICT_NEEDS_REVIEW).strip().upper()
        if verdict not in _VALID_ACCOUNT_VERDICTS:
            verdict = ACCOUNT_VERDICT_NEEDS_REVIEW
        sanitised.append({
            "issue":          item.get("issue", "Unknown issue"),
            "verdict":        verdict,
            "reason":         item.get("reason", "No reason provided."),
            "account_type":   item.get("account_type"),
            "account_number": item.get("account_number"),
        })

    return sanitised


task(name="run-mismatch-verification", task_run_name="verify: total_mismatches")
def run_mismatch_verification(
    issues: list[str],
    holdings_df: pd.DataFrame,
    file_uri: str,
    api_key: str,
    csv_dir: str,
    csv_prefix: str,
) -> list[dict]:
    """
    Description:
        Two-step LLM verification for TOTAL MISMATCHES issues:

        Step 1 — Read the true PDF total (always):
            This is a narrow single-value lookup — the model does no arithmetic.
            Corrects cases where the expected value in initial_issue_report.txt
            was itself mis-parsed from the source CSV total row.

        Step 2 — Spot-check every individual holding (always):
            Supply the parsed CSV rows for the flagged table as anchors and
            ask Gemini to confirm each holding's value against the PDF.

        Both steps are run, and the final verdict is determined based on their results:
          - Any holding corrections found in Step 2
              → VALUES INCORRECT  (patch holdings.csv, re-run loop)
          - No holding corrections AND pdf_total matches expected value
              → CONFIRMED TRUE MISMATCH (calculated sum truly differs from PDF)
          - No holding corrections AND pdf_total differs from expected value
              → CONFIRMED BAD TOTAL PARSE (expected value was mis-parsed;
                pdf_total is the authoritative value)
          - Either step failed to parse
              → NEEDS REVIEW

    Input:
        issues:      List of total_mismatches issue strings.
        holdings_df: Full parsed holdings DataFrame.
        file_uri:    Gemini File API URI from upload_pdf().
        api_key:     Gemini API key string.
        csv_dir:     Directory containing the numbered source CSVs.
        csv_prefix:  Filename prefix for source CSVs (e.g. "fidelity_table_").

    Output:
        List of verdict dicts per issue, each with keys:
          "issue"               — original issue string
          "verdict"             — one of the four verdict constants above
          "reason"              — human-readable explanation
          "pdf_total"           — value read from PDF in Step 1 (float or None)
          "phase1_corrections"  — list of per-holding correction dicts from Step 2
                                  where match=False and corrected_value is set
    """
    if not issues:
        print("  Sending 0 'total_mismatch' issues to Gemini...")
        return []

    print(f"  Sending {len(issues)} 'total_mismatch' issue(s) to Gemini...")

    all_verdicts = []

    for issue in issues:
        table_num, field = _parse_mismatch_issue(issue)

        if table_num is None or field is None:
            all_verdicts.append({
                "issue":              issue,
                "verdict":            MISMATCH_VERDICT_NEEDS_REVIEW,
                "reason":             "Could not parse table number or field from issue string.",
                "pdf_total":          None,
                "phase1_corrections": [],
            })
            continue

        # Extract the expected value from the issue string for Python classification
        expected_match = re.search(r'Expected:\s*\$?([\d,]+\.?\d*)', issue)
        expected_value = None
        if expected_match:
            try:
                expected_value = float(expected_match.group(1).replace(",", ""))
            except ValueError:
                pass

        # ── Step 1: read the true PDF total ──────────────────────────────────
        prompt_s1 = _PROMPTS["total_mismatches_read_total"].format(
            issue=issue,
            field=field,
            table_num=table_num,
        )
        
        raw_s1   = _call_gemini(_base_payload(file_uri, prompt_s1), api_key)
        raw_s1   = _strip_fences(raw_s1)
        pdf_total = None
        s1_reason = "-> Step 1 API response could not be parsed."
        s1_ok     = False

        try:
            s1_result = json.loads(raw_s1)
            pdf_total = s1_result.get("pdf_total")
            s1_reason = s1_result.get("reason", "No reason provided.")
            s1_ok     = True
            print(f"->     PDF total: {pdf_total}\n->     Context: {s1_reason}")
        except json.JSONDecodeError as exc:
            continue

        # ── Step 2: spot-check every individual holding ───────────────────────
        table_df = _load_table_holdings(table_num, field, csv_dir, csv_prefix)
        phase1_corrections = []
        s2_ok = False

        if table_df is None or table_df.empty:
            continue
        else:
            holdings_block = table_df.to_csv(index=False)
            prompt_s2 = _PROMPTS["total_mismatches_spotcheck"].format(
                issue=issue,
                field=field,
                holdings_block=holdings_block,
            )
            print(f"-> Step 2: spot-checking {len(table_df)} holding(s) for '{field}'...")

            raw_s2 = _call_gemini(_base_payload(file_uri, prompt_s2), api_key)
            raw_s2 = _strip_fences(raw_s2)

            try:
                s2_results = json.loads(raw_s2)
                s2_ok = True
                for row in s2_results:
                    if not row.get("match", True) and row.get("corrected_value") is not None:
                        phase1_corrections.append(row)
                        print(f"->      Value mismatch: '{row.get('name')}' "
                              f"parsed={row.get('parsed_value')} "
                              f"pdf={row.get('pdf_value')}")
                if not phase1_corrections:
                    print(f"->     All {len(s2_results)} holding value(s) confirmed correct.")
            except json.JSONDecodeError as exc:
                print(f"->     Step 2 JSON parse error: {exc}")

        # ── Python-side classification ────────────────────────────────────────
        if not s1_ok and not s2_ok:
            verdict = MISMATCH_VERDICT_NEEDS_REVIEW
            reason  = "Both Step 1 and Step 2 responses could not be parsed."

        elif not s2_ok:
            verdict = MISMATCH_VERDICT_NEEDS_REVIEW
            reason  = f"Step 2 holdings comparison check failed." \
                      f"\n     Context: {s1_reason}"

        elif phase1_corrections:
            # Individual holding values are wrong — patch and re-run
            verdict = MISMATCH_VERDICT_VALUES_INCORRECT
            reason  = (f"{len(phase1_corrections)} holding value(s) are incorrect "
                       f"when compared to PDF table. Correcting those values in holdings.csv. ")

        elif not s1_ok:
            # Holdings all correct but we couldn't read the PDF total
            verdict = MISMATCH_VERDICT_NEEDS_REVIEW
            reason  = ("All holding values confirmed correct, but the LLM failed to "
                       "read the PDF total in Step 1. Manual review of the summary row required.")

        else:
            # Holdings all correct and we have a PDF total — compare to expected
            total_matches = (
                pdf_total is not None
                and expected_value is not None
                and abs(pdf_total - expected_value) <= 1.0   # $1 rounding tolerance
            )

            if total_matches:
                verdict = MISMATCH_VERDICT_CONFIRMED_TRUE
                reason  = (f"All individual holding values confirmed correct against PDF."
                           f"\n     PDF total ({pdf_total}) matches expected ({expected_value}). "
                           f"\n     The calculated sum genuinely differs from the pdf total (true mismatch), "
                           f"suggesting the pdf total is incorrect.")
            else:
                verdict = MISMATCH_VERDICT_BAD_TOTAL_PARSE
                reason  = (f"All individual holding values confirmed correct. "
                           f"\n     PDF total ({pdf_total}) differs from the expected value "
                           f"({expected_value}) originally extracted from the PDF — "
                           f"the summary row was mis-parsed. "
                           f"\n     The authoritative PDF total is {pdf_total}.")

        all_verdicts.append({
            "issue":              issue,
            "verdict":            verdict,
            "reason":             reason,
            "pdf_total":          pdf_total,
            "phase1_corrections": phase1_corrections,
        })

    return all_verdicts

# Creating a new class for handling LLM validation results
class LLMValidationReport:
    """
    Description:
        Stores and renders the results of LLM-based second-pass validation.

        Organises verdicts by category and verdict type, and produces both
        console output and a saved text file that mirrors the format of the
        first-pass initial_issue_report.txt.

    Verdict types (standard categories):
        CONFIRMED      — issue is real; reflects the source PDF accurately
        FALSE POSITIVE — parser artefact; PDF data is valid
        NEEDS REVIEW   — LLM uncertain; human inspection required

    Verdict types (total_mismatches):
        VALUES CORRECT   — holdings values match PDF; mismatch is in total row
                           or is a legitimate source discrepancy
        VALUES INCORRECT — at least one individual holding value was wrong
        NEEDS REVIEW     — could not locate relevant data in PDF

    Example:
        >>> report = LLMValidationReport()
        >>> report.add_verdicts("missing_symbol", [...])
        >>> report.print_report()
    """

    _CATEGORY_LABELS = {
        "missing_symbol":   "MISSING SYMBOL/CUSIP",
        "missing_data":     "MISSING FINANCIAL DATA",
        "negative_values":  "NEGATIVE VALUES",
        "total_mismatches": "TOTAL MISMATCHES",
        "missing_account":  "MISSING ACCOUNT",
    }

    def __init__(self):
        self._verdicts: dict[str, list[dict]] = {cat: [] for cat in _LLM_CATEGORIES}

    def add_verdicts(self, category: str, verdicts: list[dict]) -> None:
        """Append a list of verdict dicts for the given category."""
        if category in self._verdicts:
            self._verdicts[category].extend(verdicts)

    def _counts(self, category: str) -> dict[str, int]:
        if category == "total_mismatches":
            valid = _VALID_MISMATCH_VERDICTS
        elif category == "missing_account":
            valid = _VALID_ACCOUNT_VERDICTS
        else:
            valid = _VALID_VERDICTS
        counts = {v: 0 for v in valid}
        for item in self._verdicts[category]:
            v = item.get("verdict", VERDICT_NEEDS_REVIEW)
            if v in counts:
                counts[v] += 1
        return counts

    def total_verdicts(self) -> int:
        return sum(len(v) for v in self._verdicts.values())

    def print_report(self) -> None:
        """Print the LLM validation report to stdout."""
        print("\nLLM-ASSISTED SELF-CORRECTION VALIDATION RESULTS:")
        print(f"Model: {_GEMINI_MODEL}\n")

        total = self.total_verdicts()
        if total == 0:
            print("Zero issues detected - no need for LLM validation.")
            return

        print(f"Total Issues Evaluated: {total}\n")

        for category, label in self._CATEGORY_LABELS.items():
            items = self._verdicts[category]
            if not items:
                continue

            counts = self._counts(category)

            if category == "total_mismatches":
                print(
                    f"{len(items)} {label} issue(s) evaluated (when compared to PDF):\n "
                    f"{counts.get(MISMATCH_VERDICT_CONFIRMED_TRUE, 0)} confirmed true total mismatch(es) - pdf sum incorrect,\n "
                    f"{counts.get(MISMATCH_VERDICT_VALUES_INCORRECT, 0)} holding value(s) incorrect,\n "
                    f"{counts.get(MISMATCH_VERDICT_BAD_TOTAL_PARSE, 0)} expected total(s) parsed incorrectly,\n "
                    f"{counts.get(MISMATCH_VERDICT_NEEDS_REVIEW, 0)} need(s) review):\n"
                )
                for item in items:
                    verdict   = item.get("verdict", MISMATCH_VERDICT_NEEDS_REVIEW)
                    issue     = item.get("issue", "")
                    reason    = item.get("reason", "")
                    pdf_total = item.get("pdf_total")
                    print(f"- [{verdict}] {issue}")
                    print(f"     -> {reason}")
                    corrections = item.get("phase1_corrections", [])
                    if corrections:
                        print(f"     -> {len(corrections)} holding value(s) corrected:")
                        for c in corrections:
                            print(f"          {c.get('name')}: "
                                  f"{c.get('parsed_value')} -> {c.get('corrected_value')}")
            elif category == "missing_account":
                print(
                    f"{len(items)} {label} issue(s) Evaluated (when compared to PDF):\n "
                    f"{counts.get(ACCOUNT_VERDICT_RESOLVED, 0)} resolved (account info found),\n "
                    f"{counts.get(ACCOUNT_VERDICT_CONFIRMED, 0)} confirmed (no account info in PDF),\n "
                    f"{counts.get(ACCOUNT_VERDICT_NEEDS_REVIEW, 0)} need(s) review):\n"
                )
                for item in items:
                    verdict = item.get("verdict", ACCOUNT_VERDICT_NEEDS_REVIEW)
                    issue   = item.get("issue", "")
                    reason  = item.get("reason", "")
                    print(f"- [{verdict}] {issue}")
                    print(f"     -> {reason}")
                    if verdict == ACCOUNT_VERDICT_RESOLVED:
                        print(f"     -> Account: {item.get('account_type')} {item.get('account_number')}")
            else:
                print(
                    f"{len(items)} {label} issue(s) Evaluated (when compared to PDF):\n "
                    f"{counts[VERDICT_CONFIRMED]} confirmed true - matches pdf,\n "
                    f"{counts[VERDICT_FALSE_POSITIVE]} incorrect value(s),\n "
                    f"{counts[VERDICT_NEEDS_REVIEW]} need(s) review):\n"
                )
                for item in items:
                    verdict = item.get("verdict", VERDICT_NEEDS_REVIEW)
                    issue   = item.get("issue", "")
                    reason  = item.get("reason", "")
                    print(f"- [{verdict}] {issue}")
                    print(f"     -> {reason}")
            print()

    def save_to_file(self, filepath: str) -> None:
        """
        Saves the LLM validation report to a UTF-8 text file.

        Input:
            filepath: Full path for the output file.
        """
        old_stdout = sys.stdout
        sys.stdout = buffer = StringIO()
        self.print_report()
        sys.stdout = old_stdout

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(buffer.getvalue())

        print(f"LLM validation report saved to {filepath}!")

### Correction & Validation tasks/FXs ###

# Keyword lists mirroring parse_tools.classify_account_type — kept local to
# avoid a circular import between llm_validation_tools and parse_tools.
_TAX_DEFERRED_KW = ['traditional ira', 'sep ira', 'simple ira', '401(k)', '403(b)', '457']
_TAX_EXEMPT_KW   = ['roth ira', 'roth 401', 'roth 403', '529', 'education', 'hsa']
_TAXABLE_KW      = ['individual', 'joint', 'tod', 'tenants', 'trust', 'custodial', 'brokerage']

def _classify_account_type_str(account_type: str) -> str:
    """
    Classify an account type label into 'Tax-Deferred', 'Tax-Exempt',
    'Taxable', or 'Unknown'. Mirrors parse_tools.classify_account_type.
    """
    if not account_type:
        return 'Unknown'
    lowered = account_type.lower()
    for kw in _TAX_DEFERRED_KW:
        if kw in lowered:
            return 'Tax-Deferred'
    for kw in _TAX_EXEMPT_KW:
        if kw in lowered:
            return 'Tax-Exempt'
    for kw in _TAXABLE_KW:
        if kw in lowered:
            return 'Taxable'
    return 'Unknown'


@task(name="apply-corrections-to-holdings")
def apply_corrections_to_holdings(
    holdings_df: pd.DataFrame,
    all_verdicts: dict[str, list[dict]],
) -> tuple[pd.DataFrame, int]:
    """
    Description:
        Scans all FALSE POSITIVE verdicts (standard categories) and
        VALUES INCORRECT verdicts (total_mismatches) and patches the corrected
        values directly into the holdings DataFrame.

        Correction logic per category:
          missing_symbol   -> writes corrected_symbol into 'Symbol/CUSIP'
          missing_data     -> writes corrected_cost_basis into 'Cost Basis'
          negative_values  -> writes abs(corrected_value) into the flagged field
                              (Quantity, Cost Basis, or Market Value — inferred
                              from the issue string "Negative <Field> = ...")
          total_mismatches -> applies phase1_corrections: each correction entry
                              contains a security name and corrected_value for
                              the mismatched field, patched by name match

        Row matching uses the security name extracted from the issue string
        (text after " for "), compared against the 'Name' column with a
        case-insensitive substring match on the first 40 characters.

    Input:
        holdings_df:  Current holdings DataFrame (not mutated in-place).
        all_verdicts: Dict mapping category -> list of verdict dicts.

    Output:
        Tuple of (patched_df, correction_count, applied_log).
        applied_log is a list of dicts — one per attempted correction — with keys:
          "issue"     — the original issue string
          "field"     — the holdings column targeted
          "security"  — the security name fragment used for row matching
          "old_value" — the value before correction (or None)
          "new_value" — the intended corrected value
          "status"    — "APPLIED" or a "FAILED — <reason>" string
    """
    df = holdings_df.copy()
    correction_count = 0
    applied_log: list[dict] = []   # records every attempted correction and its outcome

    for category, verdicts in all_verdicts.items():
        for item in verdicts:

            # ── total_mismatches: apply Step 2 corrections ────────────────────
            if category == "total_mismatches":
                # Only patch when individual holding values were found wrong.
                # CONFIRMED_TRUE and BAD_TOTAL_PARSE require no holdings changes.
                if item.get("verdict") != MISMATCH_VERDICT_VALUES_INCORRECT:
                    continue
                for correction in item.get("phase1_corrections", []):
                    name_raw       = correction.get("name", "")
                    corrected_val  = correction.get("corrected_value")
                    field          = item.get("field") or _parse_mismatch_issue(
                                         item.get("issue", ""))[1]

                    if not name_raw or corrected_val is None or not field:
                        continue

                    fragment = name_raw[:40].strip()
                    mask = df["Name"].str.contains(re.escape(fragment), case=False, na=False)
                    if not mask.any():
                        print(f"  Warning: no holdings row matched for mismatch "
                              f"correction: '{fragment}'")
                        applied_log.append({
                            "issue":    item.get("issue", ""),
                            "field":    field,
                            "security": name_raw,
                            "old_value": correction.get("parsed_value"),
                            "new_value": corrected_val,
                            "status":   "FAILED — no row matched",
                        })
                        continue

                    try:
                        corrected_float = float(corrected_val)
                    except (TypeError, ValueError):
                        print(f"  Warning: corrected_value '{corrected_val}' is not numeric")
                        applied_log.append({
                            "issue":    item.get("issue", ""),
                            "field":    field,
                            "security": name_raw,
                            "old_value": correction.get("parsed_value"),
                            "new_value": corrected_val,
                            "status":   "FAILED — value not numeric",
                        })
                        continue

                    # Map the field name to the canonical holdings column
                    col = _mismatch_field_to_column(field, df.columns)
                    if col is None:
                        print(f"  Warning: could not map field '{field}' to a "
                              f"holdings column")
                        applied_log.append({
                            "issue":    item.get("issue", ""),
                            "field":    field,
                            "security": name_raw,
                            "old_value": correction.get("parsed_value"),
                            "new_value": corrected_val,
                            "status":   f"FAILED — could not map '{field}' to holdings column",
                        })
                        continue

                    old_vals = df.loc[mask, col].tolist()
                    df.loc[mask, col] = corrected_float
                    correction_count += mask.sum()
                    print(f"  Corrected {col} -> {corrected_float} "
                          f"for {mask.sum()} row(s) matching '{fragment}'")
                    for old_val in old_vals:
                        applied_log.append({
                            "issue":    item.get("issue", ""),
                            "field":    col,
                            "security": name_raw,
                            "old_value": old_val,
                            "new_value": corrected_float,
                            "status":   "APPLIED",
                        })
                continue

            # ── standard categories: apply FALSE POSITIVE / RESOLVED corrections ─
            # missing_account uses RESOLVED instead of FALSE POSITIVE; all others
            # use FALSE POSITIVE. Skip anything that isn't a correctable verdict.
            is_correctable = (
                (category == "missing_account" and
                 item.get("verdict") == ACCOUNT_VERDICT_RESOLVED) or
                (category != "missing_account" and
                 item.get("verdict") == VERDICT_FALSE_POSITIVE)
            )
            if not is_correctable:
                continue

            issue = item.get("issue", "")
            name_match = re.search(r'\bfor\s+(.+)$', issue, re.IGNORECASE)
            if not name_match:
                print(f"  Warning: could not parse security name from issue: '{issue}'")
                continue
            fragment = name_match.group(1).strip()[:40]

            mask = df["Name"].str.contains(re.escape(fragment), case=False, na=False)
            if not mask.any():
                print(f"  Warning: no holdings row matched for correction: '{fragment}'")
                applied_log.append({
                    "issue":    issue,
                    "field":    category,
                    "security": fragment,
                    "old_value": None,
                    "new_value": None,
                    "status":   "FAILED — no row matched",
                })
                continue

            if category == "missing_symbol":
                corrected = item.get("corrected_symbol")
                if corrected is None:
                    print(f"  Warning: no corrected_symbol for: '{issue}'")
                    continue
                old_vals = df.loc[mask, "Symbol/CUSIP"].tolist()
                df.loc[mask, "Symbol/CUSIP"] = str(corrected).strip()
                correction_count += mask.sum()
                print(f"  Corrected Symbol/CUSIP -> '{corrected}' "
                      f"for {mask.sum()} row(s) matching '{fragment}'")
                for old_val in old_vals:
                    applied_log.append({
                        "issue":    issue,
                        "field":    "Symbol/CUSIP",
                        "security": fragment,
                        "old_value": old_val,
                        "new_value": str(corrected).strip(),
                        "status":   "APPLIED",
                    })

            elif category == "missing_data":
                corrected = item.get("corrected_cost_basis")
                if corrected is None:
                    print(f"  Warning: no corrected_cost_basis for: '{issue}'")
                    continue
                try:
                    corrected_float = float(corrected)
                except (TypeError, ValueError):
                    print(f"  Warning: corrected_cost_basis '{corrected}' is not numeric")
                    continue
                old_vals = df.loc[mask, "Cost Basis"].tolist()
                df.loc[mask, "Cost Basis"] = corrected_float
                correction_count += mask.sum()
                print(f"  Corrected Cost Basis -> {corrected_float} "
                      f"for {mask.sum()} row(s) matching '{fragment}'")
                for old_val in old_vals:
                    applied_log.append({
                        "issue":    issue,
                        "field":    "Cost Basis",
                        "security": fragment,
                        "old_value": old_val,
                        "new_value": corrected_float,
                        "status":   "APPLIED",
                    })

            elif category == "negative_values":
                corrected = item.get("corrected_value")
                if corrected is None:
                    print(f"  Warning: no corrected_value for: '{issue}'")
                    continue
                try:
                    corrected_float = abs(float(corrected))
                except (TypeError, ValueError):
                    print(f"  Warning: corrected_value '{corrected}' is not numeric")
                    continue
                field_match = re.search(
                    r'Negative\s+(Quantity|Cost Basis|Market Value)',
                    issue, re.IGNORECASE
                )
                if not field_match:
                    print(f"  Warning: could not infer field from: '{issue}'")
                    continue
                field = field_match.group(1)
                old_vals = df.loc[mask, field].tolist()
                df.loc[mask, field] = corrected_float
                correction_count += mask.sum()
                print(f"  Corrected {field} -> {corrected_float} "
                      f"for {mask.sum()} row(s) matching '{fragment}'")
                for old_val in old_vals:
                    applied_log.append({
                        "issue":    issue,
                        "field":    field,
                        "security": fragment,
                        "old_value": old_val,
                        "new_value": corrected_float,
                        "status":   "APPLIED",
                    })

            elif category == "missing_account":
                # RESOLVED verdict carries account_type and account_number to write
                # into both the 'Account' and 'Account Type' holdings columns.
                acct_type = item.get("account_type")
                acct_num  = item.get("account_number")
                if not acct_type or not acct_num:
                    print(f"  Warning: no account_type/account_number for: '{issue}'")
                    continue
                new_account      = f"{acct_type} {acct_num}"
                new_account_type = _classify_account_type_str(acct_type)

                old_acct_vals = df.loc[mask, "Account"].tolist()
                df.loc[mask, "Account"]      = new_account
                df.loc[mask, "Account Type"] = new_account_type
                correction_count += mask.sum()
                print(f"  Resolved Account -> '{new_account}' (type: {new_account_type}) "
                      f"for {mask.sum()} row(s) matching '{fragment}'")
                for old_val in old_acct_vals:
                    applied_log.append({
                        "issue":    issue,
                        "field":    "Account",
                        "security": fragment,
                        "old_value": old_val,
                        "new_value": new_account,
                        "status":   "APPLIED",
                    })

    return df, correction_count, applied_log

def _mismatch_field_to_column(field: str, columns) -> str | None:
    """
    Map a human-readable mismatch field name (e.g. "Est. Annual Income")
    to its canonical holdings DataFrame column name via fuzzy matching.
    """
    field_lower = field.lower()
    # Priority: exact substring containment
    for col in columns:
        if field_lower in col.lower() or col.lower() in field_lower:
            return col
    # Fallback: shared meaningful words
    field_words = set(re.findall(r'[a-z]+', field_lower)) - {"the", "a", "an", "of", "est"}
    for col in columns:
        col_words = set(re.findall(r'[a-z]+', col.lower()))
        if field_words & col_words:
            return col
    return None

def collect_resolved_issues(
    all_verdicts: dict[str, list[dict]],
    correction_count: int,
) -> frozenset[str]:
    """
    Description:
        Extract the set of issue strings that have received a terminal verdict
        in this iteration and should not be re-submitted in subsequent iterations.

        Terminal verdicts by category:
          Standard (missing_symbol, missing_data, negative_values):
            CONFIRMED      — the issue is genuine and accurately reflected in data;
                             no correction possible, no value in re-checking
            FALSE POSITIVE — issue was a parser artefact; correction was applied
                             (or attempted); the issue no longer exists in the data

          total_mismatches:
            CONFIRMED TRUE MISMATCH — all holdings verified correct; mismatch is
                                      a genuine source-document discrepancy; not
                                      correctable by the pipeline
            BAD TOTAL PARSE         — expected value was mis-parsed; authoritative
                                      PDF total recorded; no further action possible
            VALUES INCORRECT        — included only when correction_count > 0,
                                      meaning at least one cell was actually patched.
                                      If corrections failed to apply (count == 0),
                                      the issue stays in rotation so the next
                                      iteration can retry the name-matching.

        NOT included (remain in rotation):
          NEEDS REVIEW  — LLM was uncertain; must be retried
          VALUES INCORRECT with correction_count == 0 — patch failed; must retry

    Input:
        all_verdicts:     Dict mapping category → list of verdict dicts, as
                          returned by run_category_verification / run_mismatch_verification.
        correction_count: Total number of cells actually patched in this iteration
                          by apply_corrections_to_holdings. Used to guard against
                          marking VALUES INCORRECT issues as resolved when the
                          name-match failed and no correction was written.

    Output:
        frozenset of issue strings that are fully resolved.
    """
    _STANDARD_TERMINAL = {VERDICT_CONFIRMED, VERDICT_FALSE_POSITIVE}
    _MISMATCH_TERMINAL  = {MISMATCH_VERDICT_CONFIRMED_TRUE, MISMATCH_VERDICT_BAD_TOTAL_PARSE}
    _ACCOUNT_TERMINAL   = {ACCOUNT_VERDICT_RESOLVED, ACCOUNT_VERDICT_CONFIRMED}

    resolved: set[str] = set()

    for category, verdicts in all_verdicts.items():
        for item in verdicts:
            issue   = item.get("issue", "")
            verdict = item.get("verdict", "")

            if category == "total_mismatches":
                if verdict in _MISMATCH_TERMINAL:
                    resolved.add(issue)
                elif verdict == MISMATCH_VERDICT_VALUES_INCORRECT and correction_count > 0:
                    resolved.add(issue)
            elif category == "missing_account":
                if verdict in _ACCOUNT_TERMINAL:
                    resolved.add(issue)
            else:
                if verdict in _STANDARD_TERMINAL:
                    resolved.add(issue)

    return frozenset(resolved)


class ValidationLedger:
    """
    Description:
        Append-only ledger that records every verdict and correction across all
        iterations of the self-correcting validation loop.

        Unlike LLMValidationReport (which is a snapshot of one iteration),
        ValidationLedger accumulates the full history of the pipeline run and
        is written to disk after every iteration so the record is preserved even
        if the pipeline crashes mid-run.

        The saved file has three sections:
          1. Per-iteration blocks — every issue evaluated that iteration, its
             verdict, and any correction attempted with its outcome
          2. HOLDINGS CHANGES SUMMARY — every cell that was actually written to
             holdings.csv, with before/after values
          3. FINAL ISSUE DISPOSITION — every issue from the original validation
             report grouped by its ultimate status across all iterations

    Usage:
        ledger = ValidationLedger(ledger_file)
        # inside the iteration loop, after corrections are applied:
        ledger.append_iteration(iteration, all_verdicts, applied_log)
        ledger.save()   # writes to disk immediately
    """

    def __init__(self, filepath: str) -> None:
        self._filepath  = filepath
        self._iterations: list[dict] = []   # one entry per completed iteration

    def append_iteration(
        self,
        iteration: int,
        all_verdicts: dict[str, list[dict]],
        applied_log: list[dict],
    ) -> None:
        """
        Description:
            Record the results of one completed iteration.

        Input:
            iteration:    1-based iteration number.
            all_verdicts: Dict mapping category -> list of verdict dicts, as
                          returned by run_category_verification /
                          run_mismatch_verification.
            applied_log:  List of correction attempt dicts returned by
                          apply_corrections_to_holdings. Empty list if no
                          corrections were attempted this iteration.
        """
        self._iterations.append({
            "iteration":   iteration,
            "all_verdicts": all_verdicts,
            "applied_log":  applied_log,
        })

    def save(self) -> None:
        """
        Write the full ledger to the file path provided at construction.
        Safe to call after every iteration — overwrites the previous version.
        """
        lines: list[str] = []
        _hr = "─" * 72

        lines.append("VALIDATION LEDGER")
        lines.append(f"Model: {_GEMINI_MODEL}")
        lines.append(f"Total iterations recorded: {len(self._iterations)}")
        lines.append("")

        # ── Per-iteration blocks ──────────────────────────────────────────────
        for entry in self._iterations:
            iteration   = entry["iteration"]
            all_verdicts = entry["all_verdicts"]
            applied_log  = entry["applied_log"]

            total_evaluated = sum(len(v) for v in all_verdicts.values())
            lines.append(_hr)
            lines.append(f"ITERATION {iteration}  —  {total_evaluated} issue(s) evaluated")
            lines.append(_hr)

            _CATEGORY_LABELS = {
                "missing_symbol":   "MISSING SYMBOL/CUSIP",
                "missing_data":     "MISSING FINANCIAL DATA",
                "negative_values":  "NEGATIVE VALUES",
                "total_mismatches": "TOTAL MISMATCHES",
                "missing_account":  "MISSING ACCOUNT",
            }

            for category, label in _CATEGORY_LABELS.items():
                items = all_verdicts.get(category, [])
                if not items:
                    continue
                lines.append(f"\n  {label}:")
                for item in items:
                    verdict = item.get("verdict", "")
                    issue   = item.get("issue", "")
                    reason  = item.get("reason", "")
                    lines.append(f"    [{verdict}] {issue}")
                    lines.append(f"      Reason: {reason}")

                    if category == "total_mismatches":
                        pdf_total = item.get("pdf_total")
                        for c in item.get("phase1_corrections", []):
                            lines.append(
                                f"      Holding correction identified: "
                                f"{c.get('name')} | "
                                f"{c.get('parsed_value')} -> {c.get('corrected_value')}"
                            )

            # Corrections applied this iteration
            if applied_log:
                lines.append(f"\n  CORRECTIONS APPLIED THIS ITERATION:")
                applied = [e for e in applied_log if e.get("status") == "APPLIED"]
                failed  = [e for e in applied_log if e.get("status", "").startswith("FAILED")]
                if applied:
                    for e in applied:
                        lines.append(
                            f"    [WRITTEN]  {e['security']} | "
                            f"{e['field']}: {e['old_value']} -> {e['new_value']}"
                        )
                if failed:
                    for e in failed:
                        lines.append(
                            f"    [FAILED]   {e['security']} | "
                            f"{e['field']}: {e['status']}"
                        )
            else:
                lines.append(f"\n  No corrections applied this iteration.")
            lines.append("")

        # ── Holdings changes summary ──────────────────────────────────────────
        all_applied = [
            e
            for entry in self._iterations
            for e in entry["applied_log"]
            if e.get("status") == "APPLIED"
        ]
        lines.append(_hr)
        lines.append(f"HOLDINGS CHANGES SUMMARY  —  {len(all_applied)} cell(s) written")
        lines.append(_hr)
        if all_applied:
            for e in all_applied:
                iter_num = next(
                    (
                        entry["iteration"]
                        for entry in self._iterations
                        if e in entry["applied_log"]
                    ),
                    "?",
                )
                lines.append(
                    f"  Iter {iter_num} | {e['security']} | "
                    f"{e['field']}: {e['old_value']} -> {e['new_value']}"
                )
        else:
            lines.append("  No cells were written to holdings.csv.")
        lines.append("")

        # ── Final issue disposition ───────────────────────────────────────────
        # Build a map of issue -> last verdict seen across all iterations
        last_verdict: dict[str, tuple[str, str]] = {}   # issue -> (verdict, category)
        for entry in self._iterations:
            for category, verdicts in entry["all_verdicts"].items():
                for item in verdicts:
                    issue   = item.get("issue", "")
                    verdict = item.get("verdict", "")
                    if issue:
                        last_verdict[issue] = (verdict, category)

        lines.append(_hr)
        lines.append(f"FINAL ISSUE DISPOSITION  —  {len(last_verdict)} unique issue(s)")
        lines.append(_hr)

        # Group issues by their final verdict
        groups: dict[str, list[str]] = {}
        for issue, (verdict, _) in sorted(last_verdict.items()):
            groups.setdefault(verdict, []).append(issue)

        # Print in a consistent order: terminal verdicts first, NEEDS REVIEW last
        verdict_order = [
            VERDICT_CONFIRMED,
            VERDICT_FALSE_POSITIVE,
            ACCOUNT_VERDICT_RESOLVED,
            ACCOUNT_VERDICT_CONFIRMED,
            MISMATCH_VERDICT_CONFIRMED_TRUE,
            MISMATCH_VERDICT_BAD_TOTAL_PARSE,
            MISMATCH_VERDICT_VALUES_INCORRECT,
            VERDICT_NEEDS_REVIEW,
            MISMATCH_VERDICT_NEEDS_REVIEW,
            ACCOUNT_VERDICT_NEEDS_REVIEW,
        ]
        printed = set()
        for verdict in verdict_order:
            if verdict in groups:
                lines.append(f"\n  [{verdict}] ({len(groups[verdict])} issue(s)):")
                for issue in groups[verdict]:
                    lines.append(f"    - {issue}")
                printed.add(verdict)
        # Any unexpected verdict strings
        for verdict, issues in groups.items():
            if verdict not in printed:
                lines.append(f"\n  [{verdict}] ({len(issues)} issue(s)):")
                for issue in issues:
                    lines.append(f"    - {issue}")
        lines.append("")

        with open(self._filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        print(f"Validation ledger saved to {self._filepath}")


@task(name="build-llm-report")
def build_llm_report(all_verdicts: dict[str, list[dict]]) -> LLMValidationReport:
    """
    Description:
        FX assembles an LLMValidationReport from per-category verdict lists.

    Input:
        all_verdicts: Dict mapping category name → list of verdict dicts,
                      as returned by run_category_verification().

    Output:
        Populated LLMValidationReport instance.
    """
    report = LLMValidationReport()
    for category, verdicts in all_verdicts.items():
        report.add_verdicts(category, verdicts)
    return report