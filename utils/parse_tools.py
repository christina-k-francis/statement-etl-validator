"""
Helpful functions and classes for parcing financial data from extracted CSVs.
"""

import pandas as pd
import numpy as np
import glob
import json
import re
import sys
import requests
from io import StringIO
from prefect import task

### Mapping for all potential variations of column categories ###
_COLUMN_KEYWORDS = {
    # ── existing fields ──────────────────────────────────────────────────────
    'market_value': [
        'ending market value',      # Fidelity standard
        'ending market',            # Fidelity accrued-interest variant
        'ending value',             # Fidelity "Ending Value"
        'market value',             # JPM "Market Value" / UBS "Market value ($)"
        'market val',               # catch-all abbreviation
    ],
    'cost_basis': [
        'total cost basis',         # Fidelity "Total Cost Basis"
        'cost basis',               # JPM "Cost Basis" / UBS "Cost basis ($)"
        'total cost',               # catch-all
    ],
    'quantity': [
        'quantity',                 # consistent across all three brokerages
    ],

    # ── new fields ────────────────────────────────────────────────────────────
    'gain_loss': [
        'unrealized gain/loss',     # Fidelity / JPM exact
        'unrealized gain / loss',   # spacing variant
        'gain/loss',                # UBS "Unrealized gain/loss ($)"
        'gain / loss',              # spacing variant
        'unrealized gain',          # catch-all
    ],
    'annual_income': [
        'est. annual income',       # Fidelity "Est. Annual Income (EAI)"
        'est. annual inc',          # JPM "Est. Annual Inc."
        'annual income',            # UBS "Est. annual income ($)"
        'annual inc',               # short form
        'eai',                      # Fidelity parenthetical abbreviation
    ],
    'yield': [
        'est. yield',               # Fidelity "Est. Yield (EY)"
        'current yield',            # UBS "Current yield (%)"
        'coupon rate',              # Fidelity bonds tables
        'yield',                    # broad fallback
    ],
}

def resolve_column(table_columns, field_key):
    """
    Description:
        Find the best matching column name for a logical field by scanning
        actual column headers with fuzzy keyword matching.

        Each field has an ordered list of keyword fragments (see _COLUMN_KEYWORDS).
        The function returns the first actual column whose lowercased name contains
        any fragment, checked in priority order so that a more-specific match wins
        over a looser one when multiple columns could qualify.

    Input:
        table_columns: iterable of column name strings (e.g. df.columns)
        field_key:     one of the keys in _COLUMN_KEYWORDS

    Output:
        str or None: matched column name exactly as it appears in table_columns,
                     or None if no column matches
    """
    fragments = _COLUMN_KEYWORDS.get(field_key, [])
    lowered = {col: col.lower() for col in table_columns}

    for fragment in fragments:
        for col, col_lower in lowered.items():
            if fragment in col_lower:
                return col
    return None

### Account type classification mapping ###
_TAX_DEFERRED_KEYWORDS = [
    'traditional ira', 'sep ira', 'simple ira',
    '401(k)', '403(b)', '457',
]
_TAX_EXEMPT_KEYWORDS = [
    'roth ira', 'roth 401', 'roth 403',
    '529', 'education', 'hsa',
]
_TAXABLE_KEYWORDS = [
    'individual', 'joint', 'tod', 'tenants',
    'trust', 'custodial', 'brokerage',
]

def classify_account_type(account_string):
    """
    Description:
        Classify an account string into one of three tax-treatment categories:
        'Tax-Deferred', 'Tax-Exempt', or 'Taxable'.

        Classification uses priority-ordered keyword scanning against the three
        module-level lists (_TAX_DEFERRED_KEYWORDS, _TAX_EXEMPT_KEYWORDS,
        _TAXABLE_KEYWORDS).  Tax-deferred is checked first so that a string
        containing both 'traditional ira' and 'individual' resolves correctly
        to 'Tax-Deferred' rather than 'Taxable'.

        Returns 'Unknown' when no keyword matches, preserving auditability for
        account strings that do not yet have a keyword mapping.

    Input:
        account_string: str — the assembled account label, e.g.
                        'Traditional IRA 222-222222' or
                        'Individual - TOD 111-111111'

    Output:
        str: one of 'Tax-Deferred', 'Tax-Exempt', 'Taxable', or 'Unknown'
    """
    if not account_string or not isinstance(account_string, str):
        return 'Unknown'

    lowered = account_string.lower()

    for kw in _TAX_DEFERRED_KEYWORDS:
        if kw in lowered:
            return 'Tax-Deferred'

    for kw in _TAX_EXEMPT_KEYWORDS:
        if kw in lowered:
            return 'Tax-Exempt'

    for kw in _TAXABLE_KEYWORDS:
        if kw in lowered:
            return 'Taxable'

    return 'Unknown'

### Handling Account Information ###
@task(name="build-account-map")
def build_account_map(tables, acc_types, acc_nums):
    """
    Description:
        Creates a mapping of table indices to account information by finding tables
        that contain 'Beginning Account Value' marker.
    
    Input:
        tables: list of pandas DataFrames (all tables)
        acc_types: list of account types
        acc_nums: list of account numbers
        
    Output:
        dict: mapping of table index to account index
    """
    table_to_account_map = {}
    current_account_idx = 0
    
    for idx, table in enumerate(tables):
        # Check if this table contains 'Beginning Account Value'
        contains_beginning_portfolio = False
        
        # Check all cells in the table for the marker string
        for col in table.columns:
            if table[col].astype(str).str.contains('Beginning Account Value', na=False).any():
                contains_beginning_portfolio = True
                break
        
        if contains_beginning_portfolio:
            # This table marks a new account
            if current_account_idx < len(acc_types) and current_account_idx < len(acc_nums):
                table_to_account_map[idx] = current_account_idx
                current_account_idx += 1
    
    return table_to_account_map

def get_account_for_table(table_idx, table_to_account_map):
    """
    Description:
        Returns the account index for a given table index.
        The account applies from the most recent 'Beginning Account Value' marker.
    
    Input:
        table_idx: index of the table in the master tables list
        table_to_account_map: dict mapping table indices to account indices
        
    Output:
        int or None: account index, or None if no applicable account found
    """
    # Find the most recent account marker that is <= current table_idx
    applicable_account_idx = None
    for marker_table_idx in sorted(table_to_account_map.keys()):
        if marker_table_idx <= table_idx:
            applicable_account_idx = table_to_account_map[marker_table_idx]
        else:
            break
    return applicable_account_idx

### Handling Stocks/Assets ###
def extract_symbol_and_name(description):
    """
    Description:
        Extract symbol/CUSIP/ISIN and name from description.
        - Parentheses: symbol is text inside (...), name is text before
        -ISIN: 12-character alphanumeric code following 'ISIN:'
        - CUSIP: 9-character alphanumeric code following 'CUSIP:'
        - Symbol: 1-9 alphanumeric characters following 'Symbol:', where any
           character may be replaced by '.', ':', or '-' to account for international stocks
    
    Input:
        description: string containing the full description, from description column
        
    Output:
        tuple: (name, symbol) where symbol is the extracted identifier, 
                and name is everything before the matched pattern
    """
    
    if pd.isna(description) or not isinstance(description, str):
        return None, None
    
    # 1. Find text in parentheses
    match = re.search(r'\(([^)]+)\)', description)
    if match:
        symbol = match.group(1)  # Text inside parentheses
        name = description[:match.start()].strip()
        return name, symbol

    # 2. Check for ISIN pattern: exactly 12 alphanumeric characters after 'ISIN:'
    isin_match = re.search(r'ISIN:\s*([A-Z0-9]{12})', description, re.IGNORECASE)
    if isin_match:
        symbol = isin_match.group(1)
        name = description[:isin_match.start()].strip()
        return name, symbol

    # 3. Check for CUSIP pattern: exactly 9 alphanumeric characters after 'CUSIP:'
    cusip_match = re.search(r'CUSIP:\s*([A-Z0-9]{9})', description, re.IGNORECASE)
    if cusip_match:
        symbol = cusip_match.group(1)
        name = description[:cusip_match.start()].strip()
        return name, symbol

    # 4. Check for Symbol pattern: 1-9 alphanumeric characters after 'Symbol:' 
    # and potential internal characters may be '.', ':', or '-' 
    symbol_match = re.search(r'Symbol:\s*([A-Z0-9](?:[A-Z0-9.\-:]{0,7}[A-Z0-9])?)', description, re.IGNORECASE)
    if symbol_match:
        symbol = symbol_match.group(1)
        name = description[:symbol_match.start()].strip()
        return name, symbol

    # No pattern matched - return whole description as name
    return description.strip(), None
     
# Mapping for asset class types (e.g. fixed income, funds, cash, equities)
_ASSET_CLASS_HEADER_MAP = [
    # ── Fixed Income ─────────────────────────────────────────────────────────
    ('corporate bond',      'Fixed Income'),
    ('municipal bond',      'Fixed Income'),
    ('government',          'Fixed Income'),
    ('agency',              'Fixed Income'),
    ('asset backed',        'Fixed Income'),
    ('treasury',            'Fixed Income'),
    ('fixed income',        'Fixed Income'),
    # ── Funds (before broad 'bond' so "bond fund" → Funds, not Fixed Income) ─
    ('bond fund',           'Funds'),
    ('mutual fund',         'Funds'),
    ('etf',                 'Funds'),
    ('fund',                'Funds'),
    # ── Fixed Income broad fallback ──────────────────────────────────────────
    ('bond',                'Fixed Income'),
    # ── Cash & Equivalents ───────────────────────────────────────────────────
    ('cash',                'Cash & Equivalents'),
    ('money market',        'Cash & Equivalents'),
    ('sweep',               'Cash & Equivalents'),
    ('deposit',             'Cash & Equivalents'),
    # ── Equities ─────────────────────────────────────────────────────────────
    ('common stock',        'Equities'),
    ('preferred stock',     'Equities'),
    ('equities',            'Equities'),
    ('equity',              'Equities'),
    ('stock',               'Equities'),
]

# Keyword fragments used for name/symbol-based fallback classification.
_ASSET_CLASS_NAME_KEYWORDS = [
    # ── Cash & Equivalents ───────────────────────────────────────────────────
    ('money market',        'Cash & Equivalents'),
    ('cash',                'Cash & Equivalents'),
    ('sweep',               'Cash & Equivalents'),
    ('deposit',             'Cash & Equivalents'),
    # ── Funds (before Fixed Income so "bond fund" → Funds, not Fixed Income) ─
    ('etf',                 'Funds'),
    ('fund',                'Funds'),
    ('trust',               'Funds'),           # e.g. "iShares Trust"
    # ── Fixed Income ─────────────────────────────────────────────────────────
    ('bond',                'Fixed Income'),
    ('note',                'Fixed Income'),
    ('treasury',            'Fixed Income'),
    ('muni',                'Fixed Income'),
    ('fixed',               'Fixed Income'),
    ('income',              'Fixed Income'),    # e.g. "Short Duration Income"
]

def classify_asset_class_from_header(description):
    """
    Description:
        Map a raw category-header string (as found inline in the Description
        column of Fidelity-style holdings tables) to a canonical asset class
        label.

        Matching is case-insensitive substring containment against the ordered
        _ASSET_CLASS_HEADER_MAP list.  The first matching entry wins, so more-
        specific entries take precedence over broader fallbacks.

        This is the primary classifier used by the state machine in
        01_parse_holdings.py: when a category-header row is detected, this
        function maps it to a canonical label that is then carried forward as
        the current_asset_class for all subsequent holding rows in that section.

    Input:
        description: str — the raw Description cell value, e.g.
                     'Corporate Bonds', 'Common Stock', 'Cash/Money Market'

    Output:
        str or None: one of 'Equities', 'Fixed Income', 'Cash & Equivalents',
                     'Funds', or None if no entry in the map matches
    """
    if not description or not isinstance(description, str):
        return None

    lowered = description.lower()

    for fragment, label in _ASSET_CLASS_HEADER_MAP:
        if fragment in lowered:
            return label

    return None

def classify_asset_class_from_name(name, symbol=None):
    """
    Description:
        Infer asset class from a holding's name and, optionally, its ticker
        symbol.  This is the fallback classifier used when no inline category-
        header row has been seen (e.g. UBS and J.P. Morgan statements) or when
        the state-machine tracker has not yet been set for the current row.

        Matching is case-insensitive.  _ASSET_CLASS_NAME_KEYWORDS is checked
        in order so that 'bond fund' resolves to 'Funds' rather than
        'Fixed Income'.  If no keyword matches, returns 'Equities' as the
        default — individual stocks carry no descriptive keywords so the
        absence of other signals is itself the signal.

    Input:
        name:   str or None — holding name, e.g. 'Goldman Sachs High Yield Fund I'
        symbol: str or None — ticker or CUSIP, e.g. 'GS', '38141G104'

    Output:
        str: one of 'Equities', 'Fixed Income', 'Cash & Equivalents', 'Funds'
    """
    search_text = ' '.join(filter(None, [
        str(name)   if name   else '',
        str(symbol) if symbol else '',
    ])).lower()

    for fragment, label in _ASSET_CLASS_NAME_KEYWORDS:
        if fragment in search_text:
            return label

    return 'Equities'  # default: individual stocks carry no descriptive keywords

### Helpful FXs for collating pertinent, parsed data ###
def clean_currency_value(value):
    """
    Description:
        Cleans currency and numeric values by stripping '$', commas, and common
        letter suffixes, then converting to float.

        Handles parenthetical negatives: e.g. "(9,008.50)" → -9008.50,
        as commonly used in JP Morgan and other brokerage statements.

    Input:
        value: String or numeric value potentially containing currency symbols

    Output:
        float or None: Cleaned numeric value
    """
    if pd.isna(value):
        return None

    if isinstance(value, (int, float)):
        return float(value)

    if isinstance(value, str):
        stripped = value.strip()

        # Detect and handle parenthetical negative notation: (1,234.56) → -1234.56
        is_negative = stripped.startswith('(') and stripped.endswith(')')
        if is_negative:
            stripped = '-' + stripped[1:-1]

        cleaned = (stripped
                   .replace('$', '')
                   .replace(',', '')
                   .replace('c', '')
                   .replace('B', '')
                   .replace('t', '')
                   .replace('*', '')
                   .strip())

        if cleaned in ('-', '', '--') or cleaned.lower() in ('unknown', 'nan', 'n/a'):
            return None

        try:
            return float(cleaned)
        except ValueError:
            return None

    return None

def clean_percentage_value(value):
    """
    Description:
        Cleans percentage values by stripping the '%' character and converting
        to a float representing the percentage magnitude (e.g. "4.68%" → 4.68).

        Used for Est. Yield (EY) and Coupon Rate fields, which are stored as
        percentage strings rather than plain numeric values.  The returned float
        is the raw percentage number — it is NOT divided by 100.

    Input:
        value: String or numeric value potentially containing a '%' character

    Output:
        float or None: Cleaned percentage as a plain float
    """
    if pd.isna(value):
        return None

    if isinstance(value, (int, float)):
        return float(value)

    if isinstance(value, str):
        cleaned = value.replace('%', '').strip()

        if cleaned in ('-', '', '--') or cleaned.lower() in ('unknown', 'nan', 'n/a'):
            return None

        try:
            return float(cleaned)
        except ValueError:
            return None

    return None

### Helpful FXs for Data Quality Assurance and Validation ###
def validate_record(record, issue_report):
    """
    Description:
        Validate a single financial record for data quality issues.

        Note: Unrealized Gain/Loss is intentionally excluded from the negative-value
        check because losses are legitimately negative and must not be flagged as errors.

    Input:
        record:            Dict containing financial data
        issue_report: IssueReport instance
    """
    # Negative-value check — excludes Unrealized Gain/Loss (legitimately negative)
    for field in ['Quantity', 'Cost Basis', 'Market Value']:
        value = record.get(field)
        if value is not None and value < 0:
            issue_report.add_issue(
                'negative_values',
                f"Negative {field} = {value} for {record.get('Name', 'Unknown')}"
            )

    # Abnormally large stock quantities (> 1,000,000 shares)
    quantity = record.get('Quantity')
    if quantity is not None and quantity > 1_000_000:
        issue_report.add_issue(
            'abnormal_quantities',
            f"Abnormally large quantity = {quantity:,.0f} for {record.get('Name', 'Unknown')}"
        )

    # Missing Symbol/CUSIP
    if pd.isna(record.get('Symbol/CUSIP')) or record.get('Symbol/CUSIP') == '':
        issue_report.add_issue(
            'missing_symbol',
            f"Missing Symbol/CUSIP for {record.get('Name', 'Unknown')}"
        )

    # Missing Account
    if (pd.isna(record.get('Account')) or
            record.get('Account') == '' or
            record.get('Account') == 'Unknown Account'):
        issue_report.add_issue(
            'missing_account',
            f"Missing or Unknown Account for {record.get('Name', 'Unknown')}"
        )

    # Missing critical financial data
    for field in ['Quantity', 'Cost Basis', 'Market Value']:
        if record.get(field) is None:
            issue_report.add_issue(
                'missing_data',
                f"Missing {field} for {record.get('Name', 'Unknown')}"
            )

def extract_total_row(table):
    """
    Description:
        The FX extracts validation totals by summing sub-section summary rows from a table, excluding higher-level aggregates.
        Higher-level aggregates always appear in the row directly beneath a sub-section summary.

    Input:
        table: Pandas DataFrame

    Output:
        dict: Summed summary values keyed by logical field name,
              or None if no total rows are found.
              Keys: 'Quantity', 'Cost Basis', 'Market Value',
                    'Unrealized Gain/Loss', 'Est. Annual Income'
    """
    if 'Description' not in table.columns:
        return None

    # Resolve all column names once via fuzzy matching
    col_map = {
        'Quantity':             resolve_column(table.columns, 'quantity'),
        'Cost Basis':           resolve_column(table.columns, 'cost_basis'),
        'Market Value':         resolve_column(table.columns, 'market_value'),
        'Unrealized Gain/Loss': resolve_column(table.columns, 'gain_loss'),
        'Est. Annual Income':   resolve_column(table.columns, 'annual_income'),
    }

    # Find all rows starting with 'Total', excluding grand totals
    total_mask = (
        table['Description'].notna() &
        table['Description'].str.match(r'^\s*Total', case=False, na=False) &
        ~table['Description'].str.contains('Total Holdings', case=False, na=False) &
        ~table['Description'].str.contains('Total Portfolio', case=False, na=False)
    )

    total_indices = table.index[total_mask].tolist()

    if not total_indices:
        return None

    # Exclude higher-level aggregates: any Total row immediately after another
    all_table_indices = table.index.tolist()
    sub_section_indices = []

    for idx in total_indices:
        pos = all_table_indices.index(idx)
        if pos > 0 and all_table_indices[pos - 1] in total_indices:
            continue  # This is an aggregate row — skip it
        sub_section_indices.append(idx)

    if not sub_section_indices:
        return None

    # Accumulate sums across sub-section summary rows
    sums  = {field: 0.0   for field in col_map}
    found = {field: False for field in col_map}

    for idx in sub_section_indices:
        row = table.loc[idx]
        for field, col in col_map.items():
            if col is None:
                continue
            val = clean_currency_value(row.get(col))
            if val is not None:
                sums[field]  += val
                found[field]  = True

    return {field: (sums[field] if found[field] else None) for field in col_map}


def validate_totals(table_records, table_summary, table_idx, issue_report, tolerance=0.01):
    """
    Description:
        Validate that the sum of individual records matches the table's total row for all summable numeric fields.
        Checks: Quantity, Cost Basis, Market Value, Unrealized Gain/Loss, and Est. Annual Income.
        Est. Yield is intentionally excluded because it is a per-holding ratio and is not meaningfully summable.

    Input:
        table_records:     List of dicts containing individual records from the table
        table_summary:     Dict containing summary values from the total row
        table_idx:         Index of the table for reporting
        issue_report: IssueReport instance
        tolerance:         Acceptable percentage difference (default 1%)
    """
    if not table_summary or not table_records:
        return

    fields_to_check = [
        'Quantity',
        'Cost Basis',
        'Market Value',
        'Unrealized Gain/Loss',
        'Est. Annual Income',
    ]

    calculated_sums = {
        field: sum(r.get(field, 0) or 0 for r in table_records)
        for field in fields_to_check
    }

    for field in fields_to_check:
        expected   = table_summary.get(field)
        calculated = calculated_sums.get(field)

        if expected is None or expected == 0:
            continue

        diff     = abs(calculated - expected)
        pct_diff = (diff / abs(expected)) * 100

        if pct_diff > tolerance:
            issue_report.add_issue(
                'total_mismatches',
                f"Table {table_idx+1}: {field} sum/total mismatch — " #+1 hard-coded bc CSVs are numbered from 1
                f"Expected: ${expected:,.2f}, "
                f"Calculated: ${calculated:,.2f}"
            )

### Defining a class for data quality validation ###
class IssueReport:
    """
    Description:
        Class to track and report data validation issues.
        
        Tracks issues across multiple categories:
        - negative_values: Negative quantities, costs, or market values
        - abnormal_quantities: Unusually large quantities (> 1M shares)
        - missing_symbol: Records without Symbol/CUSIP
        - missing_account: Records without account information
        - missing_data: Records missing Quantity, Cost Basis, or Market Value
        - total_mismatches: Discrepancies between calculated and expected totals
    
    Example:
        >>> report = IssueReport()
        >>> report.add_issue('negative_values', 'Record 5: Negative quantity = -100')
        >>> report.print_report()
    """
    
    def __init__(self):
        self.issues = {
            'negative_values': [],
            'abnormal_quantities': [],
            'missing_symbol': [],
            'missing_account': [],
            'missing_data': [],
            'total_mismatches': [],
            # Page ranges from 00_extract_tables that could not be extracted from the PDF
            'extraction_failures': [],
        }
    
    def add_issue(self, issue_type, details):
        """
        Description:
            Add a validation issue to the report.
        
        Input:
            issue_type: Type of issue (must match a key in self.issues)
            details: Description of the issue
        """
        if issue_type in self.issues:
            self.issues[issue_type].append(details)
        else:
            raise ValueError(f"Unknown issue type: {issue_type}")
    
    def get_total_issues(self):
        """
        Description:
            Return the total count of all issues.
        """
        return sum(len(issues) for issues in self.issues.values())
    
    def has_issues(self):
        """
        Description:
            Return True if any issues were found.
        """
        return self.get_total_issues() > 0
    
    def print_report(self):
        """
        Description:
            Prints a comprehensive initial issue report to the console.
        """
        print("INITIAL ISSUE REPORT:")        
        
        total_issues = self.get_total_issues()
        
        if total_issues == 0:
            print("No validation issues found!")
            return
        
        print(f"Total Issues Found: {total_issues}\n")
        
        # Negative values
        if self.issues['negative_values']:
            print(f"NEGATIVE VALUES ({len(self.issues['negative_values'])} issues):")
            for issue in self.issues['negative_values']:
                print(f"  - {issue}")
            print()
        
        # Abnormal quantities
        if self.issues['abnormal_quantities']:
            print(f"ABNORMAL QUANTITIES ({len(self.issues['abnormal_quantities'])} issues):")
            for issue in self.issues['abnormal_quantities']:
                print(f"  - {issue}")
            print()
        
        # Missing symbols
        if self.issues['missing_symbol']:
            print(f"MISSING SYMBOL/CUSIP ({len(self.issues['missing_symbol'])} issues):")
            for issue in self.issues['missing_symbol']:
                print(f"  - {issue}")
            print()
        
        # Missing accounts
        if self.issues['missing_account']:
            print(f"MISSING ACCOUNT ({len(self.issues['missing_account'])} issues):")
            for issue in self.issues['missing_account']:
                print(f"  - {issue}")
            print()

        # Missing financial data
        if self.issues['missing_data']:
            print(f"MISSING FINANCIAL DATA ({len(self.issues['missing_data'])} issues):")
            for issue in self.issues['missing_data']:
                print(f"  - {issue}")
            print()
        
        # Total mismatches
        if self.issues['total_mismatches']:
            print(f"TOTAL MISMATCHES ({len(self.issues['total_mismatches'])} issues):")
            for issue in self.issues['total_mismatches']:
                print(f"  - {issue}")
            print()

        # Extraction failures (upstream, from 00_extract_tables)
        if self.issues['extraction_failures']:
            print(f"EXTRACTION FAILURES — PAGES NEVER PROCESSED "
                  f"({len(self.issues['extraction_failures'])} issue(s)):")
            print("  Holdings on these PDF page ranges are ABSENT from the output.")
            for issue in self.issues['extraction_failures']:
                print(f"  - {issue}")
            print()
        
    
    def save_to_file(self, filepath):
        """
        Description:
            Saves the initial issue report to a text file.
        
        Input:
            filepath: Path to the output file
        """
        
        # Capture the initial issue report output
        old_stdout = sys.stdout
        sys.stdout = report_buffer = StringIO()
        self.print_report()
        sys.stdout = old_stdout
        
        with open(filepath, 'w') as f:
            f.write(report_buffer.getvalue())


### Helpful FXs/Tasks for the parsing pipeline ###
@task(name="load-csv-tables", task_run_name="load-csv-tables: {csv_dir}")
def load_csv_tables(csv_dir: str, csv_prefix: str) -> list[pd.DataFrame]:
    """
    Description:
        Task that reads all numbered CSV files from csv_dir into DataFrames.
        Files are expected to follow the naming convention: {csv_prefix}{n}.csv

        Header row detection is automatic: the first row whose values are all
        non-numeric strings is treated as the header. This accommodates both
        Fidelity-style CSVs (where row 0 is already the header) and JP Morgan-
        style CSVs (where row 0 is a numeric index and row 1 is the header).

    Input:
        csv_dir:    Directory containing the numbered CSV files.
        csv_prefix: Filename prefix including the table number separator,
                    e.g. 'fidelity_table_' matches fidelity_table_1.csv, etc.

    Output:
        Ordered list of DataFrames, one per CSV file.
    """
    def _detect_header_row(filepath: str) -> int:
        """
        Return the index of the first row that looks like a header:
        a row where every non-empty cell EXCEPT the first is a non-numeric string.
        Skipping the first cell accommodates CSVs that have a leading numeric
        row-index column (e.g. legacy output from save_table_to_csv).
        Falls back to row 0 if no such row is found in the first 5 rows.
        """
        probe = pd.read_csv(filepath, header=None, nrows=5)
        for i, row in probe.iterrows():
            non_empty = row.dropna().astype(str)
            if non_empty.empty:
                continue
            # Skip the first cell — it may be a numeric row index artifact
            candidate_cells = non_empty.iloc[1:] if len(non_empty) > 1 else non_empty
            # A header row has no remaining cells that parse cleanly as a plain number
            if all(not re.fullmatch(r'-?\d+(\.\d+)?', cell.strip()) for cell in candidate_cells):
                return i
        return 0

    def _drop_index_column(df: pd.DataFrame) -> pd.DataFrame:
        """
        Drop a leading unnamed numeric index column if present.
        This arises when CSVs were saved with a row-index column whose header
        is an empty string (rendered as 'Unnamed: 0' by pandas).

        Guard: only drop if ALL non-null values in the column are plain integers
        (e.g. 0, 1, 2, ...). This preserves legitimate unnamed description
        columns — such as Fidelity account-summary tables — whose first column
        has no header but contains meaningful string values like
        'Beginning Account Value'. Dropping those would silently remove the
        sentinel that build_account_map depends on.
        """
        if df.columns[0] not in ('', 'Unnamed: 0'):
            return df
        col0_values = df.iloc[:, 0].dropna().astype(str)
        all_numeric = all(re.fullmatch(r'\d+', v.strip()) for v in col0_values)
        if all_numeric:
            df = df.iloc[:, 1:]
        return df

    tables = []
    for table_num in np.arange(1, len(glob.glob(f'{csv_dir}/*.csv')) + 1):
        filepath = f'{csv_dir}/{csv_prefix}{table_num}.csv'
        header_row = _detect_header_row(filepath)
        df = pd.read_csv(filepath, header=header_row)
        df = _drop_index_column(df)
        tables.append(df)
    return tables


_GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
_FILE_API_URL   = "https://generativelanguage.googleapis.com/upload/v1beta/files"
_FILE_GET_URL   = "https://generativelanguage.googleapis.com/v1beta/files/{file_id}"

_ACCOUNT_EXTRACTION_PROMPT = """
You are a financial document parser. You have been given a brokerage statement PDF.

Your task: identify ALL distinct brokerage accounts in this document and return their
account type/name and account number.

Look everywhere in the document — page headers, account summary tables, holdings
section headings, cover pages, and any table that lists account numbers.

Respond ONLY with a valid JSON array — no markdown fences, no preamble.
Each element must have exactly these keys:
  "account_type" : a short human-readable label for the account type, e.g.
                   "Individual TOD", "Traditional IRA", "Roth IRA",
                   "Joint Tenants", "Trust", "Brokerage", etc.
                   Use the exact wording from the document where possible.
  "account_number": the account number exactly as it appears in the document,
                    e.g. "111-111111" or "X12-345678"

If no account information can be found anywhere in the document, return an empty
array: []
"""

@task(name="llm-extract-account-info", task_run_name="llm-extract-account-info: {pdf_path}")
def llm_extract_account_info(pdf_path: str, api_key: str) -> tuple[list, list]:
    """
    Description:
        Calls Gemini with the full brokerage PDF to extract account type and
        account number information. Used as a pre-pass fallback when no
        structured account summary table is found in the extracted CSVs
        (e.g. J.P. Morgan statements, where account info only appears in
        page headers and is not captured as a CSV table).

        Makes a single API call: uploads the PDF to the Gemini File API,
        sends the account-extraction prompt, and parses the JSON response.

        On any failure (upload error, API error, JSON parse error), prints
        a warning and returns empty lists so the pipeline can continue.

    Input:
        pdf_path: Local path to the brokerage statement PDF.
        api_key:  Gemini API key string.

    Output:
        Tuple of (acc_types, acc_nums) — parallel lists of account type
        strings and account number strings, in document order.
        Both lists are empty if account info cannot be extracted.
    """
    import time

    print(f"  Uploading PDF to Gemini for account extraction: {pdf_path}")
    try:
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()

        upload_response = requests.post(
            _FILE_API_URL,
            params={"key": api_key},
            headers={"X-Goog-Upload-Protocol": "multipart"},
            files={
                "metadata": (None, '{"file": {"mimeType": "application/pdf"}}', "application/json"),
                "file":     (None, pdf_bytes, "application/pdf"),
            },
            timeout=120,
        )
        upload_response.raise_for_status()
        file_info = upload_response.json().get("file", {})
        file_name = file_info.get("name", "")
        file_uri  = file_info.get("uri", "")

        if not file_name:
            print("  Warning (llm_extract_account_info): PDF upload returned no file name.")
            return [], []

        # Poll until ACTIVE
        file_id = file_name.split("/")[-1]
        for _ in range(30):
            status_resp = requests.get(
                _FILE_GET_URL.format(file_id=file_id),
                params={"key": api_key},
                timeout=30,
            )
            status_resp.raise_for_status()
            if status_resp.json().get("state", "") == "ACTIVE":
                break
            time.sleep(2)
        else:
            print("  Warning (llm_extract_account_info): PDF did not reach ACTIVE state.")
            return [], []

    except Exception as exc:
        print(f"  Warning (llm_extract_account_info): PDF upload failed: {exc}")
        return [], []

    # Build and send the extraction prompt
    payload = {
        "contents": [{
            "parts": [
                {"fileData": {"mimeType": "application/pdf", "fileUri": file_uri}},
                {"text": _ACCOUNT_EXTRACTION_PROMPT},
            ]
        }],
        "generationConfig": {
            "temperature":     0.0,
            "maxOutputTokens": 1024,
            "topP":            1.0,
            "topK":            1,
        },
    }

    try:
        resp = requests.post(
            f"{_GEMINI_API_URL}?key={api_key}",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=120,
        )
        resp.raise_for_status()
        candidates = resp.json().get("candidates", [])
        if not candidates:
            print("  Warning (llm_extract_account_info): Gemini returned no candidates.")
            return [], []

        raw_text = "\n".join(
            p["text"]
            for p in candidates[0].get("content", {}).get("parts", [])
            if "text" in p
        ).strip()

        # Strip markdown fences if present
        raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text)
        raw_text = re.sub(r"\s*```$", "", raw_text)

        accounts = json.loads(raw_text)

    except json.JSONDecodeError as exc:
        print(f"  Warning (llm_extract_account_info): Could not parse JSON response: {exc}")
        return [], []
    except Exception as exc:
        print(f"  Warning (llm_extract_account_info): API call failed: {exc}")
        return [], []

    acc_types = []
    acc_nums  = []
    for entry in accounts:
        acct_type = str(entry.get("account_type", "")).strip()
        acct_num  = str(entry.get("account_number", "")).strip()
        if acct_type and acct_num:
            acc_types.append(acct_type)
            acc_nums.append(acct_num)

    if acc_types:
        print(f"  LLM account extraction found {len(acc_types)} account(s): "
              + ", ".join(f"{t} ({n})" for t, n in zip(acc_types, acc_nums)))
    else:
        print("  LLM account extraction returned no accounts.")

    return acc_types, acc_nums


@task(name="extract-account-info")
def extract_account_info(
    tables: list[pd.DataFrame],
    llm_acc_types: list | None = None,
    llm_acc_nums:  list | None = None,
) -> tuple[list, list]:
    """
    Description:
        Task that scans all tables for account type and account number information.

        If the CSV tables contain a structured account summary table (indicated
        by an 'Account Number' column), that is used as the primary source.

        If no such table is found — and llm_acc_types / llm_acc_nums are provided
        from a prior llm_extract_account_info call — those LLM-extracted values
        are returned as a fallback. This handles brokerages like J.P. Morgan
        whose account info only appears in page headers, not in extracted CSVs.

    Input:
        tables:        List of DataFrames loaded from the CSV directory.
        llm_acc_types: Optional list of account type strings from llm_extract_account_info.
        llm_acc_nums:  Optional list of account number strings from llm_extract_account_info.

    Output:
        Tuple of (acc_types, acc_nums) — parallel lists of account type strings
        and account number strings found across all tables.
    """
    acc_types = []
    acc_nums  = []
    for table in tables:
        if "Account Number" not in table.columns:
            continue
        for acc in table['Account Type/Name']:
            if pd.isnull(acc):
                continue
            elif len(acc.split('-')) > 2:
                acc_types.append('-'.join(acc.split('-')[1:]).strip())
            elif 'Portfolio' in acc:
                continue
            else:
                acc_types.append(acc.split('-')[1].strip())
        for acc in table['Account Number']:
            if pd.isna(acc):
                continue
            acc_nums.append(acc)

    # Fallback: if no account info was found in the CSVs and the caller supplied
    # LLM-extracted values, use those instead. This handles brokerages like
    # J.P. Morgan where account info only appears in PDF page headers.
    if not acc_types and llm_acc_types:
        return list(llm_acc_types), list(llm_acc_nums or [])

    return acc_types, acc_nums


@task(name="extract-holdings-records")
def extract_holdings_records(
    tables: list[pd.DataFrame],
    acc_types: list,
    acc_nums: list,
    issue_report: IssueReport,
) -> list[dict]:
    """
    Description:
        Task that iterates over all holdings tables (those with Quantity and
        Market Value columns), extracts individual holding records, classifies
        asset class and account type, and validates each record.

    Input:
        tables:            Full list of DataFrames from load_csv_tables.
        acc_types:         Account type strings from extract_account_info.
        acc_nums:          Account number strings from extract_account_info.
        issue_report: IssueReport instance to accumulate issues into.

    Output:
        List of record dicts, one per valid holding row.
    """
    table_to_account_map = build_account_map(tables, acc_types, acc_nums)

    # Identify tables that contain holdings data (Quantity + Market Value columns)
    tables_of_interest = []
    table_indices      = []
    for idx, table in enumerate(tables):
        has_quantity   = resolve_column(table.columns, 'quantity')   is not None
        has_market_value = resolve_column(table.columns, 'market_value') is not None
        if has_quantity and has_market_value:
            tables_of_interest.append(table)
            table_indices.append(idx)

    all_records = []

    for list_idx, table in enumerate(tables_of_interest):
        original_table_idx = table_indices[list_idx]
        table_records      = []

        table_summary = extract_total_row(table)

        if 'Description' not in table.columns:
            continue

        # Filter out total/summary rows
        valid_rows = table[
            table['Description'].notna() &
            ~table['Description'].str.match(r'^\s*Total', case=False, na=False)
        ].copy()

        # Resolve all field column names via fuzzy keyword matching
        market_value_col  = resolve_column(table.columns, 'market_value')
        cost_basis_col    = resolve_column(table.columns, 'cost_basis')
        quantity_col      = resolve_column(table.columns, 'quantity')
        gain_loss_col     = resolve_column(table.columns, 'gain_loss')
        annual_income_col = resolve_column(table.columns, 'annual_income')
        yield_col         = resolve_column(table.columns, 'yield')

        if market_value_col is None:
            print(f"Warning: No market value column found in table {original_table_idx}")
            continue

        valid_rows[['Name', 'Symbol/CUSIP']] = valid_rows['Description'].apply(
            lambda x: pd.Series(extract_symbol_and_name(x))
        )

        account_idx = get_account_for_table(original_table_idx, table_to_account_map)
        if account_idx is None:
            print(f"Warning: No account found for table {original_table_idx}")
            account_string = "Unknown Account"
        else:
            account_string = f"{acc_types[account_idx]} {acc_nums[account_idx]}"

        # Classify tax treatment once per table (account type is constant per table)
        account_type_string = classify_account_type(account_string)

        # Track the latest asset class label pulled from category header rows
        current_asset_class = None

        for _, row in valid_rows.iterrows():
            if pd.isna(row['Name']) or row['Name'] == '':
                continue

            record = {
                'Symbol/CUSIP':         row['Symbol/CUSIP'],
                'Name':                 row['Name'],
                'Quantity':             clean_currency_value(row.get(quantity_col)),
                'Cost Basis':           clean_currency_value(row.get(cost_basis_col)),
                'Market Value':         clean_currency_value(row.get(market_value_col)),
                'Unrealized Gain/Loss': clean_currency_value(
                                            row.get(gain_loss_col) if gain_loss_col else None
                                        ),
                'Est. Annual Income':   clean_currency_value(
                                            row.get(annual_income_col) if annual_income_col else None
                                        ),
                'Est. Yield':           clean_percentage_value(
                                            row.get(yield_col) if yield_col else None
                                        ),
                'Account':              account_string,
                'Asset Class':          (
                                            current_asset_class or
                                            classify_asset_class_from_name(
                                                row['Name'], row['Symbol/CUSIP']
                                            )
                                        ),
                'Account Type':         account_type_string,
            }

            is_category_header = (
                pd.isna(row['Symbol/CUSIP']) and
                all(record[key] is None for key in ['Quantity', 'Cost Basis', 'Market Value']) and
                any(header in str(row['Name']).lower()
                    for header in ['common stock', 'bond funds', 'preferred stock',
                                   'corporate bonds', 'municipal bonds', 'asset backed',
                                   'government', 'cash', 'money market', 'etf', 'fund',
                                   'equity', 'equities', 'fixed income', 'treasury'])
            )

            if is_category_header:
                mapped = classify_asset_class_from_header(row['Name'])
                if mapped:
                    current_asset_class = mapped
                continue

            if any(record[key] is not None for key in ['Quantity', 'Cost Basis', 'Market Value']):
                validate_record(record, issue_report)
                all_records.append(record)
                table_records.append(record)
            
        validate_totals(table_records, table_summary, original_table_idx, issue_report)

    return all_records


@task(name="build-holdings-dataframe")
def build_holdings_dataframe(records: list[dict]) -> pd.DataFrame:
    """
    Description:
        Task that assembles the final holdings DataFrame from extracted records
        and enforces the canonical column ordering.

        If records is empty, returns an empty DataFrame with the canonical
        columns pre-assigned rather than raising a KeyError. This surfaces
        a clear, actionable message instead of a cryptic pandas index error.

    Input:
        records: List of holding record dicts from extract_holdings_records.

    Output:
        DataFrame with columns in canonical order.
    """
    column_order = [
        'Symbol/CUSIP', 'Name', 'Asset Class',
        'Quantity', 'Cost Basis', 'Market Value',
        'Unrealized Gain/Loss', 'Est. Annual Income', 'Est. Yield',
        'Account Type', 'Account'
    ]

    if not records:
        print(
            "WARNING (build_holdings_dataframe): No records were extracted. "
            "The output DataFrame is empty. Check that:\n"
            "  1. The CSV files exist in the configured csv_dir.\n"
            "  2. Each holdings table has a 'Description', 'Quantity', and "
            "'Market Value' column (or recognised variants).\n"
            "  3. The header row was correctly detected by load_csv_tables."
        )
        return pd.DataFrame(columns=column_order)

    final_df = pd.DataFrame(records)

    # Identify any expected columns missing from the assembled DataFrame
    # (indicates a record dict was built with unexpected key names).
    missing = [col for col in column_order if col not in final_df.columns]
    if missing:
        raise KeyError(
            f"build_holdings_dataframe: the following canonical columns are "
            f"absent from the extracted records: {missing}. "
            f"Columns present: {list(final_df.columns)}"
        )

    return final_df[column_order]


def revalidate_holdings(
    holdings_df: pd.DataFrame,
    csv_dir: str,
    csv_prefix: str,
) -> IssueReport:
    """
    Description:
        Re-run validation checks against an already-parsed holdings DataFrame
        without re-parsing the source CSVs from scratch. Used between
        self-correction iterations in _02_llm_validation.py after holdings.csv
        has been patched with LLM-identified corrections.

        Runs the same checks as _01_parse_holdings.py:
          - validate_record (negative_values, missing_symbol, missing_data,
            abnormal_quantities, missing_account) — per holding row
          - validate_totals (total_mismatches) — per source table, using
            extract_total_row on the original source CSVs so that the
            sub-section total logic (excluding higher-level aggregates) is
            identical to the first-pass validation

        Does NOT re-run extraction_failures (those are upstream artefacts
        from _00_extract_tables.py and are unaffected by holdings patches).

    Input:
        holdings_df: Patched holdings DataFrame (from holdings.csv).
        csv_dir:     Directory containing the numbered source CSVs.
        csv_prefix:  Filename prefix for source CSVs (e.g. "fidelity_table_").

    Output:
        Fresh IssueReport reflecting the current state of holdings_df.
    """
    report = IssueReport()

    # ── Per-record checks ─────────────────────────────────────────────────────
    for _, row in holdings_df.iterrows():
        record = row.to_dict()
        validate_record(record, report)

    # ── Per-table total checks ────────────────────────────────────────────────
    # Load each source CSV, identify it as a holdings table, group holdings rows
    # from the DataFrame by table, and call validate_totals with the same
    # sub-section total logic used in the first pass.
    tables = load_csv_tables(csv_dir, csv_prefix)

    # Build a map of original_table_idx -> list of holding records for that table
    # using the same table-identification criteria as extract_holdings_records.
    table_records_map: dict[int, list[dict]] = {}

    for idx, table in enumerate(tables):
        has_quantity     = resolve_column(table.columns, 'quantity')     is not None
        has_market_value = resolve_column(table.columns, 'market_value') is not None
        if has_quantity and has_market_value:
            table_records_map[idx] = []

    # Assign each holdings row to its source table by re-parsing the CSVs and
    # matching on Name. This mirrors the name extracted by extract_symbol_and_name.
    for idx, table in enumerate(tables):
        if idx not in table_records_map:
            continue

        if 'Description' not in table.columns:
            continue

        valid_rows = table[
            table['Description'].notna() &
            ~table['Description'].str.match(r'^\s*Total', case=False, na=False)
        ]

        for _, src_row in valid_rows.iterrows():
            name, _ = extract_symbol_and_name(src_row['Description'])
            if not name:
                continue
            fragment = name[:60].strip()
            match_mask = holdings_df['Name'].str.contains(
                re.escape(fragment), case=False, na=False
            )
            for _, holding_row in holdings_df[match_mask].iterrows():
                table_records_map[idx].append(holding_row.to_dict())

    for idx, table in enumerate(tables):
        if idx not in table_records_map:
            continue
        table_summary = extract_total_row(table)
        validate_totals(
            table_records_map[idx],
            table_summary,
            idx,
            report,
        )

    return report


@task(name="save-holdings-outputs", task_run_name="save-holdings: {output_file}")
def save_holdings_outputs(
    final_df: pd.DataFrame,
    issue_report: IssueReport,
    output_file: str,
    initial_issue_file: str,
):
    """
    Description:
        Task that saves the holdings DataFrame and initial issue report as csv and txt files.

    Input:
        final_df:          Holdings DataFrame from build_holdings_dataframe.
        issue_report: Completed IssueReport.
        output_file:       Full path for the holdings CSV output.
        initial_issue_file:   Full path for the initial issue report text output.
    """
    final_df.to_csv(output_file, index=False)
    print(f"Data saved to {output_file}!")
    issue_report.save_to_file(initial_issue_file)
    print(f"Initial issue report saved to {initial_issue_file}!")