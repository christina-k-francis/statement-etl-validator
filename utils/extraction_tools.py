"""
Helpful functions and classes for extracting tabular data from financial statements, and storing them in CSVs.
"""

import io
import re
import csv
import time
import json
import base64
import zipfile
import requests
import fitz # PyMuPDF 
from PIL import Image
from pathlib import Path
from typing import NamedTuple
from prefect import task, flow, runtime

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

### Configuration ###
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"

# Retry configuration for API calls
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 5

# Image settings for PDF page conversion
PDF_DPI = 350  # Higher DPI for better digit clarity
MAX_IMAGE_DIMENSION = 2048  # Gemini's recommended max
PNG_COMPRESSION = 9  # PNG compression level (0-9, higher = smaller file, lossless)

# Batch configuration — pages per API call
BATCH_DELAY_SECONDS = 2  # Delay between batch API calls to avoid rate limits
BATCH_SIZE = 7

### Prompt Engineering Statement ###
EXTRACTION_PROMPT = """You are a precise financial document table extractor. You are being shown a set of pages from a financial statement document. Your task is to extract every table present in these pages.

CRITICAL RULES:
1. Extract ONLY data that is explicitly visible in the document. Do NOT infer, estimate, or hallucinate any values OR row labels. If a number appears on the page but has no visible description, label, or row context next to it, do NOT create a table row for it — skip it entirely.
2. If a cell is empty, leave it empty. If a value is unclear or partially obscured, output it exactly as visible with no guessing.
3. Preserve ALL original formatting of numbers and symbols exactly as shown: dollar signs ($), commas within numbers, parentheses for negatives, percentage signs (%), dashes (-), asterisks (*).
4. Preserve the EXACT numerical values — do not round, truncate, or reformat numbers.
5. Multi-line cell content (e.g., fund names spanning multiple lines, or stock descriptions with supplemental lines like "Dividend Reinvested", "Symbol: AMZN", or "EST YIELD: 2.97%") must be joined with a single space into ONE cell value on a SINGLE row. Do NOT emit a separate row for these continuation lines.
   Example of a holdings row whose description spans three visual lines in the document:
     AMAZON.COM INC
     Dividend Reinvested
     Symbol: AMZN
   Correct output (one row, description fully joined):
     AMAZON.COM INC Dividend Reinvested Symbol: AMZN|03 Feb 2021|30|3,094.08|92,822.40|...
   WRONG output (do NOT split into multiple rows):
     AMAZON.COM INC|03 Feb 2021|30|3,094.08|92,822.40|...
     Dividend Reinvested||||||
     Symbol: AMZN||||||
6. Include ALL data rows: column headers, data rows, subtotal rows, and total rows.
7. CRITICAL — CROSS-PAGE TABLES: If a table spans multiple pages (indicated by "continued" headers, repeated column headers, or continuation of the same tabular structure), combine it into ONE complete table. Do NOT split a single table into separate outputs just because it crosses a page boundary.
8. CRITICAL — CROSS-PAGE SECURITY DESCRIPTIONS: A single security row may have its description split across a page boundary. This happens when a page ends mid-description (e.g., "ANHEUSER-BUSCH INBEV SA SPONSORED ADR Dividend") and the next page begins with the remainder of that description (e.g., "Reinvested EST YIELD: 1.71% Symbol: BUD") before the financial data for that security. When you encounter this:
   - Combine the full description into ONE single row using the financial values from the FIRST page (where the security row began).
   - Do NOT emit a second row for the same security.
   - Do NOT use financial values from any OTHER security (e.g., the one listed below) to fill in the second row.
   - The continuation text on the second page is part of the description ONLY — it has no independent financial values of its own.
   Example — page 6 ends with:
     ANHEUSER-BUSCH INBEV SA SPONSORED ADR Dividend  |  01 Feb 2021  |  300  |  62.85  |  18,855.00  |  64.38  |  19,314.00  |  (459.00)
   Page 7 begins with:
     Reinvested EST YIELD: 1.71% Symbol: BUD
   Correct output (ONE row, full description, values from page 6):
     ANHEUSER-BUSCH INBEV SA SPONSORED ADR Dividend Reinvested EST YIELD: 1.71% Symbol: BUD|01 Feb 2021|300|62.85|18,855.00|64.38|19,314.00|(459.00)|...
   WRONG output (do NOT create two rows or borrow values from another security):
     ANHEUSER-BUSCH INBEV SA SPONSORED ADR Dividend|01 Feb 2021|300|62.85|18,855.00|...
     ANHEUSER-BUSCH INBEV SA Dividend Reinvested Symbol: BUD|22 Feb 2021|3,500|26.54|92,890.00|...
9. If pages contain NO tables (e.g., cover pages, disclosure text, blank pages), simply skip those pages.

WHAT TO EXCLUDE FROM TABLES:
- Do NOT include footnotes, disclaimers, or explanatory notes that appear below or beside tables (e.g., text following asterisks like "* Appreciation or depreciation..."). These are not table data.
- Do NOT include section titles, page headers, or large-font headings that appear ABOVE a table. These are visual labels on the page, not part of the tabular structure. The first row of each table should be the COLUMN HEADER ROW (the row that labels the data columns), not a section title.
  Example: if a page shows a bold heading "Change in Account Value" above a table whose columns are labeled "This Period" and "Year-to-Date", then the first row of the extracted table should be the column headers, NOT the section heading.
- Do NOT include "Contact Information" headers or similar non-columnar labels as a row. If a table starts with column values directly, begin there.
- ORPHANED / FLOATING VALUES: If a number or monetary value appears on the page outside of any table row — e.g., floating at the bottom or margin of a page with no corresponding row label, description, or column alignment — do NOT include it. Do NOT invent or guess a label for it (such as "Accrued Interest" or "Other"). Simply ignore orphaned values that are not part of the tabular structure.

BOLD AND INDENTED ROWS:
- Within a table, bold rows that have numerical values aligned with the same columns as other data rows ARE data rows — always include their values. Bold styling indicates visual emphasis (e.g., category subtotals), NOT that the row is a non-data header.
- Indented rows beneath a bold row are sub-items of that category. Both the bold parent row AND its indented children are separate data rows, each with their own values extracted from the correct columns.
  Example from an Income Summary table:
    **Taxable        $178.53     $2,839.92**     <-- bold row WITH values: extract as data row
       Dividends     178.53      1,548.74        <-- indented sub-item: extract as data row
       Interest        --          10.25          <-- indented sub-item: extract as data row
  Correct output:
    Taxable|$178.53|$2,839.92
    Dividends|178.53|1,548.74
    Interest|--|10.25
  WRONG output (do NOT do this):
    Taxable||
    Dividends|$178.53|$1,548.74

MULTI-LOT HOLDINGS (same security, multiple acquisition dates):
- Some holdings rows show a single security purchased on multiple dates. These appear as a bold summary line (no acquisition date, summed/aggregated values) followed by indented sub-rows, one per acquisition date, each with its own date and lot-level values.
- CRITICAL: Extract ONLY the bold summary row for the security. Do NOT emit the indented per-lot sub-rows as separate table rows.
- The summary row has no acquisition date. Its financial values (Quantity, Price, Market Value, Cost Basis, etc.) are the aggregated totals across all lots — use these values exactly as shown.
- Example layout in the document:
    COMCAST CORP CLASS A                  3.80451  54.11  205.86  43.05  163.77  42.09       3.80
      CLA Dividend Reinvested
      EST YIELD: 1.85%
      Symbol: CMCSA
        28 Oct 2020   3.78714  0.0137  204.92  43.01  162.90  42.02  ST
        27 Jan 2021   0.01737   0.94   50.09   43.09    0.87   0.07  ST
  Correct output (summary row only, no acquisition date):
    COMCAST CORP CLASS A CLA Dividend Reinvested EST YIELD: 1.85% Symbol: CMCSA||3.80451|54.11|205.86|43.05|163.77|42.09||3.80
  WRONG output (do NOT emit per-lot sub-rows):
    COMCAST CORP CLASS A CLA Dividend Reinvested EST YIELD: 1.85% Symbol: CMCSA|28 Oct 2020|3.78714|0.0137|204.92|43.01|162.90|42.02|ST|
    COMCAST CORP CLASS A CLA Dividend Reinvested EST YIELD: 1.85% Symbol: CMCSA|27 Jan 2021|0.01737|0.94|50.09|43.09|0.87|0.07|ST|

SUB-SECTIONS WITHIN A TABLE:
- CRITICAL: If multiple sections on the same page share the SAME column headers (same column names in the same order), they are ONE table — not separate tables. Sub-section headings like "Corporate Bonds", "Municipal Bonds", "Other Bonds" within a holdings table that all share columns like "Description | Maturity | Quantity | Price Per Unit | ..." must be included as rows within a SINGLE table output, not split into separate TABLE_START/TABLE_END blocks.
- The sub-section heading (e.g., "Corporate Bonds") should appear as a row in the table with its data cells empty.
- Only output a new TABLE_START when the column structure genuinely changes (different number of columns or different column names).
  Example: A page with "Corporate Bonds", "Municipal Bonds", and "Other Bonds" sections, all under the same "Description | Maturity | Quantity | ..." columns, should produce ONE table:
    Description|Maturity|Quantity|Price Per Unit|...
    Corporate Bonds||||||
    SABRATEK CORP NT CV|12/15/13|5,000.00|$101.250|...
    ...
    Municipal Bonds||||||
    NEW YORK NY CITY...|3/1/14|10,000.000|$107.442|...
    ...

TEXT NORMALIZATION:
- Replace special Unicode characters with plain ASCII equivalents: ™ becomes TM, ® becomes (R), © becomes (C), curly/smart quotes become straight quotes, em-dashes become --, en-dashes become -.
- Do NOT output any non-ASCII characters. Use only standard ASCII (letters, digits, basic punctuation).

OUTPUT FORMAT:
For each distinct table found in the document, output:

===TABLE_START===
<table with each row on its own line>
<cell values separated by the pipe character | >
<NO quoting needed — just use | between cells>
===TABLE_END===

CRITICAL: Use the PIPE CHARACTER ( | ) as the delimiter between cells, NOT commas. This is because financial data contains commas inside numbers (e.g., $1,234.56) and inside text (e.g., "Cards, Checking & Bill Payments"), which would break comma-separated formatting. Pipes never appear in financial data, so they are a safe delimiter.

PIPE FORMATTING RULES:
- Do NOT start a row with a leading pipe unless the first cell is intentionally empty.
- Do NOT end a row with a trailing pipe.
- Pipes appear BETWEEN cells only.
- If the first cell of a row is empty (e.g., a header row where the row-label column is blank), the line starts with a pipe: |This Period|Year-to-Date
- If the first cell has content, the line starts with that content: Beginning Account Value|$88,053.95|$76,911.26

Example of correct output format:
===TABLE_START===
|This Period|Year-to-Date
Beginning Account Value|$88,053.95|$76,911.26
Additions|$59,269.64|$107,124.70
===TABLE_END===

In the example above, the first row starts with | because its first cell (row label) is empty. The second and third rows start with text because their first cells have content.

Output each distinct table with its own TABLE_START/TABLE_END markers, in the order they appear in the document. A "distinct table" is one with its own unique set of column headers — NOT a page-split fragment of a larger table, and NOT a sub-section that shares column headers with another section on the same page.

GUIDELINES FOR FINANCIAL TABLES:
- Column headers may span multiple rows (e.g., "Unrealized" on one line and "Gain/Loss ($)" below it). Merge these into a single header row with combined text like "Unrealized Gain/Loss ($)".
- Some tables have hierarchical row labels (categories, subcategories, individual holdings). Preserve the label text as-is.
- Summary/total rows often appear in bold or with "Total" prefix — include these exactly as they appear, with their values.
- Currency values may appear as "$1,234.56" or "1,234.56" or "(1,234.56)" for negatives — preserve the exact format shown.
- Tables that continue across pages should be merged: include the column headers once at the top, then all data rows from all pages in sequence.

STACKED TWO-VALUE COLUMN REGIONS:
- Some tables display a single narrow column region whose header spans two lines and whose data cells also contain two stacked values — one per sub-row. Each sub-row maps to one of the two header lines and must be extracted into its own separate column.
- CRITICAL: The top sub-row value belongs in the first column; the bottom sub-row value belongs in the second column. These are TWO distinct columns in the output, not one.
- A "--" in either sub-row means that column has no value for this row. Output "--" in the correct column — do NOT treat it as the value for the other column.
- This pattern commonly appears as "Est. Accrued Inc." (top) and "Est. Annual Inc." (bottom) stacked within one visual column region. For each data row, extract both sub-row values into their respective columns.
  Example — a holding where Est. Accrued Inc. is "--" and Est. Annual Inc. is 1,680.00:
    Correct output:  ...|Unrealized Gain/Loss value|--|1,680.00
    WRONG output:    ...|Unrealized Gain/Loss value|1,680.00|
    WRONG output:    ...|Unrealized Gain/Loss value|--|--
- This same rule applies to summary/total rows. If the TOTAL row shows "--" for Est. Accrued Inc. and "$20,285.34" for Est. Annual Inc., output both values in their respective columns:
    Correct output:  TOTAL EQUITIES|||$755,837.61||$747,596.07|$8,241.54|--|$20,285.34
    WRONG output:    TOTAL EQUITIES|||$755,837.61||$747,596.07|$8,241.54|$20,285.34|

TOTAL / SUMMARY ROW ALIGNMENT:
- CRITICAL: In Total or summary rows (e.g., "Total Holdings", "Total Stocks", "Total Other"), the monetary values MUST be placed in the SAME columns as the data rows above them — not shifted left or right.
- Total rows often have fewer populated cells than regular data rows (e.g., a Total row may only show Ending Market Value and Total Cost Basis, with Quantity and Price Per Unit empty). Place each value directly below the column it belongs to, and leave the other cells empty.
- Example: If data rows have values in columns Description|Quantity|Price Per Unit|Ending Market Value|Total Cost Basis, and the Total row only shows $6,740.00 under Ending Market Value, output:
    Total Other (5% of account holdings)|||$6,740.00|
  NOT:
    Total Other (5% of account holdings)|$6,740.00|||
- When in doubt, look at the position of each value on the page: if a value is visually aligned under the "Ending Market Value" column header, it goes in that column — even if earlier columns are empty.

SIDE-BY-SIDE TABLES:
- Some pages display two tables placed side by side (left and right) on the same page. These are SEPARATE tables and must be extracted as separate TABLE_START/TABLE_END blocks.
- Extract the LEFT table first, then the RIGHT table.
- Even if the left and right tables share the same column headers, they are separate tables because they contain different data rows.
- Example: A "Daily Balance" page might show dates 7/01-7/15 on the left and dates 7/16-7/31 on the right, both with columns Date|Total Additions|Total Subtractions|Net Activity|Daily Balance. These must be two separate tables, not one combined table.

MULTI-STYLE TEXT WITHIN A SINGLE CELL:
- Some table cells contain text in multiple visual styles stacked vertically. For example, a cell might show a bold account type (e.g., "GENERAL INVESTMENTS") on one line and a non-bold account name (e.g., "John W. Doe - Individual TOD") on the next line. These are part of the SAME cell — join them with a space into a single cell value.
- This commonly appears in Account Summary tables with columns like "Page | Account Type/Name | Account Number | Beginning Value | Ending Value". The Account Type (bold) and Account Name (non-bold) are in the same cell and must be extracted as one combined value. Their corresponding Account Number, Beginning Value, and Ending Value are on the same row.
  Example: If the page shows:
    5    GENERAL INVESTMENTS                111-111111    $88,053.95    $103,351.18
         John W. Doe - Individual TOD
  The correct output is ONE row:
    5|GENERAL INVESTMENTS John W. Doe - Individual TOD|111-111111|$88,053.95|$103,351.18
  WRONG output (do NOT do this — splitting into two rows):
    5|GENERAL INVESTMENTS||||
    |John W. Doe - Individual TOD|111-111111|$88,053.95|$103,351.18

Extract all tables from these pages now."""

### Helpful FXs for handling PDF/ZIP page extractions ###
def _resize_image(img: Image.Image) -> Image.Image:
    """
    Description:
        FX handles resizing an image if it exceeds Gemini's recommended max dimension.

    Input:
        Image

    Output:
        Image resized to fit the predefined maximum image dimension.     
    """
    w, h = img.size
    if max(w, h) > MAX_IMAGE_DIMENSION:
        ratio = MAX_IMAGE_DIMENSION / max(w, h)
        new_size = (int(w * ratio), int(h * ratio))
        img = img.resize(new_size, Image.LANCZOS)
    return img

def _image_to_png_bytes(img: Image.Image) -> bytes:
    """
    Description:
        FX converts a PIL Image to lossless PNG bytes.
        PNG avoids JPEG compression artifacts on small digits and text,
        improving LLM accuracy on numerical values.
    """
    img = _resize_image(img)
    buffer = io.BytesIO()
    img.save(buffer, format='PNG', compress_level=PNG_COMPRESSION)
    return buffer.getvalue()


def load_pages_from_pdf(pdf_path: str) -> list[dict]:
    """
    Description:
        Convert each page of a PDF to a PNG image using PyMuPDF.
    
    Input:
        pdf_path: string pointing to the location of the pdf of interest

    Output:
        List of dicts with keys: 'page_number', 'image_bytes', 'media_type'
    """

    print(f"Converting PDF to images: {pdf_path}")
    doc = fitz.open(pdf_path)

    # Calculate zoom factor from target DPI (PyMuPDF default is 72 DPI)
    zoom = PDF_DPI / 72
    matrix = fitz.Matrix(zoom, zoom)

    pages = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        pix = page.get_pixmap(matrix=matrix)

        # Convert PyMuPDF pixmap to PIL Image
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        pages.append({
            'page_number': page_num + 1,
            'image_bytes': _image_to_png_bytes(img),
            'media_type': 'image/png'
        })

    doc.close()
    print(f"Converted {len(pages)} pages")
    return pages

def load_pages_from_zip(zip_path: str) -> list[dict]:
    """
    Description:
        Loads page images from a ZIP archive containing numbered images and optionally a manifest.json.
    
    Input: manifest.json + numbered image files (1.jpeg, 2.jpeg, ...).
    
    Output:
        List of dicts with keys: 'page_number', 'image_bytes', 'media_type'
    """
    print(f"Loading pages from ZIP: {zip_path}")
    pages = []
    
    with zipfile.ZipFile(zip_path, 'r') as zf:
        names = zf.namelist()

        if 'manifest.json' in names:
            manifest = json.loads(zf.read('manifest.json'))
            for page_info in manifest['pages']:
                img = Image.open(io.BytesIO(zf.read(page_info['image']['path'])))
                pages.append({
                    'page_number': page_info['page_number'],
                    'image_bytes': _image_to_png_bytes(img),
                    'media_type': 'image/png'
                })
        else:
            img_files = sorted(
                [n for n in names if n.lower().endswith(('.jpeg', '.jpg', '.png'))],
                key=lambda x: int(m.group(1)) if (m := re.search(r'(\d+)', x)) else 0
            )
            for i, img_file in enumerate(img_files):
                img = Image.open(io.BytesIO(zf.read(img_file)))
                pages.append({
                    'page_number': i + 1,
                    'image_bytes': _image_to_png_bytes(img),
                    'media_type': 'image/png'
                })

    print(f"Loaded {len(pages)} pages")
    return pages

@task(name="load-pages", task_run_name="load-pages: {file_path}")
def load_pages(file_path: str) -> list[dict]:
    """
    Description: 
        FX auto-detects file type and load pages accordingly.
    
    Input: 
        File types .pdf, .zip, or PDF files that are actually ZIPs.

    Output:
        List of dicts with keys corresponding to the file type.
    """
    path = Path(file_path)
    
    # Check if it's a ZIP regardless of extension
    if zipfile.is_zipfile(file_path):
        return load_pages_from_zip(file_path)
    elif path.suffix.lower() == '.pdf':
        return load_pages_from_pdf(file_path)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}. Expected .pdf or .zip")

# Defining a new classes to simplify viewing API call outcomes
class GeminiResponse(NamedTuple):
    """
    Structured result from a single Gemini API call.

    Fields:
        text:          Raw extracted text returned by the model (or "NO_TABLES_FOUND").
        finish_reason: The model's reported stop reason, e.g. "STOP", "MAX_TOKENS", "OTHER".
                       None when the call failed entirely (no candidates returned).
        page_range:    Human-readable string like "3-9" identifying which pages were sent.
    """
    text:          str
    finish_reason: str | None
    page_range:    str

# Defining a new class to document pages that failed to be extracted from the API
class BatchResult(NamedTuple):
    """
    Structured result from call_gemini_api_batched().

    Fields:
        responses:          List of GeminiResponse objects, one per successfully completed
                            (sub-)batch, in document order.
        failed_page_ranges: List of page-range strings (e.g. ["3-3", "7-7"]) whose pages
                            could NOT be extracted even after halving retries down to
                            batch_size=1. Empty list when all pages were processed.
    """
    responses:          list[GeminiResponse]
    failed_page_ranges: list[str]


### Helpful FXs for interacting with the Gemini API ###
def _is_truncated(finish_reason: str | None, response_text: str) -> bool:
    """
    Description:
        Detects whether a Gemini response was cut off before the model finished
        extracting all tables from the submitted pages.

    Input:
        finish_reason:  Value of candidates[0].finishReason from the API JSON, or
                        None if no candidates were returned.
        response_text:  The raw text content returned by the model.

    Output:
        True if the response appears truncated; False if it looks complete.
    """
    # Primary signal: API-reported finish reason
    if finish_reason is None:
        # No candidates at all — treat as truncated/failed
        return True
    if finish_reason in ("MAX_TOKENS", "OTHER"):
        return True

    # Secondary signal: unclosed TABLE_START marker
    # Only fires when finishReason looks clean (STOP) but the text disagrees
    if "===TABLE_START===" in response_text:
        start_count = response_text.count("===TABLE_START===")
        end_count   = response_text.count("===TABLE_END===")
        if start_count > end_count:
            logger.warning(
                f"finishReason='{finish_reason}' but found {start_count} TABLE_START "
                f"vs {end_count} TABLE_END — treating as truncated."
            )
            return True

    return False


@task(
    name="call-gemini-api",
    retries=0,  # retry logic is in the fx _process_pages_with_autotuning upon auto-tuning halving.
)
def call_gemini_api(pages: list[dict], api_key: str) -> GeminiResponse:
    """
    Description:
        FX sends a batch of page images to Gemini 2.5 Flash in a single API call
        and receives a table extraction response.

        Returns a GeminiResponse named tuple so that the caller can inspect
        finish_reason and decide whether to retry with a smaller batch.

    Input:
        pages:   A list of page dicts for this batch.
        api_key: Gemini API key.

    Output:
        GeminiResponse(text, finish_reason, page_range) where:
          - text          is the raw model output (or "NO_TABLES_FOUND" on hard failure)
          - finish_reason is the API's reported stop reason (e.g. "STOP", "MAX_TOKENS",
                          "OTHER"), or None if no candidates were returned / call failed
          - page_range    is a human-readable string like "3-9" for logging
    """
    page_range = (
        f"{pages[0]['page_number']}-{pages[-1]['page_number']}"
        if len(pages) > 1
        else str(pages[0]['page_number'])
    )

    # setting a custom task run name to handle formatting restraints
    runtime.task_run.name = f"gemini-api: pages {page_range}"

    parts = []
    for page in pages:
        parts.append({
            "inlineData": {
                "mimeType": page['media_type'],
                "data": base64.b64encode(page['image_bytes']).decode('utf-8')
            }
        })
    parts.append({"text": EXTRACTION_PROMPT})

    payload = {
        "contents": [{"parts": parts}],
        "generationConfig": {
            "temperature": 0.0,
            "maxOutputTokens": 65536,
            "topP": 1.0,
            "topK": 1
        },
        "safetySettings": [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]
    }

    url = f"{GEMINI_API_URL}?key={api_key}"

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"  Sending pages {page_range} to Gemini (attempt {attempt}/{MAX_RETRIES})...")
            response = requests.post(
                url, json=payload,
                headers={"Content-Type": "application/json"},
                timeout=300
            )

            if response.status_code == 200:
                result      = response.json()
                candidates  = result.get('candidates', [])
                if candidates:
                    candidate     = candidates[0]
                    finish_reason = candidate.get('finishReason')
                    text_parts    = [
                        p['text'] for p in candidate.get('content', {}).get('parts', [])
                        if 'text' in p
                    ]
                    return GeminiResponse(
                        text          = '\n'.join(text_parts),
                        finish_reason = finish_reason,
                        page_range    = page_range,
                    )
                logger.warning(f"No candidates in response: {json.dumps(result)[:500]}")
                # Return with None finish_reason so _is_truncated() treats this as failed
                return GeminiResponse(
                    text          = "NO_TABLES_FOUND",
                    finish_reason = None,
                    page_range    = page_range,
                )

            elif response.status_code == 429:
                wait = RETRY_DELAY_SECONDS * attempt
                logger.warning(f"Rate limited (429). Waiting {wait}s before retry...")
                time.sleep(wait)
                continue

            else:
                logger.error(f"API error {response.status_code}: {response.text[:500]}")
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY_SECONDS * attempt)
                    continue
                return GeminiResponse(
                    text          = "NO_TABLES_FOUND",
                    finish_reason = None,
                    page_range    = page_range,
                )

        except requests.exceptions.Timeout:
            logger.warning(f"Request timed out (attempt {attempt}/{MAX_RETRIES})")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY_SECONDS)
                continue
            return GeminiResponse(
                text          = "NO_TABLES_FOUND",
                finish_reason = None,
                page_range    = page_range,
            )

        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY_SECONDS)
                continue
            return GeminiResponse(
                text          = "NO_TABLES_FOUND",
                finish_reason = None,
                page_range    = page_range,
            )

    return GeminiResponse(
        text          = "NO_TABLES_FOUND",
        finish_reason = None,
        page_range    = page_range,
    )

def _process_pages_with_autotuning(
    pages: list[dict],
    api_key: str,
    batch_size: int,
    responses: list[GeminiResponse],
    failed_page_ranges: list[str],
) -> None:
    """
    Description:
        Recursively processes batches of pages. If the API response is truncated, 
        this FX will automatically halve the batch size — down to a floor of 1 page per call. 
        Pages that cannot be extracted even at batch_size=1 are appended to failed_page_ranges.

    Input:
        pages:              Slice of page dicts to process.
        api_key:            Gemini API key.
        batch_size:         Current batch size to attempt.
        responses:          Accumulator list — GeminiResponse objects appended here.
        failed_page_ranges: Accumulator list — failed range strings appended here.
    """
    for i in range(0, len(pages), batch_size):
        batch      = pages[i:i + batch_size]
        page_range = (
            f"{batch[0]['page_number']}-{batch[-1]['page_number']}"
            if len(batch) > 1
            else str(batch[0]['page_number'])
        )

        print(f"Processing pages {page_range} (batch_size={batch_size})...")

        gemini_response = call_gemini_api(batch, api_key)

        if _is_truncated(gemini_response.finish_reason, gemini_response.text):
            if batch_size == 1:
                # Floor reached — this single page cannot be extracted, log as failure
                logger.error(
                    f"  FAILED: pages {page_range} truncated even at batch_size=1. "
                    f"Recording as unprocessed."
                )
                failed_page_ranges.append(page_range)
            else:
                # Halve and recurse
                smaller_size = max(1, batch_size // 2)
                logger.warning(
                    f"  Truncated response for pages {page_range} "
                    f"(finishReason={gemini_response.finish_reason!r}). "
                    f"Retrying with batch_size={smaller_size}..."
                )
                _process_pages_with_autotuning(
                    batch, api_key, smaller_size, responses, failed_page_ranges
                )
        else:
            # Clean response — keep it
            responses.append(gemini_response)
            print(
                f"  OK: pages {page_range} — "
                f"finishReason={gemini_response.finish_reason!r}"
            )

        # Rate-limit courtesy delay between top-level batches (not sub-batches)
        if batch_size == BATCH_SIZE and i + batch_size < len(pages) and BATCH_DELAY_SECONDS > 0:
            print(f"  Waiting {BATCH_DELAY_SECONDS}s between batches...")
            time.sleep(BATCH_DELAY_SECONDS)


@flow(name="gemini-batched-extraction")
def call_gemini_api_batched(pages: list[dict], api_key: str) -> BatchResult:
    """
    Description:
        FX splits the full document into batches of pages and sends each batch
        as a separate API call. 

        Batches are clean partitions with NO overlap — each page appears in exactly
        one batch. Tables that span a batch boundary are stitched back together by
        merge_tables_with_shared_headers() in the caller.

    Input:
        pages:   Full list of page dicts for the entire document.
        api_key: Gemini API key.

    Output:
        BatchResult(responses, failed_page_ranges) where:
          - responses          is a list of GeminiResponse objects in document order
          - failed_page_ranges is a list of page-range strings that were never
                               successfully processed (empty list = full success)
    """
    responses:          list[GeminiResponse] = []
    failed_page_ranges: list[str]            = []

    _process_pages_with_autotuning(
        pages, api_key, BATCH_SIZE, responses, failed_page_ranges
    )

    total_batches = len(responses) + len(failed_page_ranges)
    print(
        f"Batching complete: {len(responses)} successful batch(es), "
        f"{len(failed_page_ranges)} permanently failed page range(s), "
        f"across {len(pages)} total page(s)."
    )
    if failed_page_ranges:
        logger.error(
            f"  Unprocessed page ranges (will appear in validation report): "
            f"{failed_page_ranges}"
        )

    return BatchResult(responses=responses, failed_page_ranges=failed_page_ranges)

### Helpful FXs that extract tables from the LLM API response ###
def _normalize_columns(rows: list[list[str]]) -> list[list[str]]:
    """
    Description:
        FX cleans up empty columns at the beginning and end of each table,
        detected via the pipe-delimited LLM output.
        
    Input:
        rows: list of rows (each row is a list of cell strings)

    Output:
        rows with systematic leading/trailing artifact columns removed
    """
    if not rows or len(rows) < 2:
        return rows

    # Check for trailing-pipe artifact: every row ends with empty cell
    if all(len(row) >= 2 and row[-1] == '' for row in rows):
        rows = [row[:-1] for row in rows]

    # Check for leading-pipe artifact: every row starts with empty cell
    if all(len(row) >= 2 and row[0] == '' for row in rows):
        rows = [row[1:] for row in rows]

    return rows

def _get_header_key(table: list[list[str]]) -> tuple:
    """
    Description:
        FX extracts a normalized header signature from a table for comparison. 
        Used to detect consecutive tables that share identical column headers and should be merged.

    Input:
        table: a parsed table (list of rows)

    Output:
        tuple of lowercase stripped header cell values, or empty tuple
    """
    if not table:
        return ()
    return tuple(cell.strip().lower() for cell in table[0])

def _row_fingerprint(row: list[str]) -> str:
    """
    Description:
        FX creates a normalized fingerprint for a row, used to detect
        duplicate rows across overlapping batches. The fingerprint focuses on 
        Numeric/Monetary values.

        For rows with no numeric values (section headers like
        "Corporate Bonds"), falls back to the full normalized text.

    Input:
        row: a single parsed table row (list of cell strings)

    Output:
        Normalized string fingerprint for comparison
    """
    # Extract all cells that contain numeric/monetary content
    numeric_cells = []
    for cell in row:
        c = cell.strip()
        # Match cells that contain digits (monetary values, quantities, etc.)
        if re.search(r'\d', c):
            # Normalize: strip whitespace, lowercase
            c = re.sub(r'\s+', '', c).lower()
            numeric_cells.append(c)

    if numeric_cells:
        # Use first few chars of first cell as context + all numeric values
        label = re.sub(r'\s+', ' ', row[0].strip().lower())[:20] if row[0].strip() else ''
        return label + '|' + '|'.join(numeric_cells)
    else:
        # No numeric content — use full normalized text (section headers, etc.)
        parts = []
        for cell in row:
            c = cell.strip().lower()
            c = re.sub(r'[-–—]+', '-', c)
            c = re.sub(r'\s+', ' ', c)
            if c:
                parts.append(c)
        return '||'.join(parts)

def _find_overlap_start(existing_rows: list[list[str]], incoming_rows: list[list[str]]) -> int:
    """
    Description:
        FX detects distinct rows that appear in both the existing table's tail and the incoming data.
        The best anchors are "Total" rows and summary rows, which have unique monetary values and are 
        always single-row.

        Strategy:
        1. Build fingerprints for the last ~30 rows of the existing table.
        2. Walk through incoming rows and check if each one matches an
           existing row. Track the last matched position.
        3. The first incoming row AFTER the last matched existing row
           is where new data begins.

        If no matches are found, all incoming rows are treated as new.

    Input:
        existing_rows: data rows (no header) of the existing merged table
        incoming_rows: data rows (no header) of the table being merged

    Output:
        Index into incoming_rows where NEW (non-duplicate) data begins.
        Returns 0 if no overlap detected (append everything).
    """
    if not incoming_rows or not existing_rows:
        return 0

    # Build fingerprint set for the tail of the existing table
    tail_size = min(30, len(existing_rows))
    existing_tail = existing_rows[-tail_size:]
    existing_fps = set(_row_fingerprint(r) for r in existing_tail)

    # Walk incoming rows, find the last one that matches an existing row
    last_overlap_idx = -1
    for idx, row in enumerate(incoming_rows):
        fp = _row_fingerprint(row)
        if fp in existing_fps:
            last_overlap_idx = idx

    if last_overlap_idx == -1:
        # No overlap detected
        return 0

    # Everything up to and including last_overlap_idx is duplicate
    new_start = last_overlap_idx + 1
    skipped = new_start
    if skipped > 0:
        print(f"  Overlap detected: {skipped} duplicate row(s) skipped")
    return new_start

@task(name="merge-tables")
def merge_tables_with_shared_headers(tables: list[list[list[str]]]) -> list[list[list[str]]]:
    """
    Description:
        FX detects consecutive tables that share identical column headers
        and merges them into a single table. Duplicate rows are skipped.

    Input:
        tables: list of parsed tables

    Output:
        list of tables with consecutive same-header tables merged
        and overlap duplicates removed
    """
    if len(tables) <= 1:
        return tables

    merged = [tables[0]]

    for table in tables[1:]:
        prev_header = _get_header_key(merged[-1])
        curr_header = _get_header_key(table)

        # If headers match and both have at least a header + 1 data row
        if (prev_header and curr_header and prev_header == curr_header
                and len(merged[-1]) >= 2 and len(table) >= 2):

            incoming_data = table[1:]  # skip header row
            existing_data = merged[-1][1:]  # existing data rows (no header)

            # Detect and skip overlap rows from the incoming data
            new_start = _find_overlap_start(existing_data, incoming_data)
            new_rows = incoming_data[new_start:]

            if new_rows:
                merged[-1].extend(new_rows)
                print(f"  Merged table (+{len(new_rows)} new rows): {list(curr_header)[:3]}...")
            else:
                print(f"  Skipped fully overlapping table: {list(curr_header)[:3]}...")
        else:
            merged.append(table)

    return merged

def _table_fingerprint(table: list[list[str]]) -> str:
    """
    Description:
        FX creates a fingerprint for an entire table by combining
        the fingerprints of its first few data rows (up to 5).
        Used to detect whole-table duplicates.

    Input:
        table: a parsed table (list of rows, first row is header)

    Output:
        A string fingerprint representing the table's content
    """
    if len(table) < 2:
        return ''
    # Use up to 5 data rows (skip header) for the fingerprint
    data_rows = table[1:6]
    row_fps = [_row_fingerprint(r) for r in data_rows]
    return '|||'.join(row_fps)

@task(name="deduplicate-tables")
def deduplicate_tables(tables: list[list[list[str]]]) -> list[list[list[str]]]:
    """
    Description:
        FX removes whole-table duplicates based on identical fingerprints.
        The latter table is preferentially preserved. 

    Input:
        tables: list of parsed tables (after merge step)

    Output:
        list of tables with duplicates removed (keeping the later copy)
    """
    if len(tables) <= 1:
        return tables

    # Build fingerprints, walking backwards so we keep the LAST occurrence
    seen_fps = set()
    keep_indices = []

    for i in range(len(tables) - 1, -1, -1):
        fp = _table_fingerprint(tables[i])
        if not fp:
            # Empty or single-row tables: always keep
            keep_indices.append(i)
            continue
        if fp not in seen_fps:
            seen_fps.add(fp)
            keep_indices.append(i)
        else:
            print(f"  Deduplicated table {i+1} (duplicate of a later table)")

    # Reverse to restore original order
    keep_indices.sort()
    return [tables[i] for i in keep_indices]

def _merge_continuation_rows(rows: list[list[str]]) -> list[list[str]]:
    """
    Description:
        FX detects and merges rows where only the Description cell contains text and all 
        other cells are empty. This also automatically merges rows where the Description
        starts with "CUSIP:", "ISIN:", or "Symbol:"
        
    Input:
        rows: list of rows (each row is a list of cell strings),
              including the header row as rows[0]

    Output:
        rows with continuation rows merged into the previous row's
        first cell    
    """

    if len(rows) < 2:
        return rows

    # Determine expected column count from header row
    expected_cols = len(rows[0])

    # Known section/category headings that should NEVER be merged,
    # even when all-uppercase. These are standalone row labels.
    SECTION_HEADING_PATTERNS = [
        r'^(TOTAL|COMMON STOCK|PREFERRED STOCK|MUTUAL FUNDS?|BOND FUNDS?|'
        r'SHORT-TERM FUNDS?|EXCHANGE TRADED|STOCKS?|BONDS?|OTHER|'
        r'INVESTMENT ACTIVITY|CASH MANAGEMENT|CONTRIBUTIONS?|'
        r'DISTRIBUTIONS?|ADDITIONS?|SUBTRACTIONS?|SECURITIES|'
        r'CHECKING ACTIVITY|FEES AND CHARGES|TAXES WITHHELD|'
        r'GENERAL INVESTMENTS|PERSONAL RETIREMENT|EDUCATION|'
        r'MARGIN INTEREST|BILL PAYMENTS|DEBIT CARD)\b',
    ]

    CUSIP_RE = re.compile(r'^CUSIP\s*:', re.IGNORECASE)
    ISIN_RE = re.compile(r'^ISIN\s*:', re.IGNORECASE)
    SYMBOL_RE = re.compile(r'^Symbol\s*:', re.IGNORECASE)

    merged = [rows[0]]  # always keep header as-is

    for row in rows[1:]:
        first_cell  = row[0].strip() if row else ''
        other_cells = row[1:] if len(row) > 1 else []

        if CUSIP_RE.match(first_cell) and len(merged) > 1:
            prev_row = merged[-1]
            # Append CUSIP text to preceding description
            prev_desc = prev_row[0].strip()
            separator = ' ' if prev_desc and not prev_desc.endswith(' ') else ''
            prev_row[0] = prev_desc + separator + first_cell
            # Carry over non-empty value cells into preceding row
            for col_idx, cell_val in enumerate(other_cells, start=1):
                if cell_val.strip() not in ('', '--'):
                    # Pad preceding row if needed
                    while len(prev_row) <= col_idx:
                        prev_row.append('')
                    # Only write if preceding row's cell is empty or placeholder
                    if prev_row[col_idx].strip() in ('', '--'):
                        prev_row[col_idx] = cell_val
            merged[-1] = prev_row
            continue

        if ISIN_RE.match(first_cell) and len(merged) > 1:
            prev_row = merged[-1]
            # Append ISIN text to preceding description
            prev_desc = prev_row[0].strip()
            separator = ' ' if prev_desc and not prev_desc.endswith(' ') else ''
            prev_row[0] = prev_desc + separator + first_cell
            # Carry over non-empty value cells into preceding row
            for col_idx, cell_val in enumerate(other_cells, start=1):
                if cell_val.strip() not in ('', '--'):
                    # Pad preceding row if needed
                    while len(prev_row) <= col_idx:
                        prev_row.append('')
                    # Only write if preceding row's cell is empty or placeholder
                    if prev_row[col_idx].strip() in ('', '--'):
                        prev_row[col_idx] = cell_val
            merged[-1] = prev_row
            continue

        if SYMBOL_RE.match(first_cell) and len(merged) > 1:
            prev_row = merged[-1]
            # Append Symbol text to preceding description
            prev_desc = prev_row[0].strip()
            separator = ' ' if prev_desc and not prev_desc.endswith(' ') else ''
            prev_row[0] = prev_desc + separator + first_cell
            # Carry over non-empty value cells into preceding row
            for col_idx, cell_val in enumerate(other_cells, start=1):
                if cell_val.strip() not in ('', '--'):
                    # Pad preceding row if needed
                    while len(prev_row) <= col_idx:
                        prev_row.append('')
                    # Only write if preceding row's cell is empty or placeholder
                    if prev_row[col_idx].strip() in ('', '--'):
                        prev_row[col_idx] = cell_val
            merged[-1] = prev_row
            continue

        all_others_empty = all(cell.strip() == '' for cell in other_cells)
        is_short_row     = len(row) < expected_cols and len(row) <= 1

        is_continuation = False
        if first_cell and (all_others_empty or is_short_row) and len(merged) > 1:
            is_known_section = any(
                re.search(p, first_cell, re.IGNORECASE)
                for p in SECTION_HEADING_PATTERNS
            )
            if not is_known_section:
                is_continuation = True

        if is_continuation:
            prev_desc = merged[-1][0].strip()
            separator = ' ' if prev_desc and not prev_desc.endswith(' ') else ''
            merged[-1][0] = prev_desc + separator + first_cell
        else:
            merged.append(row)

    return merged

def _merge_split_security_rows(rows: list[list[str]]) -> list[list[str]]:
    """
    Description:
        FX detects and removes spurious phantom rows created when a
        security description is split across a page boundary. The 
        duplicate-text guard prevents doubling up text repeated by the LLM.

    Input:
        rows: list of rows (each row is a list of cell strings),
              including the header row as rows[0]

    Output:
        rows with phantom cross-page continuation rows merged and discarded
    """
    if len(rows) < 2:
        return rows

    def _leading_words(text, n=3):
        words = re.sub(r'[^A-Za-z0-9 ]', ' ', text).split()
        return ' '.join(words[:n]).upper()

    def _all_values_empty(row):
        return all(cell.strip() == '' for cell in row[1:])

    def _has_any_value(row):
        return any(cell.strip() not in ('', '--') for cell in row[1:])

    # A description that starts with a digit or date token is usually a new stock or lot entry
    STARTS_LIKE_DATE_OR_NUM = re.compile(
        r'^(?:\d|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)',
        re.IGNORECASE
    )

    merged = [rows[0]]  # keep header as-is

    i = 1
    while i < len(rows):
        row  = rows[i]
        prev = merged[-1]

        first_cell = row[0].strip()
        prev_desc  = prev[0].strip()

        desc_prefix_matches = (
            first_cell
            and len(merged) > 1
            and _leading_words(first_cell) == _leading_words(prev_desc)
        )

        # Descriptions that are fully identical (same security, multi-lot) are NOT phantoms
        descs_identical = (first_cell.upper() == prev_desc.upper())

        # Path 1: empty-value phantom
        is_empty_phantom = (
            desc_prefix_matches
            and not descs_identical
            and _all_values_empty(row)
        )

        # Path 2: value-populated phantom
        is_populated_phantom = (
            desc_prefix_matches
            and not descs_identical
            and not _all_values_empty(row)
            and _has_any_value(prev)
            and not STARTS_LIKE_DATE_OR_NUM.match(first_cell)
        )

        if is_empty_phantom or is_populated_phantom:
            if is_populated_phantom:
                logger.debug(
                    f"  _merge_split_security_rows: value-populated phantom "
                    f"\'{first_cell[:60]}\' — discarding borrowed values"
                )
            # Strip the repeated leading prefix from first_cell before appending, so we don't the stock name
            prev_words = prev_desc.upper().split()
            curr_words = first_cell.split()
            overlap = 0
            for w in curr_words:
                if overlap < len(prev_words) and w.upper() == prev_words[overlap]:
                    overlap += 1
                else:
                    break
            remainder = ' '.join(curr_words[overlap:]).strip()

            if remainder and remainder.upper() not in prev_desc.upper():
                separator = ' ' if not prev_desc.endswith(' ') else ''
                merged[-1][0] = prev_desc + separator + remainder
                logger.debug(
                    f"  _merge_split_security_rows: merged "
                    f"\'{remainder[:60]}\' into preceding row"
                )
            i += 1
            continue

        merged.append(row)
        i += 1

    return merged


def _strip_cost_basis_annotations(rows: list[list[str]]) -> list[list[str]]:
    """
    Description:
        FX removes cost basis / tax annotation codes that financial statements print as 
        visual footnote markers adjacent to numeric values. 

        The header row (rows[0]) and the description column (col 0) are
        never modified.

    Input:
        rows: list of rows (each row is a list of cell strings),
              including the header row as rows[0]

    Output:
        rows with annotation codes stripped and column alignment preserved
    """
    # Matches a cell whose entire content is one of the safe codes
    STANDALONE = re.compile(r'^(?:ST|LT|MT|N)$', re.IGNORECASE)

    # Matches a trailing code appended after a number (digit or closing paren)
    TRAILING = re.compile(r'(?<=[\d)])\s+(?:ST|LT|MT|N)\s*$', re.IGNORECASE)

    # Date-like cell: contains a day+month combo or a 4-digit year
    DATE_LIKE = re.compile(
        r'(?:\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)|'
        r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2}|'
        r'\d{4})',
        re.IGNORECASE
    )

    if not rows:
        return rows

    cleaned = [rows[0]]   # header row — never modified

    for row in rows[1:]:
        # --- Pattern 1: inline N column-shift fix ---
        working = list(row)
        i = 1
        while i < len(working):
            cell = working[i].strip()
            if (re.fullmatch(r'N', cell, re.IGNORECASE)
                    and DATE_LIKE.search(working[i - 1].strip())):
                working.pop(i)        # drop N, left-shift
                working.append('')    # pad end to preserve column count
            else:
                i += 1

        # --- Patterns 2 & 3: standalone and trailing codes in value columns ---
        new_row = [working[0]]   # col 0 (description) passed through unchanged
        for cell in working[1:]:
            s = cell.strip()
            if STANDALONE.fullmatch(s):
                # Pattern 2: standalone code → empty slot (no shift)
                new_row.append('')
            else:
                # Pattern 3: trailing code after a number → strip the suffix
                new_row.append(TRAILING.sub('', s).strip())

        cleaned.append(new_row)

    return cleaned



def _fix_stacked_column_pairs(rows: list[list[str]]) -> list[list[str]]:
    """
    Description:
        FX detects stacked column pairs in any table and corrects misalignment caused by the 
        LLM inconsistently handling them.

        SAFETY GUARANTEES:
          - Header row is never modified.
          - Col 0 (Description) is never touched.
          - If no registry match is found, the function is a strict no-op.
          - Type is read from the registry, not inferred from data, so
            corrections are deterministic regardless of LLM output patterns.

    Input:
        rows: list of rows (each row is a list of cell strings),
              including the header row as rows[0]

    Output:
        rows with stacked column pair misalignment corrected
    """
    # Registry of known stacked column pairs.
    STACKED_PAIR_REGISTRY = [
        # J.P. Morgan: Est. Accrued Inc. (top) / Est. Annual Inc. (bottom)
        # Both sub-columns carry independent values → double
        ('accrued inc',   'annual inc',        'double'),
        ('est. accrued',  'est. annual',        'double'),
        ('accrued',       'annual',             'double'),
        # Fidelity: Ending Market Value (top) / Accrued Interest AI (bottom)
        # Only one value per row; second slot is always empty → single
        ('ending market', 'accrued interest',  'single'),
        ('market value',  'accrued interest',  'single'),
        ('market value',  'accrued inc',       'single'),
    ]

    CURRENCY_RE = re.compile(r'^\$[\d,]+\.\d{2}$')
    DASH_RE     = re.compile(r'^--$')

    def is_real(cell):
        return cell.strip() not in ('', '--')

    if len(rows) < 2:
        return rows

    header = rows[0]

    # --- Phase 1: detect stacked column pairs and read type from registry ---
    candidate_pairs = []  # list of col_n indices
    pair_type       = {}  # col_n → 'double' | 'single'
    for i in range(len(header) - 1):
        h1 = header[i].strip().lower()
        h2 = header[i + 1].strip().lower()
        for kw1, kw2, ptype in STACKED_PAIR_REGISTRY:
            if kw1 in h1 and kw2 in h2:
                candidate_pairs.append(i)
                pair_type[i] = ptype
                logger.debug(
                    f"  _fix_stacked_column_pairs: detected ({i},{i+1}) "
                    f"'{header[i]}'/'{header[i+1]}' → {ptype} (registry)"
                )
                break

    if not candidate_pairs:
        return rows

    data_rows     = rows[1:]
    expected_cols = len(header)

    # --- Phase 3: apply corrections ---
    fixed = [rows[0]]
    for row in data_rows:
        row  = list(row)
        desc = row[0].strip().upper() if row else ''

        # Sub-pattern C: TOTAL row left-shifted
        if desc.startswith('TOTAL'):
            mkt_col = next(
                (i for i, h in enumerate(header)
                 if 'market' in h.lower() and 'value' in h.lower()),
                None
            )
            if mkt_col is not None:
                premature_col = next(
                    (i for i in range(2, min(mkt_col, len(row)))
                     if CURRENCY_RE.match(row[i].strip())),
                    None
                )
                if premature_col is not None:
                    shift = mkt_col - premature_col
                    for _ in range(shift):
                        row.insert(premature_col, '')
                    logger.debug(
                        f'  _fix_stacked_column_pairs: Sub-C — '
                        f'TOTAL row: inserted {shift} empty cell(s) at col {premature_col}'
                    )

        # Pad for safe indexing of all overflow positions
        max_overflow = max(candidate_pairs, default=0) + 2
        while len(row) <= max_overflow:
            row.append('')

        for col_n in candidate_pairs:
            col_np1    = col_n + 1
            overflow   = col_n + 2
            v1         = row[col_n].strip()
            v2         = row[col_np1].strip()
            v_overflow = row[overflow].strip() if overflow < len(row) else ''
            ptype      = pair_type.get(col_n, 'single')

            if ptype == 'double':
                # Sub-A: spurious "--" in col N, real value already in col N+1
                if DASH_RE.match(v1) and is_real(v2):
                    row[col_n] = ''
                    logger.debug(
                        f'  _fix_stacked_column_pairs: Sub-A cleared "--" in col {col_n}'
                    )
                # Sub-B: "--" in col N+1, real value overflowed to col N+2
                elif DASH_RE.match(v2) and is_real(v_overflow):
                    row[col_np1]  = v_overflow
                    row[overflow] = ''
                    logger.debug(
                        f'  _fix_stacked_column_pairs: Sub-B moved overflow '
                        f'from col {overflow} to col {col_np1}'
                    )

            else:  # single-value
                # Value in both cols → col N is correct, clear col N+1 spillover
                if is_real(v1) and is_real(v2):
                    row[col_np1] = ''
                    logger.debug(
                        f'  _fix_stacked_column_pairs: single-value spillover '
                        f'cleared in col {col_np1}'
                    )
                # Value only in col N+1 → move to col N
                elif not is_real(v1) and is_real(v2):
                    row[col_n]   = v2
                    row[col_np1] = ''
                    logger.debug(
                        f'  _fix_stacked_column_pairs: single-value shift '
                        f"moved '{v2}' from col {col_np1} to col {col_n}"
                    )

        # Trim to expected column count, removing overflow padding
        row = row[:expected_cols]
        fixed.append(row)

    return fixed


@task(name="parse-tables-from-response")
def parse_tables_from_response(response_text: str) -> list[list[list[str]]]:
    """
    Description:
        FX parses the LLM response to extract structured tables.
        Expects pipe-delimited ( | ) rows between TABLE_START/TABLE_END markers.
        Applies defensive post-processing:
          - Strips leading/trailing empty columns caused by pipe artifacts
          - Merges continuation rows (multi-line descriptions) into single rows
          - Merges phantom cross-page description splits into the preceding row
          - Strips cost basis / tax annotation codes (ST, LT, MT, N) from value cells
          - Detects and corrects stacked column pair misalignment (single- and double-value)
          - Does NOT merge tables here (merging happens after all batches are collected)

    Input: String of LLM response

    Output:
        List of tables, where each table is list[list[str]]
        Each table is returned as a list of rows, where each row is a list of cell values.
    """
    if not response_text or "NO_TABLES_FOUND" in response_text:
        return []

    tables = []

    for block in re.split(r'===TABLE_START===', response_text):
        if '===TABLE_END===' not in block:
            continue

        table_text = block.split('===TABLE_END===')[0].strip()

        # Strip any markdown code fences the LLM might have added
        table_text = re.sub(r'^```[a-z]*\s*', '', table_text)
        table_text = re.sub(r'\s*```$', '', table_text)
        table_text = table_text.strip()

        if not table_text:
            continue

        rows = []
        for line in table_text.split('\n'):
            line = line.strip()
            if not line:
                continue

            # Split on pipe delimiter and strip whitespace from each cell
            cells = [cell.strip() for cell in line.split('|')]

            # Skip rows that are entirely empty after splitting
            if cells and any(cell != '' for cell in cells):
                rows.append(cells)

        if rows:
            # Remove systematic leading/trailing pipe artifact columns
            rows = _normalize_columns(rows)
            # Merge multi-line description continuations into single rows
            rows = _merge_continuation_rows(rows)
            # Merge phantom cross-page description splits into preceding rows
            rows = _merge_split_security_rows(rows)
            # Strip cost basis / tax annotation codes (ST, LT, MT, N)
            rows = _strip_cost_basis_annotations(rows)
            # Detect and correct stacked column pair misalignment
            rows = _fix_stacked_column_pairs(rows)
            tables.append(rows)

    return tables

@task(name="save-table-to-csv", task_run_name="save-csv: {filepath}")
def save_table_to_csv(table: list[list[str]], filepath: str):
    """
    Description:
        FX saves the parsed table (list of lists of rows) to a CSV file.
        The first row of the table is treated as the header and written as-is.
        No row index or positional numeric header row is written, so the CSV
        can be loaded directly with pd.read_csv without header-detection heuristics.

    Input:
        Table represented as a list of rows, where each row is a list of cell values.

    Output:
        CSV filepath
    """
    # Determine max columns across all rows
    max_cols = max(len(row) for row in table) if table else 0

    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # Write all rows as-is (row 0 is the column header, no index column)
        for row in table:
            writer.writerow(row + [''] * (max_cols - len(row)))