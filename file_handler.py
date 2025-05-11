import sys
import os
import re
import pandas as pd
import pyarrow.csv as pyspcsv
import pyarrow as pa
import time
import chardet
import csv
import mmap
from concurrent.futures import ThreadPoolExecutor
import io
import base64
from io import StringIO, BytesIO
import tempfile
from datetime import datetime
from pathlib import Path
import zipfile
import xml.etree.ElementTree as ET
# Note: xlrd, pyxlsb, openpyxl, xlsx2csv will be used if installed as per requirements

# --- Helper from your original code ---
def get_sheet_names_from_excel_bytes(excel_bytes, filename_ext):
    """Get sheet names from Excel file bytes."""
    # This function needs to work with in-memory bytes
    # For simplicity, we'll write to a temporary file to reuse existing logic.
    # More advanced in-memory parsing could be done with libraries directly if available.
    temp_f = tempfile.NamedTemporaryFile(delete=False, suffix=filename_ext)
    temp_f.write(excel_bytes)
    temp_f_path = temp_f.name
    temp_f.close()

    names = []
    try:
        if filename_ext == '.xls':
            import xlrd
            with xlrd.open_workbook(temp_f_path, on_demand=True) as workbook:
                names = workbook.sheet_names()
        elif filename_ext == '.xlsb':
            import pyxlsb
            with pyxlsb.open_workbook(temp_f_path) as workbook:
                names = workbook.sheets
        elif filename_ext in ('.xlsx', '.xlsm'):
            # Using openpyxl for in-memory friendly sheet name extraction if possible,
            # fallback to zipfile for wider compatibility from your original code.
            try:
                import openpyxl
                workbook = openpyxl.load_workbook(BytesIO(excel_bytes), read_only=True, data_only=True)
                names = workbook.sheetnames
            except Exception:
                with zipfile.ZipFile(BytesIO(excel_bytes)) as archive:
                    tree = ET.parse(archive.open('xl/workbook.xml'))
                    root = tree.getroot()
                    ns = {'main': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
                    sheets = root.findall(".//main:sheet", ns)
                    names = [sheet.get('name') for sheet in sheets]
    finally:
        os.unlink(temp_f_path)
    return names

class FileLoader:
    def __init__(self, file_content_string, filename, sheet_name=None):
        self.file_content_string = file_content_string # base64 encoded string
        self.filename = filename
        self.sheet_name = sheet_name
        self.file_info = {'filename': filename}
        self._file_path_ = filename # For consistency with original logic messages
        self.decoded_bytes = None

    def _decode_content(self):
        if self.decoded_bytes is None:
            content_type, content_string = self.file_content_string.split(',')
            self.decoded_bytes = base64.b64decode(content_string)
        return self.decoded_bytes

    def wait_for_file_access(self, file_path, max_attempts=5, delay=0.5): # Shorter for web
        """Placeholder, as we're mostly working in-memory or temp files."""
        return True # Simplified for this context

    def read_excel_data(self, temp_file_path, use_calamine=True):
        """Reads excel using pandas, trying calamine first."""
        df = None
        engine_used = None
        try:
            if use_calamine:
                try:
                    # Ensure calamine is available
                    pd.read_excel(temp_file_path, engine='calamine', sheet_name=self.sheet_name, nrows=1)
                    df = pd.read_excel(
                        temp_file_path,
                        engine='calamine',
                        sheet_name=self.sheet_name,
                        header=0, # Assuming header is the first row for web UI
                        dtype=str,
                        keep_default_na=False
                    )
                    engine_used = 'calamine'
                except Exception as e_calamine:
                    print(f"Calamine failed: {e_calamine}. Falling back.")
                    df = None # Reset df

            if df is None: # Fallback or if calamine not preferred/failed
                df = pd.read_excel(
                    temp_file_path,
                    sheet_name=self.sheet_name,
                    header=0,
                    dtype=str,
                    keep_default_na=False
                    # openpyxl is default for xlsx, xlrd for xls
                )
                engine_used = pd.get_option("io.excel.xlsx.reader") if self.filename.lower().endswith('.xlsx') else pd.get_option("io.excel.xls.reader")
            
            self.file_info['engine_used'] = engine_used
            return df
        except Exception as e:
            raise ValueError(f"Pandas.read_excel failed: {e}")


    def run(self):
        start_time = time.time()
        decoded_bytes = self._decode_content()
        file_ext = Path(self.filename).suffix.lower()
        output_csv_path = None # For potential excel to csv conversion

        try:
            if file_ext == '.csv':
                # Use a BytesIO object for in-memory processing
                file_like_object = BytesIO(decoded_bytes)
                # Create a temporary file path for functions that expect a path
                with tempfile.NamedTemporaryFile(delete=False, mode='wb', suffix='.csv') as tmp_file:
                    tmp_file.write(decoded_bytes)
                    self.temp_file_path = tmp_file.name
                
                df, csv_file_info = self.read_csv(self.temp_file_path)
                self.file_info.update(csv_file_info)
                os.unlink(self.temp_file_path) # Clean up temp file

            elif file_ext in ('.xlsx', '.xlsm', '.xlsb', '.xls'):
                self.file_info['sheet_name'] = self.sheet_name # Should be pre-selected via UI

                # For large Excel files, convert to CSV first (using a temp file)
                # Size threshold for conversion (e.g., 50MB for web context)
                size_mb = len(decoded_bytes) / (1024 * 1024)
                self.file_info['size_mb'] = size_mb

                if size_mb > 50 and 'xlsx2csv' in sys.modules : # Convert large Excel to CSV
                    print(f"Large Excel file ({size_mb:.2f}MB), attempting conversion to CSV.")
                    self.file_info['conversion_attempted'] = 'xlsx2csv'
                    
                    source_excel_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=file_ext)
                    source_excel_tmp.write(decoded_bytes)
                    source_excel_tmp_path = source_excel_tmp.name
                    source_excel_tmp.close()

                    output_csv_tmp_path = Path(tempfile.gettempdir()).joinpath(
                        f"{Path(self.filename).stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                    )
                    
                    try:
                        from xlsx2csv import Xlsx2csv
                        converter = Xlsx2csv(source_excel_tmp_path, outputencoding="utf-8", delimiter=',')
                        converter.convert(str(output_csv_tmp_path), sheetname=self.sheet_name)
                        
                        if output_csv_tmp_path.exists() and output_csv_tmp_path.stat().st_size > 0:
                            print(f"Successfully converted to CSV: {output_csv_tmp_path}")
                            df, csv_file_info = self.read_csv(str(output_csv_tmp_path))
                            self.file_info.update(csv_file_info)
                            self.file_info['source_type'] = 'Excel (converted to CSV)'
                            output_csv_path = output_csv_tmp_path # To delete later
                        else:
                            raise Exception("xlsx2csv conversion resulted in an empty file.")
                    except Exception as e_conv:
                        print(f"xlsx2csv conversion failed: {e_conv}. Reading Excel directly.")
                        self.file_info['conversion_failed_reason'] = str(e_conv)
                        df = self.read_excel_data(source_excel_tmp_path) # Fallback
                        self.file_info['source_type'] = 'Excel (direct read)'
                    finally:
                        os.unlink(source_excel_tmp_path)

                else: # Read Excel directly
                    print(f"Reading Excel file ({size_mb:.2f}MB) directly.")
                    source_excel_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=file_ext)
                    source_excel_tmp.write(decoded_bytes)
                    source_excel_tmp_path = source_excel_tmp.name
                    source_excel_tmp.close()
                    try:
                        df = self.read_excel_data(source_excel_tmp_path)
                    finally:
                        os.unlink(source_excel_tmp_path)
                    self.file_info['source_type'] = 'Excel (direct read)'

                if df is not None:
                    self.file_info['rows'] = df.shape[0]
                    self.file_info['columns'] = df.shape[1]

            else:
                raise ValueError(f"Unsupported file type: {file_ext}")

            self.file_info['load_time_seconds'] = time.time() - start_time
            return df, self.file_info

        except Exception as e:
            print(f"Error in FileLoader.run: {e}", file=sys.stderr) # Print to stderr for Dash logs
            self.file_info['error'] = str(e)
            self.file_info['load_time_seconds'] = time.time() - start_time
            return pd.DataFrame(), self.file_info # Return empty DataFrame on error
        finally:
            if output_csv_path and os.path.exists(output_csv_path):
                try:
                    os.unlink(output_csv_path)
                except Exception as e_del:
                    print(f"Warning: Could not delete temp CSV: {e_del}")
    
    # --- CSV Reading Logic (adapted from your FileLoaderWorker) ---
    def detect_encoding_and_dialect(self, file_path, sampling_size_bytes=1024*1024): # 1MB sample
        encoding = None
        # Simplified BOM detection from path
        with open(file_path, 'rb') as file:
            raw = file.read(128) # Small chunk for BOM
            # ... (BOM detection logic - for brevity, direct chardet usage shown)

        with open(file_path, 'rb') as f:
            sample_bytes = f.read(sampling_size_bytes)
        
        detected = chardet.detect(sample_bytes)
        encoding = detected['encoding'] if detected['encoding'] else 'utf-8'
        encoding = 'utf-8' if encoding.lower() == 'ascii' else encoding # Prefer utf-8 over ascii

        # Dialect detection with fallback
        try:
            # chardet gives bytes, Sniffer needs string
            sample_str = sample_bytes.decode(encoding, errors='replace') 
            dialect = csv.Sniffer().sniff(sample_str[:500000], delimiters=[',', ';', '\t', '|',':']) # Sniff smaller part
            # Check if delimiter is actually present in first few lines
            if not any(dialect.delimiter in line for line in sample_str.splitlines()[:20]):
                 # Fallback: count occurrences of potential delimiters in the sample
                counts = {delim: sample_str.count(delim) for delim in [',', ';', '\t', '|',':']}
                if any(c > 0 for c in counts.values()):
                    dialect.delimiter = max(counts, key=counts.get)
                else: # Default if no common delimiter found
                    dialect.delimiter = ',' 
        except (csv.Error, UnicodeDecodeError):
            dialect = csv.excel # Fallback dialect
            dialect.delimiter = ',' # Common default
        
        self.file_info['detected_encoding'] = encoding
        self.file_info['detected_delimiter'] = dialect.delimiter
        return encoding, dialect

    def read_pyarrow_contr_w_delim_fix(self, file_path, encoding, delimiter, quotechar):
        # This is a complex method. For Dash, direct pandas.read_csv with detected params is often sufficient.
        # PyArrow shines for very large files and specific optimizations.
        # If pandas fails, this could be a fallback. For now, we'll simplify.
        # You can integrate the full version if pandas proves insufficient.
        print(f"Attempting PyArrow read for: {os.path.basename(file_path)} with enc: {encoding}, delim: '{delimiter}'")
        try:
            # Basic PyArrow read, can be expanded with your original logic
            read_options = pyspcsv.ReadOptions(encoding=encoding)
            parse_options = pyspcsv.ParseOptions(delimiter=delimiter, quote_char=quotechar if quotechar else '"')
            table = pyspcsv.read_csv(file_path, read_options=read_options, parse_options=parse_options)
            df = table.to_pandas(self_destruct=True, split_blocks=True, zero_copy_only=False)
            print("PyArrow read successful.")
            self.file_info['read_method'] = 'pyarrow'
            return df
        except Exception as e_pa:
            print(f"PyArrow read failed: {e_pa}. Will try pandas.")
            self.file_info['pyarrow_error'] = str(e_pa)
            return None # Indicate failure

    def read_csv(self, file_path_str):
        file_path = Path(file_path_str)
        encoding, dialect = self.detect_encoding_and_dialect(file_path)
        
        df = self.read_pyarrow_contr_w_delim_fix(file_path, encoding, dialect.delimiter, dialect.quotechar)

        if df is None: # PyArrow failed or was skipped, try pandas
            print(f"Attempting pandas.read_csv for: {file_path.name} with enc: {encoding}, delim: '{dialect.delimiter}'")
            try:
                df = pd.read_csv(
                    file_path,
                    encoding=encoding,
                    delimiter=dialect.delimiter,
                    quotechar=dialect.quotechar if dialect.quotechar else '"',
                    escapechar=dialect.escapechar,
                    low_memory=False, # Better type inference for large files
                    dtype=str,        # Load as string initially, convert later
                    keep_default_na=False, # Keep empty strings as is
                    na_filter=False # Don't interpret 'NA', 'NULL' as NaN by default
                )
                print("pandas.read_csv successful.")
                self.file_info['read_method'] = 'pandas'
            except Exception as e_pd:
                print(f"pandas.read_csv also failed: {e_pd}")
                self.file_info['pandas_error'] = str(e_pd)
                raise ValueError(f"Failed to read CSV with both PyArrow and Pandas: {e_pd}")


        file_info_update = {
            'delimiter': dialect.delimiter,
            'encoding': encoding,
            'rows': df.shape[0],
            'columns': df.shape[1],
            'size_mb': file_path.stat().st_size / (1024 * 1024)
        }
        return df, file_info_update