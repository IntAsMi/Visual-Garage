import pandas as pd
import numpy as np

def get_column_statistics(df_column: pd.Series):
    """Generates detailed statistics for a pandas Series."""
    if df_column is None or df_column.empty:
        return {"error": "Column is empty or not provided."}

    stats = {}
    col_name = df_column.name
    stats['column_name'] = str(col_name)
    stats['dtype'] = str(df_column.dtype)
    stats['total_values'] = len(df_column)
    
    # Non-empty and empty counts
    # Pandas `count()` gives non-NA. For strings, an empty string is not NA.
    # We consider truly empty or all-whitespace strings as 'empty' for string types.
    if pd.api.types.is_string_dtype(df_column.dtype) or df_column.dtype == 'object':
        non_na_series = df_column.dropna()
        stats['non_na_count'] = len(non_na_series)
        # More specific 'empty' for strings (NaN, None, '', '   ')
        empty_conditions = non_na_series.astype(str).str.strip() == ''
        stats['empty_string_count'] = empty_conditions.sum()
        stats['truly_non_empty_count'] = stats['non_na_count'] - stats['empty_string_count']
        stats['na_count'] = df_column.isna().sum()

    else: # Numeric, datetime, etc.
        stats['non_na_count'] = df_column.count()
        stats['na_count'] = df_column.isna().sum()
        stats['truly_non_empty_count'] = stats['non_na_count'] # For non-string, non-NA is non-empty

    stats['na_percentage'] = (stats['na_count'] / stats['total_values']) * 100 if stats['total_values'] > 0 else 0
    
    unique_values = df_column.nunique(dropna=False) # Include NaNs in unique count if present
    stats['unique_values_count'] = unique_values

    if stats['truly_non_empty_count'] > 0:
        # Display some unique values if cardinality is low
        if unique_values <= 15:
            # Get value counts, including NaNs if they are part of the unique set
            counts = df_column.value_counts(dropna=False).sort_values(ascending=False)
            stats['value_counts'] = {str(k): v for k, v in counts.head(15).items()}
        else:
            # Show most frequent values even if many unique
            counts = df_column.value_counts(dropna=True).sort_values(ascending=False) # Exclude NA for most_frequent if many uniques
            stats['most_frequent_values'] = {str(k): v for k, v in counts.head(5).items()}


        # Type-specific stats
        # Attempt to convert to numeric if object/string to get numeric stats
        numeric_series = None
        if pd.api.types.is_object_dtype(df_column.dtype) or pd.api.types.is_string_dtype(df_column.dtype):
            try:
                temp_numeric = pd.to_numeric(df_column, errors='coerce').dropna()
                if not temp_numeric.empty and temp_numeric.count() > 0.5 * df_column.count(): # If at least half are numeric
                    numeric_series = temp_numeric
                    stats['coerced_to_numeric'] = True
            except Exception:
                stats['coerced_to_numeric'] = False
                pass # Cannot be coerced
        elif pd.api.types.is_numeric_dtype(df_column.dtype):
            numeric_series = df_column.dropna()

        if numeric_series is not None and not numeric_series.empty:
            stats['numeric_stats'] = {
                'mean': numeric_series.mean(),
                'median': numeric_series.median(),
                'std_dev': numeric_series.std(),
                'min': numeric_series.min(),
                'max': numeric_series.max(),
                'q1': numeric_series.quantile(0.25),
                'q3': numeric_series.quantile(0.75),
                'sum': numeric_series.sum(),
                'zeros_count': (numeric_series == 0).sum()
            }

        datetime_series = None
        if pd.api.types.is_object_dtype(df_column.dtype) or pd.api.types.is_string_dtype(df_column.dtype):
            try:
                temp_datetime = pd.to_datetime(df_column, errors='coerce').dropna()
                if not temp_datetime.empty and temp_datetime.count() > 0.5 * df_column.count(): # If at least half are datetime
                    datetime_series = temp_datetime
                    stats['coerced_to_datetime'] = True
            except Exception:
                stats['coerced_to_datetime'] = False
                pass
        elif pd.api.types.is_datetime64_any_dtype(df_column.dtype):
            datetime_series = df_column.dropna()
        
        if datetime_series is not None and not datetime_series.empty:
            stats['datetime_stats'] = {
                'earliest': datetime_series.min().isoformat() if pd.notna(datetime_series.min()) else 'N/A',
                'latest': datetime_series.max().isoformat() if pd.notna(datetime_series.max()) else 'N/A',
            }
            # Check for time component
            if not (datetime_series.dt.hour == 0).all() or \
               not (datetime_series.dt.minute == 0).all() or \
               not (datetime_series.dt.second == 0).all():
                stats['datetime_stats']['has_time_component'] = True
            else:
                stats['datetime_stats']['has_time_component'] = False


    # Format for display
    formatted_stats = [f"### Statistics for Column: `{stats.get('column_name', 'N/A')}`"]
    formatted_stats.append(f"- **Data Type (Original):** `{stats.get('dtype', 'N/A')}`")
    if stats.get('coerced_to_numeric'): formatted_stats.append(f"- _Successfully auto-coerced to numeric for stats_")
    if stats.get('coerced_to_datetime'): formatted_stats.append(f"- _Successfully auto-coerced to datetime for stats_")

    formatted_stats.append(f"- **Total Rows:** {stats.get('total_values', 0):,}")
    formatted_stats.append(f"- **Non-Empty Values (Non-NA & not just whitespace):** {stats.get('truly_non_empty_count',0):,}")
    formatted_stats.append(f"- **Missing (NA/NaN):** {stats.get('na_count', 0):,} ({stats.get('na_percentage', 0):.2f}%)")
    if 'empty_string_count' in stats and stats['empty_string_count'] > 0:
         formatted_stats.append(f"- **Empty Strings (but not NA):** {stats.get('empty_string_count',0):,}")
    formatted_stats.append(f"- **Unique Values:** {stats.get('unique_values_count', 0):,}")

    if 'value_counts' in stats:
        formatted_stats.append("- **Top Unique Values (Count):**")
        for val, count in stats['value_counts'].items():
            display_val = f"'{val}'" if len(val) < 30 else f"'{val[:27]}...'"
            if val.strip() == "": display_val = f"'' (Empty String)"
            if val == "nan" or val == "None": display_val = f"`{val}` (NA)" # If NA was stringified
            formatted_stats.append(f"  - {display_val}: {count:,}")
    elif 'most_frequent_values' in stats:
        formatted_stats.append("- **Most Frequent Values (Count):**")
        for val, count in stats['most_frequent_values'].items():
            display_val = f"'{val}'" if len(val) < 30 else f"'{val[:27]}...'"
            formatted_stats.append(f"  - {display_val}: {count:,}")
    
    if 'numeric_stats' in stats:
        ns = stats['numeric_stats']
        formatted_stats.append("- **Numeric Profile:**")
        formatted_stats.append(f"  - Mean: {ns.get('mean', 0):.2f}, Median: {ns.get('median', 0):.2f}, Sum: {ns.get('sum', 0):.2f}")
        formatted_stats.append(f"  - Min: {ns.get('min', 0):.2f}, Max: {ns.get('max', 0):.2f}, Std Dev: {ns.get('std_dev', 0):.2f}")
        formatted_stats.append(f"  - Q1 (25%): {ns.get('q1', 0):.2f}, Q3 (75%): {ns.get('q3', 0):.2f}")
        if 'zeros_count' in ns and ns['zeros_count'] > 0:
            formatted_stats.append(f"  - Contains {ns['zeros_count']:,} zero values.")


    if 'datetime_stats' in stats:
        ds = stats['datetime_stats']
        formatted_stats.append("- **Date/Time Profile:**")
        formatted_stats.append(f"  - Earliest: {ds.get('earliest', 'N/A')}")
        formatted_stats.append(f"  - Latest: {ds.get('latest', 'N/A')}")
        if ds.get('has_time_component'):
             formatted_stats.append("  - _Dates appear to include time components._")
        else:
             formatted_stats.append("  - _Dates appear to be date-only (no significant time components)._")

    if "error" in stats:
        return f"Error generating statistics: {stats['error']}"
        
    return "\n".join(formatted_stats)