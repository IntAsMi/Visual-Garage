import pandas as pd
import numpy as np # For safe evaluation context

class PandasTransformer:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy() # Work on a copy

    def apply_transformation(self, column_name, operation, new_column_name=None, params=None):
        """
        Applies a defined transformation.
        params is a dictionary of parameters for the operation.
        """
        if params is None:
            params = {}

        if not new_column_name:
            new_column_name = column_name # Modify in place if no new name

        try:
            if operation == 'to_numeric':
                self.df[new_column_name] = pd.to_numeric(self.df[column_name], errors=params.get('errors', 'coerce'))
            
            elif operation == 'to_datetime':
                self.df[new_column_name] = pd.to_datetime(self.df[column_name], errors=params.get('errors', 'coerce'), format=params.get('format', None))
            
            elif operation == 'to_string':
                self.df[new_column_name] = self.df[column_name].astype(str)

            elif operation == 'fill_na':
                fill_value = params.get('value', 0)
                # Try to convert fill_value to the column's dtype if possible
                try:
                    col_dtype = self.df[column_name].dtype
                    if pd.api.types.is_numeric_dtype(col_dtype):
                        fill_value = float(fill_value) if '.' in str(fill_value) else int(fill_value)
                    elif pd.api.types.is_datetime64_any_dtype(col_dtype):
                        fill_value = pd.to_datetime(fill_value)
                except:
                    pass # keep as is if conversion fails
                self.df[new_column_name] = self.df[column_name].fillna(fill_value)

            elif operation == 'drop_column':
                if column_name in self.df.columns:
                    self.df.drop(columns=[column_name], inplace=True)
                else:
                    return self.df, f"Column '{column_name}' not found for dropping."


            elif operation == 'rename_column':
                if not params.get('new_name_parameter'): # param name for new column name from UI
                    return self.df, "New name for renaming not provided."
                self.df.rename(columns={column_name: params['new_name_parameter']}, inplace=True)


            elif operation == 'create_from_formula':
                formula = params.get('formula_string', '')
                if not formula:
                    return self.df, "Formula not provided."
                
                # **VERY IMPORTANT SECURITY WARNING:**
                # eval() is dangerous with untrusted input.
                # For a safe version, you'd need a proper formula parser or a sandboxed environment.
                # This example uses a VERY restricted eval for simple arithmetic or column references.
                # It's NOT a full Python interpreter.
                
                # Create a limited context for eval
                # Only allow access to other columns (df['col_name']) and basic numpy functions
                # It's still risky if column names can be manipulated to inject code.
                
                # A safer approach would be to use something like `numexpr` if formulas are purely numerical
                # or a custom parser.
                
                # For this example, we'll try a limited `df.eval()` which is safer than raw `eval()`
                try:
                    # df.eval() is generally safer for column-wise operations
                    self.df[new_column_name] = self.df.eval(formula)
                except Exception as e:
                    # Fallback or more complex scenario:
                    # This is where a more robust, but also more dangerous, approach might be taken.
                    # For this example, we will NOT implement a full `eval(formula, safe_globals, local_vars)`
                    # due to inherent security risks with arbitrary string evaluation.
                    # A production system would need a dedicated formula engine or heavy sandboxing.
                    return self.df, f"Error evaluating formula with df.eval(): {e}. Ensure formula uses column names directly (e.g., 'col_A * col_B + 5'). Complex functions require a dedicated formula engine."
            
            else:
                return self.df, f"Unknown operation: {operation}"

            return self.df, f"Transformation '{operation}' on '{column_name}' applied successfully to '{new_column_name}'."
        
        except Exception as e:
            return self.df, f"Error during transformation '{operation}' on '{column_name}': {e}"

    def get_dataframe(self):
        return self.df