import pandas as pd
from pathlib import Path
import json
import sys


PATH_TO_FILE = "News_Category_Dataset_v3.json"  

def load_df(path: str) -> pd.DataFrame:
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix in ['.csv', '.tsv']:
        sep = ',' if suffix == '.csv' else '\t'
        return pd.read_csv(p, sep=sep)
    elif suffix in ['.json', '.jsonl', '.ndjson']:
        # try jsonl first
        try:
            return pd.read_json(p, lines=True)
        except ValueError:
            try:
                return pd.read_json(p)
            except Exception:
                with open(p, 'r', encoding='utf-8') as f:
                    raw = json.load(f)
                try:
                    return pd.json_normalize(raw)
                except Exception:
                    return pd.DataFrame(raw)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

def show_column_values(df: pd.DataFrame, max_unique_show: int = 50, top_n: int = 10):
    print(f"Total rows: {len(df)}\n")
    for col in df.columns:
        print("="*80)
        print(f"Column: {col}")
        # count nulls
        nulls = df[col].isnull().sum()
        print(f"  Nulls: {nulls}")
        try:
            nunique = df[col].nunique(dropna=True)
            print(f"  Unique values: {nunique}")
            # if numeric show stats
            if pd.api.types.is_numeric_dtype(df[col]):
                print("  Numeric column stats:")
                print(df[col].describe().to_string())
            # show top values (most frequent)
            print(f"  Top {top_n} values (value : count):")
            vc = df[col].value_counts(dropna=False).head(top_n)
            print(vc.to_string())
            # if many uniques and you want list of uniques:
            if nunique <= max_unique_show:
                uniques = df[col].dropna().astype(str).unique().tolist()
                print(f"  All unique values (count={len(uniques)}):")
                print(uniques)
            else:
                print(f"  (Not listing all uniques because > {max_unique_show})")
        except Exception as e:
            print("  Could not analyze column:", e)
    print("="*80)

def main():
    try:
        df = load_df(PATH_TO_FILE)
    except Exception as e:
        print("Error loading file:", e)
        sys.exit(1)

    # if single-column and contains dicts/lists, try normalize
    if len(df.columns) == 1 and df.iloc[:,0].apply(lambda x: isinstance(x, (dict, list))).any():
        try:
            df = pd.json_normalize(df.iloc[:,0].tolist())
            print("Normalized single-column JSON -> new columns:", df.columns.tolist())
        except Exception:
            pass

    # flatten nested columns if any
    if any(df[c].apply(lambda x: isinstance(x, (dict, list))).any() for c in df.columns):
        try:
            df = pd.json_normalize(df.to_dict(orient='records'))
            print("After json_normalize -> columns:", df.columns.tolist())
        except Exception:
            pass

    show_column_values(df, max_unique_show=100, top_n=10)

if __name__ == "__main__":
    main()