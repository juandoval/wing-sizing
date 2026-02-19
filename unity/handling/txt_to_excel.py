import pandas as pd
import sys

def txt_to_excel(txt_path):
    """Convert tab-separated txt file to Excel format."""
    df = pd.read_csv(txt_path, sep='\t')
    
    df = df[df['Time'] >= 0]
    
    excel_path = txt_path.replace('.txt', '.xlsx')
    df.to_excel(excel_path, index=False)
    
    print(f"Converted: {txt_path}")
    print(f"Saved to: {excel_path}")
    print(f"Rows: {len(df)}, Columns: {len(df.columns)}")

if __name__ == "__main__":
    txt_file = "unity\\handling\\take_off_43.txt"
    txt_to_excel(txt_file)
