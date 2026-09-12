from pathlib import Path

import pandas as pd

RAW_DIR=Path("data/raw")
PROCESSED_DIR=Path("data/processed")


def is_valid_docstring(docstring :str) -> bool:
    """
    Check if docstring is meaningful(not empty, not trivial TODO/NONE, at least 3 words).
    """
    
    if not isinstance(docstring,str):
        return False
    
    doc=docstring.strip()
    
    if len(doc)==0:
        return False
    
    words=doc.split()
    if len(words)<3:
        return False
    return doc.lower() not in {"todo", "none", "pass", "fixme"}



def is_valid_code(code: str) -> bool:
    """Check if code length is within reasonable bounds (10 to 2048 tokens/words)."""
    
    if not isinstance(code,str):
        return False
    
    tokens=code.strip().split()
    return 10 <= len(tokens) <= 2048
   
   
def clean_split(df: pd.DataFrame,is_train: bool =True) -> pd.DataFrame:
    """
    Clean, filter and optionally deduplicate a dataset split.
    """
    
    #Rename columns for consistency
    df=df.rename(columns={'func_documentation_string':'docstring','func_code_string':'code'})
    
    
    #Remove missing/empty values
    
    df=df.dropna(subset=["docstring","code"])
    
    
    
    #apply functions
    df=df[df["code"].apply(is_valid_code)]
    df=df[df['docstring'].apply(is_valid_docstring)]
    
    #filter boilerplate (drop __init__'s)
    if 'func_path_in_repository' in df.columns:
        df=df[~df['func_path_in_repository'].str.contains("migrations",case=False,na=False)]
        df=df[~df['func_path_in_repository'].str.contains("__init__",case=False,na=False)]    
    
    if is_train:
        df=df.drop_duplicates(subset=['code'])
    
    columns_to_keep=["code","docstring","func_name"]
    df=df[[col for col in columns_to_keep if col in df.columns]]
    return df


def preprocess_all():
    PROCESSED_DIR.mkdir(parents=True,exist_ok=True)
    
    for split in ['train','validation','test']:
        print(f"Processing {split}")
        df=pd.read_parquet(RAW_DIR / f"{split}.parquet")
        
        clean_df=clean_split(df,is_train=(split=='train'))
        print(f"{split}:{len(df)} -> {len(clean_df)} samples")
        
        clean_df.to_parquet(PROCESSED_DIR / f"{split}.parquet", index=False)
        
if __name__=='__main__':
    preprocess_all()
        