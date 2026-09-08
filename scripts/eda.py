from pathlib import Path
import numpy as np
import pandas as pd
from tokenizers import Tokenizer
from tqdm import tqdm

DATA_PATH=Path("data/processed/train.parquet")
TOKENIZER_PATH=Path("tokenizer/tokenizer.json")


def analyze_token_lengths(sample_size : int = 50000):
    print(f"Loading data from {DATA_PATH}....")
    
    df=pd.read_parquet(DATA_PATH,columns=["docstring","code"])
    
    if sample_size and sample_size < len(df):
        print(f"Sampling {sample_size:,} rows for fast analysis...")
        df=df.sample(sample_size,random_state=42)
    else:
        print(f"Analysing all {len(df):,} rows...")
        
    tokenizer=Tokenizer.from_file(str(TOKENIZER_PATH))
    tokenizer.no_padding()
    tokenizer.no_truncation()
    
    print("Tokenizing docstrings...")
    doc_encodings=tokenizer.encode_batch(df["docstring"].tolist())
    doc_lengths=np.array([len(enc.ids) for enc in doc_encodings])
    
    print("Tokenizing code snippets...")
    code_encodings=tokenizer.encode_batch(df["code"].tolist())
    code_lengths=np.array([len(enc.ids) for enc in code_encodings])
    
    percentiles=[50,75,90,95,99]
    
    print("\n" +  "="*55)
    print("Token Length Distribution")
    print("="*55)
    print(f"{'Metric':<15}{'Docstrings (Text)':<20}{'Code (Python)':<15}")
    print("-"*55)
    
    print(f"{'Mean':<15}{np.mean(doc_lengths):<20.2f}{np.mean(code_lengths):<15.2f}")
    print(f"{'Min':<15}{np.min(doc_lengths):<20}{np.min(code_lengths):<15}")
    
    
    for p in percentiles:
        print(f"{f'P{p}':<15} {np.percentile(doc_lengths, p):<20.0f} {np.percentile(code_lengths, p):<15.0f}")
        
    print(f"{'Max':<15} {np.max(doc_lengths):<20} {np.max(code_lengths):<15}")
    print("=" * 55)
    
        # Coverage analysis for candidate sequence lengths
    print("\nCODE COVERAGE (% of samples that fit without truncation):")
    for max_len in [128, 256, 384, 512]:
        coverage = (code_lengths <= max_len).mean() * 100
        print(f"  max_length = {max_len:<4} -> {coverage:6.2f}% of code functions fit completely")
        
if __name__ == "__main__":
    analyze_token_lengths(sample_size=50000)