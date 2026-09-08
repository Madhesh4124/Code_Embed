from pathlib import Path
from datasets import load_dataset


RAW_DATA_DIR=Path("data/raw")

def download_codesearchnet(output_dir: Path=RAW_DATA_DIR,language: str="python") -> None:
    """
    Download CodeSearchNet dataset for a specific programming language.

    Args:
        output_dir (Path): Directory to save the downloaded dataset.
        language (str): Programming language to download (default: "python").
    """
    output_dir.mkdir(parents=True,exist_ok=True)
    
    
    dataset=load_dataset("code-search-net/code_search_net",language,trust_remote_code=True)
    
    for split in dataset.keys():
        print(f"Saving {split}")
        dataset[split].to_parquet(output_dir / f"{split}.parquet")
    
    print("Dataset Downloaded")
    
if __name__=="__main__":
    download_codesearchnet()