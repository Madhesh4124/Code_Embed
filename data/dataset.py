from pathlib import Path
from typing import List, Optional, Dict,Union
import pandas as pd
import torch
from torch.utils.data import Dataset,DataLoader

from tokenizer.tokenizer import CodeEmbedTokenizer


PROCESSED_DIR=Path("data/processed")


class CodeSearchDataset(Dataset):
    """
    Pytorch Dataset loading CodeSearchNet processes parquet splits.
    """
    
    def __init__(self, parquet_path: Union[str,Path]):
        self.parquet_path=Path(parquet_path)
        
        if not self.parquet_path.exists():
            raise FileNotFoundError(f"Dataset split not found :{self.parquet_path}")
        
        self.df=pd.read_parquet(self.parquet_path)
        
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Dict[str,str]:
        row=self.df.iloc[idx]
        return {
            "code":row["code"],
            "docstring":row["docstring"],
            "func_name":row["func_name"] if "func_name" in row else ""
        }
            

class CodeSearchCollator:
    """
    Collator that tokenizes and dynamically pads batches of code and docstrings
    """
    
    def __init__(
        self,
        tokenizer: CodeEmbedTokenizer,
        max_length: int = 256,
        padding: bool = True
    ):
        self.tokenizer=tokenizer
        self.max_length=max_length
        self.padding=padding
        
        
    def __call__(self,batch: List[Dict[str,str]]) -> Dict[str,torch.Tensor]:
        codes = [item["code"] for item in batch]
        docstrings = [item["docstring"] for item in batch]
        func_names = [item["func_name"] for item in batch]
        
        
        #Tokenize Code batch with <CODE> modality
        code_batch =self.tokenizer.encode(
            codes,
            max_length=self.max_length,
            padding=self.padding,
            truncation=True,
            modality="code"
        )      
        
        
        #Tokenize docstring batch with <TEXT> modality
        text_batch =self.tokenizer.encode(
                      docstrings,
                      max_length=self.max_length,
                      padding=self.padding,
                      truncation=True,
                      modality="text"
        ) 
        
        return {
            "code_ids":code_batch["input_ids"],
            "code_mask":code_batch["attention_mask"],
            "text_ids":text_batch["input_ids"],
            "text_mask":text_batch["attention_mask"],
            "func_names":func_names,
        }
        
        
def create_dataloader(
    split: str = "train",
    batch_size: int = 256,
    shuffle: bool =True,
    max_length: int = 256,
    num_workers: int =0,
    tokenizer: Optional[CodeEmbedTokenizer] =None,
) -> DataLoader:
    
    """
    Factory helper to build a ready-to-use Dataloader for a given split.
    """
    if tokenizer is None:
        tokenizer=CodeEmbedTokenizer()
        
    parquet_path = PROCESSED_DIR / f"{split}.parquet"
    dataset=CodeSearchDataset(parquet_path)
    collator=CodeSearchCollator(tokenizer=tokenizer,max_length=max_length)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=torch.cuda.is_available(),
        drop_last=(split=="train")  #Drop incomplete last batch only in training
    )
    
    
if __name__=="__main__":
    
    print("Testing Dataloader on Validation set...")
    loader=create_dataloader(split="validation",batch_size=4,shuffle=False)
    batch=next(iter(loader))
    
    print("\nBatch Loaded successfully!")
    
    print("code_ids shape:  ",batch["code_ids"].shape)
    print("code_mask shape:  ",batch["code_mask"].shape)
    print("text_ids shape:  ",batch["text_ids"].shape)
    print("text_mask shape:  ",batch["text_mask"].shape)
    print("Functions:  ",batch["func_names"])