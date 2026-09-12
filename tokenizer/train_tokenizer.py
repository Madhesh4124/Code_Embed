from pathlib import Path

import pandas as pd
from tokenizers import (
    Tokenizer,
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    trainers,
)

TRAIN_DATA_DIR=Path("data/processed/train.parquet")
TOKENIZER_DIR=Path("tokenizer")

SPECIAL_TOKENS=[
    "<PAD>",
    "<UNK>",
    "<BOS>",
    "<EOS>",
    "<CODE>",
    "<TEXT>"
]

VOCAB_SIZE=16000



def batch_iterator(df: pd.DataFrame, batch_size: int = 10000):
    """
    Yield batches of text strings (both code and docstrings) for tokenzier training
    """
    
    for i in range(0,len(df),batch_size):   #step increment= batch_size
        batch=df.iloc[i : i+batch_size]
        
        yield batch["docstring"].tolist()  + batch["code"].tolist()
        # yield lets a function return values one at a time, 
        # pausing and resuming instead of finishing all at once.
        

def train_tokenizer():
    TOKENIZER_DIR.mkdir(parents=True,exist_ok=True)
    
    print("Loading the processed train dataset.....")
    
    df=pd.read_parquet(TRAIN_DATA_DIR,columns=["docstring","code"])
    
    
    #initialize bpe model with unk
    tokenizer = Tokenizer(models.BPE(unk_token="<UNK>"))

    #add normalizer
    tokenizer.normalizer=normalizers.NFKC()
    
    #add pre-tokenizer and decoder 
    tokenizer.pre_tokenizer=pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder=decoders.ByteLevel(trim_offsets=True)
    
    #set up trainer with our vocab size and special tokens
    
    trainer=trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    
    
    #Train Tokenizer
    print(f"Training BPE Tokenizer on  {len(df)} pairs (vocab_size={VOCAB_SIZE})....")
    tokenizer.train_from_iterator(batch_iterator(df),trainer=trainer)
    
    
    #save tokenizer
    save_path=TOKENIZER_DIR / "tokenizer.json"
    tokenizer.save(str(save_path))
    print(f"Tokenizer trained and saved to {save_path}!")
    

if __name__=="__main__":
    train_tokenizer()
    
