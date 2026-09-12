from pathlib import Path

import torch
from tokenizers import Tokenizer

DEFAULT_TOKENIZER_PATH = Path("tokenizer/tokenizer.json")


class CodeEmbedTokenizer:
    
    
    """wrapper around tokenizer.Tokenizer providing tensor output"""
    
    def __init__(self,tokenizer_path: str | Path=DEFAULT_TOKENIZER_PATH):
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        
        #Token IDs
        self.pad_token_id=0
        self.unk_token_id=1
        self.bos_token_id=2
        self.eos_token_id=3
        self.code_token_id=4
        self.text_token_id=5
        self.vocab_size = self.tokenizer.get_vocab_size()
        
        
    def encode(
        self,
        texts: str | list[str],
        max_length: int =256,
        padding: bool =True,
        truncation: bool =True,
        modality: str | None = None,  # "code" "text" or None
    ) -> dict[str, torch.Tensor]:
        
        """Encode string or list of strings into pytorch tensors.
        
        Returns:
        ---------
        Dict containing:
        "input_ids" : torch.Tensor of shape (batch_size, max_length) (B,L)
        "attention_mask" : torch.Tensor of shape (batch_size, max_length) (B,L)
        """
        
        #Ensure texts is a list
        is_single = isinstance(texts,str)
        text_list=[texts] if is_single else texts
        
        # Prepend modality token if requested
        if modality == "code":
            text_list = [f"<CODE> {t}" for t in text_list]
        elif modality == "text":
            text_list = [f"<TEXT> {t}" for t in text_list]
        
        #Configure padding and truncation
        if truncation:
            self.tokenizer.enable_truncation(max_length=max_length)
        else:
            self.tokenizer.no_truncation()
            
        if padding:
            self.tokenizer.enable_padding(
                length=max_length,
                pad_id=self.pad_token_id,
                pad_token="<PAD>")
        else:
            self.tokenizer.no_padding()
            
            
        
        #Encode batch
        encodings=self.tokenizer.encode_batch(text_list)
        
        input_ids=[enc.ids for enc in encodings]
        attention_mask=[enc.attention_mask for enc in encodings]
        
        return {
            "input_ids": torch.tensor(input_ids,dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask,dtype=torch.long)
        }
    
    
    def decode(
        self,
        token_ids: list[int] | torch.Tensor,
        skip_special_tokens: bool =True
    ) -> str | list[str]:
        
        """Decode token IDs back to string"""
        
        if isinstance(token_ids,torch.Tensor):
            token_ids=token_ids.tolist()
            
        
        if len(token_ids) == 0:
            return ""
        
        if isinstance(token_ids[0],list):
            #Batch of sequences
            
            return self.tokenizer.decode_batch(token_ids,skip_special_tokens=skip_special_tokens)
        
        return self.tokenizer.decode(token_ids,skip_special_tokens=skip_special_tokens)
    
    

if __name__=="__main__":
    tokenizer = CodeEmbedTokenizer()
    texts = ["def foo():\n    return 42", "Hello, world!"]
    encoded = tokenizer.encode(texts, max_length=20)
    print("Encoded:", encoded)
    
    decoded = tokenizer.decode(encoded["input_ids"])
    print("Decoded:", decoded)
        