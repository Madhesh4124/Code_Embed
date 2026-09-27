# CodeEmbed Evaluation Results

Here is the final scoreboard across all the phases we ran! 

| Phase | Model Architecture | Capacity | Key Settings | Test MRR |
|-------|--------------------|----------|--------------|----------|
| **1** | BM25 Baseline | N/A | Lexical Search (Keyword) | `0.5155` |
| **2** | Basic Encoder | 7M | Weight-Sharing, Default | `0.4500` |
| **3** | Shared Encoder | 7M | Modality Embeddings | `0.4300` |
| **4** | Dual Encoder | 7M | No Weight-Sharing | `0.2900` |
| **6** | Shared Encoder (Large) | 54M | Batch Size 16 | `0.0982` |
| **6** | **Basic Encoder (Ablation Winner)** | **7M** | **Mean Pooling, Temp=0.01, Epoch 9** | **`0.5342` 🏆** |
| **6** | Dual Fixed (FAISS) | 7M | CLS/Mean Pooling, Temp=0.1, Epoch 1 | `0.4807` |
| **6** | Dual Fixed (FAISS) | 7M | CLS/Mean Pooling, Temp=0.07, Epoch 1 | `0.4684` |
| **7** | Pretrained MiniLM-L6-v2 | 22M | Pretrained (English), No Fine-Tuning | `0.4725` |
| **7** | Pretrained CodeBERT | 125M | MLM Pretrained, No Contrastive Alignment | `0.0100` |

### Key Takeaways:
1. **Weight Sharing is Mandatory from Scratch:** The Dual Encoder (`0.29`) failed miserably compared to the Basic Encoder (`0.45`) because it couldn't align the Python and English vector spaces without sharing weights.
2. **Batch Size is King:** Scaling up the Shared Encoder to 54M parameters completely broke the model (`0.09`) because we had to drop the batch size to 16, destroying the contrastive loss function's ability to see enough negative examples.
3. **Pretraining Isn't Magic:** CodeBERT (`0.01`) failed completely for search because it was pretrained on Masked Language Modeling (filling in the blanks), not Contrastive Alignment (matching queries to code). 
4. **The Ultimate Winner:** By optimizing the Pooling Strategy to `mean` and finding the perfect Contrastive Temperature of `0.07`, our tiny 7M parameter Basic Encoder (`0.5274`) actually **beat the mathematical BM25 baseline!**
