from transformers import AutoTokenizer, AutoModel
import torch

# 這是 Hugging Face 上的路徑，執行時會自動下載
model_path = "deepchem/ChemBERTa-100M-MLM"

# 初始化
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path)

# 測試一個 SMILES
smiles = "C1CCCCC1"
inputs = tokenizer(smiles, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)

# 取得向量 (Hidden State)
# ChemBERTa 通常輸出 384 維的向量
last_hidden_state = outputs.last_hidden_state
# 取 [CLS] token (index 0) 作為整個分子的代表向量
cls_embedding = last_hidden_state[0, 0, :].numpy()

print(f"ChemBERTa 向量維度: {cls_embedding.shape}")