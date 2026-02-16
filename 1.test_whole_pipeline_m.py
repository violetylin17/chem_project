import json
import requests
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from sentence_transformers import SentenceTransformer

# ==========================================
# 1. RDKit 事實提取 (Chemical Intelligence)
# ==========================================
def extract_facts(smiles):
    print(f"[*] 正在分析 SMILES: {smiles}")
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return None
    
    facts = {
        "Formula": rdMolDescriptors.CalcMolFormula(mol),
        "MW": round(Descriptors.MolWt(mol), 2),
        "LogP": round(Descriptors.MolLogP(mol), 2),
        "Nitro_Groups": smiles.count("N(=O)=O") + smiles.count("[N+](=O)[O-]")
    }
    return facts

# ==========================================
# 2. Ollama 文本生成 (Llama 3.1 Reasoning)
# ==========================================
def generate_text_via_ollama(smiles, facts, model="aiasistentworld/Llama-3.1-8B-Instruct-STO-Master"):
    print(f"[*] 正在呼叫 Ollama ({model}) 生成描述...")
    
    prompt = f"""
    You are an expert computational chemist. 
    Analyze the following molecule based on RDKit-calculated facts:
    - SMILES: {smiles}
    - Formula: {facts['Formula']}
    - Molecular Weight: {facts['MW']}
    - LogP: {facts['LogP']}
    - Number of Nitro Groups: {facts['Nitro_Groups']}

    Task: Describe the potential energetic characteristics of this molecule. 
    Focus on how the nitro groups and LogP might influence its property (e.g., detonation velocity).
    Provide a professional summary in 3 sentences.
    """

    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False # 關閉串流，一次性取得完整結果
    }

    try:
        response = requests.post(url, json=payload, timeout=180)
        response.raise_for_status()
        return response.json().get("response", "")
    except Exception as e:
        return f"Error connecting to Ollama: {e}"

# ==========================================
# 3. 文本向量化 (BGE-M3 Embedding)
# ==========================================
def get_embedding(text):
    print("[*] 正在載入 BGE-M3 並生成向量 (首次執行會下載模型)...")
    # 如果你有 GPU，這會自動在 GPU 上跑
    model = SentenceTransformer('BAAI/bge-m3')
    embedding = model.encode(text)
    return embedding

# ==========================================
# 主流程 (Main Pipeline)
# ==========================================
if __name__ == "__main__":
    # 測試分子: TNT
    test_smiles = "CC1=C(C=C(C=C1[N+](=O)[O-])[N+](=O)[O-])[N+](=O)[O-]"
    
    # Step 1: 化學事實
    facts = extract_facts(test_smiles)
    if facts:
        print(f"  > RDKit 事實: {facts}")
        
        # Step 2: LLM 生成
        description = generate_text_via_ollama(test_smiles, facts)
        print(f"\n[+] LLM 描述:\n{description}\n")
        
        # Step 3: 向量化
        vector = get_embedding(description)
        print(f"[+] 向量生成成功！維度: {vector.shape}")
        print(f"  > 前 5 個數值: {vector[:5]}")
        
        # 儲存結果供後續分析
        result = {
            "smiles": test_smiles,
            "description": description,
            "embedding": vector.tolist()
        }
        with open("test_output.json", "w") as f:
            json.dump(result, f)
        print("\n[*] 結果已儲存至 test_output.json")
    else:
        print("[!] 無效的 SMILES 字串")