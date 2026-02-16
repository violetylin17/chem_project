import json
import requests
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from sentence_transformers import SentenceTransformer

# ==========================================
# 1. RDKit Fact Extraction (Chemical Intelligence)
# ==========================================
def extract_facts(smiles):
    print(f"[*] Analyzing SMILES: {smiles}")
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
# 2. Ollama Text Generation (Llama 3.1 Reasoning)
# ==========================================
def generate_text_via_ollama(smiles, facts, model="aiasistentworld/Llama-3.1-8B-Instruct-STO-Master"):
    print(f"[*] Calling Ollama ({model}) to generate description...")
    
    prompt = f"""
    You are an expert computational chemist.
    Analyze the following molecule based on RDKit-calculated facts:
    - SMILES: {smiles}
    - Formula: {facts['Formula']}
    - Molecular Weight: {facts['MW']}
    - LogP: {facts['LogP']}
    - Number of Nitro Groups: {facts['Nitro_Groups']}

    Task: Describe the potential energetic characteristics of this molecule.
    Focus on how the nitro groups and LogP might influence its properties (e.g., detonation velocity).
    Provide a professional summary in 3 sentences.
    """

    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False  # Disable streaming; return full result at once
    }

    try:
        response = requests.post(url, json=payload, timeout=180)
        response.raise_for_status()
        return response.json().get("response", "")
    except Exception as e:
        return f"Error connecting to Ollama: {e}"

# ==========================================
# 3. Text Embedding (BGE-M3 Embedding)
# ==========================================
def get_embedding(text):
    print("[*] Loading BGE-M3 and generating embedding (first run will download the model)...")
    model = SentenceTransformer('BAAI/bge-m3')
    embedding = model.encode(text)
    return embedding

# ==========================================
# Main Pipeline
# ==========================================
if __name__ == "__main__":
    # Test molecule: TNT
    test_smiles = "CC1=C(C=C(C=C1[N+](=O)[O-])[N+](=O)[O-])[N+](=O)[O-]"
    
    # Step 1: Chemical facts
    facts = extract_facts(test_smiles)
    if facts:
        print(f"  > RDKit Facts: {facts}")
        
        # Step 2: LLM description
        description = generate_text_via_ollama(test_smiles, facts)
        print(f"\n[+] LLM Description:\n{description}\n")
        
        # Step 3: Embedding
        vector = get_embedding(description)
        print(f"[+] Embedding generated! Dimension: {vector.shape}")
        print(f"  > First 5 values: {vector[:5]}")
        
        # Save results for later analysis
        result = {
            "smiles": test_smiles,
            "description": description,
            "embedding": vector.tolist()
        }
        with open("test_output.json", "w") as f:
            json.dump(result, f)
        print("\n[*] Results saved to test_output.json")
    else:
        print("[!] Invalid SMILES string")
