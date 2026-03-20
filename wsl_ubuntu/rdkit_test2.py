from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

def get_energetic_descriptors(smiles):
    """
    Extracts key energetic and structural descriptors from a SMILES string.
    """
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return "Invalid SMILES"

    # 1. Physicochemical Properties
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    tpsa = Descriptors.TPSA(mol)
    
    # 2. Atom Counting for Oxygen Balance (OB%) and Nitrogen Content (N%)
    atoms = {'C': 0, 'H': 0, 'O': 0, 'N': 0}
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        if symbol in atoms:
            atoms[symbol] += 1
            
    # 3. Oxygen Balance (OB%) Calculation
    # Formula: -1600/MW * (2C + H/2 - O)
    ob = (-1600 / mw) * (2 * atoms['C'] + atoms['H'] / 2 - atoms['O'])
    
    # 4. Nitrogen Content (%)
    n_content = (atoms['N'] * 14.007 / mw) * 100

    # 5. Explophore Counting (Nitro Groups)
    # SMARTS for Nitro: [N+](=O)[O-]
    nitro_pattern = Chem.MolFromSmarts("[N+](=O)[O-]")
    nitro_count = len(mol.GetSubstructMatches(nitro_pattern))
    
    # 6. Molecular Complexity & Rigidity
    csp3 = rdMolDescriptors.CalcFractionCSP3(mol)
    rings = rdMolDescriptors.CalcNumRings(mol)

    # Format findings into a structured dictionary
    return {
        "MW": round(mw, 2),
        "OB": round(ob, 2),
        "N_Content": round(n_content, 2),
        "Nitro_Count": nitro_count,
        "LogP": round(logp, 2),
        "TPSA": round(tpsa, 2),
        "CSP3": round(csp3, 2),
        "Rings": rings
    }

def generate_llm_prompt(smiles, data):
    """
    Constructs a professional scientific prompt for Llama-3.1.
    """
    facts = (
        f"SMILES: {smiles}\n"
        f"- Molecular Weight: {data['MW']} g/mol\n"
        f"- Oxygen Balance (OB%): {data['OB']}%\n"
        f"- Nitrogen Content: {data['N_Content']}%\n"
        f"- Nitro Group Count: {data['Nitro_Count']}\n"
        f"- Structural Rigidity (Fraction CSP3): {data['CSP3']}\n"
        f"- Ring Count: {data['Rings']}"
    )
    
    prompt = f"""
[HARD FACTS]
{facts}

"""
    return prompt

# --- Example Execution (Using TNT) ---
tnt_smiles = "CC1=C(C=C(C=C1[N+](=O)[O-])[N+](=O)[O-])[N+](=O)[O-]"
tnt_data = get_energetic_descriptors(tnt_smiles)

if tnt_data != "Invalid SMILES":
    final_prompt = generate_llm_prompt(tnt_smiles, tnt_data)
    print(final_prompt)