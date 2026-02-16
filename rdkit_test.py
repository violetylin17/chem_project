from rdkit import Chem
from rdkit.Chem import Descriptors

# 1. Create a molecule object from SMILES (e.g., Benzene)
smiles = "c1ccccc1"
mol = Chem.MolFromSmiles(smiles)

# 2. Calculate "Hard Facts" for your LLM Prompt
mw = Descriptors.MolWt(mol)
logp = Descriptors.MolLogP(mol)

print(f"Molecular Weight: {mw:.2f}")
print(f"LogP: {logp:.2f}")

# 3. Check for specific groups (e.g., Nitro groups for CJ velocity)
nitro_pattern = Chem.MolFromSmarts("[N+](=O)[O-]")
has_nitro = mol.HasSubstructMatch(nitro_pattern)
print(f"Contains Nitro Group: {has_nitro}")