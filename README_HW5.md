# Running Boltz for Protein–Ligand Structure Generation and Affinity Prediction

This workflow uses **Boltz** to generate 50 protein–ligand structures and predict binding affinities for OATP1B1. Several adjustments were required to ensure efficient and successful execution.

---

## Key Notes

1. **GPU Requirement**
   - Boltz must be run on a GPU for reasonable performance

2. **Environment Setup**
   - Activated a virtual environment before installing and running Boltz

3. **Project Organization**
   - Created a structured directory for:
     - Input `.yaml` files
     - Scripts
     - Output structures and predictions

4. **Ligand Input**
   - Imported a `.csv` file containing ligands and corresponding SMILES strings

5. **YAML Generation**
   - Wrote a script to automatically generate 50 `.yaml` files:
     - Same protein (OATP1B1)
     - Different ligand per file

6. **Affinity Specification**
   - Ensured each `.yaml` included an **affinity prediction block**

7. **Batch Execution**
   - Ran Boltz in a loop to process all `.yaml` files automatically

8. **Disk Quota Fix**
   - Resolved Triton cache issues by redirecting cache to scratch space:
```bash
export TRITON_CACHE_DIR=/mnt/gs21/scratch/cordahlr/triton_cache
mkdir -p $TRITON_CACHE_DIR
```

---

## Setup and Installation

### Navigate and Request GPU
```bash
cd boltz

salloc --gres=gpu:1 --mem=64G --time=02:00:00
```

### Activate Environment and Install
```bash
source boltz_env/bin/activate

pip install boltz[cuda] -U
pip install ruamel.yaml
```

---

## Project Directory Setup

```bash
mkdir -p ~/projects/oapt1b1_boltz
cd ~/projects/oapt1b1_boltz

mkdir -p inputs/yaml outputs scripts
```

### Add Protein Sequence
```bash
nano inputs/protein.fasta
```

---

## YAML Generation Script

### `scripts/generate_yaml.py`

```python
from pathlib import Path
import yaml

INPUT_DIR = Path("inputs/yaml_affinity")
OUTPUT_DIR = Path("inputs/yaml_affinity_fixed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

for yaml_file in INPUT_DIR.glob("*.yaml"):
    with open(yaml_file, "r") as f:
        data = yaml.safe_load(f)

    protein_seq = None
    ligand_smiles = None
    ligand_id = None

    for item in data.get("sequences", []):
        if "protein" in item and protein_seq is None:
            protein_seq = item["protein"]["sequence"]
            protein_id = item["protein"]["id"] if isinstance(item["protein"]["id"], str) else item["protein"]["id"][0]
        elif "ligand" in item and ligand_smiles is None:
            ligand_smiles = item["ligand"]["smiles"]
            ligand_id = item["ligand"]["id"] if isinstance(item["ligand"]["id"], str) else item["ligand"]["id"][0]

    if protein_seq is None or ligand_smiles is None:
        print(f"Skipping {yaml_file.name}: missing protein or ligand")
        continue

    new_data = {
        "version": 1,
        "sequences": [
            {
                "protein": {
                    "id": protein_id,
                    "sequence": protein_seq
                }
            },
            {
                "ligand": {
                    "id": ligand_id,
                    "smiles": ligand_smiles
                }
            }
        ],
        "properties": [
            {
                "affinity": {
                    "binder": ligand_id
                }
            }
        ]
    }

    out_file = OUTPUT_DIR / yaml_file.name
    with open(out_file, "w") as f:
        yaml.dump(new_data, f, sort_keys=False)

    print(f"✔ Wrote {out_file}")
```

### Run Script
```bash
python scripts/generate_yaml.py
```

### Verify File Count
```bash
ls inputs/yaml | wc -l
```

---

## Run Boltz on All YAML Files

```bash
for y in inputs/yaml_affinity_fixed/oapt1b1_*.yaml; do
    name=$(basename "$y" .yaml)
    echo "Running $name..."
    boltz predict "$y" --out_dir "outputs_affinity_fixed/$name" --use_msa_server
done
```

---

## Resume After GPU Crash (Last 25 Files)

```bash
for y in $(ls -1 inputs/yaml_affinity_fixed/oapt1b1_*.yaml | tail -25); do
    name=$(basename "$y" .yaml)
    echo "Running $name..."
    boltz predict "$y" --out_dir "outputs_affinity_fixed/$name" --use_msa_server
done
```

---

## Output

- Generates:
  - Protein–ligand structures
  - Predicted binding affinities
- Organized per ligand in:
  ```
  outputs_affinity_fixed/<ligand_name>/
  ```

---

## Summary

- Automated generation of 50 protein–ligand systems using YAML templating  
- Efficient batch processing with GPU acceleration  
- Integrated affinity prediction directly into workflow  
- Resolved runtime and disk quota issues for stable large-scale execution  
