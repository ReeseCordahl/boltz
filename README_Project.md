# OATP1B1 Selectivity Engineering with Boltz
## Reference https://github.com/lrg12345/boltz for any confusions

## Overview
This project focuses on engineering the human transporter OATP1B1 to improve substrate selectivity using computational methods. The goal is to reduce transport of bilirubin (an undesired metabolic waste product) while maintaining transport of estrone-3-sulfate (E3S), a model drug-like molecule.

Using structural data and machine learning, we performed saturation mutagenesis on key residues and evaluated how mutations impact ligand binding.

## Project Goals
- Reduce bilirubin binding in OATP1B1
- Preserve or maintain E3S binding
- Identify mutations that improve transporter selectivity
- Develop a computational workflow for screening mutations

## Methods

### 1. Structure Selection
Cryo-EM structures were used as templates:
- Bilirubin-bound structure (8HNC)
- E3S-bound structure (8HND)

These structures guided identification of key binding residues.
### 2. Residue Selection
Two residues were selected for mutation:
- **T357**
- **F360**

These residues are located in the bilirubin-binding pocket and were shown to minimally impact E3S transport in prior studies.
### 3. Saturation Mutagenesis
All 20 amino acids were tested at both positions:
- Total combinations: **400 mutants (20 × 20)**

Each mutant was evaluated for both bilirubin and E3S binding.
### 4. Structure Generation with Boltz
Mutant structures and ligand binding predictions were generated using a machine learning-based protein modeling tool.

#### Job Execution
Jobs were run on an HPC cluster using SLURM:

### Code used: 

    #!/bin/bash --login
    #SBATCH --job-name=boltz2
    #SBATCH --time=01:00:00
    #SBATCH --cpus-per-task=8
    #SBATCH --mem=32G
    #SBATCH --gpus=a100:1
    #SBATCH --array=0-0
    #SBATCH --output=/mnt/gs21/scratch/garlan70/boltz/logs/%x_%A_%a.out
    #SBATCH --error=/mnt/gs21/scratch/garlan70/boltz/logs/%x_%A_%a.err
 
    set -euo pipefail
 
    BASE=/mnt/gs21/scratch/garlan70/boltz
    INPUTS=$BASE/inputs/yamls
    LOGS=$BASE/logs
    OUTPUTS=$BASE/outputs
    CACHE_BASE=${SLURM_TMPDIR:-/tmp/$USER/boltz_${SLURM_ARRAY_TASK_ID}}
 
    mkdir -p "$LOGS" "$OUTPUTS" "$CACHE_BASE"
 
    # Number of distinct seeds to run per YAML
    BOLTZ_NUM_SEEDS=5
 
    # Number of diffusion samples per seed
    BOLTZ_SAMPLES_PER_SEED=5
 
    # Parallel sampling within a seed run
    MAX_PARALLEL_SAMPLES="$BOLTZ_SAMPLES_PER_SEED"
 
    # Max allowed seed value
    MAX_SEED=2147483647
 
    module purge
    module load Miniforge3
    source "$(conda info --base)/etc/profile.d/conda.sh"
 
    unset PYTHONPATH
    export PYTHONNOUSERSITE=1
 
    conda activate boltz2
 
    echo "which python: $(which python)"
    echo "which boltz:  $(which boltz)"
    echo "CACHE_BASE:   $CACHE_BASE"
 
    python -c "import sys; print('python exe:', sys.executable)"
    python -c "import site; print('usersite enabled:', site.ENABLE_USER_SITE)"
    python -c "import torch; print('cuda available:', torch.cuda.is_available(), 'count:', torch.cuda.device_count())"
 
    nvidia-smi || true
 
    # Select YAML from inputs/yamls
    mapfile -t YAMLS < <(find "$INPUTS" -maxdepth 1 -type f -name "*.yaml" | sort)
 
    echo "Found ${#YAMLS[@]} YAML files in $INPUTS"
 
    YAML="${YAMLS[$SLURM_ARRAY_TASK_ID]:-}"
 
    if [[ -z "${YAML:-}" ]]; then
    echo "No YAML found for task ${SLURM_ARRAY_TASK_ID}"
    exit 1
    fi
 
    if [[ ! -f "$YAML" ]]; then
    echo "YAML does not exist: $YAML"
    exit 1
    fi
 
    JOB_TAG="$(basename "$YAML" .yaml)"
    OUT_HOST="${OUTPUTS}/${JOB_TAG}"
 
    mkdir -p "$OUT_HOST"
 
    BASE_SEED=$(echo -n "${JOB_TAG}" | cksum | awk '{print $1 % ('"$MAX_SEED"'-1) + 1}')
 
    echo "Running task ${SLURM_ARRAY_TASK_ID}"
    echo "Input YAML:    $YAML"
    echo "Job tag:       $JOB_TAG"
    echo "Output dir:    $OUT_HOST"
    echo "Base seed:     $BASE_SEED"
    echo "Num seeds:     $BOLTZ_NUM_SEEDS"
    echo "Samples/seed:  $BOLTZ_SAMPLES_PER_SEED"
 
    sleep $((SLURM_ARRAY_TASK_ID % 15))
  
    COMPLETE_COUNT=0
    for ((i=0; i<BOLTZ_NUM_SEEDS; i++)); do
    SEED=$(( (BASE_SEED + i) % MAX_SEED ))
    [[ "$SEED" -eq 0 ]] && SEED=1
    SEED_OUT="${OUT_HOST}/seed-${SEED}"
 
    if find "$SEED_OUT" -type f \( -name "*.cif" -o -name "*.mmcif" \) | grep -q . 2>/dev/null; then
        COMPLETE_COUNT=$((COMPLETE_COUNT + 1))
    fi
    done
 
    if [[ "$COMPLETE_COUNT" -eq "$BOLTZ_NUM_SEEDS" ]]; then
    echo "All ${BOLTZ_NUM_SEEDS} seeds appear complete for ${JOB_TAG}; skipping"
    exit 0
    fi
 
# Run Boltz for multiple seeds
    for ((i=0; i<BOLTZ_NUM_SEEDS; i++)); do
    SEED=$(( (BASE_SEED + i) % MAX_SEED ))
    [[ "$SEED" -eq 0 ]] && SEED=1
 
    SEED_OUT="${OUT_HOST}/seed-${SEED}"
    SEED_CACHE="${CACHE_BASE}/seed-${SEED}"
 
    mkdir -p "$SEED_OUT" "$SEED_CACHE"
 
    if find "$SEED_OUT" -type f \( -name "*.cif" -o -name "*.mmcif" \) | grep -q . 2>/dev/null; then
        echo "==> Seed ${SEED} already completed for ${JOB_TAG}; skipping"
        continue
    fi
 
    echo "==> Running seed ${SEED} ($((i+1))/${BOLTZ_NUM_SEEDS})"
    echo "==> Seed output: ${SEED_OUT}"
    echo "==> Seed cache:  ${SEED_CACHE}"
 
    srun boltz predict "$YAML" \
      --use_msa_server \
      --cache "$SEED_CACHE" \
      --out_dir "$SEED_OUT" \
      --seed "$SEED" \
      --diffusion_samples "$BOLTZ_SAMPLES_PER_SEED" \
      --max_parallel_samples "$MAX_PARALLEL_SAMPLES" \
      --override
    done
 
    echo "Completed ${JOB_TAG}"

### 5. Affinity Analysis
Predicted binding affinities were processed using probability-weighted averages:
Combines multiple affinity predictions per structure
Normalized relative to wild-type (WT)
Generates Δ affinity values

### 6. Heatmap Visualization
Affinity results were visualized using heatmaps:
X-axis: T357 mutations
Y-axis: F360 mutations
Color: change in affinity relative to WT

Script used:

    #!/usr/bin/env python3
 
    from pathlib import Path
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    from matplotlib.patches import Rectangle
 
    # =========================
    # Paths
    # =========================
    BASE_DIR = Path("/mnt/gs21/scratch/garlan70/boltz")
    FIGURES_DIR = BASE_DIR / "figures"
 
    BILIRUBIN_CSV = FIGURES_DIR / "bilirubin_affinities.csv"
    E3S_CSV = FIGURES_DIR / "e3s_affinities.csv"
    TOP_MUTANTS_CSV = FIGURES_DIR / "top_bilirubin_selective_mutants.csv"
 
    BILIRUBIN_PNG = FIGURES_DIR / "bilirubin_affinities.png"
    E3S_PNG = FIGURES_DIR / "e3s_affinities.png"
 
    # =========================
    # Settings
    # =========================
    AA_ORDER = list("ACDEFGHIKLMNPQRSTVWY")
    WT_LABEL = "T357T_F360F"
 
    PROB_COLS = ["affinity_0", "affinity_1", "affinity_2"]
    PRED_COLS = ["affinity_pred_0", "affinity_pred_1", "affinity_pred_2"]
 
 
    def load_top_mutants(top_csv: Path) -> set[str]:
    df = pd.read_csv(top_csv)
    if "mutant" not in df.columns:
        raise ValueError(f"'mutant' column not found in {top_csv}")
    return set(df["mutant"].dropna().astype(str))
 
 
    def build_scored_dataframe(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path).copy()
 
    df["aa357"] = df["mutant"].str.extract(r"T357([A-Z])")
    df["aa360"] = df["mutant"].str.extract(r"F360([A-Z])")
 
    probs = df[PROB_COLS].to_numpy(dtype=float)
    preds = df[PRED_COLS].to_numpy(dtype=float)
 
    prob_sums = probs.sum(axis=1)
    df["affinity_weighted"] = np.where(
        prob_sums > 0,
        (probs * preds).sum(axis=1) / prob_sums,
        np.nan
    )
 
    wt_row = df.loc[df["mutant"] == WT_LABEL]
    if wt_row.empty:
        raise ValueError(f"WT row '{WT_LABEL}' not found in {csv_path}")
 
    wt_value = float(wt_row.iloc[0]["affinity_weighted"])
    df["delta_from_wt"] = df["affinity_weighted"] - wt_value
 
    return df
 
 
    def make_matrix(df: pd.DataFrame) -> np.ndarray:
    mat = np.full((20, 20), np.nan)
 
    for _, row in df.iterrows():
        aa357 = row["aa357"]
        aa360 = row["aa360"]
        if aa357 in AA_ORDER and aa360 in AA_ORDER:
            y = AA_ORDER.index(aa360)
            x = AA_ORDER.index(aa357)
            mat[y, x] = row["delta_from_wt"]
 
    return mat
 
 
    def add_top_mutant_borders(ax, top_mutants: set[str]):
    for mutant in top_mutants:
        m = pd.Series([mutant]).str.extract(r"T357([A-Z])_F360([A-Z])")
        if m.isna().any(axis=None):
            continue
 
        aa357 = m.iloc[0, 0]
        aa360 = m.iloc[0, 1]
 
        if aa357 in AA_ORDER and aa360 in AA_ORDER:
            x = AA_ORDER.index(aa357)
            y = AA_ORDER.index(aa360)
 
            rect = Rectangle(
                (x - 0.5, y - 0.5),
                1,
                1,
                fill=False,
                edgecolor="black",
                linewidth=2.5
            )
            ax.add_patch(rect)
 
 
    def plot_heatmap(csv_path: Path, out_png: Path, title: str, top_mutants: set[str]):
    df = build_scored_dataframe(csv_path)
    mat = make_matrix(df)
 
    cmap = LinearSegmentedColormap.from_list(
        "custom_heat",
        ["red", "orange", "yellow", "lightgreen", "darkgreen"]
    ).copy()
    cmap.set_bad(color="lightgray")
 
    max_abs = np.nanmax(np.abs(mat))
    norm = TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs)
 
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(mat, cmap=cmap, norm=norm, origin="upper", aspect="equal")
 
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    ax.set_xticks(range(20))
    ax.set_xticklabels(AA_ORDER)
    ax.set_yticks(range(20))
    ax.set_yticklabels(AA_ORDER)
 
    ax.set_xlabel("Position 357 Mutation")
    ax.set_ylabel("Position 360 Mutation")
    ax.set_title(title, pad=20)
 
    add_top_mutant_borders(ax, top_mutants)
 
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Probability-Weighted Affinity Change from WT")
 
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close(fig)
 
    print(f"Wrote: {out_png}")
 
 
    def main():
    top_mutants = load_top_mutants(TOP_MUTANTS_CSV)
 
    plot_heatmap(
        csv_path=BILIRUBIN_CSV,
        out_png=BILIRUBIN_PNG,
        title="Weighted Bilirubin Affinity Heatmap (Change from WT)",
        top_mutants=top_mutants,
    )
 
    plot_heatmap(
        csv_path=E3S_CSV,
        out_png=E3S_PNG,
        title="Weighted E3S Affinity Heatmap (Change from WT)",
        top_mutants=top_mutants,
    )
 
 
    if __name__ == "__main__":
    main()

## Key Results
Top candidate mutations identified:
- T357M + F360C
- T357F + F360R
- T357M + F360H
- T357Y + F360D
- T357P + F360P

These mutations:
- Reduce bilirubin binding
- Maintain or minimally affect E3S binding
