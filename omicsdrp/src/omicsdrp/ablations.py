"""Stage-1 ablation grid.

One-factor-at-a-time (OFAT) around a reference config, which is the standard way
to attribute a performance change to a single design choice. Groups:

  * ``omics``   -- omics-combination ablation (incl. the 2-modality baselines,
                   since single-omics collapses the cell branch).
  * ``encoder`` -- attention vs. plain MLP cell encoder.
  * ``drug``    -- drug representation (morgan implemented; 2 language + 2 graph
                   pretrained + classic GNN stubbed).
  * ``split``   -- mixed / unseen-cell / unseen-drug evaluation regimes.

``build_grid(groups=...)`` lets the runner pick which groups to execute so the
"skeleton today, run sequentially later" workflow can be scoped per session.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Dict, List

from .config import ExperimentConfig, DRUG_ENCODERS


def reference_config(**overrides) -> ExperimentConfig:
    base = ExperimentConfig(
        name="ref",
        omics=["SNP", "MET", "CNV", "RNA"],
        cell_encoder="attention",
        drug_encoder="morgan",
        split_mode="mixed",
    )
    return replace(base, **overrides)


def build_grid(groups: List[str] = None, **base_overrides) -> List[ExperimentConfig]:
    groups = groups or ["omics", "encoder", "drug", "split"]
    ref = reference_config(**base_overrides)
    grid: Dict[str, ExperimentConfig] = {}

    # reference always included
    grid[ref.tag()] = replace(ref, name="ref")

    if "omics" in groups:
        for omics in (["RNA", "MET"], ["RNA", "CNV"], ["RNA", "CNV", "MET"],
                      ["SNP", "MET", "CNV", "RNA"]):
            c = replace(ref, name="omics", omics=omics)
            grid[c.tag()] = c

    if "encoder" in groups:
        for enc in ("attention", "mlp"):
            c = replace(ref, name="encoder", cell_encoder=enc)
            grid[c.tag()] = c

    if "drug" in groups:
        for drug in DRUG_ENCODERS:            # morgan + 5 stubs
            c = replace(ref, name="drug", drug_encoder=drug)
            grid[c.tag()] = c

    if "split" in groups:
        for split in ("mixed", "unseen_cell", "unseen_drug"):
            c = replace(ref, name="split", split_mode=split)
            grid[c.tag()] = c

    return list(grid.values())
