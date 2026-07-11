"""Stage-1 ablation grid.

One-factor-at-a-time (OFAT) around a reference config, which is the standard way
to attribute a performance change to a single design choice. Groups:

  * ``omics``   -- omics-combination ablation (incl. the 2-modality baselines,
                   since single-omics collapses the cell branch).
  * ``encoder`` -- attention vs. plain MLP cell encoder.
  * ``drug``    -- drug representation. The default grid compares only the
                   **frozen-representation** family (morgan baseline + the 4
                   pretrained encoders): every one is "fixed representation ->
                   trained projection head", an apples-to-apples comparison.
                   The end-to-end GNNs (gin/gcn) are intentionally EXCLUDED --
                   they train the drug encoder jointly, so "how much of the gap
                   is representation vs. extra trainable capacity" is ill-defined
                   as a fair ablation. (They remain implemented and can still be
                   run manually via an explicit config.)
  * ``split``   -- mixed / unseen-cell / unseen-drug evaluation regimes.

``build_grid(groups=...)`` lets the runner pick which groups to execute so the
"skeleton today, run sequentially later" workflow can be scoped per session.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Dict, List

from .config import ExperimentConfig, DRUG_ENCODERS, DRUG_ENCODER_FAMILY

# Fair drug-representation ablation = frozen-representation encoders only
# (baseline morgan + pretrained). GNNs are excluded (end-to-end, not comparable).
ABLATION_DRUG_ENCODERS = [d for d in DRUG_ENCODERS
                          if DRUG_ENCODER_FAMILY[d] != "scratch_graph"]

# Feature ablation = RNA-anchored build-up. RNA (the transcriptomic backbone) is
# always present so the cell branch never collapses to a single modality; we add
# one, then two, then all other modalities to read off each one's marginal value.
FEATURE_OMICS_SETS = [
    ["RNA", "SNP"],                  # RNA + one
    ["RNA", "MET"],
    ["RNA", "CNV"],
    ["RNA", "SNP", "MET"],           # RNA + two
    ["RNA", "SNP", "CNV"],
    ["RNA", "MET", "CNV"],
    ["RNA", "SNP", "MET", "CNV"],    # all (== baseline reference)
]


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
        for omics in FEATURE_OMICS_SETS:
            c = replace(ref, name="omics", omics=omics)
            grid[c.tag()] = c

    if "encoder" in groups:
        for enc in ("attention", "mlp"):
            c = replace(ref, name="encoder", cell_encoder=enc)
            grid[c.tag()] = c

    if "drug" in groups:
        for drug in ABLATION_DRUG_ENCODERS:   # morgan + 4 pretrained (no GNNs)
            c = replace(ref, name="drug", drug_encoder=drug)
            grid[c.tag()] = c

    if "split" in groups:
        for split in ("mixed", "unseen_cell", "unseen_drug"):
            c = replace(ref, name="split", split_mode=split)
            grid[c.tag()] = c

    return list(grid.values())
