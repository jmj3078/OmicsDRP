# Vendored competitor repositories

These are **unmodified** upstream clones, kept here (not in git — see `.gitignore`)
so the benchmark is reproducible across sessions. Our adapters in
`omicsdrp/benchmark/adapters/<model>/` import from these but never edit them; any
required change is isolated in the adapter (a re-declared nn module or a helper).

To recreate this directory from scratch:

```bash
cd omicsdrp/benchmark/vendor
git clone https://github.com/hauldhut/GraphDRP  && git -C GraphDRP checkout ad250651f3db95d9cd7d885740e8636d609b960f
git clone https://github.com/jianglikun/DeepTTC && git -C DeepTTC  checkout a657d5698f84c693c17109351abbc5bfed55a55c
git clone https://github.com/violet-sto/TGSA    && git -C TGSA     checkout cdd9903b889112b04325bec9f61935d05d9e9179
```

| Model | Upstream | Pinned commit | License | Adapter |
|---|---|---|---|---|
| GraphDRP | https://github.com/hauldhut/GraphDRP | `ad250651f3db95d9cd7d885740e8636d609b960f` | Apache-2.0 | `adapters/graphdrp` |
| DeepTTC (DeepTTA) | https://github.com/jianglikun/DeepTTC | `a657d5698f84c693c17109351abbc5bfed55a55c` | none stated | `adapters/deeptta` |
| TGSA / TGDRP | https://github.com/violet-sto/TGSA | `cdd9903b889112b04325bec9f61935d05d9e9179` | MIT | `adapters/tgsa` |

## Excluded (audited, not used)

- **DeepCDR** (kimmo1019/DeepCDR) — Keras/TF1.13, does not run on RTX 4090 (CUDA10);
  the only faithful path was a PyTorch reimplementation. Replaced by **TGSA**, which
  is native PyTorch and fills the same multi-omics comparison axis. Not vendored.
- **DRPreter** (babaling/DRPreter) — KEGG-pathway cell branch collapses on our
  909-gene panel (only 374/909 genes intersect; 2 pathways drop to a single gene).
  Not vendored.
