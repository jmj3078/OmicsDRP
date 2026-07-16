# Vendored competitor repositories

These are **unmodified** upstream clones, kept here (not in git — see `.gitignore`)
so the benchmark is reproducible across sessions. Our adapters in
`omicsdrp/benchmark/adapters/<model>/` import from these but never edit them; any
required source change is isolated in the adapter (monkey-patch or a patched copy).

DRPreter was audited but **excluded** from the benchmark (KEGG-pathway cell branch
collapses on our 909-gene panel — only 374/909 genes intersect), so it is not vendored.

To recreate this directory from scratch:

```bash
cd omicsdrp/benchmark/vendor
git clone https://github.com/hauldhut/GraphDRP  && git -C GraphDRP checkout ad250651f3db95d9cd7d885740e8636d609b960f
git clone https://github.com/kimmo1019/DeepCDR  && git -C DeepCDR  checkout 4dc5a901d580511335b9a54ffce9fb188f9f068d
git clone https://github.com/jianglikun/DeepTTC && git -C DeepTTC  checkout a657d5698f84c693c17109351abbc5bfed55a55c
# DeepCDR: unzip data/GDSC/drug_graph_feat.zip before use
```

| Model | Upstream | Pinned commit | License |
|---|---|---|---|
| GraphDRP | https://github.com/hauldhut/GraphDRP | `ad250651f3db95d9cd7d885740e8636d609b960f` | Apache-2.0 |
| DeepCDR | https://github.com/kimmo1019/DeepCDR | `4dc5a901d580511335b9a54ffce9fb188f9f068d` | MIT |
| DeepTTC | https://github.com/jianglikun/DeepTTC | `a657d5698f84c693c17109351abbc5bfed55a55c` | none stated |
