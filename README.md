## Senescence Classifier ##

FORECASTS: FOREst-based ClAssification of Senescence in spatial TranScriptomics

Senescence Classifier built from scRNA-seq data of HCA2 fibroblast cell line (Tang et al., 2019), optimized using Visium 10x Spatial Transcriptomics data of human skin (Ganier et al., 2024).

To use, set up the environment in Anaconda using bioinf_env.yaml and import FORECASTS.py & the folder 'model' to the working directory.

Example usage:
```
from FORECASTS import *
clf = FORECASTS()

adata.obs['probs'] = clf.classify(adata, n_jobs = 8, logarithmized = False, verbose = False, normalized = True)
```
