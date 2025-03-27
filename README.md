## Senescence Classifier ##

FORECASTS: FOREst-based ClAssification of Senescence in spatial TranScriptomics.

Senescence Classifier built from scRNA-seq data of HCA2 fibroblast cell line (Tang et al., 2019), optimized using Visium 10x Spatial Transcriptomics data of human skin (Ganier et al., 2024). Implemented in Python using scikit-learn.

To get started, clone the repository:
```
git clone https://github.com/lach-lan/FORECASTS.git
cd my-repository
```

Create the Anaconda environment:
```
conda env create -f bioinf_env.yaml
```

Activate the environment:
```
conda activate bioinf_env
```

Example usage:
```
from FORECASTS import *
clf = FORECASTS()

adata.obs['probs'] = clf.classify(
    adata,
    n_jobs = 8,
    logarithmized = False,
    verbose = False,
    normalized = True
)
```
