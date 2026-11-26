#!/usr/bin/env python3
"""Writes work/cleaned/surfpro_features.npz: the SURFPRO feature table, so training machines do not need RDKit.

Usage (repository root; needs rdkit and pandas):  python3 domains/cleaning/work/export_features.py

Same features as train_surfpro.py: every RDKit 2D descriptor (Descriptors.CalcMolDescriptors, infinities set to missing,
values clipped to +/-1e6), a 512-bit Morgan count fingerprint (radius 2), the surfactant type one-hot and the CMC
temperature in C. One row per row of surfpro_clean.csv, in the same order, keyed by the Index column. Missing values
stay NaN; the loader (gpu/data.py) fills them. Columns that are constant over all rows are dropped (they carry nothing).
No labels are read or written here. Refuses to overwrite.
"""
import hashlib, json, os, sys
import numpy as np, pandas as pd, rdkit
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_surfpro import CLEAN, featurize  # noqa: E402

OUT = os.path.join(os.path.dirname(CLEAN), 'surfpro_features.npz')


def main():
    if os.path.exists(OUT):
        sys.exit(f'{OUT} exists; refusing to overwrite')
    df = pd.read_csv(CLEAN)
    X, failed, n_desc = featurize(df)
    assert not failed, failed
    keep = [c for c in X.columns if X[c].nunique(dropna=True) > 1]
    X = X[keep]
    np.savez_compressed(OUT, index=df.Index.to_numpy(np.int64), features=np.array(keep), X=X.to_numpy(np.float32))
    info = {'rows': int(len(X)), 'columns': len(keep), 'rdkit_descriptors_before_dropping_constants': int(n_desc), 'rdkit': rdkit.__version__,
            'numpy': np.__version__, 'pandas': pd.__version__, 'clean_csv_sha256': hashlib.sha256(open(CLEAN, 'rb').read()).hexdigest(),
            'npz_sha256': hashlib.sha256(open(OUT, 'rb').read()).hexdigest(), 'missing_values': int(np.isnan(X.to_numpy(float)).sum())}
    json.dump(info, open(OUT.replace('.npz', '_info.json'), 'w'), indent=1)
    print(json.dumps(info, indent=1))


if __name__ == '__main__':
    main()
