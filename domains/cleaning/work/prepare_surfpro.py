#!/usr/bin/env python3
"""Builds the cleaned SURFPRO table for two tasks: pCMC and AW_ST_CMC (air-water surface tension at the CMC).

Usage (repository root):  python3 -I domains/cleaning/work/prepare_surfpro.py
Reads work/data/S129_SURFPRO.csv (unchanged) and writes work/cleaned/surfpro_clean.csv plus
work/cleaned/surfpro_clean_info.json. An existing output with different content is never replaced;
a time-stamped name is used instead.

Reference fields. Ref_<prop> is the reference SURFPRO took the value from (often a compilation or
review; sometimes a Python-style list of several DOIs). SubRef_<prop> is the original measurement
source cited by that compilation, when given. Two group definitions are written per task:
  * group_meas_<task>: the measurement reference = SubRef if present, else the Ref token(s);
    rows sharing any token are joined by connected components (primary, paper-held-out).
  * group_union_<task>: all Ref and SubRef tokens of the row, connected components (stricter:
    whole compilations held out).
Rows with no reference for the property (50 surface-tension rows) fall back to Ref_CMC.
"""
import ast, hashlib, io, json, os, re
from collections import Counter
from datetime import datetime, timezone
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, 'data', 'S129_SURFPRO.csv')
OUT = os.path.join(HERE, 'cleaned')
TASKS = {'pCMC': ('Ref_CMC', 'SubRef_CMC'), 'AW_ST_CMC': ('Ref_AW_ST_CMC', 'SubRef_AW_ST_CMC')}


def tokens(v):
    if not isinstance(v, str) or not v.strip():
        return []
    v = v.strip()
    if v.startswith('['):
        try:
            items = ast.literal_eval(v)  # literal parsing only; no code execution
        except (ValueError, SyntaxError):
            items = re.findall(r"'([^']+)'", v)
    else:
        items = [v]
    return sorted({str(t).strip().lower() for t in items if str(t).strip()})


def components(keys):
    parent = list(range(len(keys)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    first = {}
    for i, ks in enumerate(keys):
        for k in ks:
            if k in first:
                a, b = find(i), find(first[k])
                if a != b:
                    parent[a] = b
            else:
                first[k] = i
    roots = [find(i) for i in range(len(keys))]
    lab = {r: j for j, r in enumerate(dict.fromkeys(roots))}
    return [lab[r] for r in roots]


def write_once(path, text):
    new = hashlib.sha256(text.encode()).hexdigest()
    if os.path.exists(path):
        old = hashlib.sha256(open(path, 'rb').read()).hexdigest()
        if old == new:
            return path, new, 'unchanged'
        stem, ext = os.path.splitext(path)
        path = f"{stem}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}{ext}"
    with open(path, 'x', newline='') as f:
        f.write(text)
    return path, new, 'written'


def main():
    raw = pd.read_csv(SRC)
    df = raw[['Index', 'SMILES', 'InChIKey', 'Surfactant_Type', 'Temp_Celsius', 'pCMC', 'AW_ST_CMC']].copy()
    info = {'source_sha256': hashlib.sha256(open(SRC, 'rb').read()).hexdigest(), 'rows_raw': len(raw),
            'duplicate_inchikeys': int(raw.InChIKey.duplicated().sum()),
            'surfactant_type_counts': raw.Surfactant_Type.value_counts().to_dict(), 'tasks': {}}
    for task, (ref, sub) in TASKS.items():
        has = raw[task].notna()
        meas_keys, union_keys, fallback = [], [], 0
        for _, r in raw.iterrows():
            rt, st = tokens(r[ref]), tokens(r[sub])
            if not (rt or st):
                rt = tokens(r['Ref_CMC'])
                fallback += int(pd.notna(r[task]))
            meas = st or rt
            union = sorted(set(rt) | set(st))
            meas_keys.append(meas or [f"row:{r['Index']}"])
            union_keys.append(union or [f"row:{r['Index']}"])
        df[f'meas_ref_{task}'] = ['|'.join(k) for k in meas_keys]
        df[f'all_refs_{task}'] = ['|'.join(k) for k in union_keys]
        for name, keys in (('meas', meas_keys), ('union', union_keys)):
            g = [None] * len(raw)
            idx = [i for i in range(len(raw)) if has.iloc[i]]
            comp = components([keys[i] for i in idx])
            for i, c in zip(idx, comp):
                g[i] = c
            df[f'group_{name}_{task}'] = pd.array(g, dtype='Int64')
            sizes = Counter(comp)
            info['tasks'].setdefault(task, {})[f'groups_{name}'] = {
                'n_groups': len(sizes), 'largest': max(sizes.values()), 'singletons': sum(v == 1 for v in sizes.values())}
        info['tasks'][task].update({'rows_with_label': int(has.sum()),
                                    'rows_using_Ref_CMC_fallback_for_group': fallback,
                                    'label_min_max': [float(raw[task].min()), float(raw[task].max())],
                                    'rows_with_several_ref_tokens': int(sum(len(tokens(v)) > 1 for v in raw.loc[has, ref]))})
    df = df.rename(columns={'Temp_Celsius': 'temp_C_cmc'})
    os.makedirs(OUT, exist_ok=True)
    buf = io.StringIO(); df.to_csv(buf, index=False)
    path, h, status = write_once(os.path.join(OUT, 'surfpro_clean.csv'), buf.getvalue())
    info.update({'output': os.path.relpath(path, HERE), 'output_sha256': h, 'status': status,
                 'columns': list(df.columns),
                 'note': 'Labels are surfactant physical properties (pCMC = -log10 CMC in mol/L; AW_ST_CMC in mN/m), not cleaning efficacy.'})
    write_once(os.path.join(OUT, 'surfpro_clean_info.json'), json.dumps(info, indent=1))
    print(json.dumps(info['tasks'], indent=1)); print(status, path)


if __name__ == '__main__':
    main()
