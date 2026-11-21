#!/usr/bin/env python3
"""Normalises NIST CORR-DATA into one table with explicit, non-guessing parsers.

Usage (repository root):  python3 -I domains/corrosives/work/parse_corrdata.py
Reads CORR-DATA_Database.csv and columnDefinitions.txt straight from work/data/CORR-DATA_Database.zip
in memory (the archive is never unpacked into work/data/; with --cache the CSV is also copied to the
gitignored work/cache/ for inspection). Writes work/cleaned/corrdata_normalized.csv,
work/cleaned/corrdata_references.csv and work/reproductions/corrdata_parse_report.json.
Existing outputs with different content are never replaced; a time-stamped name is used instead.

Parsing rules (anything else is left unparsed with its raw text kept):
  rate      'Rate (mm/yr) or Rating' is used first (factor 1.0). If it is not numeric, a numeric
            'Rate (mils/yr) or Rating' is converted with 1 mil/yr = 0.0254 mm/yr (1 mil = 0.001 in).
            '<x max' = upper bound, 'x min' = lower bound, 'x-y' = range. Only plain numbers are a
            point value. Letter ratings A-D are kept in a separate column; their source definitions
            (A < 0.002 in/yr = 0.0508 mm/yr; B < 0.020 in/yr = 0.508 mm/yr; C 0.020-0.050 in/yr =
            0.508-1.27 mm/yr; D > 0.050 in/yr = 1.27 mm/yr) are recorded but NOT turned into numbers.
  temp      deg C column: number, range, 'x max'/'Up to x'/'To x' (upper), 'x min' (lower). Words
            (Boiling, Room, Hot, Molten ...) are kept as a qualifier with no number. If the C text does
            not parse but the F text does, C = (F - 32) * 5/9 and temp_source = 'from_F'. Values that look
            like spreadsheet date artefacts ('5-Oct') are not reinterpreted.
  conc      number = vol % (the column's stated default unit); 'wt%', 'g/L', 'mol%', 'm' units are kept
            as written; ranges and 'x max'/'Up to x' bounds parsed; words kept as a qualifier.
  duration  'x h|d|yr' and ranges, converted to hours with 1 d = 24 h, 1 yr = 8766 h (365.25 d).
"""
import argparse, hashlib, io, json, os, re, sys, time, zipfile
from collections import Counter
from datetime import datetime, timezone
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ZIP = os.path.join(HERE, 'data', 'CORR-DATA_Database.zip')
MILS_TO_MM = 0.0254
HOURS = {'h': 1.0, 'hr': 1.0, 'd': 24.0, 'yr': 8766.0}
RATING = {'A': ('Resistant', '< 0.002 in/yr (< 0.0508 mm/yr)'), 'B': ('Good', '< 0.020 in/yr (< 0.508 mm/yr)'),
          'C': ('Questionable', '0.020-0.050 in/yr (0.508-1.27 mm/yr)'), 'D': ('Poor', '> 0.050 in/yr (> 1.27 mm/yr)')}
NUM = r'(-?\d+(?:\.\d+)?|-?\.\d+)'
GROUP_FIX = {'miscellaneous': 'Miscellaneous', 'refractroy metals and alloys': 'Refractory metals and alloys',
             'nickel and alloys': 'Nickel and alloys', 'stainless steels': 'Stainless steels'}


def parse_number_field(s):
    """Returns (value, low, high, qualifier, status) for rate-like fields."""
    t = s.strip()
    if not t:
        return None, None, None, None, 'blank'
    m = re.fullmatch(NUM, t)
    if m:
        v = float(m.group(1)); return v, v, v, 'exact', 'parsed'
    m = re.fullmatch(NUM + r'\s+max', t)
    if m:
        v = float(m.group(1)); return None, None, v, 'upper_bound', 'parsed'
    m = re.fullmatch(NUM + r'\s+min', t)
    if m:
        v = float(m.group(1)); return None, v, None, 'lower_bound', 'parsed'
    m = re.fullmatch(NUM + r'\s*-\s*' + NUM, t)
    if m:
        a, b = float(m.group(1)), float(m.group(2))
        if a <= b:
            return None, a, b, 'range', 'parsed'
    return None, None, None, None, 'unparsed'


def parse_rating(s):
    t = s.strip()
    m = re.fullmatch(r'([ABCD])\s*\((resistant|good|questionable|poor)\)', t, flags=re.I)
    if m and RATING[m.group(1).upper()][0].lower() == m.group(2).lower():
        return m.group(1).upper()
    return None


def parse_temp(s):
    t = s.strip().replace('°', '').replace('Â', '')
    if not t:
        return None, None, None, None, 'blank'
    t = re.sub(r'\s*\([^)]*\)\s*$', '', t).strip()  # drop a trailing '(F value)' echo
    if re.search(r'[A-Z][a-z]{2}', t) and re.search(r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)', t):
        return None, None, None, 'date_artefact', 'unparsed'
    m = re.fullmatch(NUM, t)
    if m:
        v = float(m.group(1)); return v, v, v, 'exact', 'parsed'
    m = re.fullmatch(NUM + r'\s*(?:-|to)\s*' + NUM, t)
    if m:
        a, b = float(m.group(1)), float(m.group(2))
        if a <= b:
            return None, a, b, 'range', 'parsed'
    m = re.fullmatch(r'(?:Up to|To)\s+' + NUM, t, flags=re.I) or re.fullmatch(NUM + r'\s+max', t)
    if m:
        return None, None, float(m.group(1)), 'upper_bound', 'parsed'
    m = re.fullmatch(NUM + r'\s+min', t)
    if m:
        return None, float(m.group(1)), None, 'lower_bound', 'parsed'
    m = re.fullmatch(r'room\s*(?:to|-)\s*' + NUM, t, flags=re.I)
    if m:
        return None, None, float(m.group(1)), 'room_to_value', 'parsed'
    m = re.fullmatch(NUM + r'\s*(?:-|to)\s*boil(?:ing)?(?: point)?', t, flags=re.I)
    if m:
        return None, float(m.group(1)), None, 'value_to_boiling', 'parsed'
    low = t.lower()
    for word, q in (('boiling', 'boiling'), ('boling', 'boiling'), ('room to boiling', 'room_to_boiling'), ('room-boiling', 'room_to_boiling'),
                    ('room', 'room'), ('ambient', 'room'), ('hot', 'hot'), ('cold', 'cold'), ('molten', 'molten'), ('red heat', 'red_heat')):
        if low == word:
            return None, None, None, q, 'qualitative'
    return None, None, None, None, 'unparsed'


def parse_conc(s):
    t = s.strip()
    if not t:
        return None, None, None, None, None, 'blank'
    m = re.fullmatch(NUM + r'\s*(wt\s*%|vol\s*%|g/L|mol\s*%|m|%)?', t, flags=re.I)
    if m:
        v = float(m.group(1)); u = m.group(2).replace(' ', '') if m.group(2) else 'vol% (column default)'
        return v, v, v, u, 'exact', 'parsed'
    m = re.fullmatch(r'~\s*' + NUM + r'(?:\s*-\s*' + NUM + r')?', t)
    if m:
        a = float(m.group(1)); b = float(m.group(2)) if m.group(2) else a
        return (a if a == b else None), a, b, 'vol% (column default)', 'approx', 'parsed'
    m = re.fullmatch(NUM + r'\s*-\s*' + NUM + r'\s*(wt\s*%|vol\s*%|g/L|mol\s*%|%)?', t, flags=re.I)
    if m:
        a, b = float(m.group(1)), float(m.group(2))
        if a <= b:
            return None, a, b, (m.group(3).replace(' ', '') if m.group(3) else 'vol% (column default)'), 'range', 'parsed'
    m = re.fullmatch(r'(?:Up to|To)\s+' + NUM + r'\s*(%)?', t, flags=re.I) or re.fullmatch(NUM + r'\s+max', t)
    if m:
        return None, None, float(m.group(1)), 'vol% (column default)', 'upper_bound', 'parsed'
    m = re.fullmatch(NUM + r'\s*%\s*(.+)', t)
    if m:
        v = float(m.group(1)); return v, v, v, '%', 'exact_with_text', 'parsed'
    words = {'saturated': 'saturated', 'saturated solution': 'saturated', 'hot saturated': 'hot_saturated', 'all concentrations': 'all',
             'all': 'all', 'anhydrous': 'anhydrous', 'concentrated': 'concentrated', 'conc': 'concentrated', 'vapor': 'vapor',
             'vapors': 'vapor', 'fused': 'fused', 'pure': 'pure', 'boiling': 'boiling', 'liquid': 'liquid', 'wet': 'wet',
             'dilute': 'dilute', 'glacial': 'glacial', 'gas': 'gas'}
    if t.lower() in words:
        return None, None, None, None, words[t.lower()], 'qualitative'
    return None, None, None, None, None, 'unparsed'


def parse_duration(s):
    t = s.strip()
    if not t:
        return None, None, None, 'blank'
    m = re.fullmatch(NUM + r'\.?\s*(?:-\s*' + NUM + r')?\s*(h|hr|d|yr)', t)
    if m:
        a = float(m.group(1)); b = float(m.group(2)) if m.group(2) else a
        if a <= b:
            return a * HOURS[m.group(3)], b * HOURS[m.group(3)], m.group(3), 'parsed'
    return None, None, None, 'unparsed'


def write_once(path, text):
    h = hashlib.sha256(text.encode()).hexdigest()
    if os.path.exists(path):
        if hashlib.sha256(open(path, 'rb').read()).hexdigest() == h:
            return path, h, 'unchanged'
        stem, ext = os.path.splitext(path)
        path = f"{stem}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}{ext}"
    with open(path, 'x', newline='', encoding='utf-8') as f:
        f.write(text)
    return path, h, 'written'


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--cache', action='store_true'); args = ap.parse_args()
    t0 = time.time()
    with zipfile.ZipFile(ZIP) as z:
        raw_bytes = z.read('CORR-DATA_Database.csv')
        coldefs = z.read('columnDefinitions.txt').decode('utf-8-sig')
    if args.cache:
        os.makedirs(os.path.join(HERE, 'cache'), exist_ok=True)
        open(os.path.join(HERE, 'cache', 'CORR-DATA_Database.csv'), 'wb').write(raw_bytes)
    defs = dict(line.split('\t', 1) for line in coldefs.splitlines() if '\t' in line)
    d = pd.read_csv(io.BytesIO(raw_bytes), dtype=str, encoding='utf-8-sig', keep_default_na=False)
    expected = ['Environment', 'Material Group', 'Material Family', 'Material', 'Rate (mm/yr) or Rating', 'Rate (mils/yr) or Rating',
                'Localized Attack', 'UNS', 'Condition/Comment', 'Concentration (Vol %)', 'Temperature (deg C)', 'Temperature (deg F)',
                'Duration', 'Reference #', 'Reference']
    assert list(d.columns) == expected, list(d.columns)
    assert set(defs) >= set(expected), set(expected) - set(defs)
    rows = []
    for i, r in d.iterrows():
        mm, mil = r['Rate (mm/yr) or Rating'], r['Rate (mils/yr) or Rating']
        v, lo, hi, q, st = parse_number_field(mm); src, factor, raw_v, raw_u = 'mm_column', 1.0, mm.strip(), 'mm/yr'
        if st != 'parsed':
            v2, lo2, hi2, q2, st2 = parse_number_field(mil)
            if st2 == 'parsed':
                v, lo, hi, q, st = [None if x is None else x * MILS_TO_MM for x in (v2, lo2, hi2)] + [q2, st2]
                src, factor, raw_v, raw_u = 'mils_column', MILS_TO_MM, mil.strip(), 'mils/yr'
        rating = parse_rating(mm) or parse_rating(mil)
        if st == 'parsed':
            rate_status = 'numeric'
        elif rating:
            rate_status = 'rating'; src = raw_v = raw_u = factor = None
        elif not mm.strip() and not mil.strip():
            rate_status = 'blank'; src = raw_v = raw_u = factor = None
        else:
            rate_status = 'unparsed'; src = raw_v = raw_u = factor = None
        # cross-check of the two rate columns where both are plain numbers
        a = parse_number_field(mm); b = parse_number_field(mil)
        consistent = 'not_comparable'
        if a[3] == 'exact' and b[3] == 'exact':
            dec = lambda t: len(t.strip().split('.')[1]) if '.' in t else 0
            tol = 0.5 * 10 ** -dec(mm) + 0.5 * 10 ** -dec(mil) * MILS_TO_MM + 0.02 * a[0]
            conv = b[0] * MILS_TO_MM
            if abs(a[0] - conv) <= tol:
                consistent = 'agree_within_rounding'
            elif a[0] > 0 and conv > 0 and 0.5 <= a[0] / conv <= 2:
                consistent = 'minor_disagreement'
            else:
                consistent = 'large_disagreement'
        tv, tlo, thi, tq, tst = parse_temp(r['Temperature (deg C)']); tsrc = 'C' if tst in ('parsed', 'qualitative') else None
        if tst not in ('parsed', 'qualitative'):
            fv, flo, fhi, fq, fst = parse_temp(r['Temperature (deg F)'])
            if fst == 'parsed':
                c = lambda x: None if x is None else round((x - 32) * 5 / 9, 2)
                tv, tlo, thi, tq, tst, tsrc = c(fv), c(flo), c(fhi), fq, 'parsed', 'from_F'
            elif fst == 'qualitative':
                tq, tst, tsrc = fq, 'qualitative', 'F'
        cv, clo, chi, cu, cq, cst = parse_conc(r['Concentration (Vol %)'])
        dlo, dhi, du, dst = parse_duration(r['Duration'])
        grp = r['Material Group'].strip()
        rows.append({
            'row_id': i, 'reference_number': r['Reference #'].strip(),
            'environment': r['Environment'].strip(), 'material_class': GROUP_FIX.get(grp.lower(), grp),
            'material_group_raw': r['Material Group'], 'material_family': r['Material Family'].strip(), 'material': r['Material'].strip(),
            'uns': r['UNS'].strip(),
            'conc_raw': r['Concentration (Vol %)'], 'conc_value': cv, 'conc_low': clo, 'conc_high': chi, 'conc_unit': cu,
            'conc_qualifier': cq, 'conc_status': cst,
            'temp_C_raw': r['Temperature (deg C)'], 'temp_F_raw': r['Temperature (deg F)'], 'temp_C_value': tv, 'temp_C_low': tlo,
            'temp_C_high': thi, 'temp_qualifier': tq, 'temp_source': tsrc, 'temp_status': tst,
            'duration_raw': r['Duration'], 'duration_h_low': dlo, 'duration_h_high': dhi, 'duration_unit': du, 'duration_status': dst,
            'rate_mm_raw': mm, 'rate_mils_raw': mil, 'rate_status': rate_status, 'rate_source_column': src,
            'rate_raw_value': raw_v, 'rate_raw_unit': raw_u, 'rate_factor_to_mm_y': factor,
            'rate_mm_y': v, 'rate_mm_y_low': lo, 'rate_mm_y_high': hi, 'rate_qualifier': q if rate_status == 'numeric' else None,
            'rate_columns_consistent': consistent,
            'rating': rating, 'rating_label': RATING[rating][0] if rating else None,
            'localized_attack': r['Localized Attack'].strip(), 'condition_comment': r['Condition/Comment'].strip()})
    out = pd.DataFrame(rows)
    refs = d[['Reference #', 'Reference']].drop_duplicates().rename(columns={'Reference #': 'reference_number', 'Reference': 'citation'})
    os.makedirs(os.path.join(HERE, 'cleaned'), exist_ok=True); os.makedirs(os.path.join(HERE, 'reproductions'), exist_ok=True)
    p1, h1, s1 = write_once(os.path.join(HERE, 'cleaned', 'corrdata_normalized.csv'), out.to_csv(index=False))
    p2, h2, s2 = write_once(os.path.join(HERE, 'cleaned', 'corrdata_references.csv'), refs.to_csv(index=False))

    cnt = lambda col: {k: int(v) for k, v in out[col].fillna('none').value_counts().items()}
    num = out[out.rate_status == 'numeric']
    exact = num[num.rate_qualifier == 'exact']
    report = {
        'run_id': datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'), 'command': 'python3 -I ' + ' '.join(sys.argv),
        'data_sha256': {'CORR-DATA_Database.zip': hashlib.sha256(open(ZIP, 'rb').read()).hexdigest(),
                        'CORR-DATA_Database.csv (in zip)': hashlib.sha256(raw_bytes).hexdigest()},
        'versions': {'python': sys.version.split()[0], 'pandas': pd.__version__},
        'outputs': {os.path.relpath(p1, HERE): h1, os.path.relpath(p2, HERE): h2}, 'output_status': [s1, s2],
        'rows': len(out), 'distinct': {'references': int(out.reference_number.nunique()), 'materials': int(out.material.nunique()),
                                       'environments': int(out.environment.nunique()), 'material_classes': int(out.material_class.nunique()),
                                       'material_families': int(out.material_family.nunique())},
        'rate_status': cnt('rate_status'),
        'numeric_rate_by_source_unit': {k: int(v) for k, v in num.rate_raw_unit.value_counts().items()},
        'numeric_rate_by_qualifier': cnt_q if (cnt_q := {k: int(v) for k, v in num.rate_qualifier.value_counts().items()}) else {},
        'numeric_exact_rate': {'rows': len(exact), 'zero_rows': int((exact.rate_mm_y == 0).sum()),
                               'references': int(exact.reference_number.nunique()),
                               'from_mm_column': int((exact.rate_source_column == 'mm_column').sum()),
                               'from_mils_column': int((exact.rate_source_column == 'mils_column').sum())},
        'rate_columns_cross_check': cnt('rate_columns_consistent'),
        'rate_cross_check_rule': 'both columns plain numbers: agree if |mm - mils*0.0254| <= half a unit in the last written digit of each plus 2%; minor if the ratio is within 0.5-2; else large. The mm column value is kept in every case.',
        'rating_counts': cnt('rating'),
        'unparsed_rate_examples': out.loc[out.rate_status == 'unparsed', 'rate_mm_raw'].value_counts().head(15).to_dict(),
        'temperature': {'status': cnt('temp_status'), 'qualifier': cnt('temp_qualifier'), 'source': cnt('temp_source'),
                        'unparsed_examples': out.loc[out.temp_status == 'unparsed', 'temp_C_raw'].value_counts().head(15).to_dict()},
        'concentration': {'status': cnt('conc_status'), 'unit': cnt('conc_unit'), 'qualifier': cnt('conc_qualifier'),
                          'unparsed_examples': out.loc[out.conc_status == 'unparsed', 'conc_raw'].value_counts().head(15).to_dict()},
        'duration': {'status': cnt('duration_status'), 'unit': cnt('duration_unit'),
                     'unparsed_examples': out.loc[out.duration_status == 'unparsed', 'duration_raw'].value_counts().head(10).to_dict()},
        'conversion_factors': {'mm/yr -> mm/yr': 1.0, 'mils/yr -> mm/yr': MILS_TO_MM, 'deg F -> deg C': '(F - 32) * 5/9',
                               'duration h': 1, 'duration d (h)': 24, 'duration yr (h)': 8766},
        'rating_definitions_from_columnDefinitions': {k: f'{v[0]}: {v[1]}' for k, v in RATING.items()},
        'column_definitions': defs,
        'elapsed_seconds': round(time.time() - t0, 2)}
    rp, _, _ = write_once(os.path.join(HERE, 'reproductions', 'corrdata_parse_report.json'), json.dumps(report, indent=1, default=str))
    print(json.dumps({k: report[k] for k in report if k not in ('column_definitions',)}, indent=1, default=str)[:6000])
    print('wrote', p1, p2, rp)


if __name__ == '__main__':
    main()
