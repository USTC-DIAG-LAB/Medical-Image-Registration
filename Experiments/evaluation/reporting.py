"""Consistent primary fields, missing values and crash-safe progress records."""
import csv
import json
import math
from pathlib import Path


def primary_metrics(row, group):
    prefix = 'organ_' if group == 'abdomen_organs' else ''
    for metric in ('asd', 'assd', 'hd95'):
        value = row.get(prefix + metric + '_mm')
        row['primary_' + metric + '_mm'] = float(value) if value is not None and math.isfinite(float(value)) else None
    row['primary_evaluation_region'] = group
    valid = row['primary_hd95_mm'] is not None
    if not valid and row.get('status') == 'success':
        row['status'] = 'missing_primary_metric'
    return valid and row.get('status') == 'success'


def clean_json(value):
    if isinstance(value, dict): return {k:clean_json(v) for k,v in value.items()}
    if isinstance(value, (list, tuple)): return [clean_json(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value): return None
    return value


def save_rows(folder, rows):
    folder = Path(folder)
    cleaned = clean_json(rows)
    tmp = folder/'metrics.json.tmp'
    tmp.write_text(json.dumps(cleaned, indent=2, allow_nan=False))
    tmp.replace(folder/'metrics.json')
    fields = sorted(set().union(*(r.keys() for r in cleaned)) - {'TM_physical'})
    tmp = folder/'metrics.csv.tmp'
    with tmp.open('w', newline='') as f:
        writer = csv.DictWriter(f, fields, extrasaction='ignore')
        writer.writeheader(); writer.writerows(cleaned)
    tmp.replace(folder/'metrics.csv')
