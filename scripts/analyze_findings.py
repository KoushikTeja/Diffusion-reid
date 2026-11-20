#!/usr/bin/env python3
import json

with open('scripts/scan_results.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

with open('scripts/full_analysis.txt', 'w', encoding='utf-8') as out:
    by_file = {}
    for item in data:
        by_file.setdefault(item['file'], []).append(item)

    for fname, items in by_file.items():
        out.write(f"\n{'='*70}\n")
        out.write(f"FILE: {fname} ({len(items)} hits)\n")
        out.write(f"{'='*70}\n")
        for it in items:
            rule_str = ", ".join(r[0] for r in it['matched_rules'])
            tok_str = ", ".join(f"'{r[1]}'" for r in it['matched_rules'])
            out.write(f"L{it['line_no']} [{tok_str}] ({rule_str}):\n  {it['line']}\n\n")

print("Wrote scripts/full_analysis.txt successfully.")
