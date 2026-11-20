#!/usr/bin/env python3
import os
import re
import json

files = [
    'FrontMatter/titlePage.tex',
    'FrontMatter/certificate.tex',
    'FrontMatter/declaration.tex',
    'FrontMatter/acknowledgements.tex',
    'FrontMatter/abstract.tex',
    'Chapters/chapter1.tex',
    'Chapters/chapter2.tex',
    'Chapters/chapter3.tex',
    'Chapters/chapter4.tex',
    'Chapters/chapter5.tex',
    'main.tex',
    'references.bib'
]

scan_patterns = {
    'PROHIBITED_BUGGED': r'\bbugged\b',
    'PROHIBITED_BROKEN': r'\bbroken\b',
    'PROHIBITED_PATHOLOGICAL': r'\bpathological\b',
    'PROHIBITED_MATHEMATICALLY_INOPTIMAL': r'mathematically\s+inoptimal',
    'PROHIBITED_INOPTIMAL': r'\binoptimal\b',
    'EXCLUDED_75_5M': r'75\.5',
    'EXCLUDED_99_3PCT': r'99\.3',
    'EXCLUDED_100X': r'(?:100\s*[xX×]|100\s*\\times)',
    'EXCLUDED_SAVING_10_HOURS': r'saving\s*>\s*10\s*hours',
    'EXCLUDED_GT_10_HOURS': r'>\s*10\s*hours',
    'SENSITIVE_BUG_BUGS_BUGGY': r'\b(?:bugs?|buggy)\b',
    'SENSITIVE_PATHOLOGY_PATHOLOGIES': r'\bpatholog(?:y|ies)\b',
    'SENSITIVE_FLAW_FLAWS_FLAWED': r'\bflaws?\b|\bflawed\b',
    'SENSITIVE_DEFECT_DEFECTS': r'\bdefects?\b|\bdefective\b',
    'SENSITIVE_CRIPPLED': r'\bcrippl(?:ed?|ing)\b',
    'SENSITIVE_EXCUSES': r'\bexcuses?\b',
    'SENSITIVE_DEBUNK': r'\bdebunk(?:ed|ing)?\b',
    'SENSITIVE_CATASTROPHIC': r'\bcatastrophic\b',
    'SENSITIVE_BLUNDER_FATAL_ERRONEOUS': r'\b(?:blunder|fatal|erroneous)\b',
    'SENSITIVE_FORENSIC': r'\bforensic\b',
    'SENSITIVE_COLLAPSE': r'\bcollapse[sd]?\b|\bcollapsing\b',
    'SENSITIVE_FAILURE_MODE': r'\bfailure\s+modes?\b',
    'SENSITIVE_MASSIVE_DEFICIT': r'\bmassive\s+deficit\b',
    'SENSITIVE_SEVERE': r'\bseverely?\b',
    'SENSITIVE_SHATTERING': r'\b(?:shattering|fractured)\b',
    'SENSITIVE_BYPASS': r'\bbypass(?:ed|ing)?\b',
}

results = []

for rel in files:
    full_path = os.path.join('report_optimized', rel)
    if not os.path.isfile(full_path):
        continue
    with open(full_path, 'r', encoding='utf-8') as f:
        for lno, line in enumerate(f, 1):
            clean = line.strip()
            if clean.startswith('%'):
                continue
            matched_rules = []
            for rule_name, pat in scan_patterns.items():
                m = re.search(pat, clean, re.IGNORECASE)
                if m:
                    matched_rules.append((rule_name, m.group(0)))
            if matched_rules:
                results.append({
                    'file': rel,
                    'line_no': lno,
                    'matched_rules': matched_rules,
                    'line': clean
                })

with open('scripts/scan_results.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2)

print(f"Total matching lines: {len(results)}")
rule_counts = {}
for r in results:
    for rule, tok in r['matched_rules']:
        rule_counts[rule] = rule_counts.get(rule, 0) + 1

for rule, count in sorted(rule_counts.items(), key=lambda x: -x[1]):
    print(f"  {rule}: {count}")
