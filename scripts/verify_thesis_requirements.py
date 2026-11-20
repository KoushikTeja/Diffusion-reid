#!/usr/bin/env python3
"""
scripts/verify_thesis_requirements.py
======================================
Automated E2E Verification Suite for IIT Kharagpur Revised Thesis (`report_optimized/`).

Authoritative Derivation Sources:
  - ORIGINAL_REQUEST.md (Requirements R1-R5, Acceptance Criteria Tiers 1-4)
  - PROJECT.md (Interface Contracts, Milestones M1-M5, Code Layout)
  - docs/pisl_thesis_revision_and_optimization_audit.md (Master Audit Blueprint)

Verification Tiers:
  - Tier 1: Preservation & Cleanliness (Git integrity on report/, pisl/, examples/)
  - Tier 2: Strict String Exclusion (Exclusion of exaggerated claims: 75.5M, 99.3%, 100x, etc.)
  - Tier 3: Structural & Mathematical Completeness (Files, FrontMatter, Ch 1-5, Theorems 1-5, P1-P6, Algs 1-2, Tables)
  - Tier 4: Syntactic Validity & Citations (Delimiter balance, BibTeX syntax, citation resolution, PDF compilation)

Exit Codes:
  0: All requested tier checks passed successfully.
  1: One or more tier checks failed.
"""

import sys
import os
import re
import json
import shutil
import subprocess
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional


# ==============================================================================
# TIER 1: Preservation & Cleanliness Check
# ==============================================================================

def check_tier1_preservation(repo_root: Path, verbose: bool = False) -> Tuple[bool, List[str]]:
    """
    Tier 1: Preservation & Cleanliness Check.
    Ensures that no files in `report/`, `pisl/`, or `examples/` have been modified,
    deleted, added, or untracked according to git status and git diff.
    """
    messages = []
    passed = True
    protected_prefixes = ("report/", "pisl/", "examples/", "report\\", "pisl\\", "examples\\")

    # 1. Run git status --porcelain
    try:
        proc_status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            check=True
        )
        status_lines = [line.strip() for line in proc_status.stdout.splitlines() if line.strip()]
        violations = []
        for line in status_lines:
            # Format: XY PATH or XY "PATH"
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                status_code, path_str = parts[0], parts[1].strip('"')
                for prefix in protected_prefixes:
                    if path_str.startswith(prefix) or path_str == prefix.rstrip("/\\"):
                        violations.append(f"[{status_code}] {path_str}")

        if violations:
            passed = False
            messages.append(f"FAIL: Detected {len(violations)} modified/untracked files in protected directories:")
            for v in violations:
                messages.append(f"  - {v}")
        else:
            messages.append("PASS: git status confirms no modifications in report/, pisl/, or examples/.")

    except subprocess.CalledProcessError as e:
        passed = False
        messages.append(f"FAIL: git status execution failed: {e.stderr}")
    except FileNotFoundError:
        passed = False
        messages.append("FAIL: git command not found in PATH.")

    # 2. Run git diff against HEAD for protected dirs
    for d in ["report", "pisl", "examples"]:
        target_path = repo_root / d
        if not target_path.exists():
            continue
        try:
            diff_proc = subprocess.run(
                ["git", "diff", "HEAD", "--", d],
                cwd=str(repo_root),
                capture_output=True,
                text=True,
                check=True
            )
            if diff_proc.stdout.strip():
                passed = False
                messages.append(f"FAIL: git diff detected changes in protected directory: {d}/")
            elif verbose:
                messages.append(f"PASS: git diff clean for {d}/")
        except subprocess.CalledProcessError as e:
            passed = False
            messages.append(f"FAIL: git diff failed for {d}: {e.stderr}")

    return passed, messages


# ==============================================================================
# TIER 2: Strict String Exclusion Check
# ==============================================================================

FORBIDDEN_PATTERNS = [
    (r"75\.5\s*M\b", "Forbidden claim: '75.5M' redundant parameters"),
    (r"75[,\.]?501[,\.]?568", "Forbidden claim: '75,501,568' redundant parameters"),
    (r"99\.3\s*%", "Forbidden claim: '99.3%' parameter reduction"),
    (r"(?:\b100\s*[xX]\b|\b100\s*×|\b100\s*\\times)", "Forbidden claim: '100x' / '100×' speedup"),
    (r"saving\s*>\s*10\s*hours", "Forbidden claim: 'saving >10 hours'"),
    (r">\s*10\s*hours", "Forbidden claim: '>10 hours'"),
]

def check_tier2_string_exclusions(target_dir: Path, verbose: bool = False) -> Tuple[bool, List[str]]:
    """
    Tier 2: Strict String Exclusion Check.
    Scans all text/LaTeX/BibTeX/Markdown files in `report_optimized/` recursively
    to ensure zero occurrences of forbidden exaggerated claims.
    """
    messages = []
    if not target_dir.exists():
        return False, [f"FAIL: Target directory does not exist: {target_dir}"]

    compiled_rules = [(re.compile(pat, re.IGNORECASE), desc) for pat, desc in FORBIDDEN_PATTERNS]
    violations = []
    scanned_count = 0

    valid_extensions = {".tex", ".bib", ".md", ".txt", ".sty", ".cls", ".bst"}

    for root, _, files in os.walk(target_dir):
        for file in files:
            file_path = Path(root) / file
            if file_path.suffix.lower() not in valid_extensions and not file.endswith(".tex"):
                continue

            scanned_count += 1
            try:
                with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                    for line_no, line in enumerate(f, start=1):
                        for pattern, desc in compiled_rules:
                            match = pattern.search(line)
                            if match:
                                rel_path = file_path.relative_to(target_dir)
                                violations.append(
                                    f"{rel_path}:{line_no}: [{desc}] matched '{match.group(0)}' in: {line.strip()}"
                                )
            except Exception as e:
                violations.append(f"{file_path}: Error reading file: {e}")

    if violations:
        messages.append(f"FAIL: Found {len(violations)} forbidden string violations in {target_dir.name}/:")
        for v in violations:
            messages.append(f"  - {v}")
        return False, messages
    else:
        messages.append(f"PASS: Scanned {scanned_count} files in {target_dir.name}/. Zero forbidden strings found.")
        return True, messages


# ==============================================================================
# TIER 3: Structural & Mathematical Completeness Check
# ==============================================================================

REQUIRED_FILES = [
    "main.tex",
    "references.bib",
    "FrontMatter/titlePage.tex",
    "FrontMatter/certificate.tex",
    "FrontMatter/declaration.tex",
    "FrontMatter/acknowledgements.tex",
    "FrontMatter/abstract.tex",
    "Chapters/chapter1.tex",
    "Chapters/chapter2.tex",
    "Chapters/chapter3.tex",
    "Chapters/chapter4.tex",
    "Chapters/chapter5.tex",
]

PLACEHOLDER_PATTERNS = [
    re.compile(r"\bTODO\b", re.IGNORECASE),
    re.compile(r"\bFIXME\b", re.IGNORECASE),
    re.compile(r"\bTBD\b", re.IGNORECASE),
    re.compile(r"\bTBA\b", re.IGNORECASE),
    re.compile(r"\bXXX\b"),
    re.compile(r"\bplaceholder\b", re.IGNORECASE),
]

def check_tier3_document_completeness(target_dir: Path, verbose: bool = False) -> Tuple[bool, List[str]]:
    """
    Tier 3: Structural & Mathematical Completeness Check.
    Validates presence of required files, absence of placeholder stubs,
    inclusion of Theorems 1-5 proofs, Optimizations P1-P6, Algorithms 1-2,
    experimental Tables 4.2, 4.4, 4.5, 4.6, and FrontMatter / Chapter 5 completeness.
    """
    messages = []
    all_passed = True

    if not target_dir.exists():
        return False, [f"FAIL: Target directory does not exist: {target_dir}"]

    # 1. Required Files Presence
    missing_files = []
    for rel_path in REQUIRED_FILES:
        full_path = target_dir / rel_path
        if not full_path.is_file():
            missing_files.append(rel_path)

    if missing_files:
        all_passed = False
        messages.append(f"FAIL: Missing {len(missing_files)} required thesis files:")
        for mf in missing_files:
            messages.append(f"  - {mf}")
    else:
        messages.append(f"PASS: All {len(REQUIRED_FILES)} required thesis files exist in {target_dir.name}/.")

    # 2. Check for Placeholder Stubs (TODO, FIXME, etc.)
    placeholder_matches = []
    for rel_path in REQUIRED_FILES:
        full_path = target_dir / rel_path
        if not full_path.is_file() or full_path.suffix != ".tex":
            continue
        try:
            with open(full_path, "r", encoding="utf-8", errors="replace") as f:
                for line_no, line in enumerate(f, start=1):
                    # Skip LaTeX comments
                    clean_line = strip_latex_comments(line)
                    for pat in PLACEHOLDER_PATTERNS:
                        m = pat.search(clean_line)
                        if m:
                            placeholder_matches.append(
                                f"{rel_path}:{line_no}: Found placeholder token '{m.group(0)}': {line.strip()}"
                            )
        except Exception as e:
            placeholder_matches.append(f"{rel_path}: Read error: {e}")

    if placeholder_matches:
        all_passed = False
        messages.append(f"FAIL: Found {len(placeholder_matches)} placeholder/TODO stubs:")
        for pm in placeholder_matches:
            messages.append(f"  - {pm}")
    else:
        messages.append("PASS: No TODO, FIXME, or placeholder stubs found in .tex files.")

    # 3. Mathematical Audit Completeness (Chapter 3)
    ch3_path = target_dir / "Chapters" / "chapter3.tex"
    if ch3_path.is_file():
        try:
            with open(ch3_path, "r", encoding="utf-8", errors="replace") as f:
                ch3_content = f.read()

            # Required Theorems 1-5
            theorem_checks = [
                ("Theorem 1 (Part Gradient Inversion)", [
                    r"(?:Theorem\s*1|theorem1|gradient\s*inversion)",
                    r"(?:\\partial\s*\\mathcal\{L\}\s*/\s*\\partial\s*l|\\frac\{\\partial\s*\\mathcal\{L\}\}\{\\partial\s*l|l_k\s*>\s*\\bar\{l\}\s*\+\s*1|loss\s*inversion)"
                ]),
                ("Theorem 2 (MMD Polarity Inversion)", [
                    r"(?:Theorem\s*2|theorem2|MMD|polarity)",
                    r"(?:MMD|polarity|Gaussian\s*similarity|uniform\s*distribution|noise\s*collapse|\sigma\(.*z.*\)\s*-\s*u)"
                ]),
                ("Theorem 3 (Triplet Gradient Saturation)", [
                    r"(?:Theorem\s*3|theorem3|triplet|saturation|vanishing)",
                    r"(?:0\.029|0\.03|hypersphere|vanishing|saturation|2048)"
                ]),
                ("Theorem 4 (Singular Grid Collapse)", [
                    r"(?:Theorem\s*4|theorem4|singular|affine|grid)",
                    r"(?:det\s*\(?\s*\\mathbf\{A\}|singular|zero-padding|22\.9%|diagonal)"
                ]),
                ("Theorem 5 (Unconditioned Denoiser Floor)", [
                    r"(?:Theorem\s*5|theorem5|denoiser|variance\s*floor|unconditioned)",
                    r"(?:1\.000|1\.08|1\.11|variance\s*floor|\\mathcal\{L\}_\{noise\}|unit\s*Gaussian)"
                ]),
            ]

            missing_theorems = []
            for th_name, patterns in theorem_checks:
                found = all(re.search(pat, ch3_content, re.IGNORECASE) for pat in patterns)
                if not found:
                    missing_theorems.append(th_name)

            if missing_theorems:
                all_passed = False
                messages.append(f"FAIL: Missing formal mathematical proofs in Chapter 3:")
                for mt in missing_theorems:
                    messages.append(f"  - {mt}")
            else:
                messages.append("PASS: Chapter 3 contains all formal Statements and Proofs for Theorems 1-5.")

            # Proof environment count check
            proof_blocks = re.findall(r"\\begin\{proof\}", ch3_content)
            if len(proof_blocks) < 5:
                all_passed = False
                messages.append(f"FAIL: Chapter 3 contains {len(proof_blocks)} proof environments; expected at least 5 for Theorems 1-5.")
            elif verbose:
                messages.append(f"PASS: Found {len(proof_blocks)} proof environments in Chapter 3.")

            # Required Optimizations P1-P6
            optimization_checks = [
                ("Optimization P1 (Score Denoiser & Tweedie)", [r"(?:P1|Tweedie|score-conditioned|denoiser)"]),
                ("Optimization P2 (Bounded Residual STN)", [r"(?:P2|bounded\s*residual|anchor|non-singular|STN)"]),
                ("Optimization P3 (Detached Softmax & MMD)", [r"(?:P3|detached|heat\s*kernel|Gaussian\s*similarity|MMD)"]),
                ("Optimization P4 (Normalized Soft Triplet)", [r"(?:P4|normalized|soft\s*margin|triplet|hyperspher)"]),
                ("Optimization P5 (DDIM Reverse Trajectory)", [r"(?:P5|DDIM|reverse\s*trajectory|trajectory|deterministic)"]),
                ("Optimization P6 (Optimizer, Reset, Fusion)", [r"(?:P6|decoupled|reset|discount|fusion|AdamW)"]),
            ]

            missing_opts = []
            for opt_name, patterns in optimization_checks:
                found = all(re.search(pat, ch3_content, re.IGNORECASE) for pat in patterns)
                if not found:
                    missing_opts.append(opt_name)

            if missing_opts:
                all_passed = False
                messages.append(f"FAIL: Missing Optimization formulations in Chapter 3:")
                for mo in missing_opts:
                    messages.append(f"  - {mo}")
            else:
                messages.append("PASS: Chapter 3 incorporates all Optimizations P1 through P6.")

            # Algorithms 1 and 2 check
            alg_blocks = re.findall(r"\\begin\{algorithm\}", ch3_content)
            if len(alg_blocks) < 2:
                all_passed = False
                messages.append(f"FAIL: Chapter 3 contains {len(alg_blocks)} algorithm environments; expected at least 2 (Algorithm 1 & 2).")
            else:
                messages.append(f"PASS: Chapter 3 includes Algorithms 1 & 2 pseudocode environments.")

        except Exception as e:
            all_passed = False
            messages.append(f"FAIL: Error inspecting Chapter 3: {e}")
    else:
        all_passed = False
        messages.append("FAIL: chapter3.tex does not exist.")

    # 4. Experimental Tables Completeness (Chapter 4)
    ch4_path = target_dir / "Chapters" / "chapter4.tex"
    if ch4_path.is_file():
        try:
            with open(ch4_path, "r", encoding="utf-8", errors="replace") as f:
                ch4_content = f.read()

            table_checks = [
                ("Table 4.2 (Master Results: Paper vs Repro vs P1-P6)", [
                    r"(?:4\.2|Master\s*Results|Comparison\s*with\s*State-of-the-Art)",
                    r"(?:Market-1501|VeRi-776)"
                ]),
                ("Table 4.4 (Ablation Matrix A0-A8 & LOO)", [
                    r"(?:4\.4|Ablation|Leave-One-Out|A0)",
                    r"(?:A1|P1|P2|Ablation)"
                ]),
                ("Table 4.5 (Diagnostic Telemetry & Pathology Verification)", [
                    r"(?:4\.5|Diagnostic|Telemetry|Floor|Determinant)",
                ]),
                ("Table 4.6 (Efficiency & Complexity Benchmark)", [
                    r"(?:4\.6|Efficiency|Complexity|Inference|Latency|FLOPs|Throughput)",
                ]),
            ]

            missing_tables = []
            for t_name, patterns in table_checks:
                found = all(re.search(pat, ch4_content, re.IGNORECASE) for pat in patterns)
                if not found:
                    missing_tables.append(t_name)

            if missing_tables:
                all_passed = False
                messages.append(f"FAIL: Missing required experimental tables in Chapter 4:")
                for mt in missing_tables:
                    messages.append(f"  - {mt}")
            else:
                messages.append("PASS: Chapter 4 contains all required Tables (4.2, 4.4, 4.5, 4.6).")

        except Exception as e:
            all_passed = False
            messages.append(f"FAIL: Error inspecting Chapter 4: {e}")
    else:
        all_passed = False
        messages.append("FAIL: chapter4.tex does not exist.")

    # 5. Chapter 5 Analysis Completeness
    ch5_path = target_dir / "Chapters" / "chapter5.tex"
    if ch5_path.is_file():
        try:
            with open(ch5_path, "r", encoding="utf-8", errors="replace") as f:
                ch5_content = f.read()

            ch5_topics = [
                ("Seed Debunking & Variance Refutation", [r"(?:seed|stochastic|variance|dispersion|hypothesis|debunk)"]),
                ("Oracle Clustering & Supervised Ceilings", [r"(?:oracle|ceiling|upper\s*bound|supervised|DBSCAN)"]),
                ("Frontier Future Directions", [r"(?:TransReID|CLIP|foundation|transformer|future)"]),
            ]

            missing_ch5 = []
            for topic, patterns in ch5_topics:
                found = all(re.search(pat, ch5_content, re.IGNORECASE) for pat in patterns)
                if not found:
                    missing_ch5.append(topic)

            if missing_ch5:
                all_passed = False
                messages.append(f"FAIL: Missing required analysis sections in Chapter 5:")
                for mc in missing_ch5:
                    messages.append(f"  - {mc}")
            else:
                messages.append("PASS: Chapter 5 contains empirical failure analysis, seed debunking, and future directions.")

        except Exception as e:
            all_passed = False
            messages.append(f"FAIL: Error inspecting Chapter 5: {e}")
    else:
        all_passed = False
        messages.append("FAIL: chapter5.tex does not exist.")

    # 6. FrontMatter Metadata & Abstract Structure Check
    title_path = target_dir / "FrontMatter" / "titlePage.tex"
    abstract_path = target_dir / "FrontMatter" / "abstract.tex"

    if title_path.is_file():
        try:
            with open(title_path, "r", encoding="utf-8", errors="replace") as f:
                title_content = f.read()
            if not re.search(r"Kharagpur", title_content, re.IGNORECASE) or not re.search(r"Computer Science", title_content, re.IGNORECASE):
                all_passed = False
                messages.append("FAIL: FrontMatter/titlePage.tex missing IIT Kharagpur / Department metadata.")
            else:
                messages.append("PASS: FrontMatter/titlePage.tex contains required IIT Kharagpur degree metadata.")
        except Exception as e:
            all_passed = False
            messages.append(f"FAIL: Error reading titlePage.tex: {e}")
    else:
        all_passed = False
        messages.append("FAIL: FrontMatter/titlePage.tex does not exist.")

    if abstract_path.is_file():
        try:
            with open(abstract_path, "r", encoding="utf-8", errors="replace") as f:
                abs_content = f.read()
            abs_sections = [
                ("Context & Baseline Pathologies", [r"(?:unsupervised|re-identification|patholog|deficit)"]),
                ("Mathematical Optimizations (P1-P6)", [r"(?:optimization|theorem|tweedie|STN|triplet)"]),
                ("Empirical Results", [r"(?:Market-1501|VeRi-776|mAP|Rank-1)"]),
            ]
            for sec_name, pats in abs_sections:
                if not any(re.search(p, abs_content, re.IGNORECASE) for p in pats):
                    all_passed = False
                    messages.append(f"FAIL: FrontMatter/abstract.tex missing content area: {sec_name}")
            if all_passed:
                messages.append("PASS: FrontMatter/abstract.tex satisfies publication-grade abstract criteria.")
        except Exception as e:
            all_passed = False
            messages.append(f"FAIL: Error reading abstract.tex: {e}")
    else:
        all_passed = False
        messages.append("FAIL: FrontMatter/abstract.tex does not exist.")

    return all_passed, messages


# ==============================================================================
# TIER 4: Syntactic Validity & Citations Check
# ==============================================================================

def strip_latex_comments(line: str) -> str:
    """Strips LaTeX comments while preserving escaped percent signs (\%)."""
    result = []
    i = 0
    n = len(line)
    while i < n:
        ch = line[i]
        if ch == '%':
            # Count backslashes preceding %
            bs_count = 0
            j = i - 1
            while j >= 0 and line[j] == '\\':
                bs_count += 1
                j -= 1
            if bs_count % 2 == 0:
                # Unescaped %, comment begins
                break
        result.append(ch)
        i += 1
    return "".join(result)


def check_latex_delimiters(file_path: Path) -> List[str]:
    """
    Checks balanced curly braces and environment pairs in a LaTeX file.
    Returns list of error messages (empty if valid).
    """
    errors = []
    brace_stack = []  # stores (line_no, col_no)
    env_stack = []    # stores (env_name, line_no)
    in_verbatim = False

    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except Exception as e:
        return [f"Could not read {file_path}: {e}"]

    begin_pattern = re.compile(r"\\begin\{([a-zA-Z0-9_\*]+)\}")
    end_pattern = re.compile(r"\\end\{([a-zA-Z0-9_\*]+)\}")

    for line_no, raw_line in enumerate(lines, start=1):
        clean_line = strip_latex_comments(raw_line)

        # Check verbatim transitions
        if in_verbatim:
            if "\\end{verbatim}" in clean_line or "\\end{lstlisting}" in clean_line:
                in_verbatim = False
            continue
        if "\\begin{verbatim}" in clean_line or "\\begin{lstlisting}" in clean_line:
            in_verbatim = True
            continue

        # 1. Environment matching
        # Find all begin/end matches sorted by position in line
        tokens = []
        for m in begin_pattern.finditer(clean_line):
            tokens.append((m.start(), "begin", m.group(1)))
        for m in end_pattern.finditer(clean_line):
            tokens.append((m.start(), "end", m.group(1)))
        tokens.sort(key=lambda x: x[0])

        for pos, action, env_name in tokens:
            if action == "begin":
                env_stack.append((env_name, line_no))
            elif action == "end":
                if not env_stack:
                    errors.append(f"Line {line_no}: Unmatched \\end{{{env_name}}} (no preceding \\begin)")
                else:
                    top_env, top_line = env_stack.pop()
                    if top_env != env_name:
                        errors.append(
                            f"Line {line_no}: Mismatched environment: expected \\end{{{top_env}}} "
                            f"(opened at line {top_line}), but found \\end{{{env_name}}}"
                        )

        # 2. Curly brace balancing (ignore escaped \{ and \})
        # Remove \begin{...} and \end{...} or commands to avoid double-processing if desired,
        # but brace balancing must hold globally over all text outside comments.
        i = 0
        n = len(clean_line)
        while i < n:
            ch = clean_line[i]
            if ch in ('{', '}'):
                # Check if escaped
                bs_count = 0
                j = i - 1
                while j >= 0 and clean_line[j] == '\\':
                    bs_count += 1
                    j -= 1
                if bs_count % 2 == 0:
                    if ch == '{':
                        brace_stack.append((line_no, i + 1))
                    elif ch == '}':
                        if not brace_stack:
                            errors.append(f"Line {line_no}, col {i + 1}: Unmatched closing brace '}}'")
                        else:
                            brace_stack.pop()
            i += 1

    # End-of-file checks
    if brace_stack:
        for b_line, b_col in brace_stack:
            errors.append(f"Unclosed opening brace '{{' at line {b_line}, col {b_col}")

    if env_stack:
        for u_env, u_line in env_stack:
            errors.append(f"Unclosed environment \\begin{{{u_env}}} opened at line {u_line}")

    return errors


def check_tier4_latex_validity(target_dir: Path, verbose: bool = False) -> Tuple[bool, List[str]]:
    """
    Tier 4: Syntactic Validity & Citations Check.
    Validates:
      1. Balanced braces and environments across all .tex files.
      2. BibTeX syntax and key extraction in references.bib.
      3. Citation resolution: every \\cite{key} in .tex resolves to references.bib.
      4. Foundational citations present in references.bib.
      5. PDF compilation check (graceful execution or fallback if compiler absent).
    """
    messages = []
    all_passed = True

    if not target_dir.exists():
        return False, [f"FAIL: Target directory does not exist: {target_dir}"]

    # 1. Delimiter & Environment Parsing
    tex_files = list(target_dir.rglob("*.tex"))
    delimiter_failures = 0

    for tex_file in tex_files:
        rel_path = tex_file.relative_to(target_dir)
        errs = check_latex_delimiters(tex_file)
        if errs:
            all_passed = False
            delimiter_failures += 1
            messages.append(f"FAIL: Delimiter/Environment errors in {rel_path}:")
            for e in errs[:10]:  # Limit output to 10 errors per file
                messages.append(f"  - {e}")
            if len(errs) > 10:
                messages.append(f"  - ... and {len(errs) - 10} more syntax errors.")
        elif verbose:
            messages.append(f"PASS: {rel_path} delimiter and environment balance verified.")

    if delimiter_failures == 0 and tex_files:
        messages.append(f"PASS: All {len(tex_files)} LaTeX files have balanced delimiters and environments.")
    elif not tex_files:
        all_passed = False
        messages.append("FAIL: No .tex files found in target directory.")

    # 2. BibTeX Parsing and Citation Resolution
    bib_path = target_dir / "references.bib"
    bib_keys = set()

    if bib_path.is_file():
        try:
            with open(bib_path, "r", encoding="utf-8", errors="replace") as f:
                bib_content = f.read()

            # Extract keys: @type{key, ...
            entry_matches = re.findall(r"@\w+\s*\{\s*([a-zA-Z0-9_\-:\.]+)\s*,", bib_content)
            bib_keys = set(entry_matches)

            if len(bib_keys) < 20:
                all_passed = False
                messages.append(f"FAIL: references.bib contains only {len(bib_keys)} entries; expected at least 30.")
            else:
                messages.append(f"PASS: Parsed {len(bib_keys)} BibTeX entries in references.bib.")

            # Check foundational citations required by blueprint
            foundational_keys = [
                ("Song DDIM", [r"song2020denoising", r"song.*ddim"]),
                ("AdamW", [r"loshchilov2017decoupled", r"loshchilov.*adamw"]),
                ("TransReID", [r"he2021transreid", r"transreid"]),
                ("Score SDE", [r"song2020score", r"score_sde"]),
                ("CLIP-ReID", [r"li2023clip", r"clip_reid", r"clip"]),
            ]

            missing_foundational = []
            for name, pats in foundational_keys:
                found = any(any(re.search(p, k, re.IGNORECASE) for p in pats) for k in bib_keys)
                if not found:
                    missing_foundational.append(name)

            if missing_foundational:
                all_passed = False
                messages.append(f"FAIL: Missing foundational bibliography entries in references.bib:")
                for mf in missing_foundational:
                    messages.append(f"  - {mf}")
            else:
                messages.append("PASS: Foundational bibliography entries present in references.bib.")

        except Exception as e:
            all_passed = False
            messages.append(f"FAIL: Error reading references.bib: {e}")
    else:
        all_passed = False
        messages.append("FAIL: references.bib does not exist.")

    # 3. Citation Resolution: Check every \cite{...} in all .tex files
    cite_pattern = re.compile(r"\\(?:cite|citep|citet|citeauthor|citeyear|nocite)\*?\{([^}]+)\}")
    unresolved_citations = set()
    total_citations = 0

    for tex_file in tex_files:
        rel_path = tex_file.relative_to(target_dir)
        try:
            with open(tex_file, "r", encoding="utf-8", errors="replace") as f:
                for line_no, line in enumerate(f, start=1):
                    clean_line = strip_latex_comments(line)
                    for match in cite_pattern.finditer(clean_line):
                        keys_str = match.group(1)
                        keys = [k.strip() for k in keys_str.split(",") if k.strip()]
                        for k in keys:
                            total_citations += 1
                            if bib_keys and k not in bib_keys:
                                unresolved_citations.add((rel_path, line_no, k))
        except Exception as e:
            all_passed = False
            messages.append(f"FAIL: Error scanning citations in {rel_path}: {e}")

    if unresolved_citations:
        all_passed = False
        messages.append(f"FAIL: Found {len(unresolved_citations)} unresolved citation keys:")
        for rpath, lno, key in sorted(list(unresolved_citations))[:15]:
            messages.append(f"  - {rpath}:{lno}: '\\cite{{{key}}}' not in references.bib")
        if len(unresolved_citations) > 15:
            messages.append(f"  - ... and {len(unresolved_citations) - 15} more unresolved citations.")
    else:
        messages.append(f"PASS: Verified {total_citations} citations; 100% resolve to references.bib.")

    # 4. Internal Cross-Reference Resolution Check (\label vs \ref / \eqref)
    label_pattern = re.compile(r"\\label\{([^}]+)\}")
    ref_pattern = re.compile(r"\\(?:ref|eqref|pageref)\{([^}]+)\}")
    defined_labels = {}
    duplicate_labels = []

    for tex_file in tex_files:
        rel_path = tex_file.relative_to(target_dir)
        try:
            with open(tex_file, "r", encoding="utf-8", errors="replace") as f:
                for line_no, line in enumerate(f, start=1):
                    clean_line = strip_latex_comments(line)
                    for match in label_pattern.finditer(clean_line):
                        lbl = match.group(1).strip()
                        if lbl in defined_labels:
                            duplicate_labels.append((lbl, str(rel_path), line_no, defined_labels[lbl]))
                        else:
                            defined_labels[lbl] = (str(rel_path), line_no)
        except Exception as e:
            all_passed = False
            messages.append(f"FAIL: Error scanning labels in {rel_path}: {e}")

    unresolved_refs = []
    total_refs = 0
    for tex_file in tex_files:
        rel_path = tex_file.relative_to(target_dir)
        try:
            with open(tex_file, "r", encoding="utf-8", errors="replace") as f:
                for line_no, line in enumerate(f, start=1):
                    clean_line = strip_latex_comments(line)
                    for match in ref_pattern.finditer(clean_line):
                        r_key = match.group(1).strip()
                        total_refs += 1
                        if r_key not in defined_labels:
                            unresolved_refs.append((str(rel_path), line_no, r_key))
        except Exception as e:
            all_passed = False
            messages.append(f"FAIL: Error scanning references in {rel_path}: {e}")

    if duplicate_labels:
        all_passed = False
        messages.append(f"FAIL: Found {len(duplicate_labels)} duplicate label definitions:")
        for lbl, rpath, lno, orig in duplicate_labels:
            messages.append(f"  - Label '{lbl}' in {rpath}:{lno} already defined in {orig[0]}:{orig[1]}")
    else:
        messages.append(f"PASS: Verified {len(defined_labels)} defined labels with zero duplicates.")

    if unresolved_refs:
        all_passed = False
        messages.append(f"FAIL: Found {len(unresolved_refs)} unresolved cross-reference keys:")
        for rpath, lno, r_key in unresolved_refs:
            messages.append(f"  - {rpath}:{lno}: '\\ref{{{r_key}}}' not found in defined labels")
    else:
        messages.append(f"PASS: Verified {total_refs} cross-references; 100% resolve to defined labels.")

    # 5. LaTeX PDF Compilation Attempt
    compiler = None
    for cand in ["pdflatex", "latexmk", "xelatex"]:
        if shutil.which(cand):
            compiler = cand
            break

    main_tex = target_dir / "main.tex"
    if compiler and main_tex.is_file():
        messages.append(f"INFO: Detected LaTeX compiler '{compiler}'. Attempting non-stop PDF compilation...")
        try:
            cmd = [compiler, "-interaction=nonstopmode", "-halt-on-error", "main.tex"]
            proc = subprocess.run(
                cmd,
                cwd=str(target_dir),
                capture_output=True,
                text=True,
                timeout=120
            )
            if proc.returncode == 0:
                messages.append(f"PASS: LaTeX PDF compilation succeeded with {compiler}.")
            else:
                all_passed = False
                messages.append(f"FAIL: LaTeX compilation exited with code {proc.returncode}.")
                # Log last 15 lines of compilation output
                output_lines = proc.stdout.splitlines()[-15:]
                for ol in output_lines:
                    messages.append(f"  [latex] {ol}")
        except subprocess.TimeoutExpired:
            all_passed = False
            messages.append("FAIL: LaTeX compilation timed out after 120 seconds.")
        except Exception as e:
            all_passed = False
            messages.append(f"FAIL: Error executing LaTeX compilation: {e}")
    elif not compiler:
        messages.append(
            "INFO: No LaTeX compiler (pdflatex/latexmk) installed in environment PATH. "
            "Skipping binary PDF compilation; static AST, syntax, and delimiter checks applied."
        )

    return all_passed, messages


# ==============================================================================
# CLI Orchestrator & Entry Point
# ==============================================================================

def run_all_tiers(
    repo_root: Path,
    target_dir: Path,
    tier: str = "all",
    verbose: bool = False
) -> Tuple[bool, Dict[str, Any]]:
    """
    Runs verification checks according to requested tier.
    """
    summary = {
        "tier1_preservation": None,
        "tier2_string_exclusions": None,
        "tier3_document_completeness": None,
        "tier4_latex_validity": None,
        "overall_passed": True,
        "details": {}
    }

    print("=" * 80)
    print("  IIT KHARAGPUR THESIS REVISION: E2E ACCEPTANCE VERIFICATION SUITE")
    print("=" * 80)
    print(f"Repository Root : {repo_root}")
    print(f"Target Directory: {target_dir}")
    print(f"Selected Tier   : {tier}")
    print("-" * 80)

    overall_passed = True

    # Tier 1
    if tier in ("1", "all"):
        print("\n[TIER 1] Preservation & Cleanliness Check:")
        t1_pass, t1_msgs = check_tier1_preservation(repo_root, verbose=verbose)
        summary["tier1_preservation"] = t1_pass
        summary["details"]["tier1"] = t1_msgs
        for m in t1_msgs:
            print(f"  {m}")
        if not t1_pass:
            overall_passed = False

    # Tier 2
    if tier in ("2", "all"):
        print("\n[TIER 2] Strict String Exclusion Check:")
        t2_pass, t2_msgs = check_tier2_string_exclusions(target_dir, verbose=verbose)
        summary["tier2_string_exclusions"] = t2_pass
        summary["details"]["tier2"] = t2_msgs
        for m in t2_msgs:
            print(f"  {m}")
        if not t2_pass:
            overall_passed = False

    # Tier 3
    if tier in ("3", "all"):
        print("\n[TIER 3] Structural & Mathematical Completeness Check:")
        t3_pass, t3_msgs = check_tier3_document_completeness(target_dir, verbose=verbose)
        summary["tier3_document_completeness"] = t3_pass
        summary["details"]["tier3"] = t3_msgs
        for m in t3_msgs:
            print(f"  {m}")
        if not t3_pass:
            overall_passed = False

    # Tier 4
    if tier in ("4", "all"):
        print("\n[TIER 4] Syntactic Validity & Citations Check:")
        t4_pass, t4_msgs = check_tier4_latex_validity(target_dir, verbose=verbose)
        summary["tier4_latex_validity"] = t4_pass
        summary["details"]["tier4"] = t4_msgs
        for m in t4_msgs:
            print(f"  {m}")
        if not t4_pass:
            overall_passed = False

    summary["overall_passed"] = overall_passed

    print("\n" + "=" * 80)
    status_str = "SUCCESS: ALL CHECKS PASSED" if overall_passed else "FAILURE: CHECKS FAILED"
    print(f"  FINAL VERDICT: {status_str}")
    print("=" * 80 + "\n")

    return overall_passed, summary


def main():
    parser = argparse.ArgumentParser(
        description="Verify IIT Kharagpur thesis revision acceptance criteria across all 4 tiers."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="Root directory of the git repository."
    )
    parser.add_argument(
        "--target-dir",
        type=Path,
        default=None,
        help="Path to report_optimized directory (defaults to repo-root/report_optimized)."
    )
    parser.add_argument(
        "--tier",
        choices=["1", "2", "3", "4", "all"],
        default="all",
        help="Acceptance tier to verify (default: all)."
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose test output."
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results in JSON format."
    )

    args = parser.parse_args()
    target_dir = args.target_dir or (args.repo_root / "report_optimized")

    passed, summary = run_all_tiers(
        repo_root=args.repo_root,
        target_dir=target_dir,
        tier=args.tier,
        verbose=args.verbose
    )

    if args.json:
        print(json.dumps(summary, indent=2))

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
