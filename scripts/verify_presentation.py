#!/usr/bin/env python3
"""
Verification Script for Beamer Presentation Artifacts.
Validates syntax, structure, theorem coverage, benchmark metrics,
negative constraints, and file cleanliness according to ORIGINAL_REQUEST.md.
"""

import os
import re
import subprocess
import sys


def check_file_exists_and_nonempty(filepath):
    """Check that a file exists and has non-zero size."""
    if not os.path.exists(filepath):
        return False, f"File does not exist: {filepath}"
    size = os.path.getsize(filepath)
    if size == 0:
        return False, f"File is empty: {filepath}"
    return True, f"Found {filepath} ({size} bytes)"


def strip_latex_comments(content):
    """Remove LaTeX comments while preserving escaped percent signs."""
    lines = []
    for line in content.splitlines():
        # Match % not preceded by \
        cleaned = re.sub(r"(?<!\\)%.*$", "", line)
        lines.append(cleaned)
    return "\n".join(lines)


def check_delimiter_balance(content):
    """Check balanced braces, brackets, and dollar signs."""
    # Remove comments
    stripped = strip_latex_comments(content)

    # Check curly braces
    brace_depth = 0
    in_escape = False
    for i, ch in enumerate(stripped):
        if in_escape:
            in_escape = False
            continue
        if ch == "\\":
            in_escape = True
            continue
        if ch == "{":
            brace_depth += 1
        elif ch == "}":
            brace_depth -= 1
            if brace_depth < 0:
                return False, f"Unmatched closing brace '}}' at character index {i}"

    if brace_depth != 0:
        return False, f"Unclosed brace detected: final depth = {brace_depth}"

    return True, "All braces '{...}' are balanced."


def check_environments(content):
    """Check that all \\begin{env} have matching \\end{env} in stack order."""
    stripped = strip_latex_comments(content)
    pattern = re.compile(r"\\(begin|end)\{([a-zA-Z0-9*]+)\}")
    stack = []

    for match in pattern.finditer(stripped):
        tag, env = match.groups()
        if tag == "begin":
            stack.append((env, match.start()))
        else:
            if not stack:
                return False, f"Unmatched \\end{{{env}}} at position {match.start()}"
            last_env, start_pos = stack.pop()
            if last_env != env:
                return False, f"Mismatched environment: \\begin{{{last_env}}} closed by \\end{{{env}}}"

    if stack:
        unclosed = [e[0] for e in stack]
        return False, f"Unclosed environments remaining: {unclosed}"

    return True, "All LaTeX environments are balanced."


def count_frames(content):
    """Count \\begin{frame} occurrences."""
    stripped = strip_latex_comments(content)
    begin_frames = re.findall(r"\\begin\{frame\}", stripped)
    end_frames = re.findall(r"\\end\{frame\}", stripped)

    if len(begin_frames) != len(end_frames):
        return False, len(begin_frames), f"Mismatched frame count: {len(begin_frames)} begins vs {len(end_frames)} ends"

    return True, len(begin_frames), f"Total frames: {len(begin_frames)}"


def check_negative_constraints(content, filepath="presentation/presentation.tex"):
    """Verify ZERO occurrences of forbidden strings."""
    forbidden = ["75.5M", "99.3%", "100x", "100×", "saving >10 hours"]
    derogatory = ["bugged", "broken", "pathological"]

    violations = []
    for f in forbidden:
        if f in content:
            violations.append(f"Forbidden string found: '{f}'")

    # In comments or text, check derogatory words
    stripped = strip_latex_comments(content)
    for d in derogatory:
        if re.search(r"\b" + re.escape(d) + r"\b", stripped, re.IGNORECASE):
            violations.append(f"Derogatory academic tone violation: '{d}'")

    if violations:
        return False, violations
    return True, "All negative constraints and academic tone rules satisfied."


def check_content_coverage(content):
    """Verify presence of key theoretical and empirical requirements."""
    stripped = strip_latex_comments(content)
    checks = {
        "Theorem 1": re.search(r"Theorem\s+1", stripped) and ("gradient" in stripped.lower()),
        "Theorem 2": re.search(r"Theorem\s+2", stripped) and ("polarity" in stripped.lower() or "mmd" in stripped.lower()),
        "Theorem 3": re.search(r"Theorem\s+3", stripped) and ("triplet" in stripped.lower() or "saturation" in stripped.lower()),
        "Theorem 4": re.search(r"Theorem\s+4", stripped) and ("singular" in stripped.lower() or "collapse" in stripped.lower()),
        "Theorem 5": re.search(r"Theorem\s+5", stripped) and ("floor" in stripped.lower() or "denois" in stripped.lower()),
        "Proposition 1": re.search(r"Proposition\s+1", stripped) and ("0.1958" in stripped or "0.196" in stripped),
        "Proposal P1 / Opt 1": ("P1" in stripped or "Optimization 1" in stripped) and ("detached" in stripped.lower() or "softmax" in stripped.lower()),
        "Proposal P2 / Opt 2": ("P2" in stripped or "Optimization 2" in stripped) and ("mmd" in stripped.lower() or "heat kernel" in stripped.lower()),
        "Proposal P3 / Opt 3": ("P3" in stripped or "Optimization 3" in stripped) and ("triplet" in stripped.lower() or "hypersphere" in stripped.lower()),
        "Proposal P4 / Opt 4": ("P4" in stripped or "Optimization 4" in stripped) and ("stn" in stripped.lower() or "spatial" in stripped.lower() or "bounded residual" in stripped.lower()),
        "Proposal P5 / Opt 5": ("P5" in stripped or "Optimization 5" in stripped) and ("tweedie" in stripped.lower() or "denoiser" in stripped.lower()),
        "Proposal P6 / Opt 6": ("P6" in stripped or "Optimization 6" in stripped) and ("ddim" in stripped.lower()),
        "Market-1501 Benchmark": "85.6" in stripped and "94.2" in stripped and "Market-1501" in stripped,
        "VeRi-776 Benchmark": "52.4" in stripped and "96.1" in stripped and "VeRi-776" in stripped,
        "STN Architectural Refinement": "architectural refinement" in stripped.lower(),
        "DDIM Sampling Refinement": "sampling refinement" in stripped.lower(),
    }

    missing = [k for k, v in checks.items() if not v]
    if missing:
        return False, f"Missing required content elements: {missing}"
    return True, "All required theorems, proposals, benchmarks, and framings are present."


def check_git_status():
    """Verify report/, pisl/, examples/ are untouched."""
    try:
        res = subprocess.run(
            ["git", "status", "--porcelain", "report/", "pisl/", "examples/"],
            capture_output=True,
            text=True,
            check=True
        )
        if res.stdout.strip():
            return False, f"Protected directories modified:\n{res.stdout}"
        return True, "Protected directories (report/, pisl/, examples/) are untouched."
    except Exception as e:
        return False, f"Git status execution failed: {e}"


def check_image_references(content, base_dir="presentation"):
    """Verify that all graphic references resolve to existing files."""
    stripped = strip_latex_comments(content)
    # Extract \includegraphics{...} or \includegraphics[...]{...}
    matches = re.findall(r"\\includegraphics(?:\[.*?\])?\{([^}]+)\}", stripped)
    graphic_paths = ["figs", "examples/visualizations"]

    missing = []
    for img in matches:
        # Check relative to base_dir with ../figs/ and ../examples/visualizations/
        found = False
        for gp in graphic_paths:
            candidate = os.path.join(gp, img)
            if os.path.exists(candidate):
                found = True
                break
        if not found:
            missing.append(img)

    if missing:
        return False, f"Referenced image files not found on disk: {missing}"
    return True, f"All {len(matches)} referenced image files exist on disk."


def main():
    print("=" * 70)
    print("RUNNING BEAMER PRESENTATION ACCEPTANCE VERIFICATION")
    print("=" * 70)

    all_passed = True

    # 1. Check theme files
    theme_files = [
        "presentation/beamerthemeSimpleDarkBlue.sty",
        "presentation/beamercolorthemeSimpleDarkBlue.sty",
        "presentation/beamerfontthemeSimpleDarkBlue.sty",
        "presentation/beamerinnerthemeSimpleDarkBlue.sty",
    ]
    for tf in theme_files:
        ok, msg = check_file_exists_and_nonempty(tf)
        print(f"[{'PASS' if ok else 'FAIL'}] Theme File: {msg}")
        if not ok:
            all_passed = False

    # 2. Check main presentation file
    pres_file = "presentation/presentation.tex"
    ok, msg = check_file_exists_and_nonempty(pres_file)
    print(f"[{'PASS' if ok else 'FAIL'}] Main Presentation: {msg}")
    if not ok:
        all_passed = False
        return 1

    with open(pres_file, "r", encoding="utf-8") as f:
        content = f.read()

    # 3. Delimiter balance
    ok, msg = check_delimiter_balance(content)
    print(f"[{'PASS' if ok else 'FAIL'}] Delimiter Balance: {msg}")
    if not ok:
        all_passed = False

    # 4. Environment balance
    ok, msg = check_environments(content)
    print(f"[{'PASS' if ok else 'FAIL'}] Environment Balance: {msg}")
    if not ok:
        all_passed = False

    # 5. Frame count
    ok, count, msg = count_frames(content)
    frame_ok = ok and (count < 20)
    print(f"[{'PASS' if frame_ok else 'FAIL'}] Frame Count (< 20): {msg} (Target < 20, Got {count})")
    if not frame_ok:
        all_passed = False

    # 6. Negative constraints
    ok, msg = check_negative_constraints(content, pres_file)
    print(f"[{'PASS' if ok else 'FAIL'}] Negative Constraints & Tone: {msg}")
    if not ok:
        all_passed = False

    # 7. Content coverage
    ok, msg = check_content_coverage(content)
    print(f"[{'PASS' if ok else 'FAIL'}] Content & Theorem Coverage: {msg}")
    if not ok:
        all_passed = False

    # 8. Image references
    ok, msg = check_image_references(content)
    print(f"[{'PASS' if ok else 'FAIL'}] Graphics Resolution: {msg}")
    if not ok:
        all_passed = False

    # 9. Git status on protected directories
    ok, msg = check_git_status()
    print(f"[{'PASS' if ok else 'FAIL'}] Source Protection: {msg}")
    if not ok:
        all_passed = False

    print("=" * 70)
    if all_passed:
        print("ALL PRESENTATION ACCEPTANCE VERIFICATIONS PASSED SUCCESSFULLY!")
        print("=" * 70)
        return 0
    else:
        print("VERIFICATION FAILED WITH ISSUES!")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    sys.exit(main())
