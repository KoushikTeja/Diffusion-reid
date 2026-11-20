"""
tests/test_thesis_acceptance.py
================================
Unit and Acceptance Test Suite for IIT Kharagpur Revised Thesis (`report_optimized/`).

Covers all 4 Acceptance Tiers:
  - Tier 1: Preservation & Cleanliness (Git integrity on report/, pisl/, examples/)
  - Tier 2: Strict String Exclusion (Exclusion of exaggerated claims: 75.5M, 99.3%, 100x, etc.)
  - Tier 3: Structural & Mathematical Completeness (Files, FrontMatter, Ch 1-5, Theorems 1-5, P1-P6, Algs 1-2, Tables)
  - Tier 4: Syntactic Validity & Citations (Delimiter balance, BibTeX syntax, citation resolution, PDF compilation)

Run with:
  python -m unittest tests/test_thesis_acceptance.py
"""

import sys
import unittest
from pathlib import Path

# Add repo root to python path to import verification module
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.verify_thesis_requirements import (
    check_tier1_preservation,
    check_tier2_string_exclusions,
    check_tier3_document_completeness,
    check_tier4_latex_validity,
    check_latex_delimiters,
    strip_latex_comments,
    FORBIDDEN_PATTERNS,
    REQUIRED_FILES
)


class TestTier1Preservation(unittest.TestCase):
    """Tier 1: Preservation & Cleanliness Checks."""

    def test_tier1_preservation_cleanliness(self):
        """Verify no files in report/, pisl/, or examples/ have been modified, deleted, or untracked."""
        passed, messages = check_tier1_preservation(REPO_ROOT, verbose=True)
        self.assertTrue(passed, msg="\n".join(messages))


class TestTier2StrictStringExclusions(unittest.TestCase):
    """Tier 2: Strict String Exclusion Checks."""

    def setUp(self):
        self.target_dir = REPO_ROOT / "report_optimized"

    def test_report_optimized_forbidden_strings(self):
        """Verify zero occurrences of forbidden claims (75.5M, 99.3%, 100x, etc.) in report_optimized/."""
        if not self.target_dir.exists():
            self.fail(f"Target directory does not exist: {self.target_dir}")
        passed, messages = check_tier2_string_exclusions(self.target_dir, verbose=True)
        self.assertTrue(passed, msg="\n".join(messages))

    def test_adversarial_forbidden_pattern_sensitivity(self):
        """
        Adversarial unit test to ensure regex patterns reliably detect forbidden claims
        and reject disguised or whitespace-padded variations.
        """
        import re
        test_cases = [
            ("Model saves 75.5M redundant parameters", True),
            ("We achieve 75.5 M parameter reduction", True),
            ("A total of 75,501,568 parameters were pruned", True),
            ("Yielding a 99.3% STN reduction", True),
            ("A 99.3 % parameter shrinkage", True),
            ("Achieves 100x DDIM speedup", True),
            ("Extraction is 100X faster", True),
            ("Sampling with 100× acceleration", True),
            ("Speedup of 100\\times in inference", True),
            ("Capable of saving >10 hours of computation", True),
            ("Saves > 10 hours in training", True),
            ("Baseline runs for 100 epochs", False),
            ("The batch size is 100", False),
            ("Training required 10 hours of runtime", False),
        ]
        compiled = [(re.compile(p, re.IGNORECASE), d) for p, d in FORBIDDEN_PATTERNS]

        for text, expected_match in test_cases:
            matched = any(pattern.search(text) for pattern, _ in compiled)
            self.assertEqual(
                matched,
                expected_match,
                msg=f"Sensitivity assertion failed for input '{text}': expected_match={expected_match}, got={matched}"
            )


class TestTier3StructuralCompleteness(unittest.TestCase):
    """Tier 3: Document Completeness & Mathematical Audit Inclusions."""

    def setUp(self):
        self.target_dir = REPO_ROOT / "report_optimized"

    def test_required_files_exist(self):
        """Verify that all 12 structural files for IIT Kharagpur thesis exist."""
        if not self.target_dir.exists():
            self.fail(f"Target directory does not exist: {self.target_dir}")
        missing = [rf for rf in REQUIRED_FILES if not (self.target_dir / rf).is_file()]
        self.assertEqual(len(missing), 0, msg=f"Missing required thesis files: {missing}")

    def test_comprehensive_document_and_mathematical_audit(self):
        """
        Full Tier 3 audit:
          - No TODOs or placeholder stubs
          - Theorems 1-5 proofs in Chapter 3
          - Optimizations P1-P6 in Chapter 3
          - Algorithms 1 & 2 in Chapter 3
          - Tables 4.2, 4.4, 4.5, 4.6 in Chapter 4
          - Analysis in Chapter 5
          - FrontMatter metadata
        """
        if not self.target_dir.exists():
            self.fail(f"Target directory does not exist: {self.target_dir}")
        passed, messages = check_tier3_document_completeness(self.target_dir, verbose=True)
        self.assertTrue(passed, msg="\n".join(messages))


class TestTier4LatexAndCitations(unittest.TestCase):
    """Tier 4: Syntactic Validity & Citations."""

    def setUp(self):
        self.target_dir = REPO_ROOT / "report_optimized"

    def test_comment_stripping_logic(self):
        """Unit test for LaTeX comment stripper handling unescaped vs escaped %."""
        self.assertEqual(strip_latex_comments("No comment here"), "No comment here")
        self.assertEqual(strip_latex_comments("Before % comment"), "Before ")
        self.assertEqual(strip_latex_comments("Escaped \\% symbol"), "Escaped \\% symbol")
        self.assertEqual(strip_latex_comments("Escaped \\% then % comment"), "Escaped \\% then ")
        self.assertEqual(strip_latex_comments("Double backslash newline \\\\% is comment"), "Double backslash newline \\\\")

    def test_synthetic_delimiter_detector(self):
        """Unit test for LaTeX delimiter checker detecting mismatched braces and environments."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)

            # Valid case
            valid_file = tmp_path / "valid.tex"
            valid_file.write_text("\\begin{equation}\n  a = \\{b + c\\};\n\\end{equation}\n", encoding="utf-8")
            self.assertEqual(check_latex_delimiters(valid_file), [])

            # Unmatched brace
            unmatched_brace = tmp_path / "unmatched_brace.tex"
            unmatched_brace.write_text("Hello {world\n", encoding="utf-8")
            errs = check_latex_delimiters(unmatched_brace)
            self.assertTrue(any("Unclosed opening brace" in e for e in errs))

            # Mismatched environment
            mismatched_env = tmp_path / "mismatched_env.tex"
            mismatched_env.write_text("\\begin{proof}\n  statement\n\\end{equation}\n", encoding="utf-8")
            errs = check_latex_delimiters(mismatched_env)
            self.assertTrue(any("Mismatched environment" in e for e in errs))

    def test_comprehensive_latex_syntax_and_citations(self):
        """
        Full Tier 4 audit:
          - Delimiters and environment parity across all .tex files
          - Valid BibTeX syntax and minimum entry count
          - Foundational citations present (Song DDIM, AdamW, TransReID, Score SDE, CLIP-ReID)
          - 100% resolution of \\cite{...} keys against references.bib
          - PDF compilation (graceful execution or fallback if compiler absent)
        """
        if not self.target_dir.exists():
            self.fail(f"Target directory does not exist: {self.target_dir}")
        passed, messages = check_tier4_latex_validity(self.target_dir, verbose=True)
        self.assertTrue(passed, msg="\n".join(messages))


if __name__ == "__main__":
    unittest.main()
