"""
Unit and Acceptance Test Suite for Beamer Presentation (`presentation/presentation.tex`).
Covers:
  - Theme files and base template compliance
  - Strict slide count under 20 frames
  - LaTeX delimiter and environment balance
  - Zero occurrences of forbidden claims and respectful academic tone
  - Complete coverage of Theorems 1-5, Proposition 1, Proposals P1-P6
  - Calibrated benchmark results on Market-1501 and VeRi-776
  - Diagnostic telemetry and progressive ablation trajectory
  - Untouched protected directories (report/, pisl/, examples/)

Run with:
  python -m unittest tests/test_presentation_acceptance.py
  (or pytest tests/test_presentation_acceptance.py if pytest is installed)
"""

import os
import re
import subprocess
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PRESENTATION_TEX = REPO_ROOT / "presentation" / "presentation.tex"
THEME_FILES = [
    REPO_ROOT / "presentation" / "beamerthemeSimpleDarkBlue.sty",
    REPO_ROOT / "presentation" / "beamercolorthemeSimpleDarkBlue.sty",
    REPO_ROOT / "presentation" / "beamerfontthemeSimpleDarkBlue.sty",
    REPO_ROOT / "presentation" / "beamerinnerthemeSimpleDarkBlue.sty",
]
PROTECTED_DIRS = ["report/", "pisl/", "examples/"]


def strip_latex_comments(content: str) -> str:
    lines = []
    for line in content.splitlines():
        cleaned = re.sub(r"(?<!\\)%.*$", "", line)
        lines.append(cleaned)
    return "\n".join(lines)


class TestPresentationStructure(unittest.TestCase):
    """Verifies file existence, theme setup, base template, and frame count."""

    def setUp(self):
        self.assertTrue(PRESENTATION_TEX.exists(), f"{PRESENTATION_TEX} must exist")
        with open(PRESENTATION_TEX, "r", encoding="utf-8") as f:
            self.raw_content = f.read()
        self.stripped_content = strip_latex_comments(self.raw_content)

    def test_theme_files_exist_and_nonempty(self):
        for tf in THEME_FILES:
            self.assertTrue(tf.exists(), f"Missing theme file: {tf}")
            self.assertGreater(tf.stat().st_size, 0, f"Theme file is empty: {tf}")

    def test_presentation_file_nonempty(self):
        self.assertGreater(PRESENTATION_TEX.stat().st_size, 0)

    def test_base_template_compliance(self):
        self.assertIn(r"\documentclass[aspectratio=169,xcolor=dvipsnames]{beamer}", self.raw_content)
        self.assertIn(r"\usetheme{SimpleDarkBlue}", self.raw_content)
        self.assertIn(r"\title{Unsupervised Learning of Intrinsic Semantics with Diffusion Model for Person Re-Identification}", self.raw_content)
        self.assertIn("Koushik Teja Viluvala", self.raw_content)
        self.assertIn("25EC65R13", self.raw_content)
        self.assertIn("Vision and Intelligent Systems", self.raw_content)
        self.assertIn("Indian Institute of Technology Kharagpur", self.raw_content)
        self.assertIn(r"\begin{document}", self.raw_content)
        self.assertIn(r"\titlepage", self.raw_content)
        self.assertIn(r"\tableofcontents", self.raw_content)
        self.assertIn(r"\end{document}", self.raw_content)

    def test_slide_count_strictly_under_20(self):
        begins = re.findall(r"\\begin\{frame\}", self.stripped_content)
        ends = re.findall(r"\\end\{frame\}", self.stripped_content)
        self.assertEqual(len(begins), len(ends), f"Mismatched frames: {len(begins)} begins vs {len(ends)} ends")
        self.assertLess(len(begins), 20, f"Frame count ({len(begins)}) exceeds maximum allowed (<20)")
        self.assertEqual(len(begins), 18, f"Expected exactly 18 frames per PROJECT.md design, got {len(begins)}")


class TestLatexSyntaxAndBalance(unittest.TestCase):
    """Validates delimiter balance and LaTeX environment matching."""

    def setUp(self):
        with open(PRESENTATION_TEX, "r", encoding="utf-8") as f:
            self.raw_content = f.read()
        self.stripped_content = strip_latex_comments(self.raw_content)

    def test_braces_balanced(self):
        depth = 0
        in_escape = False
        for i, ch in enumerate(self.stripped_content):
            if in_escape:
                in_escape = False
                continue
            if ch == "\\":
                in_escape = True
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                self.assertGreaterEqual(depth, 0, f"Unmatched closing brace at character {i}")
        self.assertEqual(depth, 0, f"Unclosed brace detected: final depth = {depth}")

    def test_environments_balanced(self):
        pattern = re.compile(r"\\(begin|end)\{([a-zA-Z0-9*]+)\}")
        stack = []
        for match in pattern.finditer(self.stripped_content):
            tag, env = match.groups()
            if tag == "begin":
                stack.append((env, match.start()))
            else:
                self.assertTrue(len(stack) > 0, f"Unmatched \\end{{{env}}} at position {match.start()}")
                last_env, _ = stack.pop()
                self.assertEqual(last_env, env, f"Mismatched environment: \\begin{{{last_env}}} closed by \\end{{{env}}}")
        self.assertEqual(len(stack), 0, f"Unclosed environments: {[e[0] for e in stack]}")


class TestNegativeConstraintsAndTone(unittest.TestCase):
    """Enforces zero occurrences of forbidden terms and respectful academic tone."""

    def setUp(self):
        with open(PRESENTATION_TEX, "r", encoding="utf-8") as f:
            self.raw_content = f.read()
        self.stripped_content = strip_latex_comments(self.raw_content)

    def test_forbidden_strings_absent(self):
        forbidden_list = [
            "75.5M",
            "99.3%",
            "100x",
            "100×",
            "saving >10 hours",
        ]
        for f in forbidden_list:
            self.assertNotIn(f, self.raw_content, f"Forbidden string '{f}' found in {PRESENTATION_TEX}")

    def test_derogatory_terms_absent(self):
        derogatory_list = ["bugged", "broken", "pathological"]
        for d in derogatory_list:
            self.assertFalse(
                re.search(r"\b" + re.escape(d) + r"\b", self.stripped_content, re.IGNORECASE),
                f"Derogatory term '{d}' found in presentation"
            )

    def test_refinement_framing(self):
        lower = self.stripped_content.lower()
        self.assertIn("architectural refinement", lower, "STN must be framed as architectural refinement")
        self.assertIn("sampling refinement", lower, "DDIM must be framed as sampling refinement")


class TestMathematicalContentCoverage(unittest.TestCase):
    """Verifies coverage of core problem formulation, Theorems 1-5, Prop 1, and Proposals P1-P6."""

    def setUp(self):
        with open(PRESENTATION_TEX, "r", encoding="utf-8") as f:
            self.raw_content = f.read()
        self.stripped_content = strip_latex_comments(self.raw_content)

    def test_problem_formulation(self):
        lower = self.stripped_content.lower()
        self.assertIn("unsupervised", lower)
        self.assertIn("cluster shattering", lower)
        self.assertTrue("dual-stream" in lower or "stream" in lower)
        self.assertIn("resnet-50", lower)
        self.assertTrue("spatial transformer" in lower or "stn" in lower)
        self.assertTrue("parameter-space diffusion" in lower or "18-d" in lower)

    def test_theorems_1_to_5_and_prop_1(self):
        # Theorem 1: Gradient Sign Inversion
        self.assertIn("Theorem 1", self.raw_content)
        self.assertIn(r"\frac{\partial \mathcal{L}_{\text{cont}}}{\partial l_k}", self.raw_content)
        self.assertTrue("gradient ascent" in self.stripped_content.lower() or "gradient" in self.stripped_content.lower())

        # Theorem 2: Supervision Polarity Inversion
        self.assertIn("Theorem 2", self.raw_content)
        self.assertIn("MMD", self.raw_content)
        self.assertIn("polarity", self.stripped_content.lower())

        # Theorem 3: Triplet Gradient Saturation
        self.assertIn("Theorem 3", self.raw_content)
        self.assertIn("2048", self.raw_content)
        self.assertTrue("0.0293" in self.raw_content or "saturation" in self.stripped_content.lower())

        # Theorem 4: Singular Coordinate Collapse
        self.assertIn("Theorem 4", self.raw_content)
        self.assertTrue(r"\det(\mathbf{A}) = 0.0" in self.raw_content or "singular" in self.stripped_content.lower())
        self.assertIn("22.9", self.raw_content)

        # Theorem 5: Bayes-Optimal Variance Floor
        self.assertIn("Theorem 5", self.raw_content)
        self.assertIn("1.000", self.raw_content)
        self.assertTrue("1.084" in self.raw_content or "floor" in self.stripped_content.lower())

        # Proposition 1: Guaranteed Non-Singularity
        self.assertIn("Proposition 1", self.raw_content)
        self.assertTrue("0.1958" in self.raw_content or "0.196" in self.raw_content)
        self.assertTrue("0.0%" in self.raw_content or "0.0\\%" in self.raw_content or "0%" in self.raw_content or "0\\%" in self.raw_content)

    def test_proposals_p1_to_p6(self):
        lower = self.stripped_content.lower()
        # Optimization 1 / P1: Detached Softmax
        self.assertTrue("p1" in lower or "optimization 1" in lower)
        self.assertTrue("detached" in lower or "softmax" in lower)
        # Optimization 2 / P2: Gaussian MMD Kernel
        self.assertTrue("p2" in lower or "optimization 2" in lower)
        self.assertTrue("heat kernel" in lower or "mmd" in lower)
        # Optimization 3 / P3: Hyperspherical Triplet
        self.assertTrue("p3" in lower or "optimization 3" in lower)
        self.assertTrue(r"\mathbb{s}^{d-1}" in lower or "hyperspher" in lower)
        # Optimization 4 / P4: Bounded Residual STN
        self.assertTrue("p4" in lower or "optimization 4" in lower)
        self.assertTrue("bounded residual" in lower or "529,170" in self.raw_content)
        # Optimization 5 / P5: Conditioned Denoiser
        self.assertTrue("p5" in lower or "optimization 5" in lower)
        self.assertTrue("tweedie" in lower or "denoiser" in lower)
        # Optimization 6 / P6: 5-Step DDIM Sampling
        self.assertTrue("p6" in lower or "optimization 6" in lower)
        self.assertTrue("ddim" in lower and "5-step" in lower)
        # Optimization 7 / P7: Training Protocol & Fusion
        self.assertTrue("tau_{merge}" in lower or "0.65" in self.raw_content or "fusion" in lower)


class TestEmpiricalBenchmarksAndTelemetry(unittest.TestCase):
    """Verifies calibrated benchmark results and telemetry values from Chapter 4."""

    def setUp(self):
        with open(PRESENTATION_TEX, "r", encoding="utf-8") as f:
            self.raw_content = f.read()
        self.stripped_content = strip_latex_comments(self.raw_content)

    def test_master_benchmarks(self):
        # Market-1501
        self.assertIn("Market-1501", self.raw_content)
        self.assertIn("85.6", self.raw_content)  # Optimized base mAP
        self.assertIn("94.2", self.raw_content)  # Optimized base R1
        self.assertIn("89.8", self.raw_content)  # Optimized RR mAP

        # VeRi-776
        self.assertIn("VeRi-776", self.raw_content)
        self.assertIn("52.4", self.raw_content)  # Optimized base mAP
        self.assertIn("96.1", self.raw_content)  # Optimized base R1
        self.assertIn("54.1", self.raw_content)  # Optimized RR mAP

    def test_ablation_and_telemetry(self):
        # A0 through A8
        self.assertIn("A0", self.raw_content)
        self.assertIn("A8", self.raw_content)
        # Telemetry
        self.assertIn("0.418", self.raw_content)  # Noise loss
        self.assertTrue("96.5%" in self.raw_content or "96.5\\%" in self.raw_content)  # VRAM reduction
        self.assertIn("529,170", self.raw_content)  # MLP parameters
        self.assertIn("618", self.raw_content)  # VeRi clusters


class TestGraphicsAndSourceCleanliness(unittest.TestCase):
    """Verifies graphic assets and untouched source directories."""

    def setUp(self):
        with open(PRESENTATION_TEX, "r", encoding="utf-8") as f:
            self.raw_content = f.read()
        self.stripped_content = strip_latex_comments(self.raw_content)

    def test_graphics_path_and_assets(self):
        self.assertIn(r"\graphicspath{{../figs/}{../examples/visualizations/}}", self.raw_content)
        self.assertIn("overview.png", self.raw_content)
        self.assertIn("heatmaps", self.raw_content)
        self.assertIn("tsne", self.raw_content)

        # Check files exist
        self.assertTrue((REPO_ROOT / "figs" / "overview.png").exists())
        self.assertTrue((REPO_ROOT / "examples" / "visualizations" / "heatmaps" / "heatmap_0_pid_1271.png").exists())
        self.assertTrue((REPO_ROOT / "examples" / "visualizations" / "veri" / "heatmaps" / "heatmap_2_pid_65.png").exists())
        self.assertTrue((REPO_ROOT / "examples" / "visualizations" / "tsne" / "tsne_plot.png").exists())
        self.assertTrue((REPO_ROOT / "examples" / "visualizations" / "veri" / "tsne" / "tsne_plot.png").exists())

    def test_protected_directories_clean(self):
        res = subprocess.run(
            ["git", "status", "--porcelain"] + PROTECTED_DIRS,
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=True
        )
        self.assertEqual(res.stdout.strip(), "", f"Protected directories modified:\n{res.stdout}")


if __name__ == "__main__":
    unittest.main()
