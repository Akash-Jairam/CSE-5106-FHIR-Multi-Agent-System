V3 consolidated results
Branch: results/v3-ful-run

Start with ANALYSIS.docx, the editable detailed handoff, or ANALYSIS.txt for
plain-text review. METHODS.txt defines cohort selection, measurement and
statistical interpretation. RUN_REGISTRY.json maps the four logical run records
to their source commits, evidence packages and derived tables. The primary full
benchmark comparison reports 2,929 completed task pairs from 94 patients.
comparison_summary.csv gives both systems' accuracy, time, tokens and estimated
cost on exactly those question IDs; use it for report tables and figures.

development/
  The paired 300-question development runs, with per-question measurements,
  scores, paired transitions, mechanism/phase summaries, source comparison and
  the original raw result archive split into Git-compatible parts.

full_v3/
  Portable canonical results and owned audit payloads for all 2,931 accepted V3
  questions. The original complete numerical delivery is preserved at
  benchmarks/v3-full-run-20261015/completed_v3/ in this same branch.

full_single_agent/
  Measurements and typed scores for 2,929 completed Single Agent questions,
  the matched 2,929 V3 rows, paired and subgroup summaries, and portable raw
  results. paired_comparison.json reports Single Agent minus V3. The main
  handoff reports V3 minus Single Agent. These 2,929 pairs form the primary
  full benchmark comparison. Pairing is posthoc and based on completion,
  without selecting by correctness.

internal/
  Operational provenance for internal development and reproduction. It is not
  part of the scientific design findings in the handoff.

Reproduction
  Read each archive manifest and reconstruction guide before loading raw data.
  Numeric CSV and JSONL tables can be used directly without restoring large raw
  files. JSON null and blank CSV cells mean unavailable, never zero. The package
  manifest inventories all consolidated artifacts with SHA256 checksums.
  From the repository root, verify all manifest leaves with:
    python3 results/v3/reproduce/verify_package.py

Scope
  These are four arm records, not four independent repetitions. The development
  questions overlap the full benchmark and must not be pooled with it. V3's
  original execution archive contains 2,931 questions. The report uses the
  same 2,929 completed pairs for both systems; run_summary.csv retains the
  original run counts and comparison_summary.csv contains reporting figures.
  Costs are estimates under recorded frozen rates; see METHODS.txt. Source
  fingerprints differ by a documented host-loop bound. This branch consolidates
  results and analysis without merging development implementation code.
