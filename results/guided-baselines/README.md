# Guided-baseline result files

The raw per-arm results behind [Guided baselines](../../docs/results/guided-baselines.md), compressed with zstd. Each `<arm>.json.zst` is one arm's results file, saved traces included, exactly as the run wrote it. Each `<arm>.json.run.json` beside it records the run's settings. `SHA256SUMS` lists checksums of the uncompressed files. All runs used commit cdbc037 (code fingerprint `78f297746133`).

| Directory | Question set | Arms | Jobs (gpt-oss:120b / qwen3:32b) |
| --- | --- | --- | --- |
| `<model>/guided_dev/` | Preflight: 50 dev questions | the two guided arms | 4600140 / 4600141 |
| `<model>/guided_held_out/` | Stage 1: the frozen 300 (`manifests/split.json`) | all five | 4600217 / 4600265 |
| `<model>/guided_held_out_v2/` | Stage 2: the fresh 300 (`manifests/pools/held_out_v2.json`) | all five | 4606603 / 4606604 |

`<model>` is `gpt-oss-120b` or `qwen3-32b`. The runs wrote these files to `output/experiments/gpt-oss:120b/` and `output/experiments/qwen3:32b/`.

The multi-agent files need a 128 MB decompression window:

```bash
cd results/guided-baselines
zstd -d --long=27 gpt-oss-120b/guided_held_out_v2/*.json.zst
sha256sum -c SHA256SUMS --ignore-missing   # macOS: shasum -a 256 -c
```

In Python, use `zstandard.ZstdDecompressor(max_window_size=2**27)`. To re-run the analysis, decompress a directory and pass it as `--output_dir` to `scripts/run_guided_baselines.py analyze` with the flags in the write-up's Reproducing section.
