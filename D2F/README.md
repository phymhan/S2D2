# D2F with S2D2

This folder contains a standalone demo integration of **S2D2** on top of **D2F-style block-diffusion decoding** for two model families:

- `llada` (LLaDA-based)
- `dream` (Dream-based)

## Setup

Install dependencies by following the official instructions of the corresponding upstream projects (D2F + model family).  
In this subfolder, we only provide the core modified files and runnable demo script.

## Quick Start

Run:

```bash
python example_d2f.py
```

### Select model family

Use `--model` to choose backend:

```bash
python example_d2f.py --model llada
python example_d2f.py --model dream
```

### Select decoding mode

Use `--generate_fn`:

- `d2f`: original D2F decoding
- `ssd`: S2D2 decoding (self-speculative)

```bash
python example_d2f.py --model llada --generate_fn d2f
python example_d2f.py --model llada --generate_fn ssd
```

> Note: in this release, S2D2 currently supports the **minimum-span policy** (`--min_ssd_span_length`).

## Important Dream Stability Notes

For Dream models, S2D2 is most reliable with:

- `--allow_resample false`
- draft-confidence thresholding enabled:
  - `--ssd_threshold_draft_confidence true`
  - `--draft_confidence_threshold 0.5`

Recommended command:

```bash
python example_d2f.py \
  --model dream \
  --generate_fn ssd \
  --allow_resample false \
  --ssd_threshold_draft_confidence true \
  --draft_confidence_threshold 0.5
```

## Useful S2D2 Flags

- `--min_ssd_span_length`: minimum draft span length to verify
- `--target_block_size`: AR-mode block size for verification (`1` means token-wise AR verifier)
- `--cache_ver` / `--draft_ver`: enable verifier/drafter cache options
- `--forward_stats true`: print decoding statistics for debugging/analysis

## Acknowledgement

This implementation builds on open-source code and model ecosystems from D2F, LLaDA, and Dream.  
We thank the original authors for sharing their code and models.
