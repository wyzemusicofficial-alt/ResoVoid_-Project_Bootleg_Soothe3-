# MEMORY.md — Persistent Project Memory (ResoVoid)

> Loaded automatically as core context. Keep concise; update when build constraints change.

## What this project is
- `ResoVoid` — dynamic resonance suppressor (Soothe2-inspired), single-crate
  nih-plug workspace (`Cargo.toml` + `xtask/` bundler). VST3 + CLAP.
- DSP: 64 log-spaced bands → per-band detector → cascade of 64 peaking
  biquads (no IFFT). GUI: egui spectrum panel + 8 draggable node markers.

## Core build constraints
- **Framework:** `nice-plug 0.3` + `nice-plug-egui 0.4` + `egui 0.36`
  (`default_fonts`). Plugin exports via `nice_export_clap!` / `nice_export_vst3!`.
- **Crate types:** `cdylib` + `rlib`. **Edition:** 2021. License GPL-3.0-only.
- **DSP crates:** `realfft 3` + `num-complex 0.4` (Hann-windowed forward FFT
  analysis only); `rtrb 0.3.3` for audio→GUI `AnalysisFrame` ring (cap 16,
  never locks on the audio thread); `atomic_float 1.1` for shared sample rate.
- **Presets:** `serde 1` + `serde_json 1`, GUI-thread only
  (`%APPDATA%\ResoVoid\presets` on Windows). Flat param-ID → value maps.
- **Real-time rules:** audio path must stay allocation-free
  (`assert_process_allocs` feature → `nice-plug/assert_process_allocs`);
  fixed-size stack arrays only per frame (incl. 13-band spectral median).
  FTZ/DAZ asserted per `process()` block; smoothers advance per sample.
- **Profiles:** release `lto = "thin"` + `strip = "symbols"`; `profiling`
  profile inherits release with debug info.

## DSP architecture (current)
- `src/dsp/analysis.rs` — 50%-overlap STFT front-end (`hop = fft/2`, Hann COLA),
  frame emitted every hop; detector/viz refresh 2× per window.
- `src/dsp/detector.rs` — spectral-median reference (`MEDIAN_HALF_WINDOW = 6`,
  13-band window) + light temporal smoothing of the *median*; `excess =
  level - (reference + threshold)`. 8 node slots (freq/depth/shape/enable)
  scale depth via shared Gaussian kernel (`σ = 1.0` octave, Bell/LowShelf/
  HighShelf) — single source of truth `node_shape()`, also used by GUI curve.
- `src/dsp/filters.rs` — `BandProcessor`: 64 peaking biquads, per-hop
  coefficient updates; optional 2× oversampling on synthesis path only.
- `src/dsp/suppressor.rs` — per-channel state (analysis + detector + bands +
  `fft_size`-sample look-ahead delay line); stereo linked (`min` per band) or
  independent; FFT-size/sample-rate switches crossfaded over 512 samples;
  reported latency = exactly one FFT window. `FFT_SIZES = [1024, 2048, 4096, 8192]`.
- `src/lib.rs` — 8 node slots are fixed-identity params (`nf0-7`, `node0-7`,
  `nen0-7`, `nsh0-7`; delete = `enabled=false`); `slnk`, `os2x`, `fft`, `delt`,
  `mix`, `out` alongside Depth/Sharpness/Selectivity/Attack/Release.

## GUI (`src/gui.rs`, `src/presets.rs`)
- Soothe3-style spectrum panel: input spectrum + per-band GR tint (cyan→violet),
  shape curve from `node_shape()`, draggable node markers (freq × depth),
  dark/light themes, preset bar (save/load JSON, throttled refresh).
- Tests live next to code (`detector`, `suppressor`, `filters`, `gui`, `presets`
  — 54 passing at last check). Run `cargo test` before push.

## Build commands (run from crate root)
- `cargo test` — fast unit tests (no plugin host needed).
- `cargo xtask bundle resovoid --release` — VST3 + CLAP in `target/bundled`.
- `cargo build --features assert_process_allocs` — verify alloc-free process path.
