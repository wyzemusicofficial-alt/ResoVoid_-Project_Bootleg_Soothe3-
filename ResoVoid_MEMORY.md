# MEMORY.md — Persistent Project Memory (nih-plug workspace)

> Loaded automatically as core context. Keep concise; update when build constraints change.

## Workspace layout
- `nih_plug template/` — minimal ResoVoid template (truce 6.3 + truce-slint, no Slint UI deps yet).
- `ResoVoid (Project Bootleg Soothe2)/` — dynamic resonance suppressor (Soothe2-inspired). Full UI, spectrum-analyzer, ringbuf.
- `spectral_orgasm/` — effect plugin, migrated to `nice-plug 0.3` + `slint 1.16.1` + `slint-baseview 0.1`; bench/installer retained, `truce.toml` removed.

## Core build constraints
- **Framework:** `spectral_orgasm` on `nice-plug 0.3` (`editor`+`vst3`+`tracing-subscriber`, `default-features = false`); `ResoVoid`/`template` still on `truce 6.3` + `truce-slint 6.3`. Backends no longer gated by `clap`/`vst3` features in `spectral_orgasm`.
- **Crate types:** `cdylib` + `rlib` in all plugins.
- **Rust edition:** 2021 everywhere.
- **Slint:** `spectral_orgasm` pinned to `=1.16.1` with `["compat-1-2","std"]` + `slint-baseview 0.1` + dual `raw-window-handle 0.5/0.6` bridge (nice-plug 0.6 handle → baseview 0.5); `ResoVoid`/`template` still on `=1.15.1` `renderer-software`. Do not bump without checking baseview compat.
- **UI sources** live in each crate's `ui/` dir; `spectral_orgasm` compiles via `slint-build =1.16.1` in `build.rs`, others via `truce-slint-build`.
- **Real-time audio rules:** `assert_no_alloc` in dev-dependencies — the process path must be allocation-free. Use `ringbuf 0.3` for cross-thread spectrum/visualization data, never locks on the audio thread.
- **DSP crates in use:** `realfft 3.5` + `plotters 0.3` (`bitmap_backend`, RGB buffer → Slint `Image::from_rgb8`) for spectrum in `spectral_orgasm`; `spectrum-analyzer`, `num-complex` in `ResoVoid`.
- **Profiles:** release uses `lto = "thin"` + `strip = "symbols"`; a `profiling` profile inherits release but keeps debug info (`strip = "none"`).
- **Benchmarks:** criterion (`0.7.0` template/spectral_orgasm, `0.5` ResoVoid) under `benches/`.

## Build commands
- `spectral_orgasm` (nice-plug): `cargo build --release` emits `target/release/spectral_orgasm.dll` + `spectral_orgasm_clap.dll` (install as `.clap`) + `spectral_orgasm_vst3.dll` (install into `*.vst3/Contents/x86_64-win/*.vst3`); standalone UI via `cargo run --bin standalone_ui --release`.
- `ResoVoid`/`template` (truce): `cargo xtask bundle <crate>` or plain `cargo build --release`.
- Windows installer for spectral_orgasm: `installer.iss` (Inno Setup, expects `target/release/*_{clap,vst3}.dll`), output in `installer_output/`; `.cargo/config.toml` xtask alias removed (no truce bundler).

## Conventions
- Author: Wyz3Music; license GPL-3.0 (only/or-later per crate — check Cargo.toml before redistributing).
- Each subdirectory is its own independent cargo project with its own lockfile/target — run cargo commands from within the specific crate directory.
