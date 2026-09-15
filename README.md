# ResoVoid

Dynamic Resonance Suppressor - A Soothe2-inspired spectral processing plugin built with Rust and nice-plug.

## Features

- **Real-time spectral processing** using STFT-style analysis (Hann-windowed real FFT, 50%-overlap windows) to drive a per-band reduction
- **Intelligent resonance detection** via a spectral-median reference (13-band window) with a selectivity-dependent threshold — sustained peaks stay flagged, not just onsets
- **Per-band dynamic suppression** with attack/release timing applied to both the baseline and the gain reduction
- **8 Soothe-style nodes** (freq × depth, Bell/Low/High-Shelf) scaling depth per region, with draggable spectrum markers
- **Stereo Link / Independent** coupling, **2× oversampling** (synthesis path), **FFT-size select** (1024–8192), **named JSON presets**
- **Soft/Hard modes** for different suppression characteristics
- **Delta monitoring** to hear only what's being removed
- **Modern GUI** with real-time spectrum visualization
- **Look-ahead latency**: the dry signal is delayed by exactly one analysis window so gain reduction is applied to the audio that produced the spectrum (reported latency matches this)

## Parameters

| Parameter | Description | Range |
|-----------|-------------|-------|
| Depth | Overall sensitivity of resonance detection | 0-100% |
| Sharpness | Width of suppression notches (Q factor) | 0-100% |
| Selectivity | How picky the detection algorithm is | 0-100% |
| Attack | How fast resonances are suppressed | 0.1-50ms |
| Release | How fast suppression releases | 10-500ms |
| Mode | Soft (smooth) vs Hard (aggressive) | Toggle |
| Delta | Listen to removed content only | Toggle |
| FFT Size | Analysis window (1024/2048/4096/8192) | Select |
| Stereo Link | Linked (min per band) vs Independent | Toggle |
| Oversampling | 2× on biquad synthesis path | Toggle |
| Nodes 1–8 | Per-region depth × freq, Bell/Shelf, enable | Drag markers / params |
| Mix | Dry/wet blend | 0-100% |
| Output | Output gain | ±12dB |

## Building

After installing [Rust](https://rustup.rs/), you can compile ResoVoid as follows:

```shell
cargo xtask bundle resovoid --release
```

This will create VST3 and CLAP plugins in the `target/bundled` directory.

## Technical Details

### Spectral Processing Pipeline

1. **STFT Analysis**: Audio fills a ring buffer; every hop (half the FFT size) the latest window is Hann-windowed and transformed with a real FFT (`realfft`).
2. **Resonance Detection**: Each band's level is compared against a spectral-median reference (median of ±6 neighboring bands at the same instant, lightly smoothed over time). Excess above `reference + threshold` (threshold shrinks with Selectivity), scaled by the node-shape depth multiplier at that frequency, triggers downward gain reduction.
3. **Gain Calculation**: The excess is scaled by Depth/Sharpness into a reduction in dB, converted to a linear per-band gain (≤ 1.0).
4. **Temporal Dynamics**: Attack/Release ballistics are applied to both the baseline follower and the final gain, preventing zipper noise when coefficients change.
5. **Gain Synthesis**: There is **no inverse FFT**. The actual audio path is a cascade of 64 peaking biquads (one per log-spaced band) whose coefficients are updated every analysis hop (twice per window). The dry signal is delayed by one analysis window (look-ahead) so the reduction lands on the audio that produced the spectrum. AIUI: analysis and synthesis are fully decoupled — no IFFT/overlap-add reconstruction exists anywhere in the codebase.

### FFT Settings

| FFT Size | Latency @44.1kHz | Frequency Resolution |
|----------|------------------|---------------------|
| 1024 | ~23ms | ~43Hz |
| 2048 | ~46ms | ~22Hz |
| 4096 | ~93ms | ~11Hz |
| 8192 | ~186ms | ~5Hz |

### Design Notes

- **50%-overlap analysis windows**: The analysis stage keeps a ring buffer and emits a Hann-windowed spectrum every hop = FFT-size / 2 samples (Hann @ 50% is COLA). Detection therefore refreshes twice per window (e.g. ~23 ms at 2048/44.1 kHz). Output-path latency is unchanged — it is set by the delay line (one full window), not by the hop.
- **Latency is genuine**: the plugin reports exactly one analysis window of latency to the host, because the dry signal is buffered by that amount (see look-ahead above). Host delay compensation will therefore stay correct.

## License

GPL-3.0 (required for VST3 compatibility)
