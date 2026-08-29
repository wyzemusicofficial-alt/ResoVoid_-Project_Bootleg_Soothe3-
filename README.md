# ResoVoid

Dynamic Resonance Suppressor - A Soothe2-inspired spectral processing plugin built with Rust and NIH-plug.

## Features

- **Real-time spectral processing** using STFT-style analysis (Hann-windowed real FFT, non-overlapping windows) to drive a per-band reduction
- **Intelligent resonance detection** via a slow spectral baseline follower with a selectivity-dependent threshold
- **Per-band dynamic suppression** with attack/release timing applied to both the baseline and the gain reduction
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

1. **STFT Analysis**: Audio is buffered into non-overlapping windows, Hann-windowed, and transformed to the frequency domain with a real FFT (`realfft`).
2. **Resonance Detection**: Each band's level is compared against a slow-moving baseline (attack/release follower). Excess above `baseline + threshold` (threshold shrinks with Selectivity) triggers downward gain reduction.
3. **Gain Calculation**: The excess is scaled by Depth/Sharpness into a reduction in dB, converted to a linear per-band gain (≤ 1.0).
4. **Temporal Dynamics**: Attack/Release ballistics are applied to both the baseline follower and the final gain, preventing zipper noise when coefficients change.
5. **Gain Synthesis**: There is **no inverse FFT**. The actual audio path is a cascade of 64 peaking biquads (one per log-spaced band) whose coefficients are updated once per analysis window. The dry signal is delayed by one analysis window (look-ahead) so the reduction lands on the audio that produced the spectrum. AIUI: analysis and synthesis are fully decoupled — no IFFT/overlap-add reconstruction exists anywhere in the codebase.

### FFT Settings

| FFT Size | Latency @44.1kHz | Frequency Resolution |
|----------|------------------|---------------------|
| 1024 | ~23ms | ~43Hz |
| 2048 | ~46ms | ~22Hz |
| 4096 | ~93ms | ~11Hz |
| 8192 | ~186ms | ~5Hz |

### Design Notes

- **Non-overlapping analysis windows**: The analysis stage collects one full FFT-size block before computing a spectrum and updating gains (no hop/overlap-add). This is a deliberate tradeoff — it is cheaper and the per-band biquad cascade smooths gain changes between updates. Detection therefore refreshes once per window (e.g. ~46 ms at 2048/44.1 kHz) rather than every hop. Overlapping analysis is a possible future enhancement but is intentionally not implemented here.
- **Latency is genuine**: the plugin reports exactly one analysis window of latency to the host, because the dry signal is buffered by that amount (see look-ahead above). Host delay compensation will therefore stay correct.

## License

GPL-3.0 (required for VST3 compatibility)
