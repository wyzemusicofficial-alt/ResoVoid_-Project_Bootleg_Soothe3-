# ResoVoid

Dynamic Resonance Suppressor - A Soothe2-inspired spectral processing plugin built with Rust and NIH-plug.

## Features

- **Real-time spectral processing** using STFT with Hann windowing and overlap-add reconstruction
- **Intelligent resonance detection** with median-filtered threshold calculation
- **Per-bin dynamic suppression** with attack/release timing
- **Perceptual weighting** (A-weighted) for enhanced detection in the 2-5kHz harshness region
- **Soft/Hard modes** for different suppression characteristics
- **Delta monitoring** to hear only what's being removed
- **Modern GUI** with real-time spectrum visualization

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

1. **STFT Analysis**: Audio is windowed (Hann), transformed to frequency domain
2. **Resonance Detection**: Per-bin magnitude compared against median-filtered threshold
3. **Gain Calculation**: Excess above threshold triggers suppression based on Depth/Sharpness
4. **Temporal Dynamics**: Attack/Release applied to gain reduction per bin
5. **STFT Synthesis**: Inverse transform with overlap-add for artifact-free reconstruction

### FFT Settings

| FFT Size | Latency @44.1kHz | Frequency Resolution |
|----------|------------------|---------------------|
| 1024 | ~23ms | ~43Hz |
| 2048 | ~46ms | ~22Hz |
| 4096 | ~93ms | ~11Hz |
| 8192 | ~186ms | ~5Hz |

## License

GPL-3.0 (required for VST3 compatibility)
