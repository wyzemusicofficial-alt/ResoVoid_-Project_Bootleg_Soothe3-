// src/dsp/mod.rs
// Digital Signal Processing modules for ResoVoid.

pub mod analysis;
pub mod bands;
pub mod detector;
pub mod filters;
pub mod suppressor;

pub use analysis::AnalysisFrame;
pub use suppressor::ResonanceSuppressor;

/// Number of analysis/reduction bands.
pub const BANDS: usize = 64;
