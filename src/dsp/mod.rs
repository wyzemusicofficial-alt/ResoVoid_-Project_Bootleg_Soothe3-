pub mod analysis;
pub mod bands;
pub mod detector;
pub mod filters;
pub mod suppressor;

pub use analysis::AnalysisFrame;
pub use suppressor::ResonanceSuppressor;

pub const BANDS: usize = 64;
