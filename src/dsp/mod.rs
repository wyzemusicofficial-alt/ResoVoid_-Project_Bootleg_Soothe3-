pub mod analysis;
pub mod bands;
pub mod detector;
pub mod filters;
pub mod suppressor;

pub use analysis::AnalysisFrame;
pub use suppressor::ResonanceSuppressor;

pub const BANDS: usize = 64;

/// Maximum number of user node slots. Slots are preallocated as fixed-identity
/// parameters (required by the host automation/recall model); a slot with
/// `enabled == false` is "deleted" and contributes zero weight.
pub const MAX_NODES: usize = 8;
