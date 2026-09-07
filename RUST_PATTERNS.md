# Rust Patterns Reference

Patterns extracted from the Bit-Melt codebase and related transcripts. Each pattern includes what it is, why it matters for audio DSP, and how it appears in this project.

---

## 1. Enum-Driven State Machines

**What it is:** Represent plugin lifecycle and message types as enums with explicit variants. The compiler enforces that all cases are handled.

**Why it matters:** Audio plugins have discrete states (background tasks, SysEx messages, error variants). Enums prevent forgetting a case and make the state space visible.

**In this project:**

```rust
// lib.rs — background tasks dispatched off the RT thread
pub enum BitMeltTask {
    LoadScales { path: PathBuf },
}

// svf.rs — filter errors with explicit variants
#[derive(Debug, Clone, PartialEq)]
pub enum SvfError {
    NegativeFreq,
    FreqAboveNyquist,
    NegativeQ,
}

// sysex.rs — SysEx protocol messages
#[derive(Debug, Clone, PartialEq)]
pub enum BitMeltSysEx {
    UpdateMatrixCell { row: u8, col: u8, weight: f32 },
}
```

**Rule:** All domain-specific message types, error types, and lifecycle phases should be enums. Never use `u32` constants or string tags where an enum fits.

---

## 2. Newtype Validation

**What it is:** Wrap primitives in single-field structs with a `parse()` constructor that validates. Invalid values become unrepresentable at compile time.

**Why it matters:** Audio parameters have valid ranges (freq 20..20000, Q 0.1..12). A raw `f32` can hold NaN, negative, or out-of-range values that silently break DSP math. Newtypes enforce correctness at construction.

**In this project:**

```rust
// types.rs
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Frequency(f32);
impl Frequency {
    pub fn parse(v: f32) -> Result<Self, &'static str> {
        if !v.is_finite() { return Err("non-finite freq"); }
        if v < 20.0 || v > 20000.0 { return Err("freq out of 20..20000"); }
        Ok(Self(v))
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QFactor(f32);
impl QFactor {
    pub fn parse(v: f32) -> Result<Self, &'static str> {
        if !v.is_finite() || v < 0.1 || v > 12.0 { return Err("Q out of 0.1..12"); }
        Ok(Self(v))
    }
}
```

**Rule:** Use newtypes for any value that has domain-specific constraints. Provide `parse()` returning `Result` and `new_clamped()` for ergonomics. Mark getters with `#[must_use]`.

---

## 3. Prelude Module

**What it is:** A `prelude.rs` file with curated `pub(crate)` re-exports so other modules write `use crate::prelude::*` instead of repeating the same imports.

**Why it matters:** Reduces import noise across the crate. Keeps dependency declarations in one place.

**In this project:**

```rust
// prelude.rs
pub(crate) use std::sync::Arc;
pub(crate) use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
pub(crate) use nice_plug::prelude::*;
pub(crate) use nice_plug::wrapper::vst3::subcategories::Vst3SubCategory;
pub(crate) use nice_plug_egui::{create_egui_editor, EguiState};
pub(crate) use parking_lot::Mutex;
pub(crate) use triple_buffer::{Input, Output, TripleBuffer};
```

**Rule:** Keep the prelude to `pub(crate)` only. Never re-export the entire crate -- select only what most modules need.

---

## 4. Macro-Driven Boilerplate Reduction

**What it is:** Define macros for repetitive patterns, especially parameter construction where the same `FloatParam::new(...).with_unit(...).with_smoother(...)` chain appears dozens of times.

**Why it matters:** Parameter boilerplate is error-prone and obscures the actual parameter definition. Macros keep each param to one line.

**In this project:**

```rust
// params.rs
macro_rules! float_param {
    ($name:expr, $default:expr, $min:expr, $max:expr, $unit:expr) => {
        FloatParam::new($name, $default, FloatRange::Linear { min: $min, max: $max })
            .with_unit($unit)
            .with_smoother(SmoothingStyle::Linear(20.0))
    };
}
macro_rules! skewed_freq_param {
    ($name:expr, $default:expr) => {
        FloatParam::new($name, $default,
            FloatRange::Skewed { min: 20.0, max: 20000.0, factor: FloatRange::skew_factor(-1.0) },
        ).with_unit(" Hz")
    };
}
```

**Rule:** Define a macro when the same 3+ line pattern repeats for more than 3 parameters. Keep macro parameters flat -- no nested match arms.

---

## 5. `#[must_use]` on Important Returns

**What it is:** Annotate functions whose return value must not be ignored with `#[must_use]`. The compiler warns if the caller drops the value.

**Why it matters:** In DSP code, forgetting to use a validated value or computed result silently breaks audio.

**In this project:**

```rust
// types.rs
#[must_use]
pub fn get(self) -> f32 { self.0 }

// svf.rs
#[must_use]
pub fn tick(&mut self, v0: f32) -> f32 { ... }
```

**Rule:** Add `#[must_use]` to: newtype getters, `parse()` constructors, `tick()`/`process_sample()` methods, and any function where ignoring the return value is almost certainly a bug.

---

## 6. `#[inline]` on Hot Functions

**What it is:** Mark functions called per-sample or per-bin with `#[inline]` so the compiler inlines them into the calling loop, eliminating call overhead.

**Why it matters:** In audio DSP, `tick()` and `process_sample()` are called tens of thousands of times per second. A single extra function call per sample adds measurable latency at small buffer sizes (32-128 samples).

**In this project:**

```rust
// dsp.rs — called per feedback path sample
#[inline]
pub fn flush_denormal(v: f32) -> f32 {
    if v.is_subnormal() { 0.0 } else { v }
}

// chroma.rs — called once per sample
#[inline]
pub fn process_sample(&mut self, x: f32, s: &ChromaSettings) -> f32 { ... }

// svf.rs — called once per sample
#[inline]
pub fn tick(&mut self, v0: f32) -> f32 { ... }
```

**Rule:** Apply `#[inline]` to: `flush_denormal`, `tick()`, `process_sample()`, `push()` on delay lines, and any function called in the innermost sample loop. Do NOT inline large functions (like `run_frame`) -- let the compiler decide.

---

## 7. Denormal Flushing

**What it is:** Subnormal floating-point values (very small numbers near zero) cause 10-100x performance penalties on x86 CPUs because they trigger microcode assists. In feedback paths, denormals accumulate naturally as signals decay.

**Why it matters:** A plugin that works fine at loud volumes can start stuttering at quiet volumes due to denormal buildup. This is the single most common cause of real-time audio thread performance cliffs.

**In this project:**

```rust
// dsp.rs — extracted helper, called unconditionally on feedback paths
#[inline]
pub fn flush_denormal(v: f32) -> f32 {
    if v.is_subnormal() { 0.0 } else { v }
}

// maim.rs — used on spectral feedback path
let prev = dsp::flush_denormal(prev);
x[k] += feedback * prev;
x[k] = dsp::flush_denormal(x[k]);
```

**Rules:**
- Call `flush_denormal()` unconditionally (not guarded by a threshold check) on every feedback path.
- Never use `#[target_feature]` for denormal flushing -- it is not portable and does not work on all backends.
- If a function writes to a state variable that feeds back (e.g., `ic1eq`, `prev_spectrum`), flush after writing.

---

## 8. Lock-Free Triple Buffering

**What it is:** A data structure that provides a single-writer, single-reader lock-free channel for passing data between threads. The writer always has a buffer to write to (no blocking), and the reader always gets the latest complete frame.

**Why it matters:** The audio thread cannot block on a mutex. GUI threads need to send data (visualizer frames, matrix updates) to the audio thread without locks. Triple buffers solve this cleanly.

**In this project:**

```rust
// lib.rs — visualizer data flow: audio writes, GUI reads
viz_producer: Input<VisualizerFrame>,       // audio thread writes here
viz_consumer: Arc<Mutex<Output<VisualizerFrame>>>, // GUI thread reads here

// lib.rs — matrix data flow: GUI writes, audio reads
matrix_producer: Arc<Mutex<Input<[[f32; 32]; 32]>>>, // GUI/SysEx writes
matrix_consumer: Output<[[f32; 32]; 32]>,             // audio reads lock-free

// In process():
let matrix = *self.matrix_consumer.read(); // zero-cost, no lock

// In process():
self.viz_producer.write(VisualizerFrame { ... });
```

**Rules:**
- Audio thread always reads, never blocks. If the writer hasn't updated, the reader gets the last complete frame.
- GUI thread writes via `try_lock()` on the mutex -- never blocks the audio thread.
- For one-directional data (audio -> GUI), use `Input`/`Output` directly. For bi-directional (matrix), wrap the writer in `Arc<Mutex<>>`.

---

## 9. Precomputed Lookup Tables

**What it is:** Move expensive computations (trig, frequency mapping) to initialization time or param-change time. Store results in arrays that are indexed at runtime.

**Why it matters:** `cos()`, `sin()`, `log2()`, and `powf()` are software routines with no hardware instruction on most CPUs. Computing them 2048 times per frame (once per MDCT bin) is the dominant cost in many audio DSP algorithms.

**In this project:**

```rust
// maim.rs — precomputed MDCT cosine table (computed once in new())
let angle = (PI / n as f32) * (i as f32 + 0.5 + half as f32 * 0.5) * (k as f32 + 0.5);
mdct_table[k * n + i] = angle.cos(); // computed once, looked up 2048x per frame

// chroma.rs — precomputed pitch-class remap LUT
fn recompute_bin_remap_lut(&mut self, s: &ChromaSettings, sr: f32, fft_size: usize) {
    for k in 0..SPECTRAL_BINS {
        // one-time trig per bin instead of per-frame
        self.bin_remap_lut[k] = k_target;
    }
}

// dsp.rs — precomputed Bark tables (behind feature flag)
pub struct BarkCache {
    pub bin_to_bark: [f32; MDCT_BINS],
    pub spreading_matrix: [[f32; MDCT_BINS]; MDCT_BINS],
}
```

**Rules:**
- LUTs are recomputed on param change or sample-rate change, never in the per-sample loop.
- Store LUTs as fixed-size arrays (`[f32; N]`) when `N` is known at compile time.
- Use a dirty flag (compare current param to cached param) to avoid redundant recomputation.

---

## 10. Pre-Allocated Buffers

**What it is:** Allocate all buffers in `new()` or `initialize()`. Never allocate in `process_sample()` or `run_frame()`. Use stack-allocated arrays (`[f32; N]`) when size is small and known.

**Why it matters:** Heap allocation (`Vec::new()`, `vec![]`) can take microseconds and may block on a lock. The audio thread has a strict budget (e.g., 32 samples at 48kHz = 0.67ms). Even one allocation per frame can cause xruns.

**In this project:**

```rust
// chroma.rs — all buffers allocated in new(), never in process_sample()
input_1024: [f32; FFT_SIZE],     // stack array
spec_1024: Vec<Complex<f32>>,     // heap, but allocated once
out_time_1024: Vec<f32>,          // heap, allocated once

// maim.rs — stack arrays for per-frame temporaries
let mut windowed = [0.0f32; MDCT_WINDOW];  // stack, not heap
let mut x = [0.0f32; MDCT_BINS];           // stack
let mut imdct = [0.0f32; MDCT_WINDOW];     // stack

// dsp.rs — DelayRing and OverlapAdd allocate in new()
pub struct DelayRing {
    buf: Vec<f32>,  // allocated once in new(cap)
    cap: usize,
    write: usize,
}
```

**Rules:**
- Use `[f32; N]` for frame-sized buffers (N <= 2048).
- Use `Vec<f32>` allocated in `new()` for variable-size buffers.
- Never use `.collect()`, `.map().collect()`, `String::new()`, or format!() in the audio callback.
- If you must allocate on the RT path, use `Vec::with_capacity()` and reuse the allocation across frames.

---

## 11. Early Return Error Handling

**What it is:** Validate inputs at function entry and return early on invalid data. Never unwrap or panic in the audio callback.

**Why it matters:** Panics unwind across the host's FFI boundary, causing undefined behavior and often a full DAW crash. Even `unwrap()` on a mutex lock can deadlock if the GUI thread holds the lock during a parameter change.

**In this project:**

```rust
// lib.rs — early return on empty buffer
if buffer.is_empty() || buffer.channels() == 0 {
    return ProcessStatus::Normal;
}

// lib.rs — sanitize NaN/Inf on input
let x_l = if x_l.is_finite() { x_l } else { 0.0 };

// maim.rs — clamp to prevent runaway
let feedback = s.feedback.clamp(0.0, 0.95);

// svf.rs — Result-based error propagation
pub fn set_params(&mut self, params: ToneParams) -> Result<(), SvfError> {
    if !p.freq.is_finite() || !p.q.is_finite() {
        return Err(SvfError::NegativeFreq);
    }
    // ...
}
```

**Rules:**
- Never `.unwrap()` in `process()` or `tick()`.
- Use `try_lock()` instead of `lock()` on the audio thread.
- Sanitize every input: `is_finite()` check, `.clamp()` to valid range.
- Return `ProcessStatus::Normal` for recoverable errors (empty buffer, invalid params).

---

## 12. Catch-Unwind on FFI Boundaries

**What it is:** Wrap code that crosses FFI boundaries (VST3 callbacks, Drop implementations) in `std::panic::catch_unwind` to prevent panics from unwinding into the host.

**Why it matters:** The VST3 API is C FFI. Rust panics that cross an FFI boundary are undefined behavior. On Windows, this manifests as CFG violations or access violations when the host's SEH handler tries to unwind through DLL code that has already been partially unloaded.

**In this project:**

```rust
// lib.rs — Drop implementation wrapped in catch_unwind
fn drop(&mut self) {
    let _ = catch_unwind(AssertUnwindSafe(|| {
        self.is_ui_active.store(false, Ordering::SeqCst);
        #[cfg(windows)]
        drain_win32_queue();
    }));
}

// lib.rs — GuardedHandle::drop also wrapped
fn drop(&mut self) {
    let _ = catch_unwind(AssertUnwindSafe(|| {
        self.flag.store(false, Ordering::SeqCst);
        drop(self._inner.take());
    }));
    #[cfg(windows)]
    drain_win32_queue(); // always runs, even if above panicked
}
```

**Rules:**
- Wrap every `Drop` that interacts with the host or UI in `catch_unwind`.
- Use `catch_unwind(AssertUnwindSafe(|| { ... }))` -- the `AssertUnwindSafe` wrapper is required because most types are not `UnwindSafe`.
- Place critical cleanup (atomic stores) inside the closure. Place best-effort cleanup (Win32 drain) outside so it always runs.

---

## 13. Atomic Telemetry

**What it is:** Use `AtomicU32` / `AtomicBool` for one-way data passing from audio thread to GUI thread. No locks, no allocation, no contention.

**Why it matters:** The audio thread cannot block. Telemetry data (worst frame time, reset count) needs to reach the GUI without any risk of blocking.

**In this project:**

```rust
// lib.rs — audio thread writes
let worst = self.chroma_l.take_worst_frame_time().as_micros() as u32;
self.worst_frame_us_atomic.store(worst, Ordering::Relaxed);

// editor.rs — GUI thread reads
let worst_us = worst_atomic.load(Ordering::Relaxed);
```

**Rules:**
- Use `Ordering::Relaxed` for telemetry -- ordering doesn't matter for approximate values.
- Use `Ordering::SeqCst` for lifecycle flags (`is_ui_active`) where visibility across threads is critical.
- Never store non-atomic data behind an atomic pointer (use triple buffer for that).

---

## 14. Trait Abstraction for DSP Chains

**What it is:** Define a `SampleProcessor` trait with a single `process_sample(x: f32) -> f32` method. Implement it for all DSP stages. Use trait objects or generics to chain stages.

**Why it matters:** Makes DSP stages composable and testable in isolation. New stages can be added without modifying the process loop.

**In this project:**

```rust
// types.rs
pub trait SampleProcessor {
    #[must_use]
    fn process_sample(&mut self, x: f32) -> f32;
    fn reset(&mut self);
}

// Blanket impl for Box<dyn SampleProcessor>
impl<T: SampleProcessor + ?Sized> SampleProcessor for Box<T> {
    fn process_sample(&mut self, x: f32) -> f32 { (**self).process_sample(x) }
    fn reset(&mut self) { (**self).reset() }
}

// SvFilter implements SampleProcessor directly
impl SampleProcessor for crate::svf::SvFilter {
    fn process_sample(&mut self, x: f32) -> f32 { self.tick(x) }
    fn reset(&mut self) { self.clear() }
}
```

**Rule:** Start with trait objects for flexibility. Profile before switching to monomorphized generics -- trait object dispatch (~1ns) is negligible compared to DSP math (~100ns per sample).

---

## 15. `#[derive]` on Domain Types

**What it is:** Use `#[derive(Debug, Clone, PartialEq)]` on all domain types. This makes them inspectable in tests, copyable where needed, and comparable for assertions.

**Why it matters:** Without `Debug`, test failure messages show opaque type names instead of field values. Without `PartialEq`, you cannot `assert_eq!` on structs.

**In this project:**

```rust
// svf.rs
#[derive(Debug, Clone, PartialEq)]
pub enum SvfError { ... }

#[derive(Clone, Copy)]
pub struct ToneParams { pub freq: f32, pub q: f32 }

// state.rs
#[derive(Clone)]
pub struct VisualizerFrame { ... }

// types.rs
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Frequency(f32);
```

**Rule:** Derive at minimum `Debug` and `Clone` on all structs. Derive `PartialEq` where test assertions are needed. Skip `Debug` only on performance-critical hot-path types where the derive would add bloat.

---

## Safety Rules Summary

### When to Use Unsafe

Use `unsafe` only for:
1. **FFI calls** (Win32 API, VST3 C ABI) -- wrap in a safe public function with bounds checks.
2. **Raw pointer dereferencing** required by C APIs.
3. **`std::mem::zeroed()`** for C struct initialization where all-zeros is valid.

In this project, `unsafe` appears only in `drain_win32_queue()` for Win32 FFI (`PeekMessageW`, `TranslateMessage`, `DispatchMessageW`). The unsafe block is wrapped in a safe function with bounds checking.

### When NOT to Use Unsafe

- Never use `unsafe` for performance. Profile first.
- Never use `unsafe` to bypass borrow checker rules. Refactor the code instead.
- Never use `unsafe` for SIMD -- use `std::simd` or `portable-simd` (stable since Rust 1.75).

### RT-Safe Rules

1. No heap allocation (`Vec::new()`, `String::new()`, `Box::new()`, `format!()`).
2. No mutex blocking (`lock()`, `parking_lot::lock()`). Use `try_lock()`.
3. No I/O (println!, eprintln!, file operations).
4. No system calls (time, sleep, thread spawn).
5. No panics (unwind across FFI = UB).
6. No `RecalculateEvent` or `set_latency_samples()` from the audio thread.
7. Flush denormals on all feedback paths.
8. Use atomics for cross-thread telemetry.
