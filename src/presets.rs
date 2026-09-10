// src/presets.rs
// Named preset load/save for ResoVoid. GUI-thread only — the audio thread is
// never involved. A preset is a flat map of nih-plug param ID (the `#[id]`
// strings in `lib.rs`) to plain value, serialized as pretty JSON.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// On-disk preset: name (redundant with the filename, kept for display) plus
/// the param ID -> value map.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Preset {
    pub name: String,
    pub values: HashMap<String, f32>,
}

impl Preset {
    pub fn new(name: &str, values: HashMap<String, f32>) -> Self {
        Self {
            name: name.to_string(),
            values,
        }
    }
}

/// Directory holding `*.json` presets: `%APPDATA%\ResoVoid\presets` on
/// Windows, `~/.config/resovoid/presets` elsewhere.
pub fn presets_dir() -> PathBuf {
    #[cfg(target_os = "windows")]
    {
        if let Ok(appdata) = std::env::var("APPDATA") {
            let mut p = PathBuf::from(appdata);
            p.push("ResoVoid");
            p.push("presets");
            return p;
        }
    }
    if let Ok(home) = std::env::var("HOME") {
        let mut p = PathBuf::from(home);
        p.push(".config");
        p.push("resovoid");
        p.push("presets");
        return p;
    }
    // Last resort: relative to the working directory.
    let mut p = PathBuf::from(".");
    p.push("resovoid_presets");
    p
}

/// File path for a preset name inside `dir`. Strips path separators so a
/// hostile/accidental name cannot escape the presets directory.
pub(crate) fn path_for(dir: &Path, name: &str) -> PathBuf {
    let safe: String = name
        .chars()
        .filter(|&c| c != '/' && c != '\\' && c != ':' && c != '\0')
        .collect();
    let stem = if safe.trim().is_empty() {
        "untitled".to_string()
    } else {
        safe.trim().to_string()
    };
    dir.join(format!("{stem}.json"))
}

/// Sorted preset names (extension stripped). Missing/unreadable dir -> empty.
pub fn list_presets() -> Vec<String> {
    list_presets_in(&presets_dir())
}

pub(crate) fn list_presets_in(dir: &Path) -> Vec<String> {
    let mut out = Vec::new();
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return out,
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("json") {
            continue;
        }
        if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
            out.push(stem.to_string());
        }
    }
    out.sort();
    out
}

/// Serialize `values` under `name` (creates the directory if needed).
pub fn save_preset(name: &str, values: HashMap<String, f32>) -> std::io::Result<()> {
    save_preset_in(&presets_dir(), name, values)
}

pub(crate) fn save_preset_in(
    dir: &Path,
    name: &str,
    values: HashMap<String, f32>,
) -> std::io::Result<()> {
    std::fs::create_dir_all(dir)?;
    let preset = Preset::new(name, values);
    let path = path_for(dir, name);
    let file = std::fs::File::create(path)?;
    serde_json::to_writer_pretty(file, &preset)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
}

/// Load a preset by name.
pub fn load_preset(name: &str) -> std::io::Result<Preset> {
    load_preset_in(&presets_dir(), name)
}

pub(crate) fn load_preset_in(dir: &Path, name: &str) -> std::io::Result<Preset> {
    let path = path_for(dir, name);
    let file = std::fs::File::open(path)?;
    serde_json::from_reader(file)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
}

/// Delete a preset by name (removes its `.json` file).
pub fn delete_preset(name: &str) -> std::io::Result<()> {
    delete_preset_in(&presets_dir(), name)
}

pub(crate) fn delete_preset_in(dir: &Path, name: &str) -> std::io::Result<()> {
    let path = path_for(dir, name);
    std::fs::remove_file(path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn tmp_dir(tag: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!("resovoid_preset_test_{tag}"));
        let _ = std::fs::remove_dir_all(&p);
        p
    }

    #[test]
    fn roundtrip_save_load() {
        let dir = tmp_dir("roundtrip");
        let mut values = HashMap::new();
        values.insert("depth".to_string(), 0.75);
        values.insert("nf0".to_string(), 440.0);
        values.insert("delt".to_string(), 1.0);
        save_preset_in(&dir, "warm", values.clone()).unwrap();

        let loaded = load_preset_in(&dir, "warm").unwrap();
        assert_eq!(loaded.name, "warm");
        assert_eq!(loaded.values, values);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn list_finds_json_only() {
        let dir = tmp_dir("list");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("a.json"), "{}").unwrap();
        std::fs::write(dir.join("b.json"), "{}").unwrap();
        std::fs::write(dir.join("notes.txt"), "hi").unwrap();
        let names = list_presets_in(&dir);
        assert_eq!(names, vec!["a".to_string(), "b".to_string()]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn list_missing_dir_is_empty() {
        let dir = tmp_dir("missing_never_created");
        let _ = std::fs::remove_dir_all(&dir);
        assert!(list_presets_in(&dir).is_empty());
    }

    #[test]
    fn name_sanitized_no_escape() {
        let dir = Path::new("/tmp/presets");
        let p = path_for(dir, "../evil:name/test");
        assert_eq!(p.file_name().unwrap().to_str().unwrap(), "..evilnametest.json");
        // No separator survives in the file name.
        assert!(!p
            .file_name()
            .unwrap()
            .to_str()
            .unwrap()
            .contains(std::path::MAIN_SEPARATOR));
    }

    #[test]
    fn load_missing_is_err() {
        let dir = tmp_dir("missing_file");
        std::fs::create_dir_all(&dir).unwrap();
        assert!(load_preset_in(&dir, "nope").is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn delete_removes_file_and_listing() {
        let dir = tmp_dir("delete");
        let mut values = HashMap::new();
        values.insert("depth".to_string(), 0.5);
        save_preset_in(&dir, "todelete", values).unwrap();
        assert!(list_presets_in(&dir).contains(&"todelete".to_string()));
        delete_preset_in(&dir, "todelete").unwrap();
        assert!(!list_presets_in(&dir).contains(&"todelete".to_string()));
        assert!(load_preset_in(&dir, "todelete").is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn delete_missing_is_err() {
        let dir = tmp_dir("delete_missing");
        std::fs::create_dir_all(&dir).unwrap();
        assert!(delete_preset_in(&dir, "nope").is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn delete_traversal_cannot_escape_dir() {
        let dir = tmp_dir("delete_traversal");
        std::fs::create_dir_all(&dir).unwrap();
        // Resolution must go through path_for: no separator survives and the
        // resolved parent stays inside `dir`.
        let p = path_for(&dir, "../evil");
        assert!(!p
            .file_name()
            .unwrap()
            .to_str()
            .unwrap()
            .contains(std::path::MAIN_SEPARATOR));
        let parent = p.parent().unwrap();
        let canonical_dir = dir.canonicalize().unwrap();
        // `dir` was just created so canonicalization succeeds; the (possibly
        // non-existent) file's parent resolves to `dir` itself when made
        // absolute, otherwise fall back to comparing against canonical dir.
        let canonical_parent = if parent.exists() {
            parent.canonicalize().unwrap()
        } else {
            canonical_dir.clone()
        };
        assert_eq!(canonical_parent, canonical_dir);
        // Deleting the traversal name fails (no such file) and creates
        // nothing outside `dir`.
        assert!(delete_preset_in(&dir, "../evil").is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
