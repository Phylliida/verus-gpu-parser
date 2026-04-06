///  Import resolution: follows `use` statements to find external function sources.
///
///  Given `use verus_fixed_point::fixed_point::prime_field::*;`, resolves:
///  1. Crate name `verus_fixed_point` → directory via Cargo.toml `[dependencies]`
///  2. Module path `fixed_point::prime_field` → `src/fixed_point/prime_field.rs`
///  3. Parses that file for function definitions
///  4. Returns function source texts as owned Strings

use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Parsed import: `use crate_name::module::path::*;` or `use crate_name::module::path::item;`
#[derive(Debug)]
struct Import {
    crate_name: String,       // e.g., "verus_fixed_point" (underscored)
    module_path: Vec<String>, // e.g., ["fixed_point", "prime_field"]
    is_glob: bool,            // true for `::*`
    items: Vec<String>,       // specific items if not glob
}

/// Find Cargo.toml by walking up from the given file path.
fn find_cargo_toml(file_path: &Path) -> Option<PathBuf> {
    let mut dir = file_path.parent()?;
    loop {
        let candidate = dir.join("Cargo.toml");
        if candidate.exists() {
            return Some(candidate);
        }
        dir = dir.parent()?;
    }
}

/// Parse Cargo.toml to extract dependency paths: crate_name → relative path.
/// Handles `foo = { path = "../foo" }` entries.
fn parse_cargo_deps(cargo_toml: &Path) -> HashMap<String, PathBuf> {
    let mut deps = HashMap::new();
    let content = match std::fs::read_to_string(cargo_toml) {
        Ok(c) => c,
        Err(_) => return deps,
    };

    let cargo_dir = cargo_toml.parent().unwrap_or(Path::new("."));
    let mut in_deps = false;

    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("[dependencies]") {
            in_deps = true;
            continue;
        }
        if trimmed.starts_with('[') && !trimmed.starts_with("[dependencies") {
            in_deps = false;
            continue;
        }
        if !in_deps { continue; }

        // Parse: crate-name = { path = "../relative/path" }
        if let Some(eq_pos) = trimmed.find('=') {
            let crate_name = trimmed[..eq_pos].trim().trim_matches('"');
            let value = trimmed[eq_pos + 1..].trim();

            if let Some(path_start) = value.find("path") {
                let after = &value[path_start + 4..];
                // Find the path string: path = "..."
                if let Some(q1) = after.find('"') {
                    let rest = &after[q1 + 1..];
                    if let Some(q2) = rest.find('"') {
                        let path_str = &rest[..q2];
                        let resolved = cargo_dir.join(path_str);
                        // Convert crate-name to crate_name (Rust uses underscores)
                        let rust_name = crate_name.replace('-', "_");
                        deps.insert(rust_name, resolved);
                    }
                }
            }
        }
    }
    deps
}

/// Parse `use` statements from source text and extract imports.
fn parse_use_statements(source: &str) -> Vec<Import> {
    let mut imports = Vec::new();

    for line in source.lines() {
        let trimmed = line.trim();
        // Skip non-use lines and comments
        if !trimmed.starts_with("use ") { continue; }
        if trimmed.starts_with("//") { continue; }

        // Strip "use " prefix and ";" suffix
        let path_str = trimmed.strip_prefix("use ")
            .and_then(|s| s.strip_suffix(';'))
            .unwrap_or("")
            .trim();

        if path_str.is_empty() { continue; }

        // Split by "::"
        let parts: Vec<&str> = path_str.split("::").collect();
        if parts.len() < 2 { continue; }

        let crate_name = parts[0].to_string();

        // Skip crate-internal and standard library imports
        if crate_name == "crate" || crate_name == "self" || crate_name == "super"
            || crate_name == "std" || crate_name == "core" { continue; }

        let last = parts.last().unwrap_or(&"");
        let is_glob = *last == "*";

        let module_path: Vec<String> = if is_glob {
            parts[1..parts.len()-1].iter().map(|s| s.to_string()).collect()
        } else {
            // Could be `use crate::a::b::item` — module is a::b, item is the function
            parts[1..parts.len()-1].iter().map(|s| s.to_string()).collect()
        };

        let items = if is_glob {
            Vec::new()
        } else {
            vec![last.to_string()]
        };

        imports.push(Import { crate_name, module_path, is_glob, items });
    }

    imports
}

/// Resolve a module path to a source file.
/// `fixed_point::prime_field` → `src/fixed_point/prime_field.rs`
/// Falls back to `src/fixed_point/prime_field/mod.rs`.
fn resolve_module_path(crate_dir: &Path, module_path: &[String]) -> Option<PathBuf> {
    let mut file_path = crate_dir.join("src");
    for part in module_path {
        file_path = file_path.join(part);
    }

    // Try .rs extension first
    let rs_path = file_path.with_extension("rs");
    if rs_path.exists() {
        return Some(rs_path);
    }

    // Try mod.rs inside directory
    let mod_path = file_path.join("mod.rs");
    if mod_path.exists() {
        return Some(mod_path);
    }

    None
}

/// Extract function source texts from a Rust/Verus source file.
/// Returns function_name → full function text (as owned String).
pub fn extract_functions_from_file(file_path: &Path) -> HashMap<String, String> {
    let source = match std::fs::read_to_string(file_path) {
        Ok(s) => s,
        Err(_) => return HashMap::new(),
    };

    let mut parser = tree_sitter::Parser::new();
    if parser.set_language(&tree_sitter_verus::LANGUAGE.into()).is_err() {
        return HashMap::new();
    }

    let tree = match parser.parse(source.as_bytes(), None) {
        Some(t) => t,
        None => return HashMap::new(),
    };

    let mut result = HashMap::new();
    collect_fn_sources(&tree.root_node(), &source, &mut result);

    // Recursively follow use statements in this file too
    let imports = parse_use_statements(&source);
    let cargo_toml = find_cargo_toml(file_path);
    if let Some(ref ct) = cargo_toml {
        let deps = parse_cargo_deps(ct);
        for import in &imports {
            if let Some(crate_dir) = deps.get(&import.crate_name) {
                if let Some(module_file) = resolve_module_path(crate_dir, &import.module_path) {
                    let sub_fns = extract_functions_from_file(&module_file);
                    for (name, text) in sub_fns {
                        result.entry(name).or_insert(text);
                    }
                }
            }
        }
    }

    result
}

/// Walk a tree-sitter tree and collect function source texts.
/// Includes preceding attribute nodes (e.g., #[gpu_base_case(...)]).
fn collect_fn_sources(node: &tree_sitter::Node, source: &str, result: &mut HashMap<String, String>) {
    let mut cursor = node.walk();
    let children: Vec<tree_sitter::Node> = node.children(&mut cursor).collect();
    let mut prev_attr: Option<String> = None;

    for child in &children {
        let kind = child.kind();
        // Track attribute items that might precede a function
        if kind == "attribute_item" {
            let attr_text = child.utf8_text(source.as_bytes()).unwrap_or("");
            if attr_text.contains("gpu_base_case") {
                prev_attr = Some(attr_text.to_string());
            }
            continue;
        }
        match kind {
            "function_item" => {
                if let Some(name_node) = child.child_by_field_name("name") {
                    let name = name_node.utf8_text(source.as_bytes()).unwrap_or("").to_string();
                    let fn_text = child.utf8_text(source.as_bytes()).unwrap_or("");
                    let is_spec = fn_text.contains("spec fn") || fn_text.contains("proof fn")
                        || fn_text.contains("open spec") || fn_text.contains("closed spec");
                    if !name.is_empty() && !is_spec
                        && !name.starts_with("proof_") && !name.starts_with("lemma_")
                        && !name.starts_with("axiom_") && !name.starts_with("broadcast_")
                    {
                        // Prepend attribute if present
                        let text = if let Some(ref attr) = prev_attr {
                            format!("{}\n{}", attr, fn_text)
                        } else {
                            fn_text.to_string()
                        };
                        result.entry(name).or_insert(text);
                    }
                }
                prev_attr = None;
            },
            _ => {
                collect_fn_sources(child, source, result);
                prev_attr = None;
            },
        }
    }
}

/// Main entry point: resolve all imports from a kernel file.
/// Returns function_name → function source text for all imported functions.
pub fn resolve_all_imports(source: &str, file_path: &str) -> HashMap<String, String> {
    let file = Path::new(file_path);
    let imports = parse_use_statements(source);
    let cargo_toml = match find_cargo_toml(file) {
        Some(ct) => ct,
        None => return HashMap::new(),
    };
    let deps = parse_cargo_deps(&cargo_toml);

    let mut all_fns: HashMap<String, String> = HashMap::new();

    for import in &imports {
        let crate_dir = match deps.get(&import.crate_name) {
            Some(d) => d,
            None => continue,
        };

        if let Some(module_file) = resolve_module_path(crate_dir, &import.module_path) {
            eprintln!("  Importing from: {}", module_file.display());
            let fns = extract_functions_from_file(&module_file);
            eprintln!("    Found {} functions", fns.len());
            for (name, text) in fns {
                all_fns.entry(name).or_insert(text);
            }
        }
    }

    all_fns
}
