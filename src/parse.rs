///  Tree-sitter based parser: Verus source → GPU Kernel IR.
///  Walks the CST and builds Kernel/Expr/Stmt types.
///  This is the trusted component.
///
///  Function call support: parses all functions in the file, then does a
///  reachability walk from the #[gpu_kernel] function to find helper functions.
///  Only reachable helpers are included in the output.

use tree_sitter::Node;
use crate::types::*;
use std::collections::{HashMap, HashSet, VecDeque};
use std::path::{Path, PathBuf};

/// Parser state: source text + variable/buffer name tracking.
struct ParseCtx<'a> {
    source: &'a str,
    var_names: Vec<String>,     // local variable name → index
    buf_decls: Vec<BufDecl>,
    builtin_names: Vec<String>, // e.g., "gid.x"
    /// Function name → fn_id mapping (shared across kernel + helpers).
    fn_name_to_id: HashMap<String, u32>,
    /// Names of functions called (collected during parsing for reachability).
    called_fns: HashSet<String>,
    /// Variables that are Vec-typed (mapped to scratch buffer on GPU).
    vec_vars: HashSet<String>,
}

impl<'a> ParseCtx<'a> {
    fn text(&self, node: &Node) -> &'a str {
        node.utf8_text(self.source.as_bytes()).unwrap_or("")
    }

    /// Get or create a variable index for the given name.
    /// Renames WGSL-invalid identifiers (bare `_` → `_unused_N`).
    fn var_idx(&mut self, name: &str) -> u32 {
        // WGSL reserved words and invalid identifiers
        let safe_name = if name == "_" {
            let n = self.var_names.len();
            format!("_unused_{}", n)
        } else if name == "self" || name == "self_val" {
            "self_val".to_string()
        } else if matches!(name, "super" | "true" | "false" | "return" | "fn" | "let" | "var"
            | "if" | "else" | "for" | "while" | "break" | "continue" | "switch" | "loop"
            | "struct" | "enum" | "type" | "const" | "override" | "diagnostic" | "enable"
            | "alias" | "bitcast" | "discard" | "fallthrough" | "default" | "case"
            | "target" | "texture" | "sampler" | "ptr" | "ref" | "function" | "private"
            | "workgroup" | "uniform" | "storage" | "handle" | "read" | "write" | "read_write"
            | "array" | "atomic" | "mat2x2" | "mat3x3" | "mat4x4" | "vec2" | "vec3" | "vec4"
            | "bool" | "f16" | "f32" | "i32" | "u32"
        ) {
            format!("{}_v", name)
        } else {
            name.to_string()
        };
        if let Some(idx) = self.var_names.iter().position(|n| n == &safe_name) {
            idx as u32
        } else {
            let idx = self.var_names.len() as u32;
            self.var_names.push(safe_name);
            idx
        }
    }

    fn buf_idx(&self, name: &str) -> Option<u32> {
        self.buf_decls.iter().position(|b| b.name == name).map(|i| i as u32)
    }

    fn fn_id(&mut self, name: &str) -> u32 {
        if let Some(&id) = self.fn_name_to_id.get(name) {
            id
        } else {
            let id = self.fn_name_to_id.len() as u32;
            self.fn_name_to_id.insert(name.to_string(), id);
            id
        }
    }
}

/// Parse a complete #[gpu_kernel] function + reachable helpers from source.
/// `file_path` is used to resolve `use` imports relative to the file's crate.
pub fn parse_gpu_kernel(source: &str, file_path: &str) -> Result<Kernel, String> {
    let mut parser = tree_sitter::Parser::new();
    parser.set_language(&tree_sitter_verus::LANGUAGE.into())
        .map_err(|e| format!("Failed to load Verus grammar: {}", e))?;

    let tree = parser.parse(source.as_bytes(), None)
        .ok_or_else(|| "Failed to parse source".to_string())?;

    let root = tree.root_node();

    // Phase 0: Resolve `use` imports → find external function source texts
    let imported_fn_sources = crate::imports::resolve_all_imports(source, file_path);
    eprintln!("Resolved {} imported functions", imported_fn_sources.len());

    // Phase 1: Find all function_item nodes in the local file
    let all_fns = find_all_functions(&root, source);

    // Phase 2: Find the #[gpu_kernel] function
    let kernel_fn_node = find_gpu_kernel_fn(&root, source)
        .ok_or_else(|| "No #[gpu_kernel] function found".to_string())?;

    // Phase 3: Parse the kernel function body
    let mut ctx = ParseCtx {
        source,
        var_names: Vec::new(),
        buf_decls: Vec::new(),
        builtin_names: Vec::new(),
        fn_name_to_id: HashMap::new(),
        called_fns: HashSet::new(),
        vec_vars: HashSet::new(),
    };

    let name = kernel_fn_node.child_by_field_name("name")
        .map(|n| n.utf8_text(source.as_bytes()).unwrap_or("kernel").to_string())
        .unwrap_or_else(|| "kernel".to_string());

    let workgroup_size = parse_workgroup_size(&kernel_fn_node, source);

    if let Some(params) = kernel_fn_node.child_by_field_name("parameters") {
        parse_parameters(&params, &mut ctx);
    }

    let body = if let Some(body_node) = kernel_fn_node.child_by_field_name("body") {
        parse_block(&body_node, &mut ctx)?
    } else {
        Stmt::Noop
    };

    // Phase 4: Reachability walk — find all transitively called functions.
    // Checks both local file functions AND imported function sources.
    let mut reachable: Vec<String> = Vec::new();
    let mut visited: HashSet<String> = HashSet::new();
    let mut queue: VecDeque<String> = ctx.called_fns.iter().cloned().collect();

    while let Some(fn_name) = queue.pop_front() {
        if visited.contains(&fn_name) { continue; }

        // Check if function exists locally or in imports
        let is_local = all_fns.contains_key(&fn_name);
        let is_imported = imported_fn_sources.contains_key(&fn_name);
        if !is_local && !is_imported { continue; }

        visited.insert(fn_name.clone());
        reachable.push(fn_name.clone());

        // Parse this function to discover ITS callees
        let callees = if is_local {
            let fn_node = &all_fns[&fn_name];
            let mut helper_ctx = ParseCtx {
                source,
                var_names: Vec::new(),
                buf_decls: Vec::new(),
                builtin_names: Vec::new(),
                fn_name_to_id: ctx.fn_name_to_id.clone(),
                called_fns: HashSet::new(),
        vec_vars: HashSet::new(),
            };
            if let Some(params) = fn_node.child_by_field_name("parameters") {
                parse_helper_parameters(&params, &mut helper_ctx);
            }
            if let Some(body_node) = fn_node.child_by_field_name("body") {
                let _ = parse_block(&body_node, &mut helper_ctx);
            }
            for (k, v) in &helper_ctx.fn_name_to_id {
                ctx.fn_name_to_id.entry(k.clone()).or_insert(*v);
            }
            helper_ctx.called_fns
        } else {
            // Imported function: parse from stored source text
            let fn_src = &imported_fn_sources[&fn_name];
            discover_callees_from_source(fn_src, &mut parser, &mut ctx)
        };

        for callee in &callees {
            if !visited.contains(callee) {
                queue.push_back(callee.clone());
            }
        }
    }

    // Phase 5: Parse reachable functions into GpuFunction structs
    let mut fn_id_map: HashMap<String, u32> = HashMap::new();
    let mut functions: Vec<GpuFunction> = Vec::new();

    for (i, fn_name) in reachable.iter().enumerate() {
        fn_id_map.insert(fn_name.clone(), i as u32);
    }

    for fn_name in &reachable {
        if all_fns.contains_key(fn_name) {
            let fn_node = &all_fns[fn_name];
            let func = parse_helper_function(fn_node, source, &fn_id_map)?;
            functions.push(func);
        } else if let Some(fn_src) = imported_fn_sources.get(fn_name) {
            let func = parse_helper_function_from_source(fn_src, &mut parser, &fn_id_map)?;
            functions.push(func);
        }
    }

    // Phase 6: Re-parse kernel body with final fn_id_map so CallStmt/Call IDs are correct
    let mut final_ctx = ParseCtx {
        source,
        var_names: Vec::new(),
        buf_decls: Vec::new(),
        builtin_names: Vec::new(),
        fn_name_to_id: fn_id_map.iter().map(|(k, v)| (k.clone(), *v)).collect(),
        called_fns: HashSet::new(),
        vec_vars: HashSet::new(),
    };

    if let Some(params) = kernel_fn_node.child_by_field_name("parameters") {
        parse_parameters(&params, &mut final_ctx);
    }

    let final_body = if let Some(body_node) = kernel_fn_node.child_by_field_name("body") {
        parse_block(&body_node, &mut final_ctx)?
    } else {
        Stmt::Noop
    };

    // Phase 7: Populate vec_buffer_map from call sites with BufSlice arguments.
    // Walk the kernel body to find calls like generic_add_limbs(&a_buf[base..], &b_buf[base..], n)
    // and map each Vec param to the buffer name it's called with.
    populate_vec_buffer_maps(&final_body, &final_ctx.buf_decls, &mut functions);

    let fn_name_to_id_vec: Vec<(String, u32)> = fn_id_map.into_iter().collect();

    Ok(Kernel {
        name,
        var_names: final_ctx.var_names,
        buf_decls: final_ctx.buf_decls,
        body: final_body,
        workgroup_size,
        builtin_names: final_ctx.builtin_names,
        functions,
        fn_name_to_id: fn_name_to_id_vec,
        scratch_size: 0,
    })
}

/// Walk a statement tree to find Call expressions with BufSlice arguments.
/// For each such call, populate the target function's vec_buffer_map.
fn populate_vec_buffer_maps(body: &Stmt, buf_decls: &[BufDecl], functions: &mut Vec<GpuFunction>) {
    match body {
        Stmt::Assign { rhs, .. } => scan_expr_for_buf_calls(rhs, buf_decls, functions),
        Stmt::CallStmt { fn_id, args, .. } => {
            // Treat as a Call expression for scanning
            let call_expr = Expr::Call(*fn_id, args.clone());
            scan_expr_for_buf_calls(&call_expr, buf_decls, functions);
        },
        Stmt::TupleDestructure { rhs, .. } => scan_expr_for_buf_calls(rhs, buf_decls, functions),
        Stmt::Seq { first, then } => {
            populate_vec_buffer_maps(first, buf_decls, functions);
            populate_vec_buffer_maps(then, buf_decls, functions);
        },
        Stmt::If { then_body, else_body, .. } => {
            populate_vec_buffer_maps(then_body, buf_decls, functions);
            populate_vec_buffer_maps(else_body, buf_decls, functions);
        },
        Stmt::For { body, .. } => populate_vec_buffer_maps(body, buf_decls, functions),
        _ => {},
    }
}

fn scan_expr_for_buf_calls(expr: &Expr, buf_decls: &[BufDecl], functions: &mut Vec<GpuFunction>) {
    match expr {
        Expr::Call(fn_id, args) => {
            let fn_idx = *fn_id as usize;
            if fn_idx < functions.len() {
                let func = &functions[fn_idx];
                let vec_params: Vec<String> = func.vec_params.clone();
                let mut mappings: Vec<(String, String)> = Vec::new();

                // Collect all BufSlice args
                let buf_slices: Vec<(u32, &Expr)> = args.iter()
                    .filter_map(|a| if let Expr::BufSlice(buf_id, offset) = a {
                        Some((*buf_id, offset.as_ref()))
                    } else { None })
                    .collect();

                // First N BufSlices map to Vec params (inputs)
                for (i, (buf_id, _)) in buf_slices.iter().enumerate() {
                    if i < vec_params.len() {
                        let buf_name = buf_decls.get(*buf_id as usize)
                            .map(|b| b.name.clone())
                            .unwrap_or_else(|| format!("buf_{}", buf_id));
                        mappings.push((vec_params[i].clone(), buf_name));
                    } else {
                        // Extra BufSlice args map to output Vecs
                        // Convention: output Vec is named "out" (from Vec::new() in body)
                        let buf_name = buf_decls.get(*buf_id as usize)
                            .map(|b| b.name.clone())
                            .unwrap_or_else(|| format!("buf_{}", buf_id));
                        mappings.push(("out".to_string(), buf_name));
                        functions[fn_idx].returns_vec = true;
                    }
                }

                if !mappings.is_empty() {
                    functions[fn_idx].vec_buffer_map = mappings;
                }
            }
            // Recurse into args
            for arg in args {
                scan_expr_for_buf_calls(arg, buf_decls, functions);
            }
        },
        Expr::BinOp(_, a, b) => {
            scan_expr_for_buf_calls(a, buf_decls, functions);
            scan_expr_for_buf_calls(b, buf_decls, functions);
        },
        Expr::TupleConstruct(elems) => {
            for e in elems { scan_expr_for_buf_calls(e, buf_decls, functions); }
        },
        _ => {},
    }
}

/// Discover callees from a function's source text (for imported functions).
fn discover_callees_from_source(
    fn_source: &str, parser: &mut tree_sitter::Parser, parent_ctx: &mut ParseCtx,
) -> HashSet<String> {
    let tree = match parser.parse(fn_source.as_bytes(), None) {
        Some(t) => t,
        None => return HashSet::new(),
    };
    let root = tree.root_node();
    // Find the function_item inside
    let fn_node = match find_first_function(&root) {
        Some(n) => n,
        None => return HashSet::new(),
    };
    let mut helper_ctx = ParseCtx {
        source: fn_source,
        var_names: Vec::new(),
        buf_decls: Vec::new(),
        builtin_names: Vec::new(),
        fn_name_to_id: parent_ctx.fn_name_to_id.clone(),
        called_fns: HashSet::new(),
        vec_vars: HashSet::new(),
    };
    if let Some(params) = fn_node.child_by_field_name("parameters") {
        parse_helper_parameters(&params, &mut helper_ctx);
    }
    if let Some(body_node) = fn_node.child_by_field_name("body") {
        let _ = parse_block(&body_node, &mut helper_ctx);
    }
    for (k, v) in &helper_ctx.fn_name_to_id {
        parent_ctx.fn_name_to_id.entry(k.clone()).or_insert(*v);
    }
    helper_ctx.called_fns
}

/// Parse a helper function from its standalone source text.
fn parse_helper_function_from_source(
    fn_source: &str, parser: &mut tree_sitter::Parser, fn_id_map: &HashMap<String, u32>,
) -> Result<GpuFunction, String> {
    let tree = parser.parse(fn_source.as_bytes(), None)
        .ok_or("Failed to parse imported function")?;
    let root = tree.root_node();
    let fn_node = find_first_function(&root)
        .ok_or("No function_item found in imported source")?;
    parse_helper_function(&fn_node, fn_source, fn_id_map)
}

/// Find the first function_item in a tree.
fn find_first_function<'a>(node: &Node<'a>) -> Option<Node<'a>> {
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        if child.kind() == "function_item" {
            return Some(child);
        }
        if let Some(f) = find_first_function(&child) {
            return Some(f);
        }
    }
    None
}

/// Find all function_item nodes in the file, keyed by name.
fn find_all_functions<'a>(node: &Node<'a>, source: &str) -> HashMap<String, Node<'a>> {
    let mut result = HashMap::new();
    collect_functions(node, source, &mut result);
    result
}

fn collect_functions<'a>(node: &Node<'a>, source: &str, result: &mut HashMap<String, Node<'a>>) {
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        match child.kind() {
            "function_item" => {
                if let Some(name_node) = child.child_by_field_name("name") {
                    let name = name_node.utf8_text(source.as_bytes()).unwrap_or("").to_string();
                    if !name.is_empty() {
                        result.insert(name, child);
                    }
                }
            },
            _ => collect_functions(&child, source, result),
        }
    }
}

/// Parse a helper function into a GpuFunction.
fn parse_helper_function(
    fn_node: &Node, source: &str, fn_id_map: &HashMap<String, u32>,
) -> Result<GpuFunction, String> {
    let mut ctx = ParseCtx {
        source,
        var_names: Vec::new(),
        buf_decls: Vec::new(),
        builtin_names: Vec::new(),
        fn_name_to_id: fn_id_map.iter().map(|(k, v)| (k.clone(), *v)).collect(),
        called_fns: HashSet::new(),
        vec_vars: HashSet::new(),
    };

    let name = fn_node.child_by_field_name("name")
        .map(|n| n.utf8_text(source.as_bytes()).unwrap_or("f").to_string())
        .unwrap_or_else(|| "f".to_string());

    // Parse parameters as regular typed variables
    let mut params = Vec::new();
    if let Some(params_node) = fn_node.child_by_field_name("parameters") {
        params = parse_typed_parameters(&params_node, &mut ctx);
    }

    // Parse return type
    let ret_type = parse_return_type(fn_node, source);

    // Create return variable
    let ret_var = ctx.var_idx("_ret");

    let body = if let Some(body_node) = fn_node.child_by_field_name("body") {
        parse_block(&body_node, &mut ctx)?
    } else {
        Stmt::Noop
    };

    let vec_params: Vec<String> = params.iter()
        .filter(|(_, ty)| matches!(ty, ParamType::VecU32))
        .map(|(name, _)| name.clone())
        .collect();
    let returns_vec = match &ret_type {
        Some(ReturnType::Tuple(_)) => false, // TODO: detect Vec in tuple returns
        _ => false,
    };

    Ok(GpuFunction {
        name,
        params,
        ret_type,
        vec_params,
        returns_vec,
        vec_buffer_map: Vec::new(),
        var_names: ctx.var_names,
        body,
        ret_var,
    })
}

/// Parse parameters as typed variables (for helper functions, not kernel).
fn parse_typed_parameters(params_node: &Node, ctx: &mut ParseCtx) -> Vec<(String, ParamType)> {
    let mut result = Vec::new();
    let mut cursor = params_node.walk();
    let children: Vec<Node> = params_node.children(&mut cursor).collect();

    for child in &children {
        let kind = child.kind();
        let text = ctx.text(child);
        // Handle &self / self parameter (trait impl methods)
        if kind == "self_parameter" || text.trim() == "&self" || text.trim() == "self"
            || text.trim() == "&mut self"
        {
            let name = "self_val".to_string();
            ctx.var_idx(&name);
            // Also register "self" as alias for "self_val" so *self resolves
            ctx.var_idx("self");
            result.push((name, ParamType::Scalar(ScalarType::U32)));
            continue;
        }
        if kind == "parameter" {
            let name = extract_param_name(child, ctx.source);
            // Detect Vec<u32> or &Vec<u32> parameters
            let param_type = if text.contains("Vec<") || text.contains("Vec <") {
                ctx.vec_vars.insert(name.clone());
                ParamType::VecU32
            } else {
                ParamType::Scalar(infer_scalar_type(text))
            };
            ctx.var_idx(&name);
            result.push((name, param_type));
        }
    }
    result
}

/// Parse helper function parameters (just add them as variables, skip attributes).
fn parse_helper_parameters(params_node: &Node, ctx: &mut ParseCtx) {
    let mut cursor = params_node.walk();
    for child in params_node.children(&mut cursor) {
        if child.kind() == "parameter" {
            let name = extract_param_name(&child, ctx.source);
            ctx.var_idx(&name);
        }
    }
}

/// Infer scalar type from parameter text like "x: u32" or "a: i32".
fn infer_scalar_type(param_text: &str) -> ScalarType {
    if param_text.contains("i32") { ScalarType::I32 }
    else if param_text.contains("f32") { ScalarType::F32 }
    else if param_text.contains("f16") { ScalarType::F16 }
    else if param_text.contains("bool") { ScalarType::Bool }
    else { ScalarType::U32 }
}

/// Parse return type from function signature.
fn parse_return_type(fn_node: &Node, source: &str) -> Option<ReturnType> {
    let text = fn_node.utf8_text(source.as_bytes()).unwrap_or("");
    // Look for "-> TYPE" pattern before the body "{"
    if let Some(arrow_pos) = text.find("->") {
        let after = &text[arrow_pos + 2..];
        let ty_text = after.split(|c: char| c == '{' || c == '\n')
            .next().unwrap_or("").trim();

        // Strip Verus return name pattern: (name: Type) → extract Type
        // e.g., "(out: (Self, Self))" → "(Self, Self)"
        let ty_text = if ty_text.starts_with('(') && ty_text.contains(':') {
            let after_colon = ty_text.split_once(':').map(|(_, t)| t.trim()).unwrap_or(ty_text);
            // Strip exactly ONE trailing ) for the outer wrapper parens
            after_colon.strip_suffix(')').unwrap_or(after_colon).trim()
        } else {
            ty_text
        };

        // Check for tuple return type: (Type, Type, ...)
        // Also handle (Self, Self) from trait impls
        if ty_text.starts_with('(') && ty_text.ends_with(')') {
            let inner = &ty_text[1..ty_text.len()-1];
            let types: Vec<ScalarType> = inner.split(',')
                .map(|s| infer_scalar_type(s.trim()))
                .collect();
            if types.len() > 1 {
                return Some(ReturnType::Tuple(types));
            }
        }

        return Some(ReturnType::Scalar(infer_scalar_type(ty_text)));
    }
    None
}

/// Recursively find a function_item with #[gpu_kernel] attribute.
fn find_gpu_kernel_fn<'a>(node: &Node<'a>, source: &str) -> Option<Node<'a>> {
    let mut cursor = node.walk();
    let children: Vec<Node<'a>> = node.children(&mut cursor).collect();
    let mut prev_has_gpu_kernel = false;

    for child in &children {
        let text = child.utf8_text(source.as_bytes()).unwrap_or("");
        if text.contains("gpu_kernel") {
            prev_has_gpu_kernel = true;
        }

        match child.kind() {
            "function_item" => {
                if text.contains("gpu_kernel") || prev_has_gpu_kernel {
                    return Some(*child);
                }
                prev_has_gpu_kernel = false;
            },
            _ => {
                if let Some(f) = find_gpu_kernel_fn(child, source) {
                    return Some(f);
                }
                if !text.contains("gpu_kernel") {
                    prev_has_gpu_kernel = false;
                }
            },
        }
    }
    None
}

/// Extract workgroup_size(X, Y, Z) from #[gpu_kernel(...)] attribute.
/// Checks both the function node text AND preceding sibling nodes (attribute_item).
fn parse_workgroup_size(fn_node: &Node, source: &str) -> (u32, u32, u32) {
    // First check the function node itself (in case attribute is embedded)
    let text = fn_node.utf8_text(source.as_bytes()).unwrap_or("");
    if let Some(result) = extract_workgroup_size(text) {
        return result;
    }
    // Check preceding siblings (attribute_item nodes before the function)
    if let Some(parent) = fn_node.parent() {
        let mut cursor = parent.walk();
        for child in parent.children(&mut cursor) {
            if child.id() == fn_node.id() { break; }
            let child_text = child.utf8_text(source.as_bytes()).unwrap_or("");
            if let Some(result) = extract_workgroup_size(child_text) {
                return result;
            }
        }
    }
    (256, 1, 1)
}

fn extract_workgroup_size(text: &str) -> Option<(u32, u32, u32)> {
    if let Some(start) = text.find("workgroup_size(") {
        let rest = &text[start + "workgroup_size(".len()..];
        if let Some(end) = rest.find(')') {
            let nums: Vec<u32> = rest[..end].split(',')
                .filter_map(|s| s.trim().parse().ok())
                .collect();
            return Some((
                nums.first().copied().unwrap_or(1),
                nums.get(1).copied().unwrap_or(1),
                nums.get(2).copied().unwrap_or(1),
            ));
        }
    }
    None
}

/// Parse function parameters — identify builtins (#[gpu_builtin]) and buffers (#[gpu_buffer]).
fn parse_parameters(params_node: &Node, ctx: &mut ParseCtx) {
    let mut cursor = params_node.walk();
    let children: Vec<Node> = params_node.children(&mut cursor).collect();

    let mut pending_attr: Option<String> = None;

    for child in &children {
        let kind = child.kind();
        let text = ctx.text(child);

        if kind == "attribute_item" {
            pending_attr = Some(text.to_string());
            continue;
        }

        if kind == "parameter" {
            let name = extract_param_name(child, ctx.source);
            let attr_text = pending_attr.take().unwrap_or_default();
            let param_text = text;

            if attr_text.contains("gpu_builtin") {
                let builtin = if attr_text.contains("thread_id_x") { "gid.x" }
                    else if attr_text.contains("thread_id_y") { "gid.y" }
                    else if attr_text.contains("thread_id_z") { "gid.z" }
                    else if attr_text.contains("workgroup_id_x") { "gid.x" }
                    else if attr_text.contains("local_id_x") { "gid.x" }
                    else { "gid.x" };
                ctx.builtin_names.push(builtin.to_string());
                ctx.var_idx(&name);
            } else if attr_text.contains("gpu_buffer") {
                let read_only = !attr_text.contains("read_write");
                let binding = ctx.buf_decls.len() as u32;
                let elem_type = if param_text.contains("f32") { ScalarType::F32 }
                    else if param_text.contains("i32") { ScalarType::I32 }
                    else { ScalarType::U32 };
                ctx.buf_decls.push(BufDecl { binding, name, read_only, elem_type });
            }
        }
    }
}

fn extract_param_name(param_node: &Node, source: &str) -> String {
    if let Some(pattern) = param_node.child_by_field_name("pattern") {
        return pattern.utf8_text(source.as_bytes()).unwrap_or("x").to_string();
    }
    let mut cursor = param_node.walk();
    for child in param_node.children(&mut cursor) {
        if child.kind() == "identifier" {
            return child.utf8_text(source.as_bytes()).unwrap_or("x").to_string();
        }
    }
    "x".to_string()
}

/// Parse a block (surrounded by braces) into a Stmt.
/// The last expression without a semicolon is treated as implicit return (__ret = expr).
fn parse_block(node: &Node, ctx: &mut ParseCtx) -> Result<Stmt, String> {
    let mut stmts = Vec::new();
    let mut cursor = node.walk();
    let children: Vec<Node> = node.children(&mut cursor).collect();

    // Find the last non-brace child to detect implicit return
    let last_meaningful = children.iter().rev()
        .find(|c| c.kind() != "}" && c.kind() != "{")
        .map(|c| c.id());

    for child in &children {
        let kind = child.kind();
        let is_last = Some(child.id()) == last_meaningful;
        match kind {
            "{" | "}" => {},
            "let_declaration" => {
                // Skip ghost/proof let bindings
                let text = ctx.text(child);
                if text.contains("ghost ") || text.contains("Ghost") {
                    continue;
                }
                stmts.push(parse_let(child, ctx)?);
            },
            "expression_statement" => stmts.push(parse_expr_stmt(child, ctx)?),
            "if_expression" => {
                if is_last {
                    // Last if-expression: treat as implicit return value
                    let if_stmt = parse_if_as_return(child, ctx)?;
                    stmts.push(if_stmt);
                } else {
                    stmts.push(parse_if(child, ctx)?);
                }
            },
            "for_expression" => stmts.push(parse_for(child, ctx)?),
            "while_expression" => stmts.push(parse_while(child, ctx)?),
            "return_expression" => {
                // return expr → assign to __ret + Return
                let mut cursor2 = child.walk();
                let ret_children: Vec<Node> = child.children(&mut cursor2).collect();
                for rc in &ret_children {
                    if rc.kind() != "return" && rc.kind() != ";" {
                        let rhs = parse_expr(rc, ctx)?;
                        let ret_var = ctx.var_idx("_ret");
                        stmts.push(Stmt::Assign { var: ret_var, rhs });
                        break;
                    }
                }
                stmts.push(Stmt::Return);
            },
            "break_expression" => stmts.push(Stmt::Break),
            "continue_expression" => stmts.push(Stmt::Continue),
            // Skip Verus-specific clauses
            "requires_clause" | "ensures_clause" | "decreases_clause"
            | "invariant_clause" | "recommends_clause" => {},
            // Skip proof blocks and ghost code
            "proof_block" | "ghost_block" | "assert_expr" => {},
            _ => {
                // Debug: show unknown node types
                eprintln!("  [parse_block] unknown node: kind={}, text={:.60}", kind,
                    ctx.text(child).replace('\n', " "));

                // Check if this is a bare expression (implicit return) at end of block
                if is_last && kind != "expression_statement" {
                    // Bare expression without ; → implicit return
                    if let Ok(expr) = parse_expr(child, ctx) {
                        let ret_var = ctx.var_idx("_ret");
                        stmts.push(Stmt::Assign { var: ret_var, rhs: expr });
                    } else if let Ok(s) = parse_expr_to_stmt(child, ctx) {
                        stmts.push(s);
                    }
                } else if let Ok(s) = parse_expr_to_stmt(child, ctx) {
                    stmts.push(s);
                }
            },
        }
    }
    Ok(Stmt::from_vec(stmts))
}

/// Parse a block as a single expression (for if-expression-as-value).
/// Extracts the last meaningful expression from { ... }.
fn parse_block_as_expr(node: &Node, ctx: &mut ParseCtx) -> Result<Expr, String> {
    let mut cursor = node.walk();
    let children: Vec<Node> = node.children(&mut cursor).collect();
    // Find the last non-brace child
    for child in children.iter().rev() {
        match child.kind() {
            "{" | "}" | ";" => continue,
            _ => return parse_expr(child, ctx),
        }
    }
    Ok(Expr::Const(0, ScalarType::U32))
}

/// Parse an if expression where the result is used as a return value.
/// Each branch's last expression gets assigned to __ret.
fn parse_if_as_return(node: &Node, ctx: &mut ParseCtx) -> Result<Stmt, String> {
    let cond_node = node.child_by_field_name("condition")
        .ok_or("if missing condition")?;
    let cond = parse_expr(&cond_node, ctx)?;

    let then_node = node.child_by_field_name("consequence")
        .ok_or("if missing then body")?;
    // Parse then-block with implicit return handling (parse_block handles it)
    let then_body = parse_block(&then_node, ctx)?;

    let else_body = if let Some(alt) = node.child_by_field_name("alternative") {
        if alt.kind() == "else_clause" {
            let mut cursor = alt.walk();
            let mut result = Stmt::Noop;
            for child in alt.children(&mut cursor) {
                if child.kind() == "block" {
                    result = parse_block(&child, ctx)?;
                } else if child.kind() == "if_expression" {
                    result = parse_if_as_return(&child, ctx)?;
                }
            }
            result
        } else {
            parse_block(&alt, ctx)?
        }
    } else {
        Stmt::Noop
    };

    Ok(Stmt::If { cond, then_body: Box::new(then_body), else_body: Box::new(else_body) })
}

fn parse_let(node: &Node, ctx: &mut ParseCtx) -> Result<Stmt, String> {
    let pattern_node = node.child_by_field_name("pattern");
    let pattern_text = pattern_node.map(|n| ctx.text(&n).to_string())
        .unwrap_or_else(|| "tmp".to_string());

    // Check for tuple destructuring: let (a, b, c) = expr
    if pattern_text.starts_with('(') && pattern_text.ends_with(')') {
        let inner = &pattern_text[1..pattern_text.len()-1];
        let names: Vec<String> = inner.split(',')
            .map(|s| s.trim().strip_prefix("mut ").unwrap_or(s.trim()).to_string())
            .filter(|s| !s.is_empty())
            .collect();
        let vars: Vec<u32> = names.iter().map(|n| ctx.var_idx(n)).collect();

        let rhs = if let Some(val) = node.child_by_field_name("value") {
            parse_expr(&val, ctx)?
        } else {
            Expr::Const(0, ScalarType::I32)
        };

        return Ok(Stmt::TupleDestructure { vars, rhs });
    }

    let name = pattern_text.strip_prefix("mut ").unwrap_or(&pattern_text).trim().to_string();

    // Check for Vec::new() — marks variable as vec-typed
    let rhs_node = node.child_by_field_name("value");
    let rhs_text = rhs_node.map(|n| ctx.text(&n).to_string()).unwrap_or_default();
    if rhs_text.contains("Vec::new()") || rhs_text.contains("Vec::<") {
        ctx.vec_vars.insert(name.clone());
        let var = ctx.var_idx(&name);
        // Vec::new() → the variable IS the scratch offset (set by caller).
        // Also create a length tracker: name_len = 0
        let len_name = format!("{}_len", name);
        let len_var = ctx.var_idx(&len_name);
        return Ok(Stmt::Assign { var: len_var, rhs: Expr::Const(0, ScalarType::U32) });
    }

    let var = ctx.var_idx(&name);

    let rhs = if let Some(val) = rhs_node {
        parse_expr(&val, ctx)?
    } else {
        Expr::Const(0, ScalarType::I32)
    };

    Ok(Stmt::Assign { var, rhs })
}

fn parse_expr_stmt(node: &Node, ctx: &mut ParseCtx) -> Result<Stmt, String> {
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        match child.kind() {
            ";" => continue,
            "if_expression" => return parse_if(&child, ctx),
            "for_expression" => return parse_for(&child, ctx),
            "return_expression" => return Ok(Stmt::Return),
            "break_expression" => return Ok(Stmt::Break),
            "continue_expression" => return Ok(Stmt::Continue),
            _ => return parse_expr_to_stmt(&child, ctx),
        }
    }
    Ok(Stmt::Noop)
}

fn parse_expr_to_stmt(node: &Node, ctx: &mut ParseCtx) -> Result<Stmt, String> {
    let text = ctx.text(node);

    // Check for barrier calls
    if text.starts_with("gpu_workgroup_barrier") {
        return Ok(Stmt::Barrier { scope: BarrierScope::Workgroup });
    }
    if text.starts_with("gpu_storage_barrier") {
        return Ok(Stmt::Barrier { scope: BarrierScope::Storage });
    }
    if text.starts_with("gpu_subgroup_barrier") {
        return Ok(Stmt::Barrier { scope: BarrierScope::Subgroup });
    }

    // Assignment: x = expr
    if node.kind() == "assignment_expression" {
        let lhs_node = node.child_by_field_name("left")
            .ok_or("assignment missing lhs")?;
        let rhs_node = node.child_by_field_name("right")
            .ok_or("assignment missing rhs")?;

        let lhs_text = ctx.text(&lhs_node);

        // Buffer write: buf[idx] = val
        if lhs_node.kind() == "index_expression" {
            let buf_name_str = lhs_node.child(0)
                .map(|n| ctx.text(&n).to_string())
                .unwrap_or_default();
            if let Some(buf) = ctx.buf_idx(&buf_name_str) {
                let idx_node = lhs_node.child(2)
                    .or_else(|| lhs_node.child_by_field_name("index"))
                    .ok_or("buffer write missing index")?;
                let idx = parse_expr(&idx_node, ctx)?;
                let val = parse_expr(&rhs_node, ctx)?;
                return Ok(Stmt::BufWrite { buf, idx, val });
            }
        }

        // Variable assignment
        let var = ctx.var_idx(lhs_text);
        let rhs = parse_expr(&rhs_node, ctx)?;
        return Ok(Stmt::Assign { var, rhs });
    }

    // Method call as statement: check for .push() on vec vars
    if node.kind() == "call_expression" {
        let func = node.child_by_field_name("function");
        if let Some(func_node) = func {
            if func_node.kind() == "field_expression" {
                let receiver = func_node.child(0);
                let method = func_node.child(2)
                    .or_else(|| func_node.child_by_field_name("field"));
                if let (Some(recv), Some(meth)) = (receiver, method) {
                    let recv_name = ctx.text(&recv).to_string();
                    let meth_name = ctx.text(&meth);
                    if meth_name == "push" && ctx.vec_vars.contains(&recv_name) {
                        let args = parse_call_args(node, ctx)?;
                        let val = args.into_iter().next().unwrap_or(Expr::Const(0, ScalarType::U32));
                        let vec_var = ctx.var_idx(&recv_name);
                        return Ok(Stmt::VecPush { vec_var, val });
                    }
                    // .len() — skip, handled elsewhere
                    if meth_name == "len" {
                        return Ok(Stmt::Noop);
                    }
                }
            }
        }
    }

    // Function call as statement: f(args) → CallStmt with result discarded
    if node.kind() == "call_expression" {
        if let Some(call_stmt) = try_parse_call_stmt(node, ctx)? {
            return Ok(call_stmt);
        }
    }

    Ok(Stmt::Noop)
}

/// Try to parse a call_expression as a CallStmt.
fn try_parse_call_stmt(node: &Node, ctx: &mut ParseCtx) -> Result<Option<Stmt>, String> {
    let func_node = node.child_by_field_name("function")
        .ok_or("call missing function")?;
    let func_name = ctx.text(&func_node).to_string();

    // Skip Verus proof/ghost functions
    if func_name.starts_with("proof") || func_name.starts_with("ghost")
        || func_name.starts_with("assert") || func_name.starts_with("reveal")
        || func_name.starts_with("lemma_") {
        return Ok(None);
    }

    let args = parse_call_args(node, ctx)?;
    let fn_id = ctx.fn_id(&func_name);
    ctx.called_fns.insert(func_name);
    let result_var = ctx.var_idx("_call_tmp");

    Ok(Some(Stmt::CallStmt { fn_id, args, result_var }))
}

/// Parse arguments from a call_expression.
fn parse_call_args(call_node: &Node, ctx: &mut ParseCtx) -> Result<Vec<Expr>, String> {
    let mut args = Vec::new();
    let args_node = call_node.child_by_field_name("arguments")
        .ok_or("call missing arguments")?;
    let mut cursor = args_node.walk();
    for child in args_node.children(&mut cursor) {
        match child.kind() {
            "(" | ")" | "," => {},
            _ => args.push(parse_expr(&child, ctx)?),
        }
    }
    Ok(args)
}

fn parse_if(node: &Node, ctx: &mut ParseCtx) -> Result<Stmt, String> {
    let cond_node = node.child_by_field_name("condition")
        .ok_or("if missing condition")?;
    let cond = parse_expr(&cond_node, ctx)?;

    let then_node = node.child_by_field_name("consequence")
        .ok_or("if missing then body")?;
    let then_body = parse_block(&then_node, ctx)?;

    let else_body = if let Some(alt) = node.child_by_field_name("alternative") {
        if alt.kind() == "else_clause" {
            let mut cursor = alt.walk();
            let mut result = Stmt::Noop;
            for child in alt.children(&mut cursor) {
                if child.kind() == "block" {
                    result = parse_block(&child, ctx)?;
                } else if child.kind() == "if_expression" {
                    result = parse_if(&child, ctx)?;
                }
            }
            result
        } else {
            parse_block(&alt, ctx)?
        }
    } else {
        Stmt::Noop
    };

    Ok(Stmt::If { cond, then_body: Box::new(then_body), else_body: Box::new(else_body) })
}

fn parse_for(node: &Node, ctx: &mut ParseCtx) -> Result<Stmt, String> {
    let pat = node.child_by_field_name("pattern")
        .map(|n| ctx.text(&n).to_string())
        .unwrap_or_else(|| "i".to_string());
    let var = ctx.var_idx(&pat);

    let value_node = node.child_by_field_name("value")
        .ok_or("for missing range")?;
    let (start, end) = parse_range(&value_node, ctx)?;

    let body_node = node.child_by_field_name("body")
        .ok_or("for missing body")?;
    let body = parse_block(&body_node, ctx)?;

    Ok(Stmt::For { var, start, end, body: Box::new(body) })
}

/// Parse a while loop. Emitted as WGSL `for (; cond;) { body }`.
fn parse_while(node: &Node, ctx: &mut ParseCtx) -> Result<Stmt, String> {
    let cond_node = node.child_by_field_name("condition")
        .ok_or("while missing condition")?;
    let cond = parse_expr(&cond_node, ctx)?;

    let body_node = node.child_by_field_name("body")
        .ok_or("while missing body")?;
    let body = parse_block(&body_node, ctx)?;

    // Emit as: for (var __w: u32 = 0u; cond; __w++) { body }
    // Using a dummy variable since WGSL requires for syntax.
    // Actually, WGSL has `loop { if (!cond) { break; } body }` but for is simpler.
    // We'll use For with a large bound and break on !cond.
    // Actually simplest: just emit as For(dummy, 0, large_n, if(!cond){break} + body)
    // But better: detect "while i < n" pattern and emit as proper for.
    //
    // For now: emit as for(; true; ) with break-if-not-cond at start.
    let break_check = Stmt::If {
        cond: Expr::UnaryOp(UnaryOp::LogicalNot, Box::new(cond)),
        then_body: Box::new(Stmt::Break),
        else_body: Box::new(Stmt::Noop),
    };
    let full_body = Stmt::Seq {
        first: Box::new(break_check),
        then: Box::new(body),
    };
    // Use For with 0..0xFFFFFFFF as a "loop" — the break handles termination
    let dummy = ctx.var_idx("_while_i");
    Ok(Stmt::For {
        var: dummy,
        start: Expr::Const(0, ScalarType::U32),
        end: Expr::Const(0xFFFFFFFF, ScalarType::U32),
        body: Box::new(full_body),
    })
}

fn parse_range(node: &Node, ctx: &mut ParseCtx) -> Result<(Expr, Expr), String> {
    let mut cursor = node.walk();
    let children: Vec<Node> = node.children(&mut cursor).collect();

    if children.len() >= 3 {
        let start = parse_expr(&children[0], ctx)?;
        let end = parse_expr(&children[2], ctx)?;
        Ok((start, end))
    } else {
        Ok((Expr::Const(0, ScalarType::U32), Expr::Const(0, ScalarType::U32)))
    }
}

/// Parse an expression node.
fn parse_expr(node: &Node, ctx: &mut ParseCtx) -> Result<Expr, String> {
    let text = ctx.text(node);

    match node.kind() {
        "integer_literal" => {
            let ty = if text.ends_with("i32") { ScalarType::I32 }
                else if text.ends_with("i64") { ScalarType::I32 }
                else if text.ends_with("f32") { ScalarType::F32 }
                else { ScalarType::U32 };
            let num_str = text.trim_end_matches("u32").trim_end_matches("i32")
                .trim_end_matches("u64").trim_end_matches("i64")
                .trim_end_matches("usize").trim_end_matches("isize")
                .trim_end_matches("u8").trim_end_matches("i8")
                .trim_end_matches("u16").trim_end_matches("i16");
            let val: i64 = num_str.parse().unwrap_or(0);
            Ok(Expr::Const(val, ty))
        },
        "float_literal" => {
            let val: f32 = text.trim_end_matches(|c: char| c.is_alphabetic())
                .parse().unwrap_or(0.0);
            Ok(Expr::FConst(val.to_bits()))
        },
        "true" => Ok(Expr::Const(1, ScalarType::Bool)),
        "false" => Ok(Expr::Const(0, ScalarType::Bool)),
        "identifier" | "scoped_identifier" | "self" => {
            // Map "self" to "self_val" (trait impl methods)
            let name = if text == "self" { "self_val" } else { text };
            if let Some(_buf) = ctx.buf_idx(name) {
                Ok(Expr::Var(ctx.var_idx(name)))
            } else {
                Ok(Expr::Var(ctx.var_idx(name)))
            }
        },
        "binary_expression" => {
            let lhs = node.child_by_field_name("left").ok_or("binop missing lhs")?;
            let rhs = node.child_by_field_name("right").ok_or("binop missing rhs")?;
            let op_node = node.child_by_field_name("operator")
                .or_else(|| node.child(1))
                .ok_or("binop missing operator")?;
            let op_text = ctx.text(&op_node);

            let op = match op_text {
                "+" => BinOp::Add, "-" => BinOp::Sub, "*" => BinOp::Mul,
                "/" => BinOp::Div, "%" => BinOp::Mod, ">>" => BinOp::Shr,
                "<<" => BinOp::Shl, "<" => BinOp::Lt, "<=" => BinOp::Le,
                ">" => BinOp::Gt, ">=" => BinOp::Ge, "==" => BinOp::Eq,
                "!=" => BinOp::Ne, "&" => BinOp::BitAnd, "|" => BinOp::BitOr,
                "^" => BinOp::BitXor, "&&" => BinOp::LogicalAnd,
                "||" => BinOp::LogicalOr,
                _ => return Err(format!("unknown binary op: {}", op_text)),
            };

            Ok(Expr::BinOp(op, Box::new(parse_expr(&lhs, ctx)?),
                               Box::new(parse_expr(&rhs, ctx)?)))
        },
        "unary_expression" | "prefix_unary_expression" => {
            let op_text = node.child(0).map(|n| ctx.text(&n)).unwrap_or("");
            let operand = node.child(1).ok_or("unary missing operand")?;
            match op_text {
                "*" => {
                    // Dereference: *expr → just expr (GPU has no references)
                    parse_expr(&operand, ctx)
                },
                "-" => Ok(Expr::UnaryOp(UnaryOp::Neg, Box::new(parse_expr(&operand, ctx)?))),
                "!" => Ok(Expr::UnaryOp(UnaryOp::LogicalNot, Box::new(parse_expr(&operand, ctx)?))),
                "~" => Ok(Expr::UnaryOp(UnaryOp::BitNot, Box::new(parse_expr(&operand, ctx)?))),
                _ => Ok(Expr::UnaryOp(UnaryOp::Neg, Box::new(parse_expr(&operand, ctx)?))),
            }
        },
        "index_expression" => {
            let base = node.child(0).ok_or("index missing base")?;
            let base_text = ctx.text(&base);
            if let Some(buf) = ctx.buf_idx(base_text) {
                let idx_node = node.child(2)
                    .ok_or("index missing index expr")?;
                let idx = parse_expr(&idx_node, ctx)?;
                Ok(Expr::ArrayRead(buf, Box::new(idx)))
            } else if ctx.vec_vars.contains(base_text) {
                // Vec indexing: a[i] → scratch[a_off + i]
                let var = ctx.var_idx(base_text);
                let idx_node = node.child(2)
                    .ok_or("vec index missing index expr")?;
                let idx = parse_expr(&idx_node, ctx)?;
                Ok(Expr::VecIndex(var, Box::new(idx)))
            } else {
                Ok(Expr::Var(ctx.var_idx(base_text)))
            }
        },
        "if_expression" => {
            // if-expression as value: if cond { A } else { B } → Select(cond, A, B)
            let cond_node = node.child_by_field_name("condition")
                .ok_or("if-expr missing condition")?;
            let cond = parse_expr(&cond_node, ctx)?;

            let then_node = node.child_by_field_name("consequence")
                .ok_or("if-expr missing then")?;
            // Parse then-block: extract the single expression value
            let then_expr = parse_block_as_expr(&then_node, ctx)?;

            let else_expr = if let Some(alt) = node.child_by_field_name("alternative") {
                if alt.kind() == "else_clause" {
                    let mut result = Expr::Const(0, ScalarType::U32);
                    let mut cursor = alt.walk();
                    for child in alt.children(&mut cursor) {
                        if child.kind() == "block" {
                            result = parse_block_as_expr(&child, ctx)?;
                        } else if child.kind() == "if_expression" {
                            result = parse_expr(&child, ctx)?;
                        }
                    }
                    result
                } else {
                    parse_block_as_expr(&alt, ctx)?
                }
            } else {
                Expr::Const(0, ScalarType::U32)
            };

            Ok(Expr::Select(Box::new(cond), Box::new(then_expr), Box::new(else_expr)))
        },
        "reference_expression" | "borrow_expression" => {
            // &expr or &mut expr — find the actual inner expression (skip & and mut tokens)
            let mut inner_opt = None;
            let mut cursor_ref = node.walk();
            for child in node.children(&mut cursor_ref) {
                let ck = child.kind();
                if ck != "&" && ck != "mut" && ck != "mutable_specifier" {
                    inner_opt = Some(child);
                    break;
                }
            }
            let inner = inner_opt.ok_or("reference missing inner")?;

            // Check for buffer slice: &buf[offset..] or &mut buf[offset..]
            if inner.kind() == "index_expression" {
                let base = inner.child(0);
                let idx = inner.child(2);
                if let (Some(base_node), Some(idx_node)) = (base, idx) {
                    let base_text = ctx.text(&base_node);
                    // Check if this is a buffer AND the index is a range expression
                    if ctx.buf_idx(base_text).is_some() && idx_node.kind() == "range_expression" {
                        let buf_id = ctx.buf_idx(base_text).unwrap();
                        // Parse the start of the range
                        let range_start = idx_node.child(0)
                            .map(|n| parse_expr(&n, ctx))
                            .transpose()?
                            .unwrap_or(Expr::Const(0, ScalarType::U32));
                        return Ok(Expr::BufSlice(buf_id, Box::new(range_start)));
                    }
                }
            }

            // Regular reference: just strip &
            parse_expr(&inner, ctx)
        },
        "parenthesized_expression" | "tuple_expression" => {
            // Could be (expr) or (a, b, c, ...)
            let mut cursor = node.walk();
            let children: Vec<Node> = node.children(&mut cursor).collect();
            let mut elems: Vec<Expr> = Vec::new();
            let mut has_comma = false;
            for child in &children {
                match child.kind() {
                    "(" | ")" => {},
                    "," => { has_comma = true; },
                    _ => elems.push(parse_expr(child, ctx)?),
                }
            }
            if has_comma || elems.len() > 1 {
                // Tuple: (a, b, c, ...)
                Ok(Expr::TupleConstruct(elems))
            } else if elems.len() == 1 {
                // Parenthesized expression: (expr)
                Ok(elems.remove(0))
            } else {
                Ok(Expr::Const(0, ScalarType::U32))
            }
        },
        "field_expression" => {
            // expr.field — could be tuple access (expr.0, expr.1) or struct field
            let base = node.child(0).ok_or("field missing base")?;
            let field = node.child(2)
                .or_else(|| node.child_by_field_name("field"))
                .ok_or("field missing field name")?;
            let field_text = ctx.text(&field);
            // If field is a number, it's a tuple access
            if let Ok(idx) = field_text.parse::<u32>() {
                let base_expr = parse_expr(&base, ctx)?;
                Ok(Expr::TupleAccess(Box::new(base_expr), idx))
            } else {
                // Struct field — treat as variable for now
                Ok(Expr::Var(ctx.var_idx(ctx.text(node))))
            }
        },
        "type_cast_expression" | "as_expression" => {
            let inner = node.child(0).ok_or("cast missing inner")?;
            let ty_text = node.child(2).map(|n| ctx.text(&n)).unwrap_or("u32");
            let ty = match ty_text {
                "f32" => ScalarType::F32, "i32" => ScalarType::I32,
                "u32" => ScalarType::U32, "f16" => ScalarType::F16,
                "usize" => ScalarType::U32,
                _ => ScalarType::U32,
            };
            Ok(Expr::Cast(ty, Box::new(parse_expr(&inner, ctx)?)))
        },
        "call_expression" => {
            let func = node.child_by_field_name("function")
                .ok_or("call missing function")?;
            let func_text = ctx.text(&func).to_string();

            // Handle method calls: x.wrapping_add(y) → x + y, x.wrapping_sub(y) → x - y, etc.
            if func.kind() == "field_expression" {
                let receiver = func.child(0).ok_or("method call missing receiver")?;
                let method = func.child(2)
                    .or_else(|| func.child_by_field_name("field"))
                    .ok_or("method call missing method name")?;
                let method_name = ctx.text(&method);
                let args = parse_call_args(node, ctx)?;
                let receiver_expr = parse_expr(&receiver, ctx)?;

                return match method_name {
                    "wrapping_add" => {
                        let rhs = args.into_iter().next().ok_or("wrapping_add needs arg")?;
                        Ok(Expr::BinOp(BinOp::WrappingAdd, Box::new(receiver_expr), Box::new(rhs)))
                    },
                    "wrapping_sub" => {
                        let rhs = args.into_iter().next().ok_or("wrapping_sub needs arg")?;
                        Ok(Expr::BinOp(BinOp::WrappingSub, Box::new(receiver_expr), Box::new(rhs)))
                    },
                    "wrapping_mul" => {
                        let rhs = args.into_iter().next().ok_or("wrapping_mul needs arg")?;
                        Ok(Expr::BinOp(BinOp::WrappingMul, Box::new(receiver_expr), Box::new(rhs)))
                    },
                    _ => {
                        // Unknown method — treat as function call with receiver as first arg
                        let method_str = method_name.to_string();
                        let fn_id = ctx.fn_id(&method_str);
                        ctx.called_fns.insert(method_str);
                        let mut all_args = vec![receiver_expr];
                        all_args.extend(args);
                        Ok(Expr::Call(fn_id, all_args))
                    },
                };
            }

            // Regular function call — strip type prefixes like T:: or Self::
            let func_name = func_text.split("::").last().unwrap_or(&func_text).to_string();
            let args = parse_call_args(node, ctx)?;
            let fn_id = ctx.fn_id(&func_name);
            ctx.called_fns.insert(func_name);
            Ok(Expr::Call(fn_id, args))
        },
        _ => {
            if let Ok(val) = text.trim().parse::<i64>() {
                Ok(Expr::Const(val, ScalarType::I32))
            } else {
                Ok(Expr::Var(ctx.var_idx(text.trim())))
            }
        },
    }
}
