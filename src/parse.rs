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
}

impl<'a> ParseCtx<'a> {
    fn text(&self, node: &Node) -> &'a str {
        node.utf8_text(self.source.as_bytes()).unwrap_or("")
    }

    /// Get or create a variable index for the given name.
    fn var_idx(&mut self, name: &str) -> u32 {
        if let Some(idx) = self.var_names.iter().position(|n| n == name) {
            idx as u32
        } else {
            let idx = self.var_names.len() as u32;
            self.var_names.push(name.to_string());
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
pub fn parse_gpu_kernel(source: &str) -> Result<Kernel, String> {
    let mut parser = tree_sitter::Parser::new();
    parser.set_language(&tree_sitter_verus::LANGUAGE.into())
        .map_err(|e| format!("Failed to load Verus grammar: {}", e))?;

    let tree = parser.parse(source.as_bytes(), None)
        .ok_or_else(|| "Failed to parse source".to_string())?;

    let root = tree.root_node();

    // Phase 1: Find all function_item nodes in the file
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

    // Phase 4: Reachability walk — find all transitively called functions
    let mut reachable: Vec<String> = Vec::new();
    let mut visited: HashSet<String> = HashSet::new();
    let mut queue: VecDeque<String> = ctx.called_fns.iter().cloned().collect();

    while let Some(fn_name) = queue.pop_front() {
        if visited.contains(&fn_name) { continue; }
        if !all_fns.contains_key(&fn_name) { continue; } // external/unknown
        visited.insert(fn_name.clone());
        reachable.push(fn_name.clone());

        // Parse this function to discover ITS callees
        let fn_node = &all_fns[&fn_name];
        let mut helper_ctx = ParseCtx {
            source,
            var_names: Vec::new(),
            buf_decls: Vec::new(), // helpers don't have buffers
            builtin_names: Vec::new(),
            fn_name_to_id: ctx.fn_name_to_id.clone(),
            called_fns: HashSet::new(),
        };
        // Parse helper params as regular variables
        if let Some(params) = fn_node.child_by_field_name("parameters") {
            parse_helper_parameters(&params, &mut helper_ctx);
        }
        if let Some(body_node) = fn_node.child_by_field_name("body") {
            let _ = parse_block(&body_node, &mut helper_ctx);
        }
        // Merge fn_name_to_id back (new functions may have been discovered)
        for (k, v) in &helper_ctx.fn_name_to_id {
            ctx.fn_name_to_id.entry(k.clone()).or_insert(*v);
        }
        // Enqueue newly discovered callees
        for callee in &helper_ctx.called_fns {
            if !visited.contains(callee) {
                queue.push_back(callee.clone());
            }
        }
    }

    // Phase 5: Parse reachable functions into GpuFunction structs
    // Assign stable fn_ids based on order discovered
    let mut fn_id_map: HashMap<String, u32> = HashMap::new();
    let mut functions: Vec<GpuFunction> = Vec::new();

    for (i, fn_name) in reachable.iter().enumerate() {
        fn_id_map.insert(fn_name.clone(), i as u32);
    }

    for fn_name in &reachable {
        let fn_node = &all_fns[fn_name];
        let func = parse_helper_function(fn_node, source, &fn_id_map)?;
        functions.push(func);
    }

    // Phase 6: Re-parse kernel body with final fn_id_map so CallStmt/Call IDs are correct
    let mut final_ctx = ParseCtx {
        source,
        var_names: Vec::new(),
        buf_decls: Vec::new(),
        builtin_names: Vec::new(),
        fn_name_to_id: fn_id_map.iter().map(|(k, v)| (k.clone(), *v)).collect(),
        called_fns: HashSet::new(),
    };

    if let Some(params) = kernel_fn_node.child_by_field_name("parameters") {
        parse_parameters(&params, &mut final_ctx);
    }

    let final_body = if let Some(body_node) = kernel_fn_node.child_by_field_name("body") {
        parse_block(&body_node, &mut final_ctx)?
    } else {
        Stmt::Noop
    };

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
    })
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
    let ret_var = ctx.var_idx("__ret");

    let body = if let Some(body_node) = fn_node.child_by_field_name("body") {
        parse_block(&body_node, &mut ctx)?
    } else {
        Stmt::Noop
    };

    Ok(GpuFunction {
        name,
        params,
        ret_type,
        var_names: ctx.var_names,
        body,
        ret_var,
    })
}

/// Parse parameters as typed variables (for helper functions, not kernel).
fn parse_typed_parameters(params_node: &Node, ctx: &mut ParseCtx) -> Vec<(String, ScalarType)> {
    let mut result = Vec::new();
    let mut cursor = params_node.walk();
    let children: Vec<Node> = params_node.children(&mut cursor).collect();

    for child in &children {
        if child.kind() == "parameter" {
            let name = extract_param_name(child, ctx.source);
            let text = ctx.text(child);
            let ty = infer_scalar_type(text);
            ctx.var_idx(&name);
            result.push((name, ty));
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
fn parse_return_type(fn_node: &Node, source: &str) -> Option<ScalarType> {
    let text = fn_node.utf8_text(source.as_bytes()).unwrap_or("");
    // Look for "-> TYPE" pattern before the body "{"
    if let Some(arrow_pos) = text.find("->") {
        let after = &text[arrow_pos + 2..];
        let ty_text = after.split(|c: char| c == '{' || c == '\n')
            .next().unwrap_or("").trim();
        // Strip Verus return name pattern: (name: Type)
        let ty_text = if ty_text.starts_with('(') && ty_text.contains(':') {
            ty_text.split(':').last().unwrap_or("").trim().trim_end_matches(')')
        } else {
            ty_text
        };
        return Some(infer_scalar_type(ty_text));
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
fn parse_workgroup_size(fn_node: &Node, source: &str) -> (u32, u32, u32) {
    let text = fn_node.utf8_text(source.as_bytes()).unwrap_or("");
    if let Some(start) = text.find("workgroup_size(") {
        let rest = &text[start + "workgroup_size(".len()..];
        if let Some(end) = rest.find(')') {
            let nums: Vec<u32> = rest[..end].split(',')
                .filter_map(|s| s.trim().parse().ok())
                .collect();
            return (
                nums.first().copied().unwrap_or(1),
                nums.get(1).copied().unwrap_or(1),
                nums.get(2).copied().unwrap_or(1),
            );
        }
    }
    (256, 1, 1)
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
            "let_declaration" => stmts.push(parse_let(child, ctx)?),
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
            "return_expression" => {
                // return expr → assign to __ret + Return
                let mut cursor2 = child.walk();
                let ret_children: Vec<Node> = child.children(&mut cursor2).collect();
                for rc in &ret_children {
                    if rc.kind() != "return" && rc.kind() != ";" {
                        let rhs = parse_expr(rc, ctx)?;
                        let ret_var = ctx.var_idx("__ret");
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
            "proof_block" | "ghost_block" => {},
            _ => {
                // Check if this is a bare expression (implicit return) at end of block
                if is_last && kind != "expression_statement" {
                    // Bare expression without ; → implicit return
                    if let Ok(expr) = parse_expr(child, ctx) {
                        let ret_var = ctx.var_idx("__ret");
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
    let name = node.child_by_field_name("pattern")
        .map(|n| ctx.text(&n).to_string())
        .unwrap_or_else(|| "tmp".to_string());
    let name = name.strip_prefix("mut ").unwrap_or(&name).trim().to_string();
    let var = ctx.var_idx(&name);

    let rhs = if let Some(val) = node.child_by_field_name("value") {
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
    let result_var = ctx.var_idx("__call_tmp");

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
        "identifier" | "scoped_identifier" => {
            if let Some(_buf) = ctx.buf_idx(text) {
                Ok(Expr::Var(ctx.var_idx(text)))
            } else {
                Ok(Expr::Var(ctx.var_idx(text)))
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
            let op = match op_text {
                "-" => UnaryOp::Neg, "!" => UnaryOp::LogicalNot,
                "~" => UnaryOp::BitNot,
                _ => UnaryOp::Neg,
            };
            Ok(Expr::UnaryOp(op, Box::new(parse_expr(&operand, ctx)?)))
        },
        "index_expression" => {
            let base = node.child(0).ok_or("index missing base")?;
            let base_text = ctx.text(&base);
            if let Some(buf) = ctx.buf_idx(base_text) {
                let idx_node = node.child(2)
                    .ok_or("index missing index expr")?;
                let idx = parse_expr(&idx_node, ctx)?;
                Ok(Expr::ArrayRead(buf, Box::new(idx)))
            } else {
                Ok(Expr::Var(ctx.var_idx(base_text)))
            }
        },
        "parenthesized_expression" => {
            let inner = node.child(1).ok_or("paren missing inner")?;
            parse_expr(&inner, ctx)
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
            // Function call as expression: f(args) → Call(fn_id, args)
            let func = node.child_by_field_name("function")
                .ok_or("call missing function")?;
            let func_name = ctx.text(&func).to_string();
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
