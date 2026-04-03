///  Tree-sitter based parser: Verus source → GPU Kernel IR.
///  Walks the CST and builds Kernel/Expr/Stmt types.
///  This is the trusted component (~300 lines).

use tree_sitter::Node;
use crate::types::*;

/// Parser state: source text + variable/buffer name tracking.
struct ParseCtx<'a> {
    source: &'a str,
    var_names: Vec<String>,     // local variable name → index
    buf_decls: Vec<BufDecl>,
    builtin_names: Vec<String>, // e.g., "gid.x"
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
}

/// Parse a complete #[gpu_kernel] function from source.
pub fn parse_gpu_kernel(source: &str) -> Result<Kernel, String> {
    let mut parser = tree_sitter::Parser::new();
    parser.set_language(&tree_sitter_verus::LANGUAGE.into())
        .map_err(|e| format!("Failed to load Verus grammar: {}", e))?;

    let tree = parser.parse(source.as_bytes(), None)
        .ok_or_else(|| "Failed to parse source".to_string())?;

    let root = tree.root_node();

    // Find the function_item inside verus! { }
    let fn_node = find_gpu_kernel_fn(&root, source)
        .ok_or_else(|| "No #[gpu_kernel] function found".to_string())?;

    parse_fn_to_kernel(&fn_node, source)
}

/// Recursively find a function_item with #[gpu_kernel] attribute.
/// In tree-sitter, attributes are often part of the function_item node
/// (as children), or may be preceding siblings. We check both.
fn find_gpu_kernel_fn<'a>(node: &Node<'a>, source: &str) -> Option<Node<'a>> {
    let mut cursor = node.walk();
    let children: Vec<Node<'a>> = node.children(&mut cursor).collect();
    let mut prev_has_gpu_kernel = false;

    for child in &children {
        // Check if this node itself contains gpu_kernel (e.g., attribute_item)
        let text = child.utf8_text(source.as_bytes()).unwrap_or("");
        if text.contains("gpu_kernel") {
            prev_has_gpu_kernel = true;
        }

        match child.kind() {
            "function_item" => {
                // The function_item in tree-sitter-verus includes its attributes
                // as children. Check the full node text.
                if text.contains("gpu_kernel") || prev_has_gpu_kernel {
                    return Some(*child);
                }
                prev_has_gpu_kernel = false;
            },
            _ => {
                // Recurse into containers
                if let Some(f) = find_gpu_kernel_fn(child, source) {
                    return Some(f);
                }
                // Reset flag if this wasn't an attribute
                if !text.contains("gpu_kernel") {
                    prev_has_gpu_kernel = false;
                }
            },
        }
    }
    None
}

/// Parse a function_item node into a Kernel.
fn parse_fn_to_kernel(fn_node: &Node, source: &str) -> Result<Kernel, String> {
    let mut ctx = ParseCtx {
        source,
        var_names: Vec::new(),
        buf_decls: Vec::new(),
        builtin_names: Vec::new(),
    };

    // Extract function name
    let name = fn_node.child_by_field_name("name")
        .map(|n| n.utf8_text(source.as_bytes()).unwrap_or("kernel").to_string())
        .unwrap_or_else(|| "kernel".to_string());

    // Parse workgroup_size from attributes
    let workgroup_size = parse_workgroup_size(fn_node, source);

    // Parse parameters — identify builtins and buffers
    if let Some(params) = fn_node.child_by_field_name("parameters") {
        parse_parameters(&params, &mut ctx);
    }

    // Parse body
    let body = if let Some(body_node) = fn_node.child_by_field_name("body") {
        parse_block(&body_node, &mut ctx)?
    } else {
        Stmt::Noop
    };

    Ok(Kernel {
        name,
        var_names: ctx.var_names,
        buf_decls: ctx.buf_decls,
        body,
        workgroup_size,
        builtin_names: ctx.builtin_names,
    })
}

/// Extract workgroup_size(X, Y, Z) from #[gpu_kernel(...)] attribute.
fn parse_workgroup_size(fn_node: &Node, source: &str) -> (u32, u32, u32) {
    let text = fn_node.utf8_text(source.as_bytes()).unwrap_or("");
    // Simple regex-free extraction: find "workgroup_size(" and parse numbers
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
    (256, 1, 1) // default
}

/// Parse function parameters — identify builtins (#[gpu_builtin]) and buffers (#[gpu_buffer]).
/// Attributes appear as `attribute_item` siblings BEFORE their `parameter` node.
fn parse_parameters(params_node: &Node, ctx: &mut ParseCtx) {
    let mut cursor = params_node.walk();
    let children: Vec<Node> = params_node.children(&mut cursor).collect();

    let mut pending_attr: Option<String> = None;

    for child in &children {
        let kind = child.kind();
        let text = ctx.text(child);

        if kind == "attribute_item" {
            // Store attribute text for the next parameter
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
    // Find the identifier in the parameter
    if let Some(pattern) = param_node.child_by_field_name("pattern") {
        return pattern.utf8_text(source.as_bytes()).unwrap_or("x").to_string();
    }
    // Fallback: scan children for identifier
    let mut cursor = param_node.walk();
    for child in param_node.children(&mut cursor) {
        if child.kind() == "identifier" {
            return child.utf8_text(source.as_bytes()).unwrap_or("x").to_string();
        }
    }
    "x".to_string()
}

/// Parse a block (surrounded by braces) into a Stmt.
fn parse_block(node: &Node, ctx: &mut ParseCtx) -> Result<Stmt, String> {
    let mut stmts = Vec::new();
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        let kind = child.kind();
        match kind {
            "{" | "}" => {},
            "let_declaration" => stmts.push(parse_let(&child, ctx)?),
            "expression_statement" => stmts.push(parse_expr_stmt(&child, ctx)?),
            "if_expression" => stmts.push(parse_if(&child, ctx)?),
            "for_expression" => stmts.push(parse_for(&child, ctx)?),
            "return_expression" => stmts.push(Stmt::Return),
            "break_expression" => stmts.push(Stmt::Break),
            "continue_expression" => stmts.push(Stmt::Continue),
            // Skip Verus-specific clauses
            "requires_clause" | "ensures_clause" | "decreases_clause"
            | "invariant_clause" | "recommends_clause" => {},
            _ => {
                // Try to parse as expression statement
                if let Ok(s) = parse_expr_to_stmt(&child, ctx) {
                    stmts.push(s);
                }
            },
        }
    }
    Ok(Stmt::from_vec(stmts))
}

fn parse_let(node: &Node, ctx: &mut ParseCtx) -> Result<Stmt, String> {
    let name = node.child_by_field_name("pattern")
        .map(|n| ctx.text(&n).to_string())
        .unwrap_or_else(|| "tmp".to_string());
    // Strip "mut " prefix if present
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
    // An expression statement wraps an inner expression + ";".
    // Unwrap and dispatch based on the inner node's kind.
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
                let idx_node = lhs_node.child(2) // skip '['
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

    // Expression as statement (e.g., function call)
    Ok(Stmt::Noop)
}

fn parse_if(node: &Node, ctx: &mut ParseCtx) -> Result<Stmt, String> {
    let cond_node = node.child_by_field_name("condition")
        .ok_or("if missing condition")?;
    let cond = parse_expr(&cond_node, ctx)?;

    let then_node = node.child_by_field_name("consequence")
        .ok_or("if missing then body")?;
    let then_body = parse_block(&then_node, ctx)?;

    let else_body = if let Some(alt) = node.child_by_field_name("alternative") {
        // Could be another if (else if) or a block (else)
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
    // for VAR in START..END { body }
    let pat = node.child_by_field_name("pattern")
        .map(|n| ctx.text(&n).to_string())
        .unwrap_or_else(|| "i".to_string());
    let var = ctx.var_idx(&pat);

    // Parse range expression (start..end)
    let value_node = node.child_by_field_name("value")
        .ok_or("for missing range")?;
    let (start, end) = parse_range(&value_node, ctx)?;

    let body_node = node.child_by_field_name("body")
        .ok_or("for missing body")?;
    let body = parse_block(&body_node, ctx)?;

    Ok(Stmt::For { var, start, end, body: Box::new(body) })
}

fn parse_range(node: &Node, ctx: &mut ParseCtx) -> Result<(Expr, Expr), String> {
    // range_expression: start..end
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
            //  Strip type suffix: u32, i32, u64, i64, usize, etc.
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
            // Check if it's a buffer name
            if let Some(buf) = ctx.buf_idx(text) {
                // Bare buffer reference — shouldn't happen in normal code
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
            // buf[idx]
            let base = node.child(0).ok_or("index missing base")?;
            let base_text = ctx.text(&base);
            if let Some(buf) = ctx.buf_idx(base_text) {
                // Find the index expression (between [ and ])
                let idx_node = node.child(2)
                    .ok_or("index missing index expr")?;
                let idx = parse_expr(&idx_node, ctx)?;
                Ok(Expr::ArrayRead(buf, Box::new(idx)))
            } else {
                // Variable indexing (not a buffer)
                Ok(Expr::Var(ctx.var_idx(base_text)))
            }
        },
        "parenthesized_expression" => {
            // (expr) — unwrap parens
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
            // Function call — for now, only barrier calls are handled as stmts
            // Other calls get inlined or become CallStmt
            let func = node.child_by_field_name("function")
                .ok_or("call missing function")?;
            let func_name = ctx.text(&func);
            // Simple: treat as variable reference for now
            Ok(Expr::Var(ctx.var_idx(func_name)))
        },
        _ => {
            // Fallback: try to parse as a number or variable
            if let Ok(val) = text.trim().parse::<i64>() {
                Ok(Expr::Const(val, ScalarType::I32))
            } else {
                Ok(Expr::Var(ctx.var_idx(text.trim())))
            }
        },
    }
}
