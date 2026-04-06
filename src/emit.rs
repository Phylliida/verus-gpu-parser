///  WGSL emitter — plain Rust version mirroring the verified emitter.
///  Structural correspondence with wgsl_emit.rs ensures correctness.
///  Supports helper functions, function calls, and tuple types (as WGSL structs).

use crate::types::*;
use std::collections::BTreeSet;

fn binop_str(op: &BinOp) -> &'static str {
    match op {
        BinOp::Add | BinOp::WrappingAdd | BinOp::FAdd => "+",
        BinOp::Sub | BinOp::WrappingSub | BinOp::FSub => "-",
        BinOp::Mul | BinOp::WrappingMul | BinOp::FMul => "*",
        BinOp::Div | BinOp::FDiv => "/",
        BinOp::Mod => "%", BinOp::Shr => ">>", BinOp::Shl => "<<",
        BinOp::Lt => "<", BinOp::Le => "<=", BinOp::Gt => ">", BinOp::Ge => ">=",
        BinOp::Eq => "==", BinOp::Ne => "!=",
        BinOp::BitAnd => "&", BinOp::BitOr => "|", BinOp::BitXor => "^",
        BinOp::LogicalAnd => "&&", BinOp::LogicalOr => "||",
    }
}

fn unaryop_str(op: &UnaryOp) -> &'static str {
    match op {
        UnaryOp::Neg | UnaryOp::FNeg => "-",
        UnaryOp::BitNot => "~",
        UnaryOp::LogicalNot => "!",
    }
}

fn scalar_type_str(ty: &ScalarType) -> &'static str {
    match ty {
        ScalarType::I32 => "i32", ScalarType::U32 => "u32",
        ScalarType::F32 => "f32", ScalarType::F16 => "f16",
        ScalarType::Bool => "bool",
    }
}

/// Generate a WGSL struct name for a tuple of given arity, e.g. "R2", "R4", "R5", "R8".
fn tuple_struct_name(arity: usize) -> String {
    format!("R{}", arity)
}

/// Generate the WGSL return type string for a ReturnType.
fn return_type_str(rt: &ReturnType) -> String {
    match rt {
        ReturnType::Scalar(ty) => scalar_type_str(ty).to_string(),
        ReturnType::Tuple(types) => tuple_struct_name(types.len()),
    }
}

fn var_name(names: &[String], idx: u32) -> String {
    names.get(idx as usize).cloned().unwrap_or_else(|| format!("v_{}", idx))
}

fn buf_name(decls: &[BufDecl], idx: u32) -> String {
    decls.get(idx as usize).map(|d| d.name.clone()).unwrap_or_else(|| format!("buf_{}", idx))
}

fn fn_name(funcs: &[GpuFunction], idx: u32) -> String {
    funcs.get(idx as usize).map(|f| {
        let base = if f.vec_buffer_map.is_empty() {
            f.name.clone()
        } else {
            let buf_suffix: Vec<&str> = f.vec_buffer_map.iter()
                .map(|(_, buf)| buf.as_str())
                .collect();
            format!("{}_{}", f.name, buf_suffix.join("_"))
        };
        // If this function is recursive, callers should use the max-depth variant
        let is_recursive = stmt_calls_fn(&f.body, idx);
        if is_recursive {
            format!("{}_d{}", base, MAX_RECURSION_DEPTH)
        } else {
            base
        }
    }).unwrap_or_else(|| format!("fn_{}", idx))
}

pub fn emit_expr(e: &Expr, var_names: &[String], buf_decls: &[BufDecl], funcs: &[GpuFunction]) -> String {
    emit_expr_ctx(e, var_names, buf_decls, funcs, &[])
}

/// Emit an expression. `vec_buf_map` maps var_name → buffer_name for Vec params.
fn emit_expr_ctx(e: &Expr, var_names: &[String], buf_decls: &[BufDecl], funcs: &[GpuFunction], vec_buf_map: &[(String, String)]) -> String {
    match e {
        Expr::Const(val, ty) => match ty {
            ScalarType::F32 | ScalarType::F16 => format!("{}.0f", val),
            ScalarType::I32 => format!("i32({})", val),
            _ => format!("{}u", val),
        },
        Expr::FConst(bits) => format!("bitcast<f32>(0x{:08x}u)", bits),
        Expr::Var(idx) => var_name(var_names, *idx),
        Expr::Builtin(idx) => var_name(var_names, *idx),
        Expr::BinOp(op, a, b) =>
            format!("({} {} {})", emit_expr_ctx(a, var_names, buf_decls, funcs, vec_buf_map),
                    binop_str(op), emit_expr_ctx(b, var_names, buf_decls, funcs, vec_buf_map)),
        Expr::UnaryOp(op, a) =>
            format!("({}{})", unaryop_str(op), emit_expr_ctx(a, var_names, buf_decls, funcs, vec_buf_map)),
        Expr::Select(c, t, f) =>
            format!("select({}, {}, {})",
                    emit_expr_ctx(f, var_names, buf_decls, funcs, vec_buf_map),
                    emit_expr_ctx(t, var_names, buf_decls, funcs, vec_buf_map),
                    emit_expr_ctx(c, var_names, buf_decls, funcs, vec_buf_map)),
        Expr::ArrayRead(buf, idx) =>
            format!("{}[{}]", buf_name(buf_decls, *buf), emit_expr_ctx(idx, var_names, buf_decls, funcs, vec_buf_map)),
        Expr::Cast(ty, inner) =>
            format!("{}({})", scalar_type_str(ty), emit_expr_ctx(inner, var_names, buf_decls, funcs, vec_buf_map)),
        Expr::Call(fn_id, args) => {
            let name = fn_name(funcs, *fn_id);
            let arg_strs: Vec<String> = args.iter()
                .map(|a| emit_expr_ctx(a, var_names, buf_decls, funcs, vec_buf_map))
                .collect();
            format!("{}({})", name, arg_strs.join(", "))
        },
        Expr::TupleConstruct(elems) => {
            let arity = elems.len();
            let elem_strs: Vec<String> = elems.iter()
                .map(|e| emit_expr_ctx(e, var_names, buf_decls, funcs, vec_buf_map))
                .collect();
            format!("{}({})", tuple_struct_name(arity), elem_strs.join(", "))
        },
        Expr::TupleAccess(base, idx) => {
            format!("{}._{}", emit_expr_ctx(base, var_names, buf_decls, funcs, vec_buf_map), idx)
        },
        Expr::ScratchRead(offset) => {
            format!("scratch[{}]", emit_expr_ctx(offset, var_names, buf_decls, funcs, vec_buf_map))
        },
        Expr::VecIndex(var, idx) => {
            let vn = var_name(var_names, *var);
            // Look up buffer name from vec_buf_map
            let buf = vec_buf_map.iter()
                .find(|(param, _)| param == &vn)
                .map(|(_, buf)| buf.as_str())
                .unwrap_or("scratch");
            format!("{}[({} + {})]", buf, vn,
                    emit_expr_ctx(idx, var_names, buf_decls, funcs, vec_buf_map))
        },
        Expr::BufSlice(buf, offset) => {
            // Buffer slice reference — should only appear as function argument,
            // not as a standalone expression. Emit as the offset.
            emit_expr_ctx(offset, var_names, buf_decls, funcs, vec_buf_map)
        },
    }
}

pub fn emit_stmt(s: &Stmt, var_names: &[String], buf_decls: &[BufDecl], funcs: &[GpuFunction], depth: usize) -> String {
    emit_stmt_ctx(s, var_names, buf_decls, funcs, depth, &[])
}

fn emit_stmt_ctx(s: &Stmt, var_names: &[String], buf_decls: &[BufDecl], funcs: &[GpuFunction], depth: usize, vec_buf_map: &[(String, String)]) -> String {
    let pad = "  ".repeat(depth);
    match s {
        Stmt::Assign { var, rhs } => {
            let vn = var_name(var_names, *var);
            // When assigning to _ret and rhs is a tuple with a Vec element,
            // and we have an output buffer, extract just the scalar part.
            if vn == "_ret" {
                if let Expr::TupleConstruct(elems) = rhs {
                    // Check if any element is a Vec var (mapped to buffer)
                    let has_output_vec = !vec_buf_map.is_empty() &&
                        vec_buf_map.iter().any(|(name, _)| name == "out");
                    if has_output_vec && elems.len() == 2 {
                        // (out_vec, scalar) → just assign scalar to _ret
                        return format!("{}{} = {};\n", pad, vn,
                            emit_expr_ctx(&elems[1], var_names, buf_decls, funcs, vec_buf_map));
                    }
                }
            }
            format!("{}{} = {};\n", pad, vn,
                    emit_expr_ctx(rhs, var_names, buf_decls, funcs, vec_buf_map))
        },
        Stmt::BufWrite { buf, idx, val } =>
            format!("{}{}[{}] = {};\n", pad, buf_name(buf_decls, *buf),
                    emit_expr_ctx(idx, var_names, buf_decls, funcs, vec_buf_map),
                    emit_expr_ctx(val, var_names, buf_decls, funcs, vec_buf_map)),
        Stmt::CallStmt { fn_id, args, result_var } => {
            let name = fn_name(funcs, *fn_id);
            let arg_strs: Vec<String> = args.iter()
                .map(|a| emit_expr_ctx(a, var_names, buf_decls, funcs, vec_buf_map))
                .collect();
            format!("{}{} = {}({});\n", pad, var_name(var_names, *result_var),
                    name, arg_strs.join(", "))
        },
        Stmt::TupleDestructure { vars, rhs } => {
            // let (a, b, c) = expr; → { var _td = expr; a = _td._0; b = _td._1; ... }
            // Use a block scope so _td doesn't conflict with other destructures.
            let mut s = format!("{}{{\n", pad);
            s.push_str(&format!("{}  var _td = {};\n", pad, emit_expr_ctx(rhs, var_names, buf_decls, funcs, vec_buf_map)));
            for (i, var) in vars.iter().enumerate() {
                s.push_str(&format!("{}  {} = _td._{};\n", pad, var_name(var_names, *var), i));
            }
            s.push_str(&format!("{}}}\n", pad));
            s
        },
        Stmt::Seq { first, then } => {
            let mut s = emit_stmt_ctx(first, var_names, buf_decls, funcs, depth, vec_buf_map);
            s.push_str(&emit_stmt_ctx(then, var_names, buf_decls, funcs, depth, vec_buf_map));
            s
        },
        Stmt::If { cond, then_body, else_body } => {
            let mut s = format!("{}if ({}) {{\n", pad, emit_expr_ctx(cond, var_names, buf_decls, funcs, vec_buf_map));
            s.push_str(&emit_stmt_ctx(then_body, var_names, buf_decls, funcs, depth + 1, vec_buf_map));
            s.push_str(&format!("{}}} else {{\n", pad));
            s.push_str(&emit_stmt_ctx(else_body, var_names, buf_decls, funcs, depth + 1, vec_buf_map));
            s.push_str(&format!("{}}}\n", pad));
            s
        },
        Stmt::For { var, start, end, body } => {
            let vn = var_name(var_names, *var);
            let mut s = format!("{}for (var {}: u32 = {}; {} < {}; {}++) {{\n",
                pad, vn, emit_expr_ctx(start, var_names, buf_decls, funcs, vec_buf_map),
                vn, emit_expr_ctx(end, var_names, buf_decls, funcs, vec_buf_map), vn);
            s.push_str(&emit_stmt_ctx(body, var_names, buf_decls, funcs, depth + 1, vec_buf_map));
            s.push_str(&format!("{}}}\n", pad));
            s
        },
        Stmt::Break => format!("{}break;\n", pad),
        Stmt::Continue => format!("{}continue;\n", pad),
        Stmt::Barrier { scope } => {
            let name = match scope {
                BarrierScope::Workgroup => "workgroupBarrier()",
                BarrierScope::Storage => "storageBarrier()",
                BarrierScope::Subgroup => "subgroupBarrier()",
            };
            format!("{}{};\n", pad, name)
        },
        Stmt::ScratchWrite { offset, val } => {
            format!("{}scratch[{}] = {};\n", pad,
                    emit_expr_ctx(offset, var_names, buf_decls, funcs, vec_buf_map),
                    emit_expr_ctx(val, var_names, buf_decls, funcs, vec_buf_map))
        },
        Stmt::VecPush { vec_var, val } => {
            let vn = var_name(var_names, *vec_var);
            let buf = vec_buf_map.iter()
                .find(|(param, _)| param == &vn)
                .map(|(_, b)| b.as_str())
                .unwrap_or("scratch");
            let len_var = format!("{}_len", vn);
            format!("{}{}[({} + {})] = {};\n{}{} = {} + 1u;\n",
                    pad, buf, vn, len_var, emit_expr_ctx(val, var_names, buf_decls, funcs, vec_buf_map),
                    pad, len_var, len_var)
        },
        Stmt::Return => format!("{}return;\n", pad),
        Stmt::Noop => String::new(),
    }
}

/// Collect all tuple arities used in the kernel and its functions.
fn collect_tuple_arities(k: &Kernel) -> BTreeSet<usize> {
    let mut arities = BTreeSet::new();
    for f in &k.functions {
        if let Some(ReturnType::Tuple(types)) = &f.ret_type {
            arities.insert(types.len());
        }
        collect_arities_from_stmt(&f.body, &mut arities);
    }
    collect_arities_from_stmt(&k.body, &mut arities);
    arities
}

fn collect_arities_from_stmt(s: &Stmt, arities: &mut BTreeSet<usize>) {
    match s {
        Stmt::Assign { rhs, .. } => collect_arities_from_expr(rhs, arities),
        Stmt::BufWrite { idx, val, .. } => {
            collect_arities_from_expr(idx, arities);
            collect_arities_from_expr(val, arities);
        },
        Stmt::CallStmt { args, .. } => {
            for a in args { collect_arities_from_expr(a, arities); }
        },
        Stmt::TupleDestructure { rhs, .. } => collect_arities_from_expr(rhs, arities),
        Stmt::ScratchWrite { offset, val } => {
            collect_arities_from_expr(offset, arities);
            collect_arities_from_expr(val, arities);
        },
        Stmt::VecPush { val, .. } => collect_arities_from_expr(val, arities),
        Stmt::Seq { first, then } => {
            collect_arities_from_stmt(first, arities);
            collect_arities_from_stmt(then, arities);
        },
        Stmt::If { cond, then_body, else_body } => {
            collect_arities_from_expr(cond, arities);
            collect_arities_from_stmt(then_body, arities);
            collect_arities_from_stmt(else_body, arities);
        },
        Stmt::For { start, end, body, .. } => {
            collect_arities_from_expr(start, arities);
            collect_arities_from_expr(end, arities);
            collect_arities_from_stmt(body, arities);
        },
        _ => {},
    }
}

fn collect_arities_from_expr(e: &Expr, arities: &mut BTreeSet<usize>) {
    match e {
        Expr::TupleConstruct(elems) => {
            arities.insert(elems.len());
            for el in elems { collect_arities_from_expr(el, arities); }
        },
        Expr::TupleAccess(base, _) => collect_arities_from_expr(base, arities),
        Expr::BinOp(_, a, b) => {
            collect_arities_from_expr(a, arities);
            collect_arities_from_expr(b, arities);
        },
        Expr::UnaryOp(_, a) => collect_arities_from_expr(a, arities),
        Expr::Select(c, t, f) => {
            collect_arities_from_expr(c, arities);
            collect_arities_from_expr(t, arities);
            collect_arities_from_expr(f, arities);
        },
        Expr::ArrayRead(_, idx) => collect_arities_from_expr(idx, arities),
        Expr::Cast(_, inner) => collect_arities_from_expr(inner, arities),
        Expr::Call(_, args) => {
            for a in args { collect_arities_from_expr(a, arities); }
        },
        Expr::ScratchRead(offset) => collect_arities_from_expr(offset, arities),
        Expr::VecIndex(_, idx) => collect_arities_from_expr(idx, arities),
        Expr::BufSlice(_, offset) => collect_arities_from_expr(offset, arities),
        _ => {},
    }
}

/// Emit a WGSL struct definition for a tuple of given arity.
fn emit_tuple_struct(arity: usize) -> String {
    let mut s = format!("struct {} {{\n", tuple_struct_name(arity));
    for i in 0..arity {
        s.push_str(&format!("  _{}: u32,\n", i));
    }
    s.push_str("}\n\n");
    s
}

/// Check if an expression tree contains a Call to a specific fn_id.
fn expr_calls_fn(e: &Expr, fn_id: u32) -> bool {
    match e {
        Expr::Call(id, args) => {
            if *id == fn_id { return true; }
            args.iter().any(|a| expr_calls_fn(a, fn_id))
        },
        Expr::BinOp(_, a, b) => expr_calls_fn(a, fn_id) || expr_calls_fn(b, fn_id),
        Expr::UnaryOp(_, a) => expr_calls_fn(a, fn_id),
        Expr::Select(c, t, f) => expr_calls_fn(c, fn_id) || expr_calls_fn(t, fn_id) || expr_calls_fn(f, fn_id),
        Expr::TupleConstruct(elems) => elems.iter().any(|e| expr_calls_fn(e, fn_id)),
        Expr::TupleAccess(base, _) => expr_calls_fn(base, fn_id),
        Expr::ArrayRead(_, idx) => expr_calls_fn(idx, fn_id),
        Expr::Cast(_, inner) => expr_calls_fn(inner, fn_id),
        Expr::VecIndex(_, idx) => expr_calls_fn(idx, fn_id),
        Expr::ScratchRead(offset) => expr_calls_fn(offset, fn_id),
        Expr::BufSlice(_, offset) => expr_calls_fn(offset, fn_id),
        _ => false,
    }
}

fn stmt_calls_fn(s: &Stmt, fn_id: u32) -> bool {
    match s {
        Stmt::Assign { rhs, .. } => expr_calls_fn(rhs, fn_id),
        Stmt::TupleDestructure { rhs, .. } => expr_calls_fn(rhs, fn_id),
        Stmt::CallStmt { fn_id: id, args, .. } => *id == fn_id || args.iter().any(|a| expr_calls_fn(a, fn_id)),
        Stmt::Seq { first, then } => stmt_calls_fn(first, fn_id) || stmt_calls_fn(then, fn_id),
        Stmt::If { cond, then_body, else_body } => expr_calls_fn(cond, fn_id) || stmt_calls_fn(then_body, fn_id) || stmt_calls_fn(else_body, fn_id),
        Stmt::For { start, end, body, .. } => expr_calls_fn(start, fn_id) || expr_calls_fn(end, fn_id) || stmt_calls_fn(body, fn_id),
        Stmt::VecPush { val, .. } => expr_calls_fn(val, fn_id),
        Stmt::ScratchWrite { offset, val } => expr_calls_fn(offset, fn_id) || expr_calls_fn(val, fn_id),
        _ => false,
    }
}

const MAX_RECURSION_DEPTH: u32 = 6; // supports up to N=2^6=64 limbs

/// Emit a helper function as WGSL. `fn_idx` is this function's index in all_funcs.
fn emit_function(f: &GpuFunction, all_funcs: &[GpuFunction], fn_idx: usize) -> String {
    // Check for self-recursion
    let is_recursive = stmt_calls_fn(&f.body, fn_idx as u32);
    if is_recursive {
        return emit_recursive_unrolled(f, all_funcs, fn_idx);
    }
    emit_function_single(f, all_funcs, fn_idx, None, &None)
}

/// Emit all depth-stratified copies of a recursive function.
fn emit_recursive_unrolled(f: &GpuFunction, all_funcs: &[GpuFunction], fn_idx: usize) -> String {
    let mut s = String::new();

    // Use the annotated #[gpu_base_case(name)] from the function
    let base_case_name = f.base_case.clone();

    // Emit depth 0 through MAX_RECURSION_DEPTH
    for depth in 0..=MAX_RECURSION_DEPTH {
        s.push_str(&emit_function_single(f, all_funcs, fn_idx, Some(depth), &base_case_name));
    }
    s
}

/// Find the base case function for recursion unrolling.
/// Scans the function body for Call expressions to non-self functions.
fn find_base_case_fn(f: &GpuFunction, all_funcs: &[GpuFunction], fn_idx: usize) -> Option<String> {
    find_non_self_call(&f.body, all_funcs, fn_idx as u32)
}

fn find_non_self_call(s: &Stmt, funcs: &[GpuFunction], self_id: u32) -> Option<String> {
    match s {
        Stmt::Assign { rhs, .. } => find_non_self_call_expr(rhs, funcs, self_id),
        Stmt::TupleDestructure { rhs, .. } => find_non_self_call_expr(rhs, funcs, self_id),
        Stmt::Seq { first, then } =>
            find_non_self_call(first, funcs, self_id).or_else(|| find_non_self_call(then, funcs, self_id)),
        Stmt::If { then_body, else_body, .. } =>
            find_non_self_call(then_body, funcs, self_id).or_else(|| find_non_self_call(else_body, funcs, self_id)),
        Stmt::For { body, .. } => find_non_self_call(body, funcs, self_id),
        _ => None,
    }
}

fn find_non_self_call_expr(e: &Expr, funcs: &[GpuFunction], self_id: u32) -> Option<String> {
    match e {
        Expr::Call(fn_id, args) => {
            if *fn_id != self_id {
                // Found a non-self call — this is likely the base case
                return Some(fn_name(funcs, *fn_id));
            }
            for a in args { if let Some(r) = find_non_self_call_expr(a, funcs, self_id) { return Some(r); } }
            None
        },
        Expr::BinOp(_, a, b) => find_non_self_call_expr(a, funcs, self_id).or_else(|| find_non_self_call_expr(b, funcs, self_id)),
        Expr::TupleConstruct(elems) => elems.iter().find_map(|e| find_non_self_call_expr(e, funcs, self_id)),
        _ => None,
    }
}

/// Emit a single function, optionally at a specific recursion depth.
/// When depth is Some(d), self-calls are replaced:
///   d > 0: self-call → call to _d{d-1}
///   d == 0: self-call → call to base case (schoolbook)
fn emit_function_single(f: &GpuFunction, all_funcs: &[GpuFunction], fn_idx: usize, depth: Option<u32>, base_case: &Option<String>) -> String {
    let mut s = String::new();

    let ret_ty = if f.returns_vec {
        "u32".to_string()
    } else {
        f.ret_type.as_ref()
            .map(|rt| return_type_str(rt))
            .unwrap_or_else(|| "u32".to_string())
    };

    // Build name with buffer suffix and depth suffix
    let base_name = if f.vec_buffer_map.is_empty() {
        f.name.clone()
    } else {
        let buf_suffix: Vec<&str> = f.vec_buffer_map.iter()
            .map(|(_, buf)| buf.as_str())
            .collect();
        format!("{}_{}", f.name, buf_suffix.join("_"))
    };
    let fn_display_name = match depth {
        Some(d) => format!("{}_d{}", base_name, d),
        None => base_name.clone(),
    };

    // Signature
    let mut param_strs: Vec<String> = f.params.iter()
        .map(|(name, ty)| match ty {
            ParamType::Scalar(sty) => format!("{}: {}", name, scalar_type_str(sty)),
            ParamType::VecU32 => format!("{}: u32", name),
        })
        .collect();
    if f.returns_vec {
        param_strs.push("out: u32".to_string());
    }
    s.push_str(&format!("fn {}({}) -> {} {{\n", fn_display_name, param_strs.join(", "), ret_ty));

    // Local variables — skip params and special names to avoid redefinition
    let param_names: std::collections::HashSet<&str> = f.params.iter()
        .map(|(name, _)| name.as_str()).collect();
    let param_count = f.params.len();
    for i in param_count..f.var_names.len() {
        let vn = &f.var_names[i];
        if vn == "_ret" || vn == "_call_tmp" || vn == "out" { continue; }
        if param_names.contains(vn.as_str()) { continue; }
        s.push_str(&format!("  var {}: u32;\n", vn));
    }
    s.push_str(&format!("  var _ret: {};\n", ret_ty));

    // Body — emit with depth-aware call replacement
    let empty_bufs: Vec<BufDecl> = Vec::new();
    let recursion_ctx = depth.map(|d| RecursionCtx {
        self_fn_idx: fn_idx as u32,
        current_depth: d,
        base_name: base_name.clone(),
        base_case_name: base_case.clone(),
    });
    s.push_str(&emit_stmt_depth(&f.body, &f.var_names, &empty_bufs, all_funcs, 1,
                                 &f.vec_buffer_map, &recursion_ctx));

    s.push_str("  return _ret;\n");
    s.push_str("}\n\n");
    s
}

/// Context for recursion unrolling.
struct RecursionCtx {
    self_fn_idx: u32,
    current_depth: u32,
    base_name: String,
    /// Name of the base-case function (used at depth 0 to replace self-calls).
    base_case_name: Option<String>,
}

/// Emit a statement with recursion-aware call replacement.
fn emit_stmt_depth(s: &Stmt, var_names: &[String], buf_decls: &[BufDecl], funcs: &[GpuFunction],
                   depth: usize, vec_buf_map: &[(String, String)], rec: &Option<RecursionCtx>) -> String {
    let pad = "  ".repeat(depth);
    match s {
        Stmt::Assign { var, rhs } => {
            let vn = var_name(var_names, *var);
            if vn == "_ret" {
                if let Expr::TupleConstruct(elems) = rhs {
                    let has_output_vec = !vec_buf_map.is_empty() &&
                        vec_buf_map.iter().any(|(name, _)| name == "out");
                    if has_output_vec && elems.len() == 2 {
                        return format!("{}{} = {};\n", pad, vn,
                            emit_expr_depth(&elems[1], var_names, buf_decls, funcs, vec_buf_map, rec));
                    }
                }
            }
            format!("{}{} = {};\n", pad, vn,
                    emit_expr_depth(rhs, var_names, buf_decls, funcs, vec_buf_map, rec))
        },
        Stmt::BufWrite { buf, idx, val } =>
            format!("{}{}[{}] = {};\n", pad, buf_name(buf_decls, *buf),
                    emit_expr_depth(idx, var_names, buf_decls, funcs, vec_buf_map, rec),
                    emit_expr_depth(val, var_names, buf_decls, funcs, vec_buf_map, rec)),
        Stmt::CallStmt { fn_id, args, result_var } => {
            let name = resolve_fn_name(funcs, *fn_id, rec);
            let arg_strs: Vec<String> = args.iter()
                .map(|a| emit_expr_depth(a, var_names, buf_decls, funcs, vec_buf_map, rec))
                .collect();
            format!("{}{} = {}({});\n", pad, var_name(var_names, *result_var),
                    name, arg_strs.join(", "))
        },
        Stmt::TupleDestructure { vars, rhs } => {
            let mut s = format!("{}{{\n", pad);
            s.push_str(&format!("{}  var _td = {};\n", pad,
                emit_expr_depth(rhs, var_names, buf_decls, funcs, vec_buf_map, rec)));
            for (i, var) in vars.iter().enumerate() {
                s.push_str(&format!("{}  {} = _td._{};\n", pad, var_name(var_names, *var), i));
            }
            s.push_str(&format!("{}}}\n", pad));
            s
        },
        Stmt::Seq { first, then } => {
            let mut s = emit_stmt_depth(first, var_names, buf_decls, funcs, depth, vec_buf_map, rec);
            s.push_str(&emit_stmt_depth(then, var_names, buf_decls, funcs, depth, vec_buf_map, rec));
            s
        },
        Stmt::If { cond, then_body, else_body } => {
            let mut s = format!("{}if ({}) {{\n", pad,
                emit_expr_depth(cond, var_names, buf_decls, funcs, vec_buf_map, rec));
            s.push_str(&emit_stmt_depth(then_body, var_names, buf_decls, funcs, depth + 1, vec_buf_map, rec));
            s.push_str(&format!("{}}} else {{\n", pad));
            s.push_str(&emit_stmt_depth(else_body, var_names, buf_decls, funcs, depth + 1, vec_buf_map, rec));
            s.push_str(&format!("{}}}\n", pad));
            s
        },
        Stmt::For { var, start, end, body } => {
            let vn = var_name(var_names, *var);
            let mut s = format!("{}for (var {}: u32 = {}; {} < {}; {}++) {{\n",
                pad, vn, emit_expr_depth(start, var_names, buf_decls, funcs, vec_buf_map, rec),
                vn, emit_expr_depth(end, var_names, buf_decls, funcs, vec_buf_map, rec), vn);
            s.push_str(&emit_stmt_depth(body, var_names, buf_decls, funcs, depth + 1, vec_buf_map, rec));
            s.push_str(&format!("{}}}\n", pad));
            s
        },
        Stmt::VecPush { vec_var, val } => {
            let vn = var_name(var_names, *vec_var);
            let buf = vec_buf_map.iter()
                .find(|(param, _)| param == &vn)
                .map(|(_, b)| b.as_str())
                .unwrap_or("scratch");
            let len_var = format!("{}_len", vn);
            format!("{}{}[({} + {})] = {};\n{}{} = {} + 1u;\n",
                    pad, buf, vn, len_var,
                    emit_expr_depth(val, var_names, buf_decls, funcs, vec_buf_map, rec),
                    pad, len_var, len_var)
        },
        Stmt::ScratchWrite { offset, val } =>
            format!("{}scratch[{}] = {};\n", pad,
                    emit_expr_depth(offset, var_names, buf_decls, funcs, vec_buf_map, rec),
                    emit_expr_depth(val, var_names, buf_decls, funcs, vec_buf_map, rec)),
        Stmt::Break => format!("{}break;\n", pad),
        Stmt::Continue => format!("{}continue;\n", pad),
        Stmt::Barrier { scope } => {
            let name = match scope {
                BarrierScope::Workgroup => "workgroupBarrier()",
                BarrierScope::Storage => "storageBarrier()",
                BarrierScope::Subgroup => "subgroupBarrier()",
            };
            format!("{}{};\n", pad, name)
        },
        Stmt::Return => format!("{}return;\n", pad),
        Stmt::Noop => String::new(),
    }
}

/// Emit an expression with recursion-aware call replacement.
fn emit_expr_depth(e: &Expr, var_names: &[String], buf_decls: &[BufDecl], funcs: &[GpuFunction],
                   vec_buf_map: &[(String, String)], rec: &Option<RecursionCtx>) -> String {
    match e {
        Expr::Call(fn_id, args) => {
            let name = resolve_fn_name(funcs, *fn_id, rec);
            let arg_strs: Vec<String> = args.iter()
                .map(|a| emit_expr_depth(a, var_names, buf_decls, funcs, vec_buf_map, rec))
                .collect();
            format!("{}({})", name, arg_strs.join(", "))
        },
        // For all other expressions, delegate to emit_expr_ctx
        _ => emit_expr_ctx(e, var_names, buf_decls, funcs, vec_buf_map),
    }
}

/// Resolve function name, handling recursion depth replacement.
/// At depth K>0, self-calls become _d{K-1}.
/// At depth 0, self-calls are replaced with the first non-recursive
/// function called in the same body (typically the base case like schoolbook).
fn resolve_fn_name(funcs: &[GpuFunction], fn_id: u32, rec: &Option<RecursionCtx>) -> String {
    if let Some(ctx) = rec {
        if fn_id == ctx.self_fn_idx {
            if ctx.current_depth == 0 {
                // Find a non-recursive fallback function called in the body.
                // This is typically the base case (e.g., schoolbook for karatsuba).
                if let Some(ref fallback) = ctx.base_case_name {
                    return fallback.clone();
                }
                // Last resort: still use _d0 (will cause WGSL error if actually recursive)
                return format!("{}_d0", ctx.base_name);
            } else {
                return format!("{}_d{}", ctx.base_name, ctx.current_depth - 1);
            }
        }
    }
    fn_name(funcs, fn_id)
}

pub fn emit_kernel(k: &Kernel) -> String {
    let mut s = String::new();

    // Tuple struct definitions
    let arities = collect_tuple_arities(k);
    for arity in &arities {
        s.push_str(&emit_tuple_struct(*arity));
    }

    // Scratch buffer (workgroup shared memory for Vec-backed operations)
    if k.scratch_size > 0 {
        s.push_str(&format!("var<workgroup> scratch: array<u32, {}>;\n\n", k.scratch_size));
    }

    // Buffer declarations
    for buf in &k.buf_decls {
        let access = if buf.read_only { "read" } else { "read_write" };
        s.push_str(&format!("@group(0) @binding({}) var<storage, {}> {}: array<{}>;\n",
            buf.binding, access, buf.name, scalar_type_str(&buf.elem_type)));
    }
    s.push('\n');

    // Helper functions
    for (i, f) in k.functions.iter().enumerate() {
        s.push_str(&emit_function(f, &k.functions, i));
    }

    // Entry point
    s.push_str(&format!("@compute @workgroup_size({}, {}, {})\n",
        k.workgroup_size.0, k.workgroup_size.1, k.workgroup_size.2));
    s.push_str(&format!("fn {}(\n  @builtin(global_invocation_id) gid: vec3<u32>,\n) {{\n",
        k.name));

    // Builtin extraction
    for (i, bn) in k.builtin_names.iter().enumerate() {
        s.push_str(&format!("  let {} = {};\n", var_name(&k.var_names, i as u32), bn));
    }

    // Declare local variables
    let builtin_count = k.builtin_names.len();
    let buf_names: Vec<&str> = k.buf_decls.iter().map(|b| b.name.as_str()).collect();
    for i in builtin_count..k.var_names.len() {
        let vn = &k.var_names[i];
        if !buf_names.contains(&vn.as_str()) && vn != "_ret" && vn != "_call_tmp" {
            s.push_str(&format!("  var {}: u32;\n", vn));
        }
    }

    // Body
    s.push_str(&emit_stmt(&k.body, &k.var_names, &k.buf_decls, &k.functions, 1));
    s.push_str("}\n");
    s
}
