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
        if f.vec_buffer_map.is_empty() {
            f.name.clone()
        } else {
            let buf_suffix: Vec<&str> = f.vec_buffer_map.iter()
                .map(|(_, buf)| buf.as_str())
                .collect();
            format!("{}_{}", f.name, buf_suffix.join("_"))
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
        Stmt::Assign { var, rhs } =>
            format!("{}{} = {};\n", pad, var_name(var_names, *var),
                    emit_expr_ctx(rhs, var_names, buf_decls, funcs, vec_buf_map)),
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
            // let (a, b, c) = expr; → { var __td = expr; a = __td._0; b = __td._1; ... }
            // Use a block scope so __td doesn't conflict with other destructures.
            let mut s = format!("{}{{\n", pad);
            s.push_str(&format!("{}  var __td = {};\n", pad, emit_expr_ctx(rhs, var_names, buf_decls, funcs, vec_buf_map)));
            for (i, var) in vars.iter().enumerate() {
                s.push_str(&format!("{}  {} = __td._{};\n", pad, var_name(var_names, *var), i));
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

/// Emit a helper function as WGSL.
fn emit_function(f: &GpuFunction, all_funcs: &[GpuFunction]) -> String {
    let mut s = String::new();

    let ret_ty = f.ret_type.as_ref()
        .map(|rt| return_type_str(rt))
        .unwrap_or_else(|| "u32".to_string());

    // Build monomorphized name: fn_name + buffer names for Vec params
    let fn_display_name = if f.vec_buffer_map.is_empty() {
        f.name.clone()
    } else {
        let buf_suffix: Vec<&str> = f.vec_buffer_map.iter()
            .map(|(_, buf)| buf.as_str())
            .collect();
        format!("{}_{}", f.name, buf_suffix.join("_"))
    };

    // Signature — Vec params become offset parameters (u32)
    let param_strs: Vec<String> = f.params.iter()
        .map(|(name, ty)| match ty {
            ParamType::Scalar(sty) => format!("{}: {}", name, scalar_type_str(sty)),
            ParamType::VecU32 => format!("{}: u32", name), // offset into buffer
        })
        .collect();
    s.push_str(&format!("fn {}({}) -> {} {{\n", fn_display_name, param_strs.join(", "), ret_ty));

    // Declare local variables (skip params)
    let param_count = f.params.len();
    for i in param_count..f.var_names.len() {
        let vn = &f.var_names[i];
        if vn == "__ret" || vn == "__call_tmp" { continue; }
        s.push_str(&format!("  var {}: u32;\n", vn));
    }

    // Return value
    s.push_str(&format!("  var __ret: {};\n", ret_ty));

    // Body — use vec_buffer_map for buffer-backed Vec params
    let empty_bufs: Vec<BufDecl> = Vec::new();
    s.push_str(&emit_stmt_ctx(&f.body, &f.var_names, &empty_bufs, all_funcs, 1, &f.vec_buffer_map));

    s.push_str("  return __ret;\n");
    s.push_str("}\n\n");
    s
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
    for f in &k.functions {
        s.push_str(&emit_function(f, &k.functions));
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
        if !buf_names.contains(&vn.as_str()) && vn != "__ret" && vn != "__call_tmp" {
            s.push_str(&format!("  var {}: u32;\n", vn));
        }
    }

    // Body
    s.push_str(&emit_stmt(&k.body, &k.var_names, &k.buf_decls, &k.functions, 1));
    s.push_str("}\n");
    s
}
