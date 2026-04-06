///  WGSL emitter — plain Rust version mirroring the verified emitter.
///  Structural correspondence with wgsl_emit.rs ensures correctness.
///  Now supports helper function emission and function calls.

use crate::types::*;

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

fn var_name(names: &[String], idx: u32) -> String {
    names.get(idx as usize).cloned().unwrap_or_else(|| format!("v_{}", idx))
}

fn buf_name(decls: &[BufDecl], idx: u32) -> String {
    decls.get(idx as usize).map(|d| d.name.clone()).unwrap_or_else(|| format!("buf_{}", idx))
}

fn fn_name(funcs: &[GpuFunction], idx: u32) -> String {
    funcs.get(idx as usize).map(|f| f.name.clone()).unwrap_or_else(|| format!("fn_{}", idx))
}

pub fn emit_expr(e: &Expr, var_names: &[String], buf_decls: &[BufDecl], funcs: &[GpuFunction]) -> String {
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
            format!("({} {} {})", emit_expr(a, var_names, buf_decls, funcs),
                    binop_str(op), emit_expr(b, var_names, buf_decls, funcs)),
        Expr::UnaryOp(op, a) =>
            format!("({}{})", unaryop_str(op), emit_expr(a, var_names, buf_decls, funcs)),
        Expr::Select(c, t, f) =>
            format!("select({}, {}, {})",
                    emit_expr(f, var_names, buf_decls, funcs),
                    emit_expr(t, var_names, buf_decls, funcs),
                    emit_expr(c, var_names, buf_decls, funcs)),
        Expr::ArrayRead(buf, idx) =>
            format!("{}[{}]", buf_name(buf_decls, *buf), emit_expr(idx, var_names, buf_decls, funcs)),
        Expr::Cast(ty, inner) =>
            format!("{}({})", scalar_type_str(ty), emit_expr(inner, var_names, buf_decls, funcs)),
        Expr::Call(fn_id, args) => {
            let name = fn_name(funcs, *fn_id);
            let arg_strs: Vec<String> = args.iter()
                .map(|a| emit_expr(a, var_names, buf_decls, funcs))
                .collect();
            format!("{}({})", name, arg_strs.join(", "))
        },
    }
}

pub fn emit_stmt(s: &Stmt, var_names: &[String], buf_decls: &[BufDecl], funcs: &[GpuFunction], depth: usize) -> String {
    let pad = "  ".repeat(depth);
    match s {
        Stmt::Assign { var, rhs } =>
            format!("{}{} = {};\n", pad, var_name(var_names, *var),
                    emit_expr(rhs, var_names, buf_decls, funcs)),
        Stmt::BufWrite { buf, idx, val } =>
            format!("{}{}[{}] = {};\n", pad, buf_name(buf_decls, *buf),
                    emit_expr(idx, var_names, buf_decls, funcs),
                    emit_expr(val, var_names, buf_decls, funcs)),
        Stmt::CallStmt { fn_id, args, result_var } => {
            let name = fn_name(funcs, *fn_id);
            let arg_strs: Vec<String> = args.iter()
                .map(|a| emit_expr(a, var_names, buf_decls, funcs))
                .collect();
            format!("{}{} = {}({});\n", pad, var_name(var_names, *result_var),
                    name, arg_strs.join(", "))
        },
        Stmt::Seq { first, then } => {
            let mut s = emit_stmt(first, var_names, buf_decls, funcs, depth);
            s.push_str(&emit_stmt(then, var_names, buf_decls, funcs, depth));
            s
        },
        Stmt::If { cond, then_body, else_body } => {
            let mut s = format!("{}if ({}) {{\n", pad, emit_expr(cond, var_names, buf_decls, funcs));
            s.push_str(&emit_stmt(then_body, var_names, buf_decls, funcs, depth + 1));
            s.push_str(&format!("{}}} else {{\n", pad));
            s.push_str(&emit_stmt(else_body, var_names, buf_decls, funcs, depth + 1));
            s.push_str(&format!("{}}}\n", pad));
            s
        },
        Stmt::For { var, start, end, body } => {
            let vn = var_name(var_names, *var);
            let mut s = format!("{}for (var {}: u32 = {}; {} < {}; {}++) {{\n",
                pad, vn, emit_expr(start, var_names, buf_decls, funcs),
                vn, emit_expr(end, var_names, buf_decls, funcs), vn);
            s.push_str(&emit_stmt(body, var_names, buf_decls, funcs, depth + 1));
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
        Stmt::Return => format!("{}return;\n", pad),
        Stmt::Noop => String::new(),
    }
}

/// Emit a helper function as WGSL.
fn emit_function(f: &GpuFunction, all_funcs: &[GpuFunction]) -> String {
    let mut s = String::new();

    let ret_ty = f.ret_type.map(|t| scalar_type_str(&t)).unwrap_or("u32");

    // Signature
    let param_strs: Vec<String> = f.params.iter()
        .map(|(name, ty)| format!("{}: {}", name, scalar_type_str(ty)))
        .collect();
    s.push_str(&format!("fn {}({}) -> {} {{\n", f.name, param_strs.join(", "), ret_ty));

    // Declare local variables (skip params, they're declared in signature)
    let param_count = f.params.len();
    for i in param_count..f.var_names.len() {
        let vn = &f.var_names[i];
        if vn == "__ret" { continue; } // declared as return value
        s.push_str(&format!("  var {}: {};\n", vn, ret_ty));
    }

    // Return value
    s.push_str(&format!("  var __ret: {};\n", ret_ty));

    // Body — use empty buf_decls since helpers don't access buffers directly
    let empty_bufs: Vec<BufDecl> = Vec::new();
    s.push_str(&emit_stmt(&f.body, &f.var_names, &empty_bufs, all_funcs, 1));

    s.push_str(&format!("  return __ret;\n"));
    s.push_str("}\n\n");
    s
}

pub fn emit_kernel(k: &Kernel) -> String {
    let mut s = String::new();

    // Buffer declarations
    for buf in &k.buf_decls {
        let access = if buf.read_only { "read" } else { "read_write" };
        s.push_str(&format!("@group(0) @binding({}) var<storage, {}> {}: array<{}>;\n",
            buf.binding, access, buf.name, scalar_type_str(&buf.elem_type)));
    }
    s.push('\n');

    // Helper functions (emitted before the kernel entry point)
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

    // Declare local variables (skip builtins and buffer names)
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
