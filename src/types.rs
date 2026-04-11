///  GPU kernel IR types — mirrors the verified RtExpr/RtStmt/RtKernel types.
///  These are plain Rust (not Verus) for use in the parser.
///  Structural correspondence with the verified types is by construction.

#[derive(Debug, Clone, Copy)]
pub enum ScalarType { I32, U32, F32, F16, Bool }

#[derive(Debug, Clone, Copy)]
pub enum BinOp {
    Add, Sub, Mul, Div, Mod, Shr, Shl,
    WrappingAdd, WrappingSub, WrappingMul,
    FAdd, FSub, FMul, FDiv,
    Lt, Le, Gt, Ge, Eq, Ne,
    BitAnd, BitOr, BitXor,
    LogicalAnd, LogicalOr,
}

#[derive(Debug, Clone, Copy)]
pub enum UnaryOp { Neg, FNeg, BitNot, LogicalNot }

#[derive(Debug, Clone, Copy)]
pub enum BarrierScope { Workgroup, Storage, Subgroup }

#[derive(Debug, Clone)]
pub enum Expr {
    Const(i64, ScalarType),
    FConst(u32),                    // f32 bits
    Var(u32),                       // local index
    Builtin(u32),                   // local_idx
    BinOp(BinOp, Box<Expr>, Box<Expr>),
    UnaryOp(UnaryOp, Box<Expr>),
    Select(Box<Expr>, Box<Expr>, Box<Expr>),
    ArrayRead(u32, Box<Expr>),
    Cast(ScalarType, Box<Expr>),
    Call(u32, Vec<Expr>),           // fn_id, args → returns value
    TupleConstruct(Vec<Expr>),      // (a, b, c, ...) → struct construction
    TupleAccess(Box<Expr>, u32),    // expr.0, expr.1, ... → member access
    /// Read from scratch buffer: scratch[offset_expr]
    ScratchRead(Box<Expr>),
    /// Read from scratch at vec_var's offset + index: scratch[vec_off + idx]
    VecIndex(u32, Box<Expr>),       // (vec_var_id, index_expr)
    /// Buffer slice reference: &buf[offset..] → (buf_id, offset_expr)
    /// Used as argument to Vec-param functions for monomorphization.
    BufSlice(u32, Box<Expr>),       // (buf_id, offset_expr)
}

#[derive(Debug, Clone)]
pub enum Stmt {
    Assign { var: u32, rhs: Expr },
    BufWrite { buf: u32, idx: Expr, val: Expr },
    CallStmt { fn_id: u32, args: Vec<Expr>, result_var: u32 },
    /// let (a, b, c) = expr; → destructure tuple into multiple vars
    TupleDestructure { vars: Vec<u32>, rhs: Expr },
    /// Write to scratch buffer: scratch[offset_expr] = val
    ScratchWrite { offset: Expr, val: Expr },
    /// Vec push: scratch[vec_off + vec_len] = val; vec_len++
    VecPush { vec_var: u32, val: Expr },
    Seq { first: Box<Stmt>, then: Box<Stmt> },
    If { cond: Expr, then_body: Box<Stmt>, else_body: Box<Stmt> },
    For { var: u32, start: Expr, end: Expr, body: Box<Stmt> },
    Break,
    Continue,
    Barrier { scope: BarrierScope },
    Return,
    Noop,
}

impl Stmt {
    /// Build a Seq from a Vec of statements.
    pub fn from_vec(mut stmts: Vec<Stmt>) -> Stmt {
        if stmts.is_empty() { return Stmt::Noop; }
        if stmts.len() == 1 { return stmts.remove(0); }
        let first = stmts.remove(0);
        Stmt::Seq {
            first: Box::new(first),
            then: Box::new(Stmt::from_vec(stmts)),
        }
    }
}

#[derive(Debug, Clone)]
pub struct BufDecl {
    pub binding: u32,
    pub name: String,
    pub read_only: bool,
    pub elem_type: ScalarType,
}

/// Parameter type: scalar or Vec (buffer-backed array region).
#[derive(Debug, Clone)]
pub enum ParamType {
    Scalar(ScalarType),
    /// Vec<u32> parameter — on GPU, this maps to an offset into the scratch buffer.
    /// The function receives `name_off: u32` and accesses `scratch[name_off + i]`.
    VecU32,
}

/// Return type: either a single scalar or a tuple of scalars.
#[derive(Debug, Clone)]
pub enum ReturnType {
    Scalar(ScalarType),
    Tuple(Vec<ScalarType>),
}

/// A helper function callable from the kernel or other helpers.
#[derive(Debug, Clone)]
pub struct GpuFunction {
    pub name: String,
    pub params: Vec<(String, ParamType)>,   // (name, type)
    pub ret_type: Option<ReturnType>,
    /// Names of params that are Vec<u32> (for scratch buffer mapping).
    pub vec_params: Vec<String>,
    /// If the function returns a Vec<u32> as part of its return type,
    /// the caller provides an output scratch offset as an extra parameter.
    pub returns_vec: bool,
    /// Buffer names that back each Vec param (for monomorphized variants).
    /// Empty for the "template" function; filled for monomorphized copies.
    pub vec_buffer_map: Vec<(String, String)>,  // (param_name, buffer_name)
    pub var_names: Vec<String>,
    pub body: Stmt,
    pub ret_var: u32,
    /// Base case function for recursion unrolling. Parsed from #[gpu_base_case(name)].
    /// At depth 0, self-calls are replaced with this function.
    pub base_case: Option<String>,
    /// Slice aliases from `let x = vslice(buf, off)`. Maps local var_name
    /// to the buffer name the slice refers to. Used during monomorphization
    /// so that `f(x, ...)` (where x is a slice alias) resolves to
    /// `f_variant_backed_by_<buf>(...)`.
    pub slice_aliases: std::collections::HashMap<String, String>,
}

#[derive(Debug, Clone)]
pub struct Kernel {
    pub name: String,
    pub var_names: Vec<String>,
    pub buf_decls: Vec<BufDecl>,
    pub body: Stmt,
    pub workgroup_size: (u32, u32, u32),
    pub builtin_names: Vec<String>,
    pub functions: Vec<GpuFunction>,        // helper functions (reachable from kernel)
    pub fn_name_to_id: Vec<(String, u32)>,  // name → index into functions
    /// Size of workgroup scratch buffer (for Vec-backed operations).
    /// If > 0, emits `var<workgroup> scratch: array<u32, SIZE>;`
    pub scratch_size: u32,
    /// Variables annotated with #[gpu_local(N)] — stored as thread-local
    /// arrays instead of scratch buffer offsets. (var_name, array_size)
    pub local_arrays: Vec<(String, u32)>,
    /// Variables annotated with #[gpu_skip] — offset computations suppressed,
    /// treated as plain scalar variables (e.g., sign values).
    pub skipped_vars: Vec<String>,
}
