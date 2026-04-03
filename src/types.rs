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
}

#[derive(Debug, Clone)]
pub enum Stmt {
    Assign { var: u32, rhs: Expr },
    BufWrite { buf: u32, idx: Expr, val: Expr },
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

#[derive(Debug, Clone)]
pub struct Kernel {
    pub name: String,
    pub var_names: Vec<String>,
    pub buf_decls: Vec<BufDecl>,
    pub body: Stmt,
    pub workgroup_size: (u32, u32, u32),
    pub builtin_names: Vec<String>,
}
