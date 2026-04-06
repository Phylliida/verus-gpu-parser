/// Test kernel with helper function calls.
/// The parser should discover `add_limbs` and `sub_if_gte` as reachable helpers.

fn add_limbs(a: u32, b: u32) -> u32 {
    a + b
}

fn sub_if_gte(val: u32, threshold: u32) -> u32 {
    if val >= threshold {
        val - threshold
    } else {
        val
    }
}

/// This function should NOT be included (not reachable from kernel).
fn unused_helper(x: u32) -> u32 {
    x * 2u32
}

fn add_mod(a: u32, b: u32, p: u32) -> u32 {
    let sum = add_limbs(a, b);
    sub_if_gte(sum, p)
}

#[gpu_kernel(workgroup_size(256, 1, 1))]
fn modular_add_kernel(
    #[gpu_builtin(thread_id_x)] tid: u32,
    #[gpu_buffer(0, read)] a: &[u32],
    #[gpu_buffer(1, read)] b: &[u32],
    #[gpu_buffer(2, read_write)] out: &mut [u32],
) {
    let p = 4294967291u32;
    let result = add_mod(a[tid], b[tid], p);
    out[tid] = result;
}
