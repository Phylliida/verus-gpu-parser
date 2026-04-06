/// Test kernel that calls verified generic_add_limbs from verus-fixed-point.
/// The transpiler follows `use` imports and transpiles the function directly.

use verus_fixed_point::fixed_point::limb_ops::*;

#[gpu_kernel(workgroup_size(256, 1, 1))]
fn test_add_buffers(
    #[gpu_builtin(thread_id_x)] tid: u32,
    #[gpu_buffer(0, read)] a_buf: &[u32],
    #[gpu_buffer(1, read)] b_buf: &[u32],
    #[gpu_buffer(2, read_write)] out_buf: &mut [u32],
) {
    let n = 4u32;
    let base = tid * n;

    let (result, carry) = generic_add_limbs(&a_buf[base..], &b_buf[base..], n);

    for i in 0u32..n {
        out_buf[base + i] = result[i];
    }
}
