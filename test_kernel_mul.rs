/// Test: import generic_mul_karatsuba (recursive) from verus-fixed-point.
/// The transpiler should unroll the recursion into depth-stratified copies.

use verus_fixed_point::fixed_point::limb_ops::*;

#[gpu_kernel(workgroup_size(256, 1, 1))]
fn test_mul_kernel(
    #[gpu_builtin(thread_id_x)] tid: u32,
    #[gpu_buffer(0, read)] a_buf: &[u32],
    #[gpu_buffer(1, read)] b_buf: &[u32],
    #[gpu_buffer(2, read_write)] out_buf: &mut [u32],
) {
    let n = 4u32;
    let base = tid * n;

    // Call verified Karatsuba multiply — transpiler unrolls recursion
    let carry = generic_mul_karatsuba(&a_buf[base..], &b_buf[base..], n, &mut out_buf[base..]);
}
