/// Test kernel calling verified generic_add_limbs with buffer-backed Vecs.
/// Input Vecs: &a_buf[base..], &b_buf[base..]
/// Output Vec: &mut out_buf[base..] (extra arg maps to the returned Vec)

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

    // Call generic_add_limbs: a_buf → input a, b_buf → input b, out_buf → output Vec
    let carry = generic_add_limbs(&a_buf[base..], &b_buf[base..], n, &mut out_buf[base..]);
}
