/// Test kernel: call generic_add_limbs with buffer slice arguments.
/// The transpiler should monomorphize based on which buffers are passed.

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

    // Call generic_add_limbs with buffer slices.
    // The transpiler sees &a_buf[base..] and &b_buf[base..] as BufSlice args,
    // and monomorphizes generic_add_limbs for these specific buffers.
    let (result, carry) = generic_add_limbs(&a_buf[base..], &b_buf[base..], n);

    // Write result to output buffer
    for i in 0u32..n {
        out_buf[base + i] = result[i];
    }
}
