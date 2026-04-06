/// Test kernel that imports and uses verified functions from verus-fixed-point.
/// The transpiler follows `use` statements to find generic_add_limbs, etc.

use verus_fixed_point::fixed_point::limb_ops::*;

/// Simple kernel: add two 4-limb values from input buffers, store result.
/// Uses the VERIFIED generic_add_limbs from verus-fixed-point.
#[gpu_kernel(workgroup_size(256, 1, 1))]
fn test_add_kernel(
    #[gpu_builtin(thread_id_x)] tid: u32,
    #[gpu_buffer(0, read)] a_buf: &[u32],
    #[gpu_buffer(1, read)] b_buf: &[u32],
    #[gpu_buffer(2, read_write)] out_buf: &mut [u32],
) {
    let n = 4u32;
    let base = tid * n;

    // Read limbs from buffers into scratch-backed Vecs
    // (The transpiler maps Vec operations to scratch buffer access)
    let a0 = a_buf[base + 0u32];
    let a1 = a_buf[base + 1u32];
    let a2 = a_buf[base + 2u32];
    let a3 = a_buf[base + 3u32];

    let b0 = b_buf[base + 0u32];
    let b1 = b_buf[base + 1u32];
    let b2 = b_buf[base + 2u32];
    let b3 = b_buf[base + 3u32];

    // Call the verified add3 function from LimbOps
    let mut carry = 0u32;

    let (r0, c0) = add3(a0, b0, carry);
    carry = c0;
    let (r1, c1) = add3(a1, b1, carry);
    carry = c1;
    let (r2, c2) = add3(a2, b2, carry);
    carry = c2;
    let (r3, c3) = add3(a3, b3, carry);

    out_buf[base + 0u32] = r0;
    out_buf[base + 1u32] = r1;
    out_buf[base + 2u32] = r2;
    out_buf[base + 3u32] = r3;
}
