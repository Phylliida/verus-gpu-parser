use vstd::prelude::*;

verus! {

#[gpu_kernel(workgroup_size(256, 1, 1))]
fn vector_add(
    #[gpu_builtin(thread_id_x)] tid: u32,
    #[gpu_buffer(0, read)] a: &[i32],
    #[gpu_buffer(1, read)] b: &[i32],
    #[gpu_buffer(2, read_write)] out: &mut [i32],
)
    requires tid < 1024
{
    if tid < 1024u32 {
        out[tid] = a[tid] + b[tid];
    }
}

} // verus!
