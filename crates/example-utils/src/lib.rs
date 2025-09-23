// 03-launch-kernel.rs from cudarc
pub fn run_a_kernel() -> Result<(), Box<dyn std::error::Error>> {
    use cudarc::{
        driver::{CudaContext, LaunchConfig, PushKernelArg},
        nvrtc::compile_ptx,
    };
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let ptx = compile_ptx(
        "
extern \"C\" __global__ void sin_kernel(float *out, const float *inp, const size_t numel) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numel) {
        out[i] = sin(inp[i]);
    }
}",
    )?;
    let module = ctx.load_module(ptx)?;
    let sin = module.load_function("sin_kernel")?;
    let in_host = [1.0, 2.0, 3.0];
    let in_device = stream.memcpy_stod(&in_host)?;
    let mut out_device = in_device.clone();
    let n = 3i32;
    let cfg = LaunchConfig::for_num_elems(n as u32);
    let mut launch_args = stream.launch_builder(&sin);
    launch_args.arg(&mut out_device);
    launch_args.arg(&in_device);
    launch_args.arg(&n);
    unsafe { launch_args.launch(cfg) }?;
    let a_host_2 = stream.memcpy_dtov(&in_device)?;
    assert_eq!(&in_host, a_host_2.as_slice());
    Ok(())
}
