use tracing::level_filters::LevelFilter;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_max_level(LevelFilter::TRACE)
        .init();

    // Set up context and activate a callback:
    let _cuptir = cuptir::Context::builder()
        .with_activity_kinds([
            // cudarc::cupti::sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL,
            // cudarc::cupti::sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_MEMORY2,
            // cudarc::cupti::sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_MEMCPY2,
            cudarc::cupti::sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_MEMORY_POOL,
        ])
        .with_activity_latency_timestamps(true)
        .with_activity_record_handler(|record| {
            println!("{record:?}");
            Ok(())
        })
        .build()?;

    // CUDA usage goes here, e.g.:
    let device = candle_core::Device::new_cuda(0)?;
    let a = candle_core::Tensor::randn(0f32, 1., (2, 3), &device)?;
    let b = candle_core::Tensor::randn(0f32, 1., (3, 4), &device)?;
    let c = a.matmul(&b)?;

    println!("{c}");

    Ok(())
}
