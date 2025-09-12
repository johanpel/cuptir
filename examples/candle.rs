use tracing::level_filters::LevelFilter;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_max_level(LevelFilter::TRACE)
        .init();

    // Set up context and activate a callback:
    let _cuptir = cuptir::Context::builder()
        .with_activity_kinds([
            cuptir::activity::Kind::Driver,
            cuptir::activity::Kind::Runtime,
            cuptir::activity::Kind::InternalLaunchApi,
            cuptir::activity::Kind::ConcurrentKernel,
            cuptir::activity::Kind::Memory2,
            cuptir::activity::Kind::Memcpy2,
            cuptir::activity::Kind::MemoryPool,
        ])
        .with_activity_latency_timestamps(true)
        .with_activity_record_buffer_callback(|buffer| {
            // Normally, this callback should aim to return as quickly as possible in
            // order to keep the profiling overhead as low as possible. It is very
            // likely a bad idea to do something similar to the below if you care about
            // performance. It is recommended to hand this buffer off to some
            // asynchronous processing logic.
            buffer.into_iter().try_for_each(|record| {
                match record {
                    Ok(r) => match serde_json::to_string(&r) {
                        Ok(json) => println!("{json}"),
                        Err(e) => tracing::warn!("json error: {e}"),
                    },
                    Err(e) => tracing::warn!("erroneous record: {e}"),
                }
                Ok::<(), cuptir::error::CuptirError>(())
            })?;
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
