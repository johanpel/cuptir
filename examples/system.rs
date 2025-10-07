fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::level_filters::LevelFilter::TRACE)
        .init();

    use cuptir::activity;

    let _context = cuptir::Context::builder()
        .with_activity(
            activity::Builder::new()
                .with_kinds([
                    activity::Kind::Device,
                    // activity::Kind::Environment,
                    activity::Kind::Pcie,
                ])
                .with_record_buffer_handler(|buffer| {
                    buffer
                        .into_iter()
                        .for_each(|record| println!("{:#?}", record.unwrap()));
                    Ok(())
                }),
        )
        .build()?;

    cudarc::driver::CudaContext::new(0)?;

    std::thread::sleep(std::time::Duration::from_secs(5));

    Ok(())
}
