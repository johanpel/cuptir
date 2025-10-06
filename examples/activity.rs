use cuptir_example_utils::run_a_kernel;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    use cuptir::activity;

    let _context = cuptir::Context::builder()
        .with_activity(
            activity::Builder::new()
                .with_kinds([activity::Kind::ConcurrentKernel])
                .with_record_buffer_handler(|buffer| {
                    buffer
                        .into_iter()
                        .for_each(|record| println!("{:#?}", record.unwrap()));
                    Ok(())
                }),
        )
        .build()?;

    run_a_kernel()?;

    Ok(())
}
