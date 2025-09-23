use cuptir::{
    Context,
    activity::{self},
    error::CuptirError,
};
use cuptir_example_utils::run_a_kernel;

fn setup() -> Result<Context, CuptirError> {
    cuptir::ContextBuilder::new()
        .with_activity(
            activity::Builder::new()
                .with_kinds([activity::Kind::ConcurrentKernel])
                .latency_timestamps(true)
                .with_record_buffer_handler(move |buffer| {
                    buffer.into_iter().try_for_each(|record| {
                        match record {
                            Ok(rec) => println!("{rec:?}"),
                            Err(e) => tracing::warn!("erroneous record: {e}"),
                        }
                        Ok::<(), cuptir::error::CuptirError>(())
                    })?;
                    Ok(())
                }),
        )
        .build()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let context = setup()?;

    run_a_kernel()?;

    context.activity_flush_all(true)?;
    Ok(())
}
