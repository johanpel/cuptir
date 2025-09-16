use std::fs::File;

use colored::Colorize;
use cuptir::callback::{self, driver::FunctionArguments};
use tracing::level_filters::LevelFilter;

/// This example shows how to use the CUPTI Activity API wrappers, as well as the CUPTI
/// Callback API wrappers. It enables activity records and callbacks for a small number
/// of CUDA functions, and prints them to the standard output.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Set up tracing the wrappers.
    let trace = File::create("trace.log")?;
    tracing_subscriber::fmt()
        .with_writer(trace)
        .with_max_level(LevelFilter::TRACE)
        .with_ansi(false)
        .init();

    // Set up context.
    //
    // Normally, the callbacks should aim to return as quickly as possible in order to
    // keep the profiling overhead as low as possible. It is very likely a bad idea to
    // do something similar to the below if you care about performance. Especially for
    // the buffers of the activity API. It is recommended to hand them off to some
    // asynchronous processing logic.
    let _cuptir = cuptir::Context::builder()
        .with_activity_kinds([cuptir::activity::Kind::ConcurrentKernel])
        .enable_activity_latency_timestamps(true)
        .with_activity_record_buffer_handler(move |buffer| {
            buffer.into_iter().try_for_each(|record| {
                match record {
                    Ok(r) => match serde_json::to_string(&r) {
                        Ok(json) => println!("{}: {json}", "activity".cyan()),
                        Err(e) => tracing::warn!("json error: {e}"),
                    },
                    Err(e) => tracing::warn!("erroneous record: {e}"),
                }
                Ok::<(), cuptir::error::CuptirError>(())
            })?;
            Ok(())
        })
        // We could enable callbacks for all APIs from a domains at once as follows:
        // .with_callback_domains([callback::Domain::RuntimeApi, callback::Domain::DriverApi])
        // However, this is very spammy.
        //
        // Instead, enable single callbacks of interest, e.g.:
        .with_callbacks_for_driver([callback::driver::Function::cuMemcpyDtoHAsync_v2])
        .with_callback_handler(move |data| {
            match data {
                callback::Data::DriverApi(driver) => {
                    if let Some(args) = driver.arguments {
                        match args {
                            // TODO: provide safe wrappers for function parameters
                            FunctionArguments::cuMemcpyDtoHAsync_v2(params) => {
                                let params: &cuptir::sys::cuMemcpyDtoHAsync_v2_params_st =
                                    unsafe { &*params };
                                println!(
                                    "{}: {}, bytes: {}",
                                    "callback".purple(),
                                    driver.function_name().unwrap(),
                                    params.ByteCount,
                                );
                            }
                            _ => (),
                        }
                    }
                }
                _ => (),
            };
            Ok(())
        })
        .build()?;

    // CUDA usage goes here, e.g. the example from the Candle crate:
    let device = candle_core::Device::new_cuda(0)?;
    let a = candle_core::Tensor::randn(0f32, 1., (2, 3), &device)?;
    let b = candle_core::Tensor::randn(0f32, 1., (3, 4), &device)?;
    let c = a.matmul(&b)?;

    println!("{c}");

    Ok(())
}
