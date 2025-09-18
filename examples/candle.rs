use std::fs::File;

use colored::Colorize;
use cuptir::{
    callback::{self},
    driver,
};
use tracing::level_filters::LevelFilter;

/// This example shows how to use the CUPTI Activity API wrappers, as well as the CUPTI
/// Callback API wrappers. It enables activity records and callbacks for a small number
/// of CUDA functions, and prints them to the standard output.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Set up tracing the wrappers.
    let trace = File::create("example_trace.log")?;
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
    let activity = cuptir::activity::Builder::new()
        .with_kinds([cuptir::activity::Kind::ConcurrentKernel])
        .latency_timestamps(true)
        .with_record_buffer_handler(move |buffer| {
            buffer.into_iter().try_for_each(|record| {
                match record {
                    Ok(r) => match serde_json::to_string(&r) {
                        Ok(json) => println!("{}: {json}", "activity record".cyan()),
                        Err(e) => tracing::warn!("json error: {e}"),
                    },
                    Err(e) => tracing::warn!("erroneous record: {e}"),
                }
                Ok::<(), cuptir::error::CuptirError>(())
            })?;
            Ok(())
        });

    let callback = cuptir::callback::Builder::new()
        .with_driver_functions([
            driver::Function::cuMemcpyDtoHAsync_v2,
            driver::Function::cuLaunchKernel,
        ])
        .with_handler(move |data| {
            match data {
                callback::Data::DriverApi(callback_data) => match callback_data.arguments {
                    Some(driver::FunctionParams::cuMemcpyDtoHAsync_v2(params)) => {
                        let params: &cuptir::cudarc::cupti::sys::cuMemcpyDtoHAsync_v2_params_st =
                            unsafe { &*params };
                        println!(
                            "{}: {}, bytes: {}, site: {:?}, correlation id: {}",
                            "callback".purple(),
                            callback_data.function_name().unwrap(),
                            params.ByteCount,
                            callback_data.site,
                            callback_data.correlation_id
                        );
                    }
                    Some(driver::FunctionParams::cuLaunchKernel(params)) => {
                        let params: &cuptir::cudarc::cupti::sys::cuLaunchKernel_params =
                            unsafe { &*params };
                        print!(
                            "{}: kernel launched\n\tname: {}\n\tblock dims: {}, {}, {}\n",
                            "callback".purple(),
                            callback_data.symbol_name.unwrap_or_default(),
                            params.blockDimX,
                            params.blockDimY,
                            params.blockDimZ
                        );
                    }
                    _ => (),
                },
                _ => (),
            };
            Ok(())
        });

    let _context = cuptir::ContextBuilder::new()
        .with_activity(activity)
        .with_callback(callback)
        .build()?;

    // CUDA usage goes here, e.g. the example from the Candle crate:
    let device = candle_core::Device::new_cuda(0)?;
    let a = candle_core::Tensor::randn(0f32, 1., (2, 3), &device)?;
    let b = candle_core::Tensor::randn(0f32, 1., (3, 4), &device)?;
    let c = a.matmul(&b)?;

    tracing::info!("{c}");

    Ok(())
}
