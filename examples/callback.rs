use cuptir::{
    Context,
    callback::{self},
    driver,
    error::CuptirError,
};
use cuptir_example_utils::run_a_kernel;

fn callback_handler(
    data: cuptir::callback::Data,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    if let callback::Data::DriverApi(data) = data
        && let Some(driver::FunctionParams::cuLaunchKernel(params)) = data.arguments
    {
        let params: &cuptir::cudarc::cupti::sys::cuLaunchKernel_params = unsafe { &*params };
        println!(
            "{:?}: {:?}, block dims: {}, {}, {}",
            data.site,
            data.symbol_name.unwrap_or_default(),
            params.blockDimX,
            params.blockDimY,
            params.blockDimZ,
        );
    };
    Ok(())
}

fn setup() -> Result<Context, CuptirError> {
    cuptir::ContextBuilder::new()
        .with_callback(
            callback::Builder::new()
                .with_driver_functions([driver::Function::cuLaunchKernel])
                .with_handler(callback_handler),
        )
        .build()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let _cuptir = setup()?;

    run_a_kernel()?;

    Ok(())
}
