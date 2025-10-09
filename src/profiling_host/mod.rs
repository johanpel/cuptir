use std::ptr::null_mut;

use cudarc::{
    cupti::{result, sys},
    driver,
};

use crate::{CuptirError, enums::ProfilerSupportLevel, utils::try_str_from_ffi};

fn format_device_supported(
    params: &sys::CUpti_Profiler_DeviceSupported_Params,
) -> Result<String, CuptirError> {
    Ok(format!(
        "architecture={:?}, sli={:?}, vGpu={:?}, confidentialCompute={:?}, cmp={:?}, wsl={:?}",
        ProfilerSupportLevel::try_from(params.architecture)?,
        ProfilerSupportLevel::try_from(params.sli)?,
        ProfilerSupportLevel::try_from(params.vGpu)?,
        ProfilerSupportLevel::try_from(params.confidentialCompute)?,
        ProfilerSupportLevel::try_from(params.cmp)?,
        ProfilerSupportLevel::try_from(params.wsl)?
    ))
}

fn check_device_supported(cu_device: cudarc::driver::sys::CUdevice) -> Result<(), CuptirError> {
    use sys::CUpti_Profiler_Support_Level as Support;
    // Can't use size_of for the entire struct, because this includes padding.
    // Using it will error out CUPTI
    let struct_size = std::mem::offset_of!(sys::CUpti_Profiler_DeviceSupported_Params, api)
        + size_of::<sys::CUpti_Profiler_API>();
    let mut params = sys::CUpti_Profiler_DeviceSupported_Params {
        // inputs:
        structSize: struct_size,
        pPriv: null_mut(),
        cuDevice: cu_device,
        api: sys::CUpti_Profiler_API::CUPTI_PROFILER_PM_SAMPLING,
        // outputs:
        isSupported: Support::CUPTI_PROFILER_CONFIGURATION_UNKNOWN,
        architecture: Support::CUPTI_PROFILER_CONFIGURATION_UNKNOWN,
        sli: Support::CUPTI_PROFILER_CONFIGURATION_UNKNOWN,
        vGpu: Support::CUPTI_PROFILER_CONFIGURATION_UNKNOWN,
        confidentialCompute: Support::CUPTI_PROFILER_CONFIGURATION_UNKNOWN,
        cmp: Support::CUPTI_PROFILER_CONFIGURATION_UNKNOWN,
        wsl: Support::CUPTI_PROFILER_CONFIGURATION_UNKNOWN,
    };
    unsafe { result::profiling::device_supported(&mut params) }?;
    if params.isSupported == Support::CUPTI_PROFILER_CONFIGURATION_SUPPORTED {
        tracing::trace!(
            "PM Sampling is supported: {}",
            format_device_supported(&params)?
        );
        Ok(())
    } else {
        Err(CuptirError::Builder(format!(
            "PM Sampling is not supported: {:?}, {}",
            ProfilerSupportLevel::try_from(params.isSupported)?,
            format_device_supported(&params).unwrap_or_default()
        )))
    }
}

fn init_profiler() -> Result<(), CuptirError> {
    let mut params = sys::CUpti_Profiler_Initialize_Params {
        structSize: size_of::<sys::CUpti_Profiler_Initialize_Params>(),
        pPriv: null_mut(),
    };

    unsafe { result::profiling::initialize(&mut params) }?;
    Ok(())
}

fn get_chip_name(device_ordinal: i32) -> Result<String, CuptirError> {
    // Can't use sizeof for the entire struct, because this includes padding.
    let struct_size = std::mem::offset_of!(sys::CUpti_Device_GetChipName_Params, pChipName)
        + size_of::<*const core::ffi::c_char>();

    let mut chip_name_params = sys::CUpti_Device_GetChipName_Params {
        structSize: struct_size,
        pPriv: null_mut(),
        deviceIndex: device_ordinal as usize,
        pChipName: null_mut(),
    };
    unsafe { result::device::get_chip_name(&mut chip_name_params) }?;
    unsafe { try_str_from_ffi(chip_name_params.pChipName) }
        .map(ToOwned::to_owned)
        .ok_or(CuptirError::Builder(format!(
            "Chip name at ptr {:x} is null or not valid UTF-8",
            chip_name_params.pChipName as usize
        )))
}

fn get_counter_availability(device_ordinal: i32) -> Result<Vec<u8>, CuptirError> {
    // Can't use sizeof for the entire struct, because this includes padding.
    // TODO: consider writing a proc macro for this.
    let struct_size = std::mem::offset_of!(
        sys::CUpti_PmSampling_GetCounterAvailability_Params,
        pCounterAvailabilityImage
    ) + size_of::<*mut u8>();

    // First, get the size of the image:
    let mut params = sys::CUpti_PmSampling_GetCounterAvailability_Params {
        structSize: struct_size,
        pPriv: null_mut(),
        deviceIndex: device_ordinal as usize,
        counterAvailabilityImageSize: 0,
        pCounterAvailabilityImage: null_mut(), // means we want it to populate counterAvailabilityImageSize
    };
    unsafe { result::pm_sampling::get_counter_availability(&mut params) }?;

    // Get the image
    let mut image = vec![0u8; params.counterAvailabilityImageSize];
    params.pCounterAvailabilityImage = image.as_mut_ptr();
    unsafe { result::pm_sampling::get_counter_availability(&mut params) }?;

    Ok(image)
}

pub(crate) struct Context {
    object: *mut sys::CUpti_Profiler_Host_Object,
}

impl Context {}

#[derive(Debug, Default)]
pub struct ContextBuilder {
    device_ordinal: i32,
}

impl ContextBuilder {
    pub fn new() -> Self {
        Default::default()
    }

    pub fn device_index(mut self, index: i32) -> Self {
        self.device_ordinal = index;
        self
    }

    /// Requires the driver to be initialized.
    pub(crate) fn build(self) -> Result<Context, CuptirError> {
        cudarc::driver::result::init()?;

        let cu_device = cudarc::driver::result::device::get(self.device_ordinal)?;
        check_device_supported(cu_device)?;
        init_profiler()?;
        let chip_name = get_chip_name(self.device_ordinal)?;
        tracing::trace!("chip name: {chip_name}");
        // Safety: chip name was converted from a C-string to a Rust string, where, if it would
        // contain a 0 byte, it would have been interpreted as the string terminator, so it should
        // not contain any 0 byte itself.
        let chip_name_c =
            std::ffi::CString::new(chip_name).expect("unexpected 0 byte in chip name");

        let availability_image = get_counter_availability(self.device_ordinal)?;

        let mut params = sys::CUpti_Profiler_Host_Initialize_Params {
            structSize: size_of::<sys::CUpti_Profiler_Host_Initialize_Params>(),
            pPriv: null_mut(),
            profilerType: sys::CUpti_ProfilerType::CUPTI_PROFILER_TYPE_PM_SAMPLING,
            pChipName: chip_name_c.as_ptr(),
            pCounterAvailabilityImage: availability_image.as_ptr(),
            pHostObject: null_mut(),
        };
        // This requires elevated priviliges:
        tracing::trace!("initializing profiler host");
        unsafe { result::profiler_host::initialize(&mut params) }?;

        let context = Context {
            object: params.pHostObject,
        };

        Ok(context)
    }
}

#[cfg(test)]
mod test {
    use crate::profiling_host::ContextBuilder;

    #[test]
    fn pm_sampling() -> std::result::Result<(), Box<dyn std::error::Error>> {
        tracing_subscriber::fmt()
            .with_max_level(tracing::level_filters::LevelFilter::TRACE)
            .init();

        let context = cudarc::driver::CudaContext::new(0)?;

        ContextBuilder::new().device_index(0).build()?;

        Ok(())
    }
}
