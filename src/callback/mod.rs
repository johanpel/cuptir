use std::sync::OnceLock;

use cudarc::{cupti::result, cupti::sys};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use strum::FromRepr;

use crate::{error::CuptirError, try_demangle_from_ffi, try_str_from_ffi};

pub mod driver;
pub mod runtime;

#[derive(Clone, Copy, Debug, FromRepr, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[repr(u32)]
pub enum Domain {
    DriverApi = sys::CUpti_CallbackDomain::CUPTI_CB_DOMAIN_DRIVER_API as u32,
    RuntimeApi = sys::CUpti_CallbackDomain::CUPTI_CB_DOMAIN_RUNTIME_API as u32,
    Resource = sys::CUpti_CallbackDomain::CUPTI_CB_DOMAIN_RESOURCE as u32,
    Synchronize = sys::CUpti_CallbackDomain::CUPTI_CB_DOMAIN_SYNCHRONIZE as u32,
    Nvtx = sys::CUpti_CallbackDomain::CUPTI_CB_DOMAIN_NVTX as u32,
    State = sys::CUpti_CallbackDomain::CUPTI_CB_DOMAIN_STATE as u32,
}

impl From<Domain> for sys::CUpti_CallbackDomain {
    #[rustfmt::skip]
    fn from(value: Domain) -> Self {
        use sys::CUpti_CallbackDomain as c;
        match value {
            Domain::DriverApi => c::CUPTI_CB_DOMAIN_DRIVER_API,
            Domain::RuntimeApi => c::CUPTI_CB_DOMAIN_RUNTIME_API,
            Domain::Resource => c::CUPTI_CB_DOMAIN_RESOURCE,
            Domain::Synchronize => c::CUPTI_CB_DOMAIN_SYNCHRONIZE,
            Domain::Nvtx => c::CUPTI_CB_DOMAIN_NVTX,
            Domain::State => c::CUPTI_CB_DOMAIN_STATE
        }
    }
}

impl TryFrom<sys::CUpti_CallbackDomain> for Domain {
    type Error = CuptirError;

    fn try_from(value: sys::CUpti_CallbackDomain) -> Result<Self, Self::Error> {
        match value {
            sys::CUpti_CallbackDomain::CUPTI_CB_DOMAIN_INVALID
            | sys::CUpti_CallbackDomain::CUPTI_CB_DOMAIN_SIZE
            | sys::CUpti_CallbackDomain::CUPTI_CB_DOMAIN_FORCE_INT => {
                Err(CuptirError::SentinelEnum(value as u32))
            }
            other => Self::from_repr(other as u32).ok_or(CuptirError::Corrupted),
        }
    }
}

pub type CallbackHandlerFn =
    dyn Fn(Data) -> Result<(), Box<dyn std::error::Error + Send + Sync>> + Send + Sync;

/// Globally accessible handler for callbacks from the CUPTI Callback API.
///
/// Because the buffer complete callback doesn't have a way of passing custom data, e.g.
/// a reference to the Context, this is needed to get to the Rust callback for  record
/// processing.
pub(crate) static CALLBACK_HANDLER: OnceLock<Box<CallbackHandlerFn>> = OnceLock::new();

/// Sets the activity record handler. This can be only called once.
pub(crate) fn set_callback_handler(
    callback_handler: Box<CallbackHandlerFn>,
) -> Result<(), CuptirError> {
    CALLBACK_HANDLER
        .set(callback_handler)
        .map_err(|_| CuptirError::CallbackHandler("can only be set once".into()))
}

/// Calls the global record handler function if it is installed.
fn handle_callback(callback: Data) -> Result<(), CuptirError> {
    if let Some(handler) = CALLBACK_HANDLER.get() {
        handler(callback).map_err(|e| CuptirError::CallbackHandler(e.to_string()))
    } else {
        tracing::warn!("callback received, but no callback handler is installed");
        Ok(())
    }
}

// Obtain the name of a callback within a specific domain.
pub(crate) fn callback_name(
    domain: sys::CUpti_CallbackDomain,
    id: u32,
) -> Result<String, CuptirError> {
    let mut name_ptr: *const std::os::raw::c_char = std::ptr::null_mut();
    unsafe {
        result::get_callback_name(domain, id, &mut name_ptr)?;
    }
    Ok(if let Some(name) = unsafe { try_str_from_ffi(name_ptr) } {
        name.to_owned()
    } else {
        "<unnamed>".to_string()
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[repr(u32)]
pub enum Site {
    Enter = sys::CUpti_ApiCallbackSite::CUPTI_API_ENTER as u32,
    Exit = sys::CUpti_ApiCallbackSite::CUPTI_API_EXIT as u32,
}

impl TryFrom<sys::CUpti_ApiCallbackSite> for Site {
    type Error = CuptirError;

    fn try_from(value: sys::CUpti_ApiCallbackSite) -> Result<Self, Self::Error> {
        match value {
            sys::CUpti_ApiCallbackSite::CUPTI_API_ENTER => Ok(Site::Enter),
            sys::CUpti_ApiCallbackSite::CUPTI_API_EXIT => Ok(Site::Exit),
            sys::CUpti_ApiCallbackSite::CUPTI_API_CBSITE_FORCE_INT => {
                Err(CuptirError::SentinelEnum(value as u32))
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
pub struct ApiData {
    pub callback_site: Site,
    pub function_name: Option<String>,
    // pub functionParams: *const ::core::ffi::c_void,
    // pub functionReturnValue: *mut ::core::ffi::c_void,
    pub symbol_name: Option<String>,
    // pub context: CUcontext,
    pub context_uid: u32,
    // pub correlationData: *mut u64,
    pub correlation_id: u32,
}

impl ApiData {
    // We need to be careful with the source field in ApiData. CUPTI docs mention that
    // the entry is only valid for driver and runtime "launch" callbacks, where it
    // returns the name of the kernel. Unfortunately, it doesn't zero-initialize the
    // struct, so we can't simply check by nullptr whether this holds anything valid. We
    // have to check the callback ids, so they are needed for the conversion, hence
    // we're not trivially implementing TryFrom.

    fn try_from_cupti_callback_data(
        data: &sys::CUpti_CallbackData,
        with_symbol_name: bool,
    ) -> Result<Self, CuptirError> {
        Ok(Self {
            callback_site: data.callbackSite.try_into()?,
            function_name: unsafe { try_demangle_from_ffi(data.functionName) },
            symbol_name: if with_symbol_name {
                unsafe { try_demangle_from_ffi(data.symbolName) }
            } else {
                None
            },
            context_uid: data.contextUid,
            correlation_id: data.correlationId,
        })
    }

    fn try_from_driver(
        id: sys::CUpti_driver_api_trace_cbid_enum,
        data: &sys::CUpti_CallbackData,
    ) -> Result<Self, CuptirError> {
        use sys::CUpti_driver_api_trace_cbid_enum as D;
        Self::try_from_cupti_callback_data(
            data,
            matches!(
                id,
                D::CUPTI_DRIVER_TRACE_CBID_cuLaunch
                    | D::CUPTI_DRIVER_TRACE_CBID_cuLaunchCooperativeKernel_ptsz
                    | D::CUPTI_DRIVER_TRACE_CBID_cuLaunchCooperativeKernel
                    | D::CUPTI_DRIVER_TRACE_CBID_cuLaunchCooperativeKernelMultiDevice
                    | D::CUPTI_DRIVER_TRACE_CBID_cuLaunchGrid
                    | D::CUPTI_DRIVER_TRACE_CBID_cuLaunchGridAsync
                    | D::CUPTI_DRIVER_TRACE_CBID_cuLaunchHostFunc_ptsz
                    | D::CUPTI_DRIVER_TRACE_CBID_cuLaunchHostFunc
                    | D::CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel_ptsz
                    | D::CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel
                    | D::CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx_ptsz
                    | D::CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx
            ),
        )
    }

    fn try_from_runtime(
        id: sys::CUpti_runtime_api_trace_cbid_enum,
        data: &sys::CUpti_CallbackData,
    ) -> Result<Self, CuptirError> {
        use sys::CUpti_runtime_api_trace_cbid_enum as R;
        Self::try_from_cupti_callback_data(
            data,
            matches!(
                id,
                R::CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_ptsz_v7000
                    | R::CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020
                    | R::CUPTI_RUNTIME_TRACE_CBID_cudaLaunchCooperativeKernel_ptsz_v9000
                    | R::CUPTI_RUNTIME_TRACE_CBID_cudaLaunchCooperativeKernel_v9000
                    | R::CUPTI_RUNTIME_TRACE_CBID_cudaLaunchCooperativeKernelMultiDevice_v9000
                    | R::CUPTI_RUNTIME_TRACE_CBID_cudaLaunchHostFunc_ptsz_v10000
                    | R::CUPTI_RUNTIME_TRACE_CBID_cudaLaunchHostFunc_v10000
                    | R::CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_ptsz_v7000
                    | R::CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000
                    | R::CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernelExC_ptsz_v11060
                    | R::CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernelExC_v11060
            ),
        )
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
pub struct DriverApiCallbackData {
    /// The callback id in the driver domain
    callback_id: u32,
    callback_data: ApiData,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
pub struct RuntimeApiCallbackData {
    /// The callback id in the runtime domain
    callback_id: u32,
    callback_data: ApiData,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
pub enum Data {
    DriverApi(DriverApiCallbackData),
    RuntimeApi(RuntimeApiCallbackData),
    // TODO: the other domains
    Resource,
    Synchronize,
    Nvtx,
    State,
}

pub(crate) unsafe extern "C" fn handler(
    _userdata: *mut ::core::ffi::c_void,
    domain: sys::CUpti_CallbackDomain,
    cbid: sys::CUpti_CallbackId,
    cbdata: *const ::core::ffi::c_void,
) {
    if cbdata.is_null() {
        tracing::warn!("cupti callback handler called with nullptr for data");
        return;
    }

    if let Ok(domain) = Domain::try_from(domain) {
        let maybe_callback = match domain {
            Domain::DriverApi => {
                let data: &sys::CUpti_CallbackData =
                    unsafe { &*(cbdata as *const sys::CUpti_CallbackData) };
                ApiData::try_from_driver(
                    unsafe {
                        std::mem::transmute::<
                            u32,
                            cudarc::cupti::sys::CUpti_driver_api_trace_cbid_enum,
                        >(cbid)
                    },
                    data,
                )
                .map(|callback_data| DriverApiCallbackData {
                    callback_id: cbid,
                    callback_data,
                })
                .map(Data::DriverApi)
            }
            Domain::RuntimeApi => {
                let data: &sys::CUpti_CallbackData =
                    unsafe { &*(cbdata as *const sys::CUpti_CallbackData) };
                ApiData::try_from_runtime(
                    unsafe {
                        std::mem::transmute::<
                            u32,
                            cudarc::cupti::sys::CUpti_runtime_api_trace_cbid_enum,
                        >(cbid)
                    },
                    data,
                )
                .map(|callback_data| RuntimeApiCallbackData {
                    callback_id: cbid,
                    callback_data,
                })
                .map(Data::RuntimeApi)
            }
            Domain::Resource => Ok(Data::Resource),
            Domain::Synchronize => Ok(Data::Synchronize),
            Domain::Nvtx => Ok(Data::Nvtx),
            Domain::State => Ok(Data::State),
        };

        match maybe_callback {
            Ok(callback) => {
                if let Err(error) = handle_callback(callback) {
                    tracing::warn!("cupti callback handler error: {error}")
                }
            }
            Err(error) => {
                tracing::warn!("cupti callback data conversion error: {error}")
            }
        }
    }
}
