use std::sync::OnceLock;

use cudarc::{cupti::result, cupti::sys};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use strum::FromRepr;

use crate::{error::CuptirError, try_string_from_ffi};

#[derive(Clone, Copy, Debug, FromRepr, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[repr(u32)]
pub enum Domain {
    DriverApi = 1,
    RuntimeApi = 2,
    Resource = 3,
    Synchronize = 4,
    Nvtx = 5,
    State = 6,
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
    dyn Fn(Callback) -> Result<(), Box<dyn std::error::Error + Send + Sync>> + Send + Sync;

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
fn handle_callback(callback: Callback) -> Result<(), CuptirError> {
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
    Ok(if let Some(name) = try_string_from_ffi(name_ptr) {
        name
    } else {
        "<unnamed>".to_string()
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[repr(u32)]
pub enum CallbackSite {
    Enter,
    Exit,
}

impl TryFrom<sys::CUpti_ApiCallbackSite> for CallbackSite {
    type Error = CuptirError;

    fn try_from(value: sys::CUpti_ApiCallbackSite) -> Result<Self, Self::Error> {
        match value {
            sys::CUpti_ApiCallbackSite::CUPTI_API_ENTER => Ok(CallbackSite::Enter),
            sys::CUpti_ApiCallbackSite::CUPTI_API_EXIT => Ok(CallbackSite::Exit),
            sys::CUpti_ApiCallbackSite::CUPTI_API_CBSITE_FORCE_INT => {
                Err(CuptirError::SentinelEnum(value as u32))
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
pub struct CallbackData {
    pub callback_site: CallbackSite,
    pub function_name: Option<String>,
    // pub functionParams: *const ::core::ffi::c_void,
    // pub functionReturnValue: *mut ::core::ffi::c_void,
    // pub symbol_name: Option<String>,
    // pub context: CUcontext,
    pub context_uid: u32,
    // pub correlationData: *mut u64,
    pub correlation_id: u32,
}

impl TryFrom<&sys::CUpti_CallbackData> for CallbackData {
    type Error = CuptirError;

    fn try_from(value: &sys::CUpti_CallbackData) -> Result<Self, Self::Error> {
        Ok(Self {
            callback_site: value.callbackSite.try_into()?,
            function_name: try_string_from_ffi(value.functionName),
            // TODO: figure this doc out:
            // This entry is valid only for driver and runtime launch callbacks, where it returns the name of the kernel.
            // symbol_name: try_string_from_ffi(value.symbolName),
            context_uid: value.contextUid,
            correlation_id: value.correlationId,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
pub enum Callback {
    DriverApi(CallbackData),
    RuntimeApi(CallbackData),
    // TODO:
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
                CallbackData::try_from(data).map(Callback::DriverApi)
            }
            Domain::RuntimeApi => {
                let data: &sys::CUpti_CallbackData =
                    unsafe { &*(cbdata as *const sys::CUpti_CallbackData) };
                CallbackData::try_from(data).map(Callback::RuntimeApi)
            }
            Domain::Resource => Ok(Callback::Resource),
            Domain::Synchronize => Ok(Callback::Synchronize),
            Domain::Nvtx => Ok(Callback::Nvtx),
            Domain::State => Ok(Callback::State),
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
