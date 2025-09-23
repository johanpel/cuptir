//! Safe wrappers around the CUPTI Callback API.
use std::collections::HashSet;
use std::sync::OnceLock;

use cudarc::{cupti::result, cupti::sys};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use tracing::trace;

use crate::{
    driver,
    enums::CallbackDomain,
    error::CuptirError,
    runtime,
    utils::{try_demangle_from_ffi, try_str_from_ffi},
};

pub type Domain = crate::enums::CallbackDomain;
pub type Site = crate::enums::ApiCallbackSite;

pub type HandlerFn =
    dyn Fn(Data) -> Result<(), Box<dyn std::error::Error + Send + Sync>> + Send + Sync;

/// Globally accessible handler for callbacks from the CUPTI Callback API.
///
/// Because the buffer complete callback doesn't have a way of passing custom data, e.g.
/// a reference to the Context, this is needed to get to the Rust callback for  record
/// processing.
static HANDLER: OnceLock<Box<HandlerFn>> = OnceLock::new();

/// Sets the activity record handler. This can be only called once.
fn set_handler(handler: Box<HandlerFn>) -> Result<(), CuptirError> {
    HANDLER
        .set(handler)
        .map_err(|_| CuptirError::CallbackHandler("can only be set once".into()))
}

/// Calls the global record handler function if it is installed.
fn handle_callback(callback: Data) -> Result<(), CuptirError> {
    if let Some(handler) = HANDLER.get() {
        handler(callback).map_err(|e| CuptirError::CallbackHandler(e.to_string()))
    } else {
        tracing::warn!("callback received, but no callback handler is installed");
        Ok(())
    }
}

// Obtain the name of a callback within a specific domain.
pub(crate) fn callback_name(domain: CallbackDomain, id: u32) -> Result<String, CuptirError> {
    let mut name_ptr: *const std::os::raw::c_char = std::ptr::null_mut();
    unsafe {
        result::get_callback_name(domain.into(), id, &mut name_ptr)?;
    }
    Ok(if let Some(name) = unsafe { try_str_from_ffi(name_ptr) } {
        name.to_owned()
    } else {
        "<unnamed>".to_string()
    })
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
pub struct DriverApiCallbackData {
    pub function: driver::Function,
    // TODO: figure out how to best serialize and expose this field in the public API
    #[cfg_attr(feature = "serde", serde(skip))]
    pub arguments: Option<driver::FunctionParams>,
    pub site: Site,
    // pub function_name: Option<String>,
    // pub functionParams: *const ::core::ffi::c_void,
    // pub functionReturnValue: *mut ::core::ffi::c_void,
    pub symbol_name: Option<String>,
    // pub context: CUcontext,
    pub context_uid: u32,
    // pub correlationData: *mut u64,
    pub correlation_id: u32,
}

impl DriverApiCallbackData {
    fn try_new(
        driver_callback_id: u32,
        value: &sys::CUpti_CallbackData,
    ) -> Result<Self, CuptirError> {
        let function = driver::Function::try_from(unsafe {
            std::mem::transmute::<u32, sys::CUpti_driver_api_trace_cbid>(driver_callback_id)
        })?;
        let site = value.callbackSite.try_into()?;

        // CUPTI docs mention that the symbol_name is only valid for callbacks on
        // "launch" functions, where it returns the name of the kernel. Unfortunately,
        // it doesn't zero-initialize the struct, so we can't simply check by nullptr
        // whether this holds anything valid. We have to check the callback ids, so they
        // are needed for the conversion, hence we're not trivially implementing
        // TryFrom.
        let symbol_name = if matches!(
            function,
            driver::Function::cuLaunch
                | driver::Function::cuLaunchCooperativeKernel_ptsz
                | driver::Function::cuLaunchCooperativeKernel
                | driver::Function::cuLaunchCooperativeKernelMultiDevice
                | driver::Function::cuLaunchGrid
                | driver::Function::cuLaunchGridAsync
                | driver::Function::cuLaunchHostFunc_ptsz
                | driver::Function::cuLaunchHostFunc
                | driver::Function::cuLaunchKernel_ptsz
                | driver::Function::cuLaunchKernel
                | driver::Function::cuLaunchKernelEx_ptsz
                | driver::Function::cuLaunchKernelEx
        ) {
            unsafe { try_demangle_from_ffi(value.symbolName) }
        } else {
            None
        };
        Ok(Self {
            function,
            arguments: driver::FunctionParams::try_new(function, value.functionParams).ok(),
            site,
            symbol_name,
            context_uid: value.contextUid,
            correlation_id: value.correlationId,
        })
    }

    pub fn function_name(&self) -> Result<String, CuptirError> {
        callback_name(Domain::DriverApi, self.function as u32)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
pub struct RuntimeApiCallbackData {
    pub function: runtime::Function,
    #[cfg_attr(feature = "serde", serde(skip))]
    pub arguments: Option<runtime::FunctionParams>,
    pub site: Site,
    // pub function_name: Option<String>,
    // pub functionParams: *const ::core::ffi::c_void,
    // pub functionReturnValue: *mut ::core::ffi::c_void,
    pub symbol_name: Option<String>,
    // pub context: CUcontext,
    pub context_uid: u32,
    // pub correlationData: *mut u64,
    pub correlation_id: u32,
}

impl RuntimeApiCallbackData {
    fn try_new(
        runtime_callback_id: u32,
        value: &sys::CUpti_CallbackData,
    ) -> Result<Self, CuptirError> {
        let function = runtime::Function::try_from(unsafe {
            std::mem::transmute::<u32, sys::CUpti_runtime_api_trace_cbid>(runtime_callback_id)
        })?;
        let site = value.callbackSite.try_into()?;

        // CUPTI docs mention that the entry is only valid for "launch" callbacks.
        let symbol_name = if matches!(
            function,
            runtime::Function::cudaLaunch_ptsz_v7000
                | runtime::Function::cudaLaunch_v3020
                | runtime::Function::cudaLaunchCooperativeKernel_ptsz_v9000
                | runtime::Function::cudaLaunchCooperativeKernel_v9000
                | runtime::Function::cudaLaunchCooperativeKernelMultiDevice_v9000
                | runtime::Function::cudaLaunchHostFunc_ptsz_v10000
                | runtime::Function::cudaLaunchHostFunc_v10000
                | runtime::Function::cudaLaunchKernel_ptsz_v7000
                | runtime::Function::cudaLaunchKernel_v7000
                | runtime::Function::cudaLaunchKernelExC_ptsz_v11060
                | runtime::Function::cudaLaunchKernelExC_v11060
        ) {
            unsafe { try_demangle_from_ffi(value.symbolName) }
        } else {
            None
        };
        Ok(Self {
            function,
            arguments: runtime::FunctionParams::try_new(function, value.functionParams).ok(),
            site,
            symbol_name,
            context_uid: value.contextUid,
            correlation_id: value.correlationId,
        })
    }

    pub fn function_name(&self) -> Result<String, CuptirError> {
        callback_name(Domain::RuntimeApi, self.function as u32)
    }
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
                DriverApiCallbackData::try_new(cbid, data).map(Data::DriverApi)
            }
            Domain::RuntimeApi => {
                let data: &sys::CUpti_CallbackData =
                    unsafe { &*(cbdata as *const sys::CUpti_CallbackData) };
                RuntimeApiCallbackData::try_new(cbid, data).map(Data::RuntimeApi)
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

#[derive(Default)]
pub struct Builder {
    enabled_domains: HashSet<Domain>,
    enabled_driver_functions: HashSet<driver::Function>,
    enabled_runtime_functions: HashSet<runtime::Function>,
    handler: Option<Box<HandlerFn>>,
}

impl Builder {
    pub fn new() -> Self {
        Default::default()
    }

    /// Add the supplied domains to the set of activated callback domains.
    ///
    /// Note this enables all callbacks in the specified domain and should be used with
    /// care.
    pub fn with_domains(mut self, domains: impl IntoIterator<Item = Domain>) -> Self {
        self.enabled_domains.extend(domains);
        self
    }

    pub fn with_driver_functions(
        mut self,
        callbacks: impl IntoIterator<Item = driver::Function>,
    ) -> Self {
        self.enabled_driver_functions.extend(callbacks);
        self
    }

    pub fn with_runtime_functions(
        mut self,
        callbacks: impl IntoIterator<Item = runtime::Function>,
    ) -> Self {
        self.enabled_runtime_functions.extend(callbacks);
        self
    }

    /// Set the handler for the Callback API.
    ///
    /// The handler function should return as quickly as possible to minimize profiling
    /// overhead.
    pub fn with_handler<F>(mut self, handler: F) -> Self
    where
        F: Fn(Data) -> Result<(), Box<dyn std::error::Error + Send + Sync>> + Send + Sync + 'static,
    {
        self.handler = Some(Box::new(handler));
        self
    }
}

#[derive(Debug)]
pub(crate) struct Context {
    subscriber_handle: sys::CUpti_SubscriberHandle,
    user_data: *mut std::ffi::c_void,

    enabled_domains: Vec<Domain>,
    enabled_driver_functions: Vec<driver::Function>,
    enabled_runtime_functions: Vec<runtime::Function>,
}

impl Builder {
    /// Build the [Context].
    ///
    /// This function can fail if there is another Context or any other type of CUPTI
    /// subscriber.
    pub(crate) fn build(self) -> Result<Context, CuptirError> {
        let mut context = Context {
            subscriber_handle: std::ptr::null_mut(),
            user_data: std::ptr::null_mut(),

            enabled_domains: self.enabled_domains.into_iter().collect(),
            enabled_driver_functions: self.enabled_driver_functions.into_iter().collect(),
            enabled_runtime_functions: self.enabled_runtime_functions.into_iter().collect(),
        };

        // Subscribe. This can fail if theres some other thing trying to use CUPTI
        // already, which is a requirement that is recommended to be checked before
        // doing anything else. So even if we don't use the Callback API we still
        // need to pass a handler.
        let anything_enabled = !(context.enabled_domains.is_empty()
            && context.enabled_driver_functions.is_empty()
            && context.enabled_runtime_functions.is_empty());
        if let Some(client_handler) = self.handler {
            if !anything_enabled {
                return Err(CuptirError::Builder(
                    "callback handler provided, but no callback domains or functions were enabled"
                        .to_string(),
                ));
            } else {
                set_handler(client_handler)?;
                trace!("subscribing");
                unsafe {
                    result::subscribe(
                        &mut context.subscriber_handle,
                        Some(handler),
                        context.user_data as *mut _,
                    )?;
                }
            }
        } else if anything_enabled {
            return Err(CuptirError::Builder(
                "callback domains and/or functions are enabled, but no callback handler is set"
                    .into(),
            ));
        } else {
            // No callbacks were enabled and neither was handler installed, but cupti
            // clients are supposed to subscribe, so we subscribe without any handler.
            unsafe {
                trace!("subscribing");
                result::subscribe(&mut context.subscriber_handle, None, context.user_data)?;
            }
        }

        // Enable the callback domains, if needed.
        if !context.enabled_domains.is_empty() {
            trace!("enabling callback domains: {:?}", context.enabled_domains);
            context
                .enabled_domains
                .iter()
                .try_for_each(|domain| unsafe {
                    result::enable_domain(1, context.subscriber_handle, (*domain).into())
                })?;
        }

        // Enable the callback for specific driver and runtime API functions
        if !context.enabled_driver_functions.is_empty() {
            trace!(
                "enabling callback for driver functions: {:?}",
                context.enabled_driver_functions
            );
            context
                .enabled_driver_functions
                .iter()
                .try_for_each(|api| unsafe {
                    result::enable_callback(
                        1,
                        context.subscriber_handle,
                        sys::CUpti_CallbackDomain::CUPTI_CB_DOMAIN_DRIVER_API,
                        (*api) as u32,
                    )
                })?;
        }

        if !context.enabled_runtime_functions.is_empty() {
            trace!(
                "enabling callback for runtime functions: {:?}",
                context.enabled_runtime_functions
            );
            context
                .enabled_runtime_functions
                .iter()
                .try_for_each(|api| unsafe {
                    result::enable_callback(
                        1,
                        context.subscriber_handle,
                        sys::CUpti_CallbackDomain::CUPTI_CB_DOMAIN_RUNTIME_API,
                        (*api) as u32,
                    )
                })?;
        }

        Ok(context)
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        if !self.subscriber_handle.is_null() {
            trace!("unsubscribing");
            if let Err(error) = unsafe { result::unsubscribe(self.subscriber_handle) } {
                tracing::warn!("unable to unsubscribe: {error}");
            }
        }

        if !self.user_data.is_null() {
            let _user_data = unsafe { Box::from_raw(self.user_data as *mut usize) };
        }
    }
}
