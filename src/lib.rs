use core::ffi;
use std::{collections::HashSet, ffi::CStr, ptr::NonNull};

use cudarc::{cupti::result as cupti, cupti::sys};
use tracing::trace;

use crate::error::CuptirError;

pub mod activity;
pub mod callback;
pub mod error;

/// Global context for cuptir.
///
/// Only one instance of this struct can exist at a time, because CUPTI can only handle
/// one subscriber.
///
/// This [Context] should be created before any other CUDA interactions take place in
/// the program.
// TODO: probably split this out over multiple contexts, one per CUPTI module
#[derive(Debug)]
#[repr(C)]
pub struct Context {
    // Activity API stuff
    enabled_activity_kinds: Vec<activity::Kind>,

    // Callback API stuff
    enabled_callback_domains: Vec<callback::Domain>,
    sys_subscriber_handle: sys::CUpti_SubscriberHandle,
    user_data: *mut u8, // User data on the heap, TODO make generic
}

impl Context {
    /// Return a builder to help set up the [Context].
    pub fn builder() -> ContextBuilder {
        ContextBuilder::default()
    }

    /// Flush the activity buffers, potentially triggering the activity record handler.
    pub fn flush_activity() -> Result<(), CuptirError> {
        trace!("flushing activity buffer");
        cupti::activity::flush_all(
            sys::CUpti_ActivityFlag::CUPTI_ACTIVITY_FLAG_FLUSH_FORCED as u32,
        )?;
        Ok(())
    }
}

/// Builder to help initialize a [Context].
#[derive(Default)]
pub struct ContextBuilder {
    enabled_activity_kinds: HashSet<activity::Kind>,
    activity_record_buffer_handler: Option<Box<activity::RecordBufferHandlerFn>>,
    activity_latency_timestamps: bool,

    enabled_callback_domains: HashSet<callback::Domain>,
    callback_handler: Option<Box<callback::CallbackHandlerFn>>,
}

impl ContextBuilder {
    /// Add the supplied activity kinds to the set of activated activity kinds.
    pub fn with_activity_kinds(mut self, kinds: impl IntoIterator<Item = activity::Kind>) -> Self {
        self.enabled_activity_kinds.extend(kinds);
        self
    }

    /// Set the activity record buffer handler function.
    ///
    /// The handler function should return as quickly as possible to minimize profiling
    /// overhead.
    pub fn with_activity_record_buffer_handler<F>(mut self, handler: F) -> Self
    where
        F: Fn(activity::RecordBuffer) -> Result<(), Box<dyn std::error::Error + Send + Sync>>
            + Send
            + Sync
            + 'static,
    {
        self.activity_record_buffer_handler = Some(Box::new(handler));
        self
    }

    /// Enable latency timestamps (`queued` and `submitted`) for
    /// [activity::KernelRecord] event records.
    ///
    /// Disabled by default.
    pub fn enable_activity_latency_timestamps(mut self, enabled: bool) -> Self {
        self.activity_latency_timestamps = enabled;
        self
    }

    /// Add the supplied domains to the set of activated callback domains.
    pub fn with_callback_domains(
        mut self,
        domains: impl IntoIterator<Item = callback::Domain>,
    ) -> Self {
        self.enabled_callback_domains.extend(domains);
        self
    }

    /// Set the handler for the Callback API.
    ///
    /// The handler function should return as quickly as possible to minimize profiling
    /// overhead.
    pub fn with_callback_handler<F>(mut self, handler: F) -> Self
    where
        F: Fn(callback::Callback) -> Result<(), Box<dyn std::error::Error + Send + Sync>>
            + Send
            + Sync
            + 'static,
    {
        self.callback_handler = Some(Box::new(handler));
        self
    }

    /// Build the [Context].
    ///
    /// This function can fail if there is another Context or any other type of CUPTI
    /// subscriber.
    pub fn build(self) -> Result<Context, CuptirError> {
        let mut context = Context {
            enabled_activity_kinds: self.enabled_activity_kinds.into_iter().collect(),
            sys_subscriber_handle: std::ptr::null_mut(),
            user_data: std::ptr::null_mut(),

            enabled_callback_domains: self.enabled_callback_domains.into_iter().collect(),
        };

        // Subscribe. This can fail if theres some other thing trying to use CUPTI
        // already, which is a requirement that is recommended to be checked before
        // doing anything else. So even if we don't use the callback API we still
        // need to pass a handler.
        trace!("subscribing context");
        if let Some(callback_handler) = self.callback_handler {
            if context.enabled_callback_domains.is_empty() {
                return Err(CuptirError::Builder(format!(
                    "callback handler provided, but no domains were enabled"
                )));
            } else {
                callback::set_callback_handler(callback_handler)?;
                unsafe {
                    cupti::subscribe(
                        &mut context.sys_subscriber_handle,
                        Some(callback::handler),
                        context.user_data as *mut _,
                    )?;
                }
            }
        } else {
            if !context.enabled_callback_domains.is_empty() {
                return Err(CuptirError::Builder(format!(
                    "callback domains {:?} are enabled, but no callback handler is set",
                    context.enabled_callback_domains
                )));
            } else {
                unsafe {
                    cupti::subscribe(
                        &mut context.sys_subscriber_handle,
                        None,
                        context.user_data as *mut _,
                    )?;
                }
            }
        }

        // Enable the callback domains
        trace!(
            "enabling callback domains: {:?}",
            context.enabled_callback_domains
        );
        context
            .enabled_callback_domains
            .iter()
            .try_for_each(|domain| unsafe {
                cupti::enable_domain(1, context.sys_subscriber_handle, (*domain).into())
            })?;

        // Set the activity record handler, if any.
        if let Some(activity_record_buffer_handler) = self.activity_record_buffer_handler {
            trace!("setting record handler");
            activity::set_record_buffer_handler(activity_record_buffer_handler)?;
        }

        if self.activity_latency_timestamps {
            trace!("enabling latency timestamps");
            cupti::activity::enable_latency_timestamps(1)?;
        }

        trace!("registering activity buffer callbacks");
        cupti::activity::register_callbacks(
            Some(activity::buffer_requested_callback),
            Some(activity::buffer_complete_callback),
        )?;

        trace!(
            "enabling activity kinds: {:?}",
            context.enabled_activity_kinds
        );
        context
            .enabled_activity_kinds
            .iter()
            .try_for_each(|activity_kind| cupti::activity::enable((*activity_kind).into()))?;

        Ok(context)
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        if let Err(error) = Self::flush_activity() {
            tracing::warn!("unable to flush activity buffer: {error}")
        }

        trace!(
            "disabling activity kinds: {:?}",
            self.enabled_activity_kinds
        );
        if let Err(error) = self
            .enabled_activity_kinds
            .iter()
            .try_for_each(|activity_kind| cupti::activity::disable((*activity_kind).into()))
        {
            tracing::warn!("unable to disable CUPTI activity: {error}");
        }

        if !self.sys_subscriber_handle.is_null() {
            trace!("unsubscribing context");
            if let Err(error) = unsafe { cupti::unsubscribe(self.sys_subscriber_handle) } {
                tracing::warn!("unable to unsubscribe CUPTI client: {error}");
            }
        }

        if !self.user_data.is_null() {
            unsafe {
                let _user_data = Box::from_raw(self.user_data);
            }
        }
    }
}

// TODO: figure out whether we should deallocate ourselves across usages. Docs aren't
// super clear on this.
fn try_string_from_ffi(c_string: *const ffi::c_char) -> Option<String> {
    NonNull::new(c_string as *mut _).map(|p| {
        unsafe { CStr::from_ptr(p.as_ptr()) }
            .to_string_lossy()
            .into_owned()
    })
}

// TODO: optimize
fn try_demangle_from_ffi(c_string: *const ffi::c_char) -> Option<String> {
    try_string_from_ffi(c_string).map(|string| {
        cpp_demangle::Symbol::new(string.as_str())
            .map(|symbol| symbol.to_string())
            .ok()
            .unwrap_or(string)
    })
}

#[cfg(test)]
mod tests {
    use serial_test::serial;

    use super::*;

    #[test]
    #[serial]
    fn make_context() -> std::result::Result<(), CuptirError> {
        let _context = Context::builder()
            .with_activity_kinds([activity::Kind::ConcurrentKernel])
            .build()?;
        Ok(())
    }

    #[test]
    #[serial]
    fn make_multiple_contexts_fails() {
        let context0 = Context::builder().build();
        let context1 = Context::builder().build();

        assert!(context0.is_ok());
        assert!(context1.is_err());
        assert!(
            context1.unwrap_err()
                == CuptirError::Cupti(cupti::CuptiError(
                    sys::CUptiResult::CUPTI_ERROR_MULTIPLE_SUBSCRIBERS_NOT_SUPPORTED
                ))
        );
    }

    #[test]
    #[serial]
    fn activity_record_buffer_handler() -> std::result::Result<(), Box<dyn std::error::Error>> {
        use std::sync::{Arc, Mutex};
        let records: Arc<Mutex<Vec<activity::Record>>> = Arc::new(Mutex::new(vec![]));

        let recs_cb = Arc::clone(&records);
        let context = Context::builder()
            .with_activity_record_buffer_handler(move |buffer| {
                recs_cb.lock().unwrap().extend(
                    buffer
                        .into_iter()
                        .filter_map(|maybe_record| maybe_record.ok()),
                );
                Ok(())
            })
            .with_activity_kinds([activity::Kind::Driver])
            .build();

        // Init the driver and get the count. This should result in exactly one event.
        cudarc::driver::result::init()?;
        let _ = cudarc::driver::result::device::get_count()?;

        drop(context);

        let recs = records.lock().unwrap();
        assert_eq!(recs.len(), 1);
        match &recs[0] {
            activity::Record::DriverApi(rec) => assert_eq!(rec.name, "cuDeviceGetCount"),
            _ => panic!("unexpected record kind"),
        }

        Ok(())
    }
}
