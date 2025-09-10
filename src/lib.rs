#![doc = include_str!("../README.md")]

use std::collections::HashSet;

use cudarc::{cupti::result as cupti, cupti::sys as ffi};
use tracing::trace;

use crate::error::CuptirError;

pub mod activity;
pub mod error;

/// Global context for cuptir.
///
/// Only one instance of this struct can exist at a time, because CUPTI can only handle
/// one subscriber.
///
/// This [Context] should be created before any other CUDA interactions take place in
/// the program.
#[derive(Debug)]
#[repr(C)]
pub struct Context {
    enabled_activity_kinds: Vec<ffi::CUpti_ActivityKind>,
    ffi_subscriber_handle: ffi::CUpti_SubscriberHandle,
    user_data: *mut u8, // User data on the heap, TODO make generic
}

impl Context {
    /// Return a builder to help set up the [Context].
    pub fn builder() -> ContextBuilder {
        ContextBuilder::default()
    }
}

/// Builder to help initialize a [Context].
#[derive(Default)]
pub struct ContextBuilder {
    enabled_activity_kinds: HashSet<ffi::CUpti_ActivityKind>,
    activity_record_handler: Option<Box<activity::RecordHandler>>,
    activity_latency_timestamps: bool,
}

impl ContextBuilder {
    /// Add the supplied activity kind to the set of activated activity kinds.
    pub fn with_activity_kind(mut self, kind: ffi::CUpti_ActivityKind) -> Self {
        self.enabled_activity_kinds.insert(kind);
        self
    }

    /// Add the supplied activity kinds to the set of activated activity kinds.
    pub fn with_activity_kinds(
        mut self,
        kinds: impl IntoIterator<Item = ffi::CUpti_ActivityKind>,
    ) -> Self {
        self.enabled_activity_kinds.extend(kinds);
        self
    }

    /// Supply the activity record handler function.
    pub fn with_activity_record_handler<F>(mut self, handler: F) -> Self
    where
        F: Fn(activity::Record) -> Result<(), Box<dyn std::error::Error + Send + Sync>>
            + Send
            + Sync
            + 'static,
    {
        self.activity_record_handler = Some(Box::new(handler));
        self
    }

    /// Enable or disable latency timestamps (`queued` and `submitted`) for
    /// [activity::KernelRecord] event records.
    ///
    /// Disabled by default.
    pub fn with_activity_latency_timestamps(mut self, enabled: bool) -> Self {
        self.activity_latency_timestamps = enabled;
        self
    }

    /// Build the [Context].
    ///
    /// This function can fail if there is another Context or another type of CUPTI
    /// subscriber, such as NSight Systems or the DCGM.
    pub fn build(self) -> Result<Context, CuptirError> {
        let mut context = Context {
            enabled_activity_kinds: self.enabled_activity_kinds.into_iter().collect(),
            ffi_subscriber_handle: std::ptr::null_mut(),
            user_data: std::ptr::null_mut(),
        };

        // Subscribe. This can fail if theres some other thing trying to use CUPTI
        // already, which is a requirement that is recommended to be checked before
        // doing anything else. So even if we don't use the callback API we still
        // need to pass a handler.
        trace!("subscribing context");
        unsafe {
            cupti::subscribe(
                &mut context.ffi_subscriber_handle,
                None,
                context.user_data as *mut _,
            )?;
        }

        // Set the activity record handler, if any.
        if let Some(activity_record_handler) = self.activity_record_handler {
            trace!("setting record handler");
            activity::set_record_handler(activity_record_handler)?;
        }

        trace!("registering activity buffer callbacks");
        cupti::activity::register_callbacks(
            Some(activity::buffer_requested_callback),
            Some(activity::buffer_complete_callback),
        )?;

        if self.activity_latency_timestamps {
            trace!("enabling latency timestamps");
            cupti::activity::enable_latency_timestamps(1)?;
        }

        trace!(
            "enabling activity kinds: {:?}",
            context.enabled_activity_kinds
        );
        context
            .enabled_activity_kinds
            .iter()
            .try_for_each(|activity_kind| cupti::activity::enable(*activity_kind))?;

        Ok(context)
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        trace!("flushing activity buffer");
        if let Err(error) = cupti::activity::flush_all(
            ffi::CUpti_ActivityFlag::CUPTI_ACTIVITY_FLAG_FLUSH_FORCED as u32,
        ) {
            tracing::warn!("unable to flush CUPTI activity: {error}");
        }

        trace!(
            "disabling activity kinds: {:?}",
            self.enabled_activity_kinds
        );
        if let Err(error) = self
            .enabled_activity_kinds
            .iter()
            .try_for_each(|activity_kind| cupti::activity::disable(*activity_kind))
        {
            tracing::warn!("unable to disable CUPTI activity: {error}");
        }

        if !self.ffi_subscriber_handle.is_null() {
            trace!("unsubscribing context");
            if let Err(error) = unsafe { cupti::unsubscribe(self.ffi_subscriber_handle) } {
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

#[cfg(test)]
mod tests {
    use serial_test::serial;

    use super::*;

    #[test]
    #[serial]
    fn make_context() -> std::result::Result<(), CuptirError> {
        let _context = Context::builder()
            .with_activity_kind(ffi::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL)
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
                    ffi::CUptiResult::CUPTI_ERROR_MULTIPLE_SUBSCRIBERS_NOT_SUPPORTED
                ))
        );
    }

    #[test]
    #[serial]
    fn activity_record_handler() -> std::result::Result<(), Box<dyn std::error::Error>> {
        use std::sync::{Arc, Mutex};
        let records: Arc<Mutex<Vec<activity::Record>>> = Arc::new(Mutex::new(vec![]));

        let recs_cb = Arc::clone(&records);
        let context = Context::builder()
            .with_activity_record_handler(move |record| {
                recs_cb.lock().unwrap().push(record);
                Ok(())
            })
            .with_activity_kind(ffi::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_DRIVER)
            .build();

        // Init the driver and get the count. This should result in exactly one event.
        cudarc::driver::result::init()?;
        let device_count = cudarc::driver::result::device::get_count()?;
        println!("Device count: {device_count}");

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
