use crate::error::CuptirError;

pub mod activity;
pub mod callback;
pub mod driver;
pub mod error;
pub mod runtime;
mod utils;

// CUPTI exposes pretty much the entirety of public types from the CUDA runtime and
// driver. In order to be able to work with these types, expose cudarc for now :tm:.
pub use cudarc;

// Generated enums
pub mod enums {
    include!(concat!(env!("OUT_DIR"), "/enums.rs"));
}
// Generated meta structs
pub mod function_params {
    include!(concat!(env!("OUT_DIR"), "/function_params.rs"));
}

/// Global context for cuptir.
///
/// Only one instance of this struct can exist at a time, because CUPTI can only handle
/// one subscriber.
///
/// This [Context] should be created before any other CUDA interactions take place in
/// the program.
// TODO: probably split this out over multiple contexts, one per CUPTI module
#[derive(Debug)]
pub struct Context {
    activity: Option<activity::Context>,

    // This needs to be dropped last, so this field should be the last field of this
    // struct.
    _callback: callback::Context,
}

impl Context {
    pub fn builder() -> ContextBuilder {
        Default::default()
    }

    /// Enable the unified memory counters of the activity API set when building the
    /// context.
    ///
    /// This can only be performed after CUDA driver initialization, hence it is not
    /// performed when using [ContextBuilder::build].
    ///
    /// This requires the activity context to be initialized, see
    /// [ContextBuilder::with_activity].
    pub fn enable_activity_unified_memory_counters(&self) -> Result<(), CuptirError> {
        if let Some(activity) = &self.activity {
            activity.enable_unified_memory_counters()
        } else {
            Err(CuptirError::Activity(
                "activity context is not initialized".into(),
            ))
        }
    }
}

/// Builder to help initialize a [Context].
#[derive(Default)]
pub struct ContextBuilder {
    activity: Option<activity::Builder>,
    callback: Option<callback::Builder>,
}

impl ContextBuilder {
    pub fn new() -> Self {
        Default::default()
    }

    /// Enable the CUPTI activity API, see [activity].
    pub fn with_activity(mut self, builder: activity::Builder) -> Self {
        self.activity = Some(builder);
        self
    }

    /// Enable the CUPTI callback API, see [callback].
    pub fn with_callback(mut self, builder: callback::Builder) -> Self {
        self.callback = Some(builder);
        self
    }

    /// Build the [Context].
    ///
    /// This function can fail if there is another Context or any other type of CUPTI
    /// subscriber.
    pub fn build(self) -> Result<Context, CuptirError> {
        // Callback needs to be built first because it subscribes as a CUPTI client,
        // which needs to be done before anything else.
        let callback = self.callback.unwrap_or_default().build()?;

        let activity = if let Some(activity) = self.activity {
            activity.build()?
        } else {
            None
        };

        Ok(Context {
            activity,
            _callback: callback,
        })
    }
}

#[cfg(test)]
mod tests {
    use serial_test::serial;

    use super::*;

    use cudarc::cupti::result::CuptiError;
    use cudarc::cupti::sys;

    #[test]
    #[serial]
    fn make_context() -> std::result::Result<(), CuptirError> {
        let _context = Context::builder().build()?;
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
                == CuptirError::Cupti(CuptiError(
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
        let context = ContextBuilder::new()
            .with_activity(
                activity::Builder::new()
                    .with_record_buffer_handler(move |buffer| {
                        recs_cb.lock().unwrap().extend(
                            buffer
                                .into_iter()
                                .filter_map(|maybe_record| maybe_record.ok()),
                        );
                        Ok(())
                    })
                    .with_kinds([activity::Kind::Driver]),
            )
            .build()?;

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
