//! Safe wrappers for the CUDA Profiling Tools Interface (CUPTI)
//!  
//! Cuptir provides safe wrappers around the bindings of the CUDA Profiling Tools
//! Interface (CUPTI) provided by [cudarc](https://docs.rs/cudarc/latest/cudarc/).
//!
//! These wrappers are experimental and explorative. At some point they could be
//! upstreamed to `cudarc`.
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

/// Generated enums
pub mod enums {
    include!(concat!(env!("OUT_DIR"), "/enums.rs"));
}
/// Generated function parameter structs
pub mod function_params {
    include!(concat!(env!("OUT_DIR"), "/function_params.rs"));
}

/// Context managing CUPTI.
///
/// Only one instance of this struct can exist at a time, because CUPTI can only handle
/// one subscriber.
///
/// This [Context] should be created before any other CUDA interactions take place in
/// the program. There is one exception for functionality related to the Unified Virtual
/// Memory (UVM) counters of the Activity API. Also see:
/// [Context::activity_enable_unified_memory_counters].
#[derive(Debug)]
pub struct Context {
    // While somewhat counter-intuitive, following CUPTI examples, drop this first to
    // call cupti unsubscribe, after which they flush the activity API.
    _callback: callback::Context,

    activity: Option<activity::Context>,
}

impl Context {
    /// Return a [`ContextBuilder`] which should be used to create a [`Context`].
    pub fn builder() -> ContextBuilder {
        Default::default()
    }

    /// Enable the unified memory counters of the activity API set when building the
    /// context.
    ///
    /// This can only be performed after CUDA driver initialization, hence it is not
    /// performed when using [`ContextBuilder::build`].
    ///
    /// This requires the activity context to be initialized, see
    /// [ContextBuilder::with_activity].
    pub fn activity_enable_unified_memory_counters(&mut self) -> Result<(), CuptirError> {
        if let Some(activity) = &mut self.activity {
            activity.enable_unified_memory_counters()
        } else {
            Err(CuptirError::Activity(
                "enabling unified memory counters requires initializing with an activity api context"
                    .into(),
            ))
        }
    }

    /// Request to deliver activity records.
    ///
    /// This function has no effect if the activity API is unused. To enable it, see
    /// [`ContextBuilder::with_activity`].
    ///
    /// When forced is false, only complete records are flushed. When forced is true,
    /// incomplete records may be flushed.
    pub fn activity_flush_all(&self, forced: bool) -> Result<(), CuptirError> {
        if self.activity.is_some() {
            activity::Context::flush_all(forced)
        } else {
            Ok(())
        }
    }
}

/// Builder to help initialize a [`Context`].
#[derive(Default)]
pub struct ContextBuilder {
    activity: Option<activity::Builder>,
    callback: Option<callback::Builder>,
}

impl ContextBuilder {
    pub fn new() -> Self {
        Default::default()
    }

    /// Enable the CUPTI activity API, see [`activity`].
    pub fn with_activity(mut self, builder: activity::Builder) -> Self {
        self.activity = Some(builder);
        self
    }

    /// Enable the CUPTI callback API, see [`callback`].
    pub fn with_callback(mut self, builder: callback::Builder) -> Self {
        self.callback = Some(builder);
        self
    }

    /// Build the [`Context`].
    ///
    /// This function will fail if there is another [`Context`] or any other type of
    /// CUPTI subscriber.
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

    type TestResult = std::result::Result<(), Box<dyn std::error::Error>>;

    #[test]
    #[serial]
    fn make_context() -> TestResult {
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
}
