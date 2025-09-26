//! Error types used by this crate.

use thiserror::Error;

/// Cuptir error kinds
#[derive(Clone, Error, Debug, PartialEq, Eq)]
pub enum CuptirError {
    /// An error from the underlying cudarc::cupti module.
    #[error("cupti error {0}")]
    Cupti(#[from] cudarc::cupti::result::CuptiError),
    /// A nullptr check failed.
    #[error("unexpected nullptr")]
    NullPointer,
    /// A value returned through the FFI was corrupted, typically an invalid enum value.
    #[error("<corrupted>")]
    Corrupted,
    /// A feature is not supported in this version.
    #[error("not implemented")]
    NotImplemented,
    /// A value returned through the FFI was a sentinel value.
    #[error("unexpected sentinel enum variant: {0}")]
    SentinelEnum(u32),
    /// An error related to the record buffer handler of the Activity API.
    #[error("activity record buffer handler: {0}")]
    ActivityRecordBufferHandler(String),
    /// An error related to the Activity API.
    #[error("activity error: {0}")]
    Activity(String),
    /// An error related to the callback handler of the Callback API.
    #[error("callback handler: {0}")]
    CallbackHandler(String),
    /// An error during symbol demangling.
    #[error("demangle: {0}")]
    Demangle(#[from] cpp_demangle::error::Error),
    /// An error attempting to build some context. Typically a misconfiguration issue.
    #[error("builder: {0}")]
    Builder(String),
    /// A certain pattern is not supported by CUPTI.
    ///
    /// For example, obtaining function parameters from callbacks for specific functions
    /// may not be possible when CUPTI doesn't have definitions for that function.
    #[error("not supported by CUPTI")]
    CuptiNotSupported,
}
