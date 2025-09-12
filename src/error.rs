use thiserror::Error;

#[derive(Clone, Error, Debug, PartialEq, Eq)]
pub enum CuptirError {
    #[error("cupti error {0}")]
    Cupti(#[from] cudarc::cupti::result::CuptiError),
    #[error("unexpected nullptr")]
    NullPointer,
    #[error("<corrupted>")]
    Corrupted,
    #[error("not implemented")]
    NotImplemented,
    #[error("unexpected sentinel enum variant: {0}")]
    SentinelEnum(u32),
    #[error("acivity record buffer handler: {0}")]
    ActivityRecordBufferHandler(String),
    #[error("callback handler: {0}")]
    CallbackHandler(String),
    #[error("demangle: {0}")]
    Demangle(#[from] cpp_demangle::error::Error),
    #[error("builder: {0}")]
    Builder(String),
}
