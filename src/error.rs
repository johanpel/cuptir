use thiserror::Error;

#[derive(Error, Debug, PartialEq, Eq)]
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
    #[error("acivity record handler: {0}")]
    AcivityRecordHandler(String),
    #[error("demangle: {0}")]
    Demangle(#[from] cpp_demangle::error::Error),
}
