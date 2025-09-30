use cudarc::cupti::sys;

use crate::{
    error::CuptirError,
    utils::{try_demangle_from_ffi, try_str_from_ffi},
};

pub type Kind = crate::enums::ActivityMemoryKind;
pub type OperationType = crate::enums::ActivityMemoryOperationType;

/// Memory activity record
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
pub struct Record {
    pub memory_operation_type: OperationType,
    pub memory_kind: Kind,
    pub correlation_id: u32,
    pub address: u64,
    pub bytes: u64,
    pub timestamp: u64,
    pub pc: u64,
    pub process_id: super::ProcessId,
    pub device_id: super::DeviceId,
    pub context_id: super::ContextId,
    pub stream_id: super::StreamId,
    pub name: Option<String>,
    pub is_async: u32,
    // TODO: this union:
    // pub memory_pool_config: sys::CUpti_ActivityMemory4__bindgen_ty_1,
    pub source: Option<String>,
}

impl TryFrom<&sys::CUpti_ActivityMemory4> for Record {
    type Error = CuptirError;

    fn try_from(value: &sys::CUpti_ActivityMemory4) -> Result<Self, Self::Error> {
        Ok(Self {
            memory_operation_type: value.memoryOperationType.try_into()?,
            memory_kind: value.memoryKind.try_into()?,
            correlation_id: value.correlationId,
            address: value.address,
            bytes: value.bytes,
            timestamp: value.timestamp,
            pc: value.PC,
            process_id: value.processId,
            device_id: value.deviceId,
            context_id: value.contextId,
            stream_id: value.streamId,
            name: unsafe { try_demangle_from_ffi(value.name) },
            is_async: value.isAsync,
            source: unsafe { try_str_from_ffi(value.source) }.map(ToOwned::to_owned),
        })
    }
}
