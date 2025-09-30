//! Types related Unified Virtual Memory (UVM) a.k.a. Unified Memory

use cudarc::cupti::sys;

use crate::error::CuptirError;

pub type CounterScope = crate::enums::ActivityUnifiedMemoryCounterScope;
pub type CounterKind = crate::enums::ActivityUnifiedMemoryCounterKind;
pub type AccessType = crate::enums::ActivityUnifiedMemoryAccessType;
pub type MigrationCause = crate::enums::ActivityUnifiedMemoryMigrationCause;

/// Configuration for enabling UVM counter activity records
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct CounterConfig {
    pub scope: CounterScope,
    pub kind: CounterKind,
    pub device_id: u32,
    pub enable: bool,
}

impl From<CounterConfig> for sys::CUpti_ActivityUnifiedMemoryCounterConfig {
    fn from(value: CounterConfig) -> Self {
        sys::CUpti_ActivityUnifiedMemoryCounterConfig {
            scope: value.scope.into(),
            kind: value.kind.into(),
            deviceId: value.device_id,
            enable: if value.enable { 1 } else { 0 },
        }
    }
}

/// Data of UVM counter records representing transfers
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
pub struct BytesTransfer {
    pub memory_region_bytes: u64,
    pub start: super::Timestamp,
    pub end: super::Timestamp,
    pub virtual_base_address: u64,
    pub source_device_id: super::DeviceId,
    pub destination_device_id: super::DeviceId,
    pub stream_id: super::StreamId,
    pub process_id: super::ProcessId,
    pub migration_cause: MigrationCause,
}

impl BytesTransfer {
    /// Construct a new BytesTransfer struct assuming the provided record holds
    /// valid data for a bytes transfer kind of record.
    fn try_from_record_unchecked(
        rec: &sys::CUpti_ActivityUnifiedMemoryCounter3,
    ) -> Result<Self, CuptirError> {
        Ok(Self {
            memory_region_bytes: rec.value,
            start: rec.start,
            end: rec.end,
            virtual_base_address: rec.address,
            source_device_id: rec.srcId,
            destination_device_id: rec.dstId,
            stream_id: rec.streamId,
            process_id: rec.processId,
            migration_cause: MigrationCause::try_from(rec.flags)?,
        })
    }
}

/// A UVM counter activity record
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
pub enum CounterRecord {
    BytesTransferHtoD(BytesTransfer),
    BytesTransferDtoH(BytesTransfer),
    CpuPageFaultCount,
    GpuPageFault,
    Thrashing,
    Throttling,
    RemoteMap,
    // TODO: figure out whether [BytesTransfer] is the right choice here. Although plausible, the
    // docs don't mention anything about how to properly interpret the fields.
    BytesTransferDtoD(BytesTransfer),
    Count,
    Unknown,
}

impl TryFrom<&sys::CUpti_ActivityUnifiedMemoryCounter3> for CounterRecord {
    type Error = CuptirError;

    fn try_from(value: &sys::CUpti_ActivityUnifiedMemoryCounter3) -> Result<Self, Self::Error> {
        let kind: CounterKind = value.counterKind.try_into()?;
        Ok(match kind {
            CounterKind::Unknown => Self::Unknown,
            CounterKind::BytesTransferHtod => {
                Self::BytesTransferHtoD(BytesTransfer::try_from_record_unchecked(value)?)
            }
            CounterKind::BytesTransferDtoh => {
                Self::BytesTransferDtoH(BytesTransfer::try_from_record_unchecked(value)?)
            }
            CounterKind::BytesTransferDtod => {
                Self::BytesTransferDtoD(BytesTransfer::try_from_record_unchecked(value)?)
            }
            // TODO: the ones below:
            CounterKind::CpuPageFaultCount => Self::CpuPageFaultCount,
            CounterKind::GpuPageFault => Self::GpuPageFault,
            CounterKind::Thrashing => Self::Thrashing,
            CounterKind::Throttling => Self::Throttling,
            CounterKind::RemoteMap => Self::RemoteMap,
        })
    }
}
