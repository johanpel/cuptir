//! Memory pool activity record support

use cudarc::cupti::sys;

use crate::error::CuptirError;

pub type OperationType = crate::enums::ActivityMemoryPoolOperationType;
pub type PoolType = crate::enums::ActivityMemoryPoolType;

/// Memory pool activity record.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
pub struct Record {
    /// The memory pool operation requested by the user.
    pub memory_pool_operation_type: OperationType,
    /// The type of the memory pool.
    pub memory_pool_type: PoolType,
    /// The correlation ID of the memory pool operation.
    pub correlation_id: super::CorrelationId,
    /// The ID of the process to which this record belongs to.
    pub process_id: super::ProcessId,
    /// The ID of the device where the memory pool is created.
    pub device_id: super::DeviceId,
    /// The minimum bytes to keep of the memory pool. Only valid for trims.
    pub min_bytes_to_keep: Option<usize>,
    /// The virtual address of the allocation.
    pub address: u64,
    /// The size of the memory pool operation in bytes.
    pub size: Option<u64>,
    /// The release threshold of the memory pool.
    pub release_threshold: Option<u64>,
    /// The start timestamp for the memory operation, in ns.
    pub timestamp: super::Timestamp,
    /// The utilized size of the memory pool.
    pub utilized_size: Option<u64>,
}

impl TryFrom<&sys::CUpti_ActivityMemoryPool2> for Record {
    type Error = CuptirError;

    fn try_from(value: &sys::CUpti_ActivityMemoryPool2) -> Result<Self, Self::Error> {
        let op = value.memoryPoolOperationType.try_into()?;
        let pool_type = value.memoryPoolType.try_into()?;
        Ok(Self {
            memory_pool_operation_type: op,
            memory_pool_type: pool_type,
            correlation_id: value.correlationId,
            process_id: value.processId,
            device_id: value.deviceId,
            min_bytes_to_keep: matches!(op, OperationType::Trimmed).then_some(value.minBytesToKeep),
            address: value.address,
            size: matches!(pool_type, PoolType::Local).then_some(value.size),
            release_threshold: matches!(pool_type, PoolType::Local)
                .then_some(value.releaseThreshold),
            timestamp: value.timestamp,
            utilized_size: matches!(pool_type, PoolType::Local).then_some(value.utilizedSize),
        })
    }
}
