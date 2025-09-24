//! Safe wrappers around the CUPTI Activity API
//!
//! # Notes on activity kinds.
//!
//! Please refer to [the CUPTI
//! documentation](https://docs.nvidia.com/cupti/api/group__CUPTI__ACTIVITY__API.html#_CPPv418CUpti_ActivityKind)
//! to understand which activity kinds relate to which type of record.
use std::{
    alloc::{Layout, alloc, dealloc},
    collections::HashSet,
    num::NonZero,
    sync::RwLock,
};

use cudarc::cupti::{
    result,
    sys::{self, CUpti_ActivityFlag},
};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use tracing::{trace, warn};

use crate::{
    callback::callback_name,
    driver,
    enums::{DriverFunc, RuntimeFunc},
    error::CuptirError,
    runtime,
    utils::{try_demangle_from_ffi, try_str_from_ffi},
};

pub type ChannelType = crate::enums::ChannelType;
pub type FuncShmemLimitConfig = crate::enums::FuncShmemLimitConfig;
pub type Kind = crate::enums::ActivityKind;
pub type MemcpyKind = crate::enums::ActivityMemcpyKind;
pub type MemoryKind = crate::enums::ActivityMemoryKind;
pub type MemoryOperationType = crate::enums::ActivityMemoryOperationType;
pub type MemoryPoolOperationType = crate::enums::ActivityMemoryPoolOperationType;
pub type MemoryPoolType = crate::enums::ActivityMemoryPoolType;
pub type PartitionedGlobalCacheConfig = crate::enums::ActivityPartitionedGlobalCacheConfig;

/// Default CUPTI buffer size.
///
/// CUPTI docs recommended this to be between 1 and 10 MiB.
const CUPTI_BUFFER_SIZE: usize = 8 * 1024 * 1024;
/// Default CUPTI buffer alignment.
const CUPTI_BUFFER_ALIGN: usize = 8;

/// A buffer with activity records.
#[derive(Debug, Default)]
pub struct RecordBuffer {
    /// Pointer to the raw bytes.
    ptr: *mut u8,
    /// The size of the allocation.
    size: usize,
    /// The number of valid bytes within the buffer.
    valid_size: usize,
}

impl RecordBuffer {
    /// Attempt to construct a new RecordBuffer.
    ///
    /// If the supplied `ptr` is a nullptr, this function will return an error.
    ///
    /// # Safety
    /// This resulting [RecordBuffer] takes ownership over the buffer and will
    /// deallocate it when dropped.
    fn try_new(ptr: *mut u8, size: usize, valid_size: usize) -> Result<Self, CuptirError> {
        if ptr.is_null() {
            Err(CuptirError::NullPointer)
        } else {
            Ok(Self {
                ptr,
                size,
                valid_size,
            })
        }
    }
}

impl Drop for RecordBuffer {
    fn drop(&mut self) {
        let layout = Layout::from_size_align(self.size, CUPTI_BUFFER_ALIGN).unwrap();
        unsafe {
            dealloc(self.ptr, layout);
        }
    }
}

impl IntoIterator for RecordBuffer {
    type Item = Result<Record, CuptirError>;
    type IntoIter = RecordBufferIterator;

    fn into_iter(self) -> Self::IntoIter {
        RecordBufferIterator {
            buffer: self,
            current_record_ptr: std::ptr::null_mut(),
        }
    }
}

/// An iterator over activity [Record]s in a [RecordBuffer].
pub struct RecordBufferIterator {
    buffer: RecordBuffer,
    current_record_ptr: *mut sys::CUpti_Activity,
}

impl Iterator for RecordBufferIterator {
    type Item = Result<Record, CuptirError>;

    fn next(&mut self) -> Option<Self::Item> {
        // Safety: the buffer can only be constructed with a non-null ptr, plus CUPTI
        // would return a graceful error if it were null.
        let result = unsafe {
            result::activity::get_next_record(
                self.buffer.ptr,
                self.buffer.valid_size,
                &mut self.current_record_ptr,
            )
        };
        if let Err(error) = &result {
            match error.0 {
                sys::CUptiResult::CUPTI_ERROR_MAX_LIMIT_REACHED
                | sys::CUptiResult::CUPTI_ERROR_INVALID_KIND
                | sys::CUptiResult::CUPTI_ERROR_NOT_INITIALIZED => None,
                _ => {
                    warn!("unexpected error in record buffer iterator: {error:?}");
                    None
                }
            }
        } else if self.current_record_ptr.is_null() {
            None
        } else {
            Some(unsafe { Record::try_from_record_ptr(self.current_record_ptr) })
        }
    }
}

/// Type of callback function used to handle [RecordBuffer]s.
pub type RecordBufferHandlerFn =
    dyn Fn(RecordBuffer) -> Result<(), Box<dyn std::error::Error + Send + Sync>> + Send + Sync;

/// Globally accessible callback to handle activity record buffers.
///
/// Because the buffer complete callback doesn't have a way of passing custom data, e.g.
/// a reference to the Context, this is needed to get to the Rust callback for record
/// processing. This is a RwLock because this allows dropping e.g. captured
/// [std::sync::Arc]s such that after the [Context] drops, inner values can be taken out
/// of the Arc.
pub(crate) static RECORD_BUFFER_HANDLER: RwLock<Option<Box<RecordBufferHandlerFn>>> =
    RwLock::new(None);

/// Sets the activity record handler. This can be only called once.
pub(crate) fn set_record_buffer_handler(
    activity_record_buffer_handler: Option<Box<RecordBufferHandlerFn>>,
) -> Result<(), CuptirError> {
    let mut lock = RECORD_BUFFER_HANDLER.try_write().map_err(|e| {
        CuptirError::ActivityRecordBufferHandler(format!(
            "Unable to set record buffer handler: {e}"
        ))
    })?;
    // Practically prevent multiple activity contexts from existing at the same time
    if lock.is_some() && activity_record_buffer_handler.is_some() {
        Err(CuptirError::ActivityRecordBufferHandler(
            "cannot set activity record buffer handler twice without reset".into(),
        ))
    } else {
        *lock = activity_record_buffer_handler;
        Ok(())
    }
}

/// Calls the global record handler function if it is installed.
fn handle_record_buffer(record_buffer: RecordBuffer) -> Result<(), CuptirError> {
    let lock = RECORD_BUFFER_HANDLER.read().map_err(|e| {
        CuptirError::ActivityRecordBufferHandler(format!(
            "Unable to access record buffer handler: {e}"
        ))
    })?;
    if let Some(handler) = lock.as_ref() {
        handler(record_buffer).map_err(|e| CuptirError::ActivityRecordBufferHandler(e.to_string()))
    } else {
        warn!("activity records received, but no callback is installed");
        Ok(())
    }
}

/// A time stamp in nanoseconds.
///
/// CUPTI docs are not clear about the epoch to which this timestamp is relative. It
/// might be relative to the Unix epoch.
pub type Timestamp = u64;

pub type ProcessId = u32;
pub type ThreadId = u32;
pub type CorrelationId = u32;
pub type StreamId = u32;
pub type DeviceId = u32;

/// Properties shared across of [DriverApiRecord] and [RuntimeApiRecord] activity records.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
pub struct ApiProps {
    pub start: Timestamp,
    pub end: Timestamp,
    pub process_id: ProcessId,
    pub thread_id: ThreadId,
    pub correlation_id: CorrelationId,
    pub return_value: u32,
}

impl From<&sys::CUpti_ActivityAPI> for ApiProps {
    fn from(value: &sys::CUpti_ActivityAPI) -> Self {
        Self {
            start: value.start,
            end: value.end,
            process_id: value.processId,
            thread_id: value.threadId,
            correlation_id: value.correlationId,
            return_value: value.returnValue,
        }
    }
}

/// A CUDA driver API record
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
pub struct DriverApiRecord {
    pub function: DriverFunc,
    pub props: ApiProps,
}

impl DriverApiRecord {
    pub fn function_name(&self) -> Result<String, CuptirError> {
        callback_name(crate::callback::Domain::DriverApi, self.function as u32)
    }
}

/// A CUDA runtime API record
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
pub struct RuntimeApiRecord {
    pub function: RuntimeFunc,
    pub props: ApiProps,
}

impl RuntimeApiRecord {
    pub fn function_name(&self) -> Result<String, CuptirError> {
        callback_name(crate::callback::Domain::RuntimeApi, self.function as u32)
    }
}

/// Internal launch API record
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
pub struct InternalLaunchApiRecord {
    pub props: ApiProps,
}

/// Kernel activity record
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
pub struct KernelRecord {
    // TODO: this union:
    // pub cache_config: sys::CUpti_ActivityKernel9__bindgen_ty_1,
    pub shared_memory_config: u8,
    pub registers_per_thread: u16,
    pub partitioned_global_cache_requested: PartitionedGlobalCacheConfig,
    pub partitioned_global_cache_executed: PartitionedGlobalCacheConfig,
    pub start: Timestamp,
    pub end: Timestamp,
    pub completed: Timestamp,
    pub device_id: u32,
    pub context_id: u32,
    pub stream_id: u32,
    pub grid_x: i32,
    pub grid_y: i32,
    pub grid_z: i32,
    pub block_x: i32,
    pub block_y: i32,
    pub block_z: i32,
    pub static_shared_memory: i32,
    pub dynamic_shared_memory: i32,
    pub local_memory_per_thread: u32,
    pub local_memory_total: u32,
    pub correlation_id: u32,
    pub grid_id: i64,
    pub name: Option<String>,
    pub queued: Option<Timestamp>,
    pub submitted: Option<Timestamp>,
    pub launch_type: u8,
    pub is_shared_memory_carveout_requested: u8,
    pub shared_memory_carveout_requested: u8,
    pub padding: u8,
    pub shared_memory_executed: u32,
    pub graph_node_id: u64,
    pub shmem_limit_config: FuncShmemLimitConfig,
    pub graph_id: u32,
    // TODO: this CUDA runtime API type:
    // pub p_access_policy_window: *mut CUaccessPolicyWindow,
    pub channel_id: u32,
    pub cluster_x: u32,
    pub cluster_y: u32,
    pub cluster_z: u32,
    pub cluster_scheduling_policy: u32,
    pub local_memory_total_v2: u64,
    pub max_potential_cluster_size: u32,
    pub max_active_clusters: u32,
    pub channel_type: ChannelType,
}

impl TryFrom<&sys::CUpti_ActivityKernel9> for KernelRecord {
    type Error = CuptirError;

    fn try_from(value: &sys::CUpti_ActivityKernel9) -> Result<Self, Self::Error> {
        Ok(KernelRecord {
            shared_memory_config: value.sharedMemoryConfig,
            registers_per_thread: value.registersPerThread,
            partitioned_global_cache_requested: PartitionedGlobalCacheConfig::try_from(
                value.partitionedGlobalCacheRequested,
            )?,
            partitioned_global_cache_executed: PartitionedGlobalCacheConfig::try_from(
                value.partitionedGlobalCacheExecuted,
            )?,
            start: value.start,
            end: value.end,
            completed: value.completed,
            device_id: value.deviceId,
            context_id: value.contextId,
            stream_id: value.streamId,
            grid_x: value.gridX,
            grid_y: value.gridY,
            grid_z: value.gridZ,
            block_x: value.blockX,
            block_y: value.blockY,
            block_z: value.blockZ,
            static_shared_memory: value.staticSharedMemory,
            dynamic_shared_memory: value.dynamicSharedMemory,
            local_memory_per_thread: value.localMemoryPerThread,
            local_memory_total: value.localMemoryTotal,
            correlation_id: value.correlationId,
            grid_id: value.gridId,
            name: unsafe { try_demangle_from_ffi(value.name) },
            queued: if value.queued == sys::CUPTI_TIMESTAMP_UNKNOWN as u64 {
                None
            } else {
                Some(value.queued)
            },
            submitted: if value.submitted == sys::CUPTI_TIMESTAMP_UNKNOWN as u64 {
                None
            } else {
                Some(value.submitted)
            },
            launch_type: value.launchType,
            is_shared_memory_carveout_requested: value.isSharedMemoryCarveoutRequested,
            shared_memory_carveout_requested: value.sharedMemoryCarveoutRequested,
            padding: value.padding,
            shared_memory_executed: value.sharedMemoryExecuted,
            graph_node_id: value.graphNodeId,
            shmem_limit_config: value.shmemLimitConfig.try_into()?,
            graph_id: value.graphId,
            channel_id: value.channelID,
            cluster_x: value.clusterX,
            cluster_y: value.clusterY,
            cluster_z: value.clusterZ,
            cluster_scheduling_policy: value.clusterSchedulingPolicy,
            local_memory_total_v2: value.localMemoryTotal_v2,
            max_potential_cluster_size: value.maxPotentialClusterSize,
            max_active_clusters: value.maxActiveClusters,
            channel_type: value.channelType.try_into()?,
        })
    }
}

/// Memcpy activity record
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
pub struct MemcpyRecord {
    pub copy_kind: MemcpyKind,
    pub src_kind: MemoryKind,
    pub dst_kind: MemoryKind,
    pub is_async: bool,
    pub bytes: u64,
    pub start: Timestamp,
    pub end: Timestamp,
    pub device_id: DeviceId,
    pub context_id: u32,
    pub stream_id: StreamId,
    pub correlation_id: u32,
    pub runtime_correlation_id: u32,
    pub graph_node_id: u64,
    pub graph_id: u32,
    pub channel_id: u32,
    pub copy_count: u64,
    pub channel_type: ChannelType,
}

impl TryFrom<&sys::CUpti_ActivityMemcpy6> for MemcpyRecord {
    type Error = CuptirError;

    fn try_from(value: &sys::CUpti_ActivityMemcpy6) -> Result<Self, Self::Error> {
        Ok(Self {
            copy_kind: MemcpyKind::try_from(value.copyKind as u32)?,
            src_kind: MemoryKind::try_from(value.srcKind as u32)?,
            dst_kind: MemoryKind::try_from(value.dstKind as u32)?,
            is_async: value.flags
                & CUpti_ActivityFlag::CUPTI_ACTIVITY_FLAG_MEMCPY_ASYNC as u32 as u8
                != 0,
            bytes: value.bytes,
            start: value.start,
            end: value.end,
            device_id: value.deviceId,
            context_id: value.contextId,
            stream_id: value.streamId,
            correlation_id: value.correlationId,
            runtime_correlation_id: value.runtimeCorrelationId,
            graph_node_id: value.graphNodeId,
            graph_id: value.graphId,
            channel_id: value.channelID,
            channel_type: value.channelType.try_into()?,
            copy_count: value.copyCount,
        })
    }
}

/// Memory activity record
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
pub struct MemoryRecord {
    pub memory_operation_type: MemoryOperationType,
    pub memory_kind: MemoryKind,
    pub correlation_id: u32,
    pub address: u64,
    pub bytes: u64,
    pub timestamp: u64,
    pub pc: u64,
    pub process_id: u32,
    pub device_id: u32,
    pub context_id: u32,
    pub stream_id: u32,
    pub name: Option<String>,
    pub is_async: u32,
    // TODO: this union:
    // pub memory_pool_config: sys::CUpti_ActivityMemory4__bindgen_ty_1,
    pub source: Option<String>,
}

impl TryFrom<&sys::CUpti_ActivityMemory4> for MemoryRecord {
    type Error = CuptirError;

    fn try_from(value: &sys::CUpti_ActivityMemory4) -> Result<Self, Self::Error> {
        Ok(MemoryRecord {
            memory_operation_type: value.memoryOperationType.try_into()?,
            memory_kind: MemoryKind::try_from(value.memoryKind)?,
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

/// Memory pool activity record.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
pub struct MemoryPoolRecord {
    pub memory_pool_operation_type: MemoryPoolOperationType,
    pub memory_pool_type: MemoryPoolType,
    pub correlation_id: u32,
    pub process_id: u32,
    pub device_id: u32,
    pub min_bytes_to_keep: usize,
    pub address: u64,
    pub size: u64,
    pub release_threshold: u64,
    pub timestamp: u64,
    pub utilized_size: u64,
}

impl TryFrom<&sys::CUpti_ActivityMemoryPool2> for MemoryPoolRecord {
    type Error = CuptirError;

    fn try_from(value: &sys::CUpti_ActivityMemoryPool2) -> Result<Self, Self::Error> {
        Ok(MemoryPoolRecord {
            memory_pool_operation_type: value.memoryPoolOperationType.try_into()?,
            memory_pool_type: value.memoryPoolType.try_into()?,
            correlation_id: value.correlationId,
            process_id: value.processId,
            device_id: value.deviceId,
            min_bytes_to_keep: value.minBytesToKeep,
            address: value.address,
            size: value.size,
            release_threshold: value.releaseThreshold,
            timestamp: value.timestamp,
            utilized_size: value.utilizedSize,
        })
    }
}

/// Types related Unified Virtual Memory (UVM) a.k.a. Unified Memory
pub mod uvm {
    use super::*;

    pub type CounterScope = crate::enums::ActivityUnifiedMemoryCounterScope;
    pub type CounterKind = crate::enums::ActivityUnifiedMemoryCounterKind;
    pub type AccessType = crate::enums::ActivityUnifiedMemoryAccessType;
    pub type MigrationCause = crate::enums::ActivityUnifiedMemoryMigrationCause;

    /// Configuration for enabling UVM counter activity records
    #[derive(Clone, Debug, PartialEq, Eq, Hash)]
    pub struct CounterConfig {
        pub scope: uvm::CounterScope,
        pub kind: uvm::CounterKind,
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
        pub start: Timestamp,
        pub end: Timestamp,
        pub virtual_base_address: u64,
        pub source_device_id: DeviceId,
        pub destination_device_id: DeviceId,
        pub stream_id: StreamId,
        pub process_id: ProcessId,
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
        // TODO: figure out whether [BytesTransfer] is the right choice here. Although
        // plausible, the docs don't mention anything about how to properly interpret
        // the fields.
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
}

/// An activity record.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
pub enum Record {
    DriverApi(DriverApiRecord),
    RuntimeApi(RuntimeApiRecord),
    InternalLaunchApi(InternalLaunchApiRecord),
    Kernel(KernelRecord),
    Memcpy(MemcpyRecord),
    Memory(MemoryRecord),
    MemoryPool(MemoryPoolRecord),
    UnifiedMemoryCounter(uvm::CounterRecord),
}

impl Record {
    unsafe fn try_from_record_ptr(
        record_ptr: *mut sys::CUpti_Activity,
    ) -> Result<Self, CuptirError> {
        if record_ptr.is_null() {
            return Err(CuptirError::NullPointer);
        }

        // Safety: null check is done at the start of this function, so record dereferences
        // should be safe.
        let kind = unsafe { *record_ptr }.kind;
        match kind {
            sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_DRIVER => {
                let api_record = unsafe { &*(record_ptr as *const sys::CUpti_ActivityAPI) };
                Ok(Record::DriverApi(DriverApiRecord {
                    function: DriverFunc::try_from(api_record.cbid)?,
                    props: api_record.into(),
                }))
            }
            sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_RUNTIME => {
                let api_record = unsafe { &*(record_ptr as *const sys::CUpti_ActivityAPI) };
                Ok(Record::RuntimeApi(RuntimeApiRecord {
                    function: RuntimeFunc::try_from(api_record.cbid)?,
                    props: api_record.into(),
                }))
            }
            sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_INTERNAL_LAUNCH_API => {
                let api_record = unsafe { &*(record_ptr as *const sys::CUpti_ActivityAPI) };
                Ok(Record::InternalLaunchApi(InternalLaunchApiRecord {
                    props: api_record.into(),
                }))
            }
            sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL => {
                let value = unsafe { &*(record_ptr as *const sys::CUpti_ActivityKernel9) };
                Ok(Record::Kernel(value.try_into()?))
            }
            sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_MEMCPY => {
                let memcpy_record = unsafe { &*(record_ptr as *const sys::CUpti_ActivityMemcpy6) };
                Ok(Record::Memcpy(memcpy_record.try_into()?))
            }
            sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_MEMORY2 => {
                let value = unsafe { &*(record_ptr as *const sys::CUpti_ActivityMemory4) };
                Ok(Record::Memory(value.try_into()?))
            }
            sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_MEMORY_POOL => {
                let memory_pool_record =
                    unsafe { &*(record_ptr as *const sys::CUpti_ActivityMemoryPool2) };
                Ok(Record::MemoryPool(memory_pool_record.try_into()?))
            }
            sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_UNIFIED_MEMORY_COUNTER => {
                let unified_memory_counter_record =
                    unsafe { &*(record_ptr as *const sys::CUpti_ActivityUnifiedMemoryCounter3) };
                Ok(Record::UnifiedMemoryCounter(
                    unified_memory_counter_record.try_into()?,
                ))
            }
            _ => {
                trace!("unimplemented activity kind: {kind:?}");
                Err(CuptirError::NotImplemented)
            }
        }
    }
}

/// Callback CUPTI uses to request a new activity record buffer.
#[unsafe(no_mangle)]
pub(crate) extern "C" fn buffer_requested_callback(
    buffer: *mut *mut u8,
    size: *mut usize,
    max_num_records: *mut usize,
) {
    trace!("buffer requested");
    let layout = Layout::from_size_align(CUPTI_BUFFER_SIZE, CUPTI_BUFFER_ALIGN).unwrap();
    unsafe {
        // Safety: ownership of the memory allocated here is transferred to the
        // RecordBuffer constructed in [buffer_complete_callback], and freed when the
        // RecordBuffer is dropped.
        let ptr = alloc(layout);
        *buffer = ptr;
        *size = CUPTI_BUFFER_SIZE;
        *max_num_records = 0; // means: fill this with as many records as possible
    }
}

/// Callback CUPTI uses to flush an activity record buffer.
#[unsafe(no_mangle)]
pub(crate) extern "C" fn buffer_complete_callback(
    _context: cudarc::driver::sys::CUcontext,
    stream_id: u32,
    buffer: *mut u8,
    size: usize,
    valid_size: usize,
) {
    trace!("buffer complete - stream id: {stream_id}, size: {size}, valid: {valid_size}");
    // Safety: RecordBuffer::try_new will fail only if buffer is a nullptr, in which
    // case somehow no memory was allocated for the buffer that could be freed.
    if let Err(error) =
        RecordBuffer::try_new(buffer, size, valid_size).and_then(handle_record_buffer)
    {
        warn!("processing activity buffer failed: {error}");
    }
}

/// Context for the CUPTI Activity API
#[derive(Debug, Default)]
pub(crate) struct Context {
    enabled_kinds: Vec<Kind>,
    enabled_driver_functions: Vec<driver::Function>,
    enabled_runtime_functions: Vec<runtime::Function>,

    unified_memory_counter_configs: Vec<sys::CUpti_ActivityUnifiedMemoryCounterConfig>,
}

impl Context {
    pub(crate) fn flush_all(forced: bool) -> Result<(), CuptirError> {
        trace!("flushing activity buffer");
        result::activity::flush_all(if forced {
            sys::CUpti_ActivityFlag::CUPTI_ACTIVITY_FLAG_FLUSH_FORCED as u32
        } else {
            0
        })?;
        Ok(())
    }

    pub(crate) fn enable_unified_memory_counters(&self) -> Result<(), CuptirError> {
        if !self.unified_memory_counter_configs.is_empty() {
            unsafe {
                sys::cuptiActivityConfigureUnifiedMemoryCounter(
                    self.unified_memory_counter_configs.as_ptr() as *mut _,
                    self.unified_memory_counter_configs.len() as u32,
                )
            }
            .result()?;
            Ok(result::activity::enable(Kind::UnifiedMemoryCounter.into())?)
        } else {
            Err(CuptirError::Activity("enabling unified memory counters requires enabling the activity kind and supplying counter configurations".into()))
        }
    }
}

/// Builder to help configure the CUPTI Activity API
#[derive(Default)]
pub struct Builder {
    enabled_kinds: HashSet<Kind>,

    enabled_driver_functions: HashSet<driver::Function>,
    disabled_driver_functions: HashSet<driver::Function>,

    enabled_runtime_functions: HashSet<runtime::Function>,
    disabled_runtime_functions: HashSet<runtime::Function>,

    record_buffer_handler: Option<Box<RecordBufferHandlerFn>>,

    latency_timestamps: bool,
    allocation_source: bool,
    disable_all_sync_records: bool,

    unified_memory_counter_configs: HashSet<uvm::CounterConfig>,

    flush_period: Option<NonZero<u32>>,
}

impl Builder {
    pub fn new() -> Self {
        Default::default()
    }

    /// Add the supplied activity kinds to the set of activated activity kinds.
    pub fn with_kinds(mut self, kinds: impl IntoIterator<Item = Kind>) -> Self {
        self.enabled_kinds.extend(kinds);
        self
    }

    /// Enable the collection of activity records for the supplied driver API functions.
    pub fn with_driver_functions(
        mut self,
        functions: impl IntoIterator<Item = driver::Function>,
    ) -> Self {
        self.enabled_driver_functions.extend(functions);
        self
    }

    /// Disable the collection of activity records for the supplied driver API functions.
    pub fn without_driver_functions(
        mut self,
        functions: impl IntoIterator<Item = driver::Function>,
    ) -> Self {
        self.disabled_driver_functions.extend(functions);
        self
    }

    /// Enable the collection of activity records for the supplied driver API functions.
    pub fn with_runtime_functions(
        mut self,
        functions: impl IntoIterator<Item = runtime::Function>,
    ) -> Self {
        self.enabled_runtime_functions.extend(functions);
        self
    }

    /// Disable the collection of activity records for the supplied driver API functions.
    pub fn without_runtime_functions(
        mut self,
        functions: impl IntoIterator<Item = runtime::Function>,
    ) -> Self {
        self.enabled_runtime_functions.extend(functions);
        self
    }

    /// Set the activity record buffer handler function.
    ///
    /// The handler function should return as quickly as possible to minimize profiling
    /// overhead.
    pub fn with_record_buffer_handler<F>(mut self, handler: F) -> Self
    where
        F: Fn(RecordBuffer) -> Result<(), Box<dyn std::error::Error + Send + Sync>>
            + Send
            + Sync
            + 'static,
    {
        self.record_buffer_handler = Some(Box::new(handler));
        self
    }

    /// Set whether latency timestamps should be enabled, which deliver the `queued` and
    /// `submitted` fields for [KernelRecord] event records.
    ///
    /// Disabled by default.
    pub fn latency_timestamps(mut self, enabled: bool) -> Self {
        self.latency_timestamps = enabled;
        self
    }

    /// Set whether to enables tracking the source library for memory allocation
    /// requests.
    ///
    /// Disabled by default.
    pub fn allocation_source(mut self, enabled: bool) -> Self {
        self.allocation_source = enabled;
        self
    }

    /// Set whether to enable collecting records for all synchronization operations.
    ///
    /// Enabled by default.
    pub fn all_sync_records(mut self, enabled: bool) -> Self {
        self.disable_all_sync_records = !enabled;
        self
    }

    /// Set the unified memory counter configurations.
    pub fn with_unified_memory_counter_configs(
        mut self,
        configs: impl IntoIterator<Item = uvm::CounterConfig>,
    ) -> Self {
        self.unified_memory_counter_configs.extend(configs);
        self
    }

    /// Set the flush period in milliseconds for the underlying CUPTI worker thread.
    ///  
    /// If interval is None, use CUPTI's internal heuristics to determine when to flush,
    /// which is the default mode of operation.
    pub fn flush_period(mut self, milliseconds: Option<NonZero<u32>>) -> Self {
        self.flush_period = milliseconds;
        self
    }

    fn toggle_activity<T, F>(
        which: &str,
        items: &[T],
        enable_func: F,
        enable: bool,
    ) -> Result<(), CuptirError>
    where
        u32: From<T>,
        T: Copy + std::fmt::Debug,
        F: Fn(u32, u8) -> Result<(), result::CuptiError>,
    {
        if !items.is_empty() {
            trace!(
                "{} activity record collection for {} functions: {:?}",
                if enable { "enabling" } else { "disabling" },
                which,
                items
            );
            items.iter().try_for_each(|cupti_func| {
                enable_func(u32::from(*cupti_func), if enable { 1 } else { 0 })
            })?;
        }
        Ok(())
    }

    pub(crate) fn build(self) -> Result<Option<Context>, CuptirError> {
        let anything_enabled = !(self.enabled_kinds.is_empty()
            && self.enabled_driver_functions.is_empty()
            && self.enabled_runtime_functions.is_empty()
            && self.unified_memory_counter_configs.is_empty());

        if anything_enabled {
            if let Some(record_buffer_handler) = self.record_buffer_handler {
                trace!("registering activity buffer callbacks");
                set_record_buffer_handler(Some(record_buffer_handler))?;
                result::activity::register_callbacks(
                    Some(buffer_requested_callback),
                    Some(buffer_complete_callback),
                )?;
            } else {
                return Err(CuptirError::Builder(
                    "collection of activity kinds or functions was enabled but no record buffer handler was set".into(),
                ));
            }

            if self.latency_timestamps {
                trace!("enabling latency timestamps");
                result::activity::enable_latency_timestamps(1)?;
            }
            if self.allocation_source {
                trace!("enabling allocation source tracking");
                result::activity::enable_allocation_source(1)?;
            }
            if self.disable_all_sync_records {
                trace!("disabling all sync records collection");
                result::activity::enable_all_sync_records(0)?;
            } else {
                trace!("enable all sync records collection");
                result::activity::enable_all_sync_records(0)?;
            }

            if let Some(interval) = self.flush_period {
                trace!("setting activity buffer flush period to {interval}");
                result::activity::flush_period(interval.into())?;
            }

            let enabled_kinds: Vec<_> = self.enabled_kinds.into_iter().collect();

            if !enabled_kinds.is_empty() {
                trace!(
                    "enabling activity record collection for kinds: {:?}",
                    enabled_kinds
                );
                enabled_kinds.iter().try_for_each(|kind| {
                    // CUPTI_ACTIVITY_KIND_UNIFIED_MEMORY_COUNTER can only be enabled
                    // after driver initialization, otherwise enabling it here results
                    // in a CUPTI_ERROR_NOT_READY. Error out here to inform the client
                    // code about what they should do instead.
                    //
                    // TODO: We may be able to do this automatically for the
                    // client based on callbacks in the future.
                    if *kind == Kind::UnifiedMemoryCounter {
                        Err(CuptirError::Builder("enabling unified memory counter activity records must be performed explicitly after CUDA driver initialization through the crate-level Context".into()))
                    } else {
                        Ok(result::activity::enable((*kind).into())?)
                    }
                })?;
            }

            let enabled_driver_functions = self
                .enabled_driver_functions
                .into_iter()
                .collect::<Vec<_>>();
            let enabled_runtime_functions = self
                .enabled_runtime_functions
                .into_iter()
                .collect::<Vec<_>>();

            Self::toggle_activity(
                "driver",
                enabled_driver_functions.as_slice(),
                result::activity::enable_driver_api,
                true,
            )?;
            Self::toggle_activity(
                "driver",
                self.disabled_driver_functions
                    .into_iter()
                    .collect::<Vec<_>>()
                    .as_slice(),
                result::activity::enable_driver_api,
                false,
            )?;
            Self::toggle_activity(
                "runtime",
                enabled_runtime_functions.as_slice(),
                result::activity::enable_runtime_api,
                true,
            )?;
            Self::toggle_activity(
                "runtime",
                self.disabled_runtime_functions
                    .into_iter()
                    .collect::<Vec<_>>()
                    .as_slice(),
                result::activity::enable_runtime_api,
                false,
            )?;

            Ok(Some(Context {
                enabled_kinds,
                enabled_driver_functions,
                enabled_runtime_functions,
                unified_memory_counter_configs: self
                    .unified_memory_counter_configs
                    .into_iter()
                    .map(Into::into)
                    .collect(),
            }))
        } else if self.record_buffer_handler.is_some() {
            Err(CuptirError::Builder("An activity record buffer handler is installed but no activity kind or driver/runtime API functions activity is enabled".into()))
        } else {
            Ok(None)
        }
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        if let Err(error) = Self::flush_all(true) {
            warn!("unable to flush activity buffer: {error}")
        }

        if !self.enabled_kinds.is_empty() {
            trace!(
                "disabling activity record collection for kinds: {:?}",
                self.enabled_kinds
            );
            if let Err(error) = self
                .enabled_kinds
                .iter()
                .try_for_each(|activity_kind| result::activity::disable((*activity_kind).into()))
            {
                warn!("unable to disable activity kind: {error}");
            }
        }

        if !self.enabled_driver_functions.is_empty() {
            trace!(
                "disabling activity record collection for functions: {:?}",
                self.enabled_driver_functions
            );
            if let Err(error) = self
                .enabled_driver_functions
                .iter()
                .try_for_each(|func| result::activity::enable_driver_api(func.into(), 0))
            {
                warn!("unable to disable activity for driver function: {error}");
            }
        }

        if !self.enabled_runtime_functions.is_empty() {
            trace!(
                "disabling activity record collection for runtime functions: {:?}",
                self.enabled_runtime_functions
            );
            if let Err(error) = self
                .enabled_runtime_functions
                .iter()
                .try_for_each(|func| result::activity::enable_runtime_api(func.into(), 0))
            {
                warn!("unable to disable activity for runtime function: {error}");
            }
        }

        // Unset the record buffer handler function. We would ideally tell CUPTI to stop
        // using the buffer completion callbacks by resetting them to nullptrs or
        // something, or through some explicit API for it, but that does not exist, so
        // we will handle more records coming in somehow in [handle_record_buffer].
        trace!("resetting activity record buffer handler");
        if let Err(e) = set_record_buffer_handler(None) {
            warn!("unable to reset activity record buffer handler: {e}")
        }
    }
}

#[cfg(test)]
pub(crate) mod test {
    use std::sync::{Arc, Mutex};

    use cuptir_example_utils::run_a_kernel;
    use serial_test::serial;

    use crate::enums::{DriverFunc, RuntimeFunc};

    use super::*;

    type TestResult = Result<(), Box<dyn std::error::Error>>;

    pub(crate) fn get_records<F>(
        builder: Builder,
        func: F,
    ) -> Result<Vec<Record>, Box<dyn std::error::Error>>
    where
        F: Fn(&mut Context) -> TestResult,
    {
        let records: Arc<Mutex<Vec<Record>>> = Arc::new(Mutex::new(vec![]));
        let records_cb = Arc::clone(&records);

        let callback = crate::callback::Builder::new().build();
        let mut activity = builder
            .with_record_buffer_handler(move |buffer| {
                records_cb.lock().unwrap().extend(
                    buffer
                        .into_iter()
                        .filter_map(|maybe_record| maybe_record.ok()),
                );
                Ok(())
            })
            .build()?
            .unwrap();

        func(&mut activity)?;

        drop(activity);
        drop(callback);

        Ok(Arc::into_inner(records).unwrap().into_inner()?)
    }

    #[test]
    fn record_buffer() {
        // A nullptr is rejected
        assert!(RecordBuffer::try_new(std::ptr::null_mut(), 0, 0).is_err());

        // Iterator returns none if valid size is 0.
        const ELEMENTS: usize = 128;
        let buffer = RecordBuffer::try_new(
            Box::into_raw(Box::new([0u32; ELEMENTS])) as *mut u8,
            ELEMENTS * std::mem::size_of::<u32>(),
            0,
        )
        .unwrap();
        let mut iter = buffer.into_iter();
        assert!(iter.next().is_none());
    }

    #[test]
    #[serial]
    fn builder() -> TestResult {
        // Building the default results in no activity context.
        assert!(Builder::new().build().unwrap().is_none());
        // Can't build with some kinds enabled but without handler
        assert!(
            Builder::new()
                .with_kinds([Kind::ConcurrentKernel])
                .build()
                .is_err()
        );
        // Can't build with a handler but without any kind or function enabled.
        assert!(
            Builder::new()
                .with_record_buffer_handler(|_| Ok(()))
                .build()
                .is_err()
        );
        // Can build with a kind enabled and a handler.
        Builder::new()
            .with_kinds([Kind::ConcurrentKernel])
            .with_record_buffer_handler(|_| Ok(()))
            .build()?;
        // Can build with a driver function enabled and a handler.
        Builder::new()
            .with_driver_functions([DriverFunc::cuMemcpy])
            .with_record_buffer_handler(|_| Ok(()))
            .build()?;
        // Can build with a kind enabled and a runtime function.
        Builder::new()
            .with_runtime_functions([RuntimeFunc::cudaMemcpy_v3020])
            .with_record_buffer_handler(|_| Ok(()))
            .build()?;

        Ok(())
    }

    #[test]
    #[serial]
    fn make_multiple_contexts_fails() -> TestResult {
        let _a = Builder::new()
            .with_kinds([Kind::ConcurrentKernel])
            .with_record_buffer_handler(|_| Ok(()))
            .build()?;

        let _b = Builder::new()
            .with_kinds([Kind::ConcurrentKernel])
            .with_record_buffer_handler(|_| Ok(()))
            .build();
        match _b.unwrap_err() {
            CuptirError::ActivityRecordBufferHandler(_) => (),
            _ => panic!("unexpected error"),
        };
        Ok(())
    }

    #[test]
    #[serial]
    fn driver_and_runtime_kinds() -> TestResult {
        let recs = get_records(
            Builder::new().with_kinds([Kind::Driver, Kind::Runtime]),
            |_| {
                // Do a runtime thing. This causes a large amount driver records to be
                // generated probably as part of lazy runtime initialization.
                let ptr = unsafe { cudarc::runtime::result::malloc_sync(1) }?;

                // Do a driver thing.
                unsafe { cudarc::driver::result::free_sync(ptr as u64) }?;
                Ok(())
            },
        )?;

        dbg!(&recs);

        let mut num_cuda_malloc = 0;
        let mut num_cu_mem_free = 0;

        for rec in recs.iter() {
            match rec {
                Record::RuntimeApi(r) => {
                    if r.function == RuntimeFunc::cudaMalloc_v3020 {
                        assert_eq!(r.function_name().unwrap(), "cudaMalloc_v3020");
                        num_cuda_malloc += 1;
                    }
                }
                Record::DriverApi(d) => {
                    if d.function == DriverFunc::cuMemFree_v2 {
                        assert_eq!(d.function_name().unwrap(), "cuMemFree_v2");
                        num_cu_mem_free += 1;
                    }
                }
                // Don't care about other records for now :tm:
                _ => (),
            }
        }

        assert_eq!(num_cuda_malloc, 1);
        assert_eq!(num_cu_mem_free, 1);

        Ok(())
    }

    #[test]
    #[serial]
    fn kernel() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let recs = get_records(
            Builder::new()
                .with_kinds([Kind::ConcurrentKernel])
                .latency_timestamps(true),
            |_| Ok(run_a_kernel()?),
        )?;

        assert_eq!(recs.len(), 1);
        match &recs[0] {
            Record::Kernel(rec) => {
                assert!(rec.name.as_ref().is_some_and(|name| name.contains("sin")));
            }
            _ => panic!("unexpected record kind"),
        }
        Ok(())
    }

    #[test]
    #[serial]
    fn mem_record_allocate_and_copy() -> TestResult {
        const SIZE: u64 = 1337;

        let recs = get_records(
            Builder::new().with_kinds([Kind::Memcpy, Kind::Memory2]),
            |_| {
                // Round-trip some bytes.
                let context = cudarc::driver::CudaContext::new(0)?;
                let stream = context.default_stream();
                let host_buffer_a = vec![42u8; SIZE as usize];
                let mut host_buffer_b = vec![0u8; SIZE as usize];
                // Record 1: device allocation
                let mut device_buffer = unsafe { stream.alloc::<u8>(SIZE as usize) }?;
                // Record 2: memcpy
                stream.memcpy_htod(&host_buffer_a, &mut device_buffer)?;
                // Record 3: memcpy
                stream.memcpy_dtoh(&device_buffer, &mut host_buffer_b)?;
                // Record 4: device free
                drop(device_buffer);
                stream.synchronize()?;
                // Sanity check
                assert!(host_buffer_b.into_iter().all(|v| v == 42u8));
                Ok(())
            },
        )?;

        assert_eq!(recs.len(), 4);

        if let Record::Memory(alloc) = &recs[0] {
            assert_eq!(alloc.bytes, SIZE);
            assert_eq!(alloc.memory_kind, MemoryKind::Device);
            assert_eq!(alloc.memory_operation_type, MemoryOperationType::Allocation);
        } else {
            panic!();
        }
        if let Record::Memcpy(h2d) = &recs[1] {
            assert_eq!(h2d.bytes, SIZE);
            assert_eq!(h2d.copy_kind, MemcpyKind::Htod);
            assert!(h2d.is_async);
        } else {
            panic!();
        }
        if let Record::Memcpy(d2h) = &recs[2] {
            assert_eq!(d2h.bytes, SIZE);
            assert_eq!(d2h.copy_kind, MemcpyKind::Dtoh);
            assert!(d2h.is_async);
        } else {
            panic!();
        }
        if let Record::Memory(free) = &recs[3] {
            assert_eq!(free.bytes, SIZE);
            assert_eq!(free.memory_kind, MemoryKind::Device);
            assert_eq!(free.memory_operation_type, MemoryOperationType::Release);
        } else {
            panic!();
        }

        Ok(())
    }

    #[test]
    #[serial]
    fn unified_memory() -> TestResult {
        let recs = get_records(
            Builder::new().with_unified_memory_counter_configs(
                [
                    uvm::CounterKind::BytesTransferHtod,
                    uvm::CounterKind::BytesTransferDtoh,
                    uvm::CounterKind::CpuPageFaultCount,
                    uvm::CounterKind::GpuPageFault,
                ]
                .into_iter()
                .map(|kind| uvm::CounterConfig {
                    scope: uvm::CounterScope::ProcessSingleDevice,
                    kind,
                    device_id: 0,
                    enable: true,
                }),
            ),
            |activity: &mut Context| {
                let cuda = cudarc::driver::CudaContext::new(0)?;
                let stream = cuda.default_stream();

                activity.enable_unified_memory_counters()?;

                const SIZE: usize = 1024;

                // Cause host-to-device memory page migrations
                {
                    let mut slice = unsafe { cuda.alloc_unified::<u8>(SIZE, true) }?;
                    let host_slice = slice.as_mut_slice()?;
                    host_slice.fill(42);
                    stream.memset_zeros(&mut slice)?;
                    stream.synchronize()?;
                }
                // Cause device-to-host memory page migrations
                {
                    let mut slice = unsafe { cuda.alloc_unified::<u8>(SIZE, true) }?;
                    stream.memset_zeros(&mut slice)?;
                    stream.synchronize()?;
                    let host_slice = slice.as_mut_slice()?;
                    host_slice.fill(42);
                }
                Ok(())
            },
        )?;

        dbg!(&recs);

        // Without completely unraveling the implementation details of UVM by checking
        // every record, check whether we observe at least one page fault on host and
        // gpu and at least one migration in both directions.
        let mut num_page_faults_cpu = 0;
        let mut num_page_faults_gpu = 0;
        let mut num_migrations_h2d = 0;
        let mut num_migrations_d2h = 0;

        for rec in recs.into_iter() {
            match rec {
                Record::UnifiedMemoryCounter(counter_record) => match counter_record {
                    uvm::CounterRecord::BytesTransferHtoD(_) => num_migrations_h2d += 1,
                    uvm::CounterRecord::BytesTransferDtoH(_) => num_migrations_d2h += 1,
                    uvm::CounterRecord::CpuPageFaultCount => num_page_faults_cpu += 1,
                    uvm::CounterRecord::GpuPageFault => num_page_faults_gpu += 1,
                    _ => (),
                },
                _ => (),
            }
        }

        assert!(num_page_faults_cpu > 1);
        assert!(num_page_faults_gpu > 1);
        assert!(num_migrations_h2d > 1);
        assert!(num_migrations_d2h > 1);

        Ok(())
    }

    #[test]
    pub fn record_nullptr_deref_errors_out() {
        assert!(
            unsafe {
                Record::try_from_record_ptr(std::ptr::null_mut() as *mut sys::CUpti_Activity)
            }
            .is_err()
        )
    }
}
