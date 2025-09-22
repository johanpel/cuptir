use std::{
    alloc::{Layout, alloc, dealloc},
    collections::HashSet,
    num::NonZero,
    sync::OnceLock,
};

use cudarc::cupti::{result, sys};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use tracing::{trace, warn};

use crate::{
    callback::callback_name,
    driver,
    error::CuptirError,
    runtime,
    utils::{try_demangle_from_ffi, try_str_from_ffi},
};

pub type ChannelType = crate::enums::ChannelType;
pub type FuncShmemLimitConfig = crate::enums::FuncShmemLimitConfig;
pub type Kind = crate::enums::ActivityKind;
pub type MemoryKind = crate::enums::ActivityMemoryKind;
pub type MemoryOperationType = crate::enums::ActivityMemoryOperationType;
pub type MemoryPoolOperationType = crate::enums::ActivityMemoryPoolOperationType;
pub type MemoryPoolType = crate::enums::ActivityMemoryPoolType;
pub type PartitionedGlobalCacheConfig = crate::enums::ActivityPartitionedGlobalCacheConfig;
pub type UnifiedMemoryCounterScope = crate::enums::ActivityUnifiedMemoryCounterScope;
pub type UnifiedMemoryCounterKind = crate::enums::ActivityUnifiedMemoryCounterKind;

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
                | sys::CUptiResult::CUPTI_ERROR_INVALID_KIND => None,
                sys::CUptiResult::CUPTI_ERROR_NOT_INITIALIZED => {
                    warn!("cupti is not initialized");
                    None
                }
                _ => {
                    warn!("unexpected error in record buffer iterator: {error:?}");
                    None
                }
            }
        } else if !self.current_record_ptr.is_null() {
            Some(unsafe { Record::try_from_record_ptr(self.current_record_ptr) })
        } else {
            None
        }
    }
}

/// Type of callback function used to handle [RecordBuffer]s.
pub type RecordBufferHandlerFn =
    dyn Fn(RecordBuffer) -> Result<(), Box<dyn std::error::Error + Send + Sync>> + Send + Sync;

/// Globally accessible callback to handle activity record buffers.
///
/// Because the buffer complete callback doesn't have a way of passing custom data, e.g.
/// a reference to the Context, this is needed to get to the Rust callback for  record
/// processing.
pub(crate) static RECORD_BUFFER_HANDLER: OnceLock<Box<RecordBufferHandlerFn>> = OnceLock::new();

/// Sets the activity record handler. This can be only called once.
pub(crate) fn set_record_buffer_handler(
    activity_record_buffer_handler: Box<RecordBufferHandlerFn>,
) -> Result<(), CuptirError> {
    RECORD_BUFFER_HANDLER
        .set(activity_record_buffer_handler)
        .map_err(|_| CuptirError::ActivityRecordBufferHandler("can only be set once".into()))
}

/// Calls the global record handler function if it is installed.
fn handle_record_buffer(record_buffer: RecordBuffer) -> Result<(), CuptirError> {
    if let Some(handler) = RECORD_BUFFER_HANDLER.get() {
        handler(record_buffer).map_err(|e| CuptirError::ActivityRecordBufferHandler(e.to_string()))
    } else {
        warn!("activity records received, but no callback is installed");
        Ok(())
    }
}

/// A time stamp
pub type Timestamp = u64;

pub type ProcessId = u32;
pub type ThreadId = u32;
pub type CorrelationId = u32;

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
    pub name: String,
    pub props: ApiProps,
}

/// A CUDA runtime API record
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
pub struct RuntimeApiRecord {
    pub name: String,
    pub props: ApiProps,
}

/// An internal launch API record
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
pub struct InternalLaunchApiRecord {
    pub props: ApiProps,
}

/// A kernel record
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

/// A memcpy record
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
pub struct MemcpyRecord {
    pub copy_kind: u8,
    pub src_kind: u8,
    pub dst_kind: u8,
    pub flags: u8,
    pub bytes: u64,
    pub start: u64,
    pub end: u64,
    pub device_id: u32,
    pub context_id: u32,
    pub stream_id: u32,
    pub correlation_id: u32,
    pub runtime_correlation_id: u32,
    pub pad: u32,
    pub graph_node_id: u64,
    pub graph_id: u32,
    pub channel_id: u32,
    pub pad2: u32,
    pub copy_count: u64,
    pub channel_type: ChannelType,
}

impl TryFrom<&sys::CUpti_ActivityMemcpy6> for MemcpyRecord {
    type Error = CuptirError;

    fn try_from(value: &sys::CUpti_ActivityMemcpy6) -> Result<Self, Self::Error> {
        Ok(Self {
            copy_kind: value.copyKind,
            src_kind: value.srcKind,
            dst_kind: value.dstKind,
            flags: value.flags,
            bytes: value.bytes,
            start: value.start,
            end: value.end,
            device_id: value.deviceId,
            context_id: value.contextId,
            stream_id: value.streamId,
            correlation_id: value.correlationId,
            runtime_correlation_id: value.runtimeCorrelationId,
            pad: value.pad,
            graph_node_id: value.graphNodeId,
            graph_id: value.graphId,
            channel_id: value.channelID,
            channel_type: value.channelType.try_into()?,
            pad2: value.pad2,
            copy_count: value.copyCount,
        })
    }
}

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
    pub pad1: u32,
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
            pad1: value.pad1,
            source: unsafe { try_str_from_ffi(value.source) }.map(ToOwned::to_owned),
        })
    }
}

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

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct UnifiedMemoryCounterConfig {
    pub scope: UnifiedMemoryCounterScope,
    pub kind: UnifiedMemoryCounterKind,
    pub device_id: u32,
    pub enable: bool,
}

impl From<UnifiedMemoryCounterConfig> for sys::CUpti_ActivityUnifiedMemoryCounterConfig {
    fn from(value: UnifiedMemoryCounterConfig) -> Self {
        sys::CUpti_ActivityUnifiedMemoryCounterConfig {
            scope: value.scope.into(),
            kind: value.kind.into(),
            deviceId: value.device_id,
            enable: if value.enable { 1 } else { 0 },
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
pub struct UnifiedMemoryCounterRecord {
    pub counter_kind: UnifiedMemoryCounterKind,
    pub value: u64,
    pub start: u64,
    pub end: u64,
    pub address: u64,
    pub src_id: u32,
    pub dst_id: u32,
    pub stream_id: u32,
    pub process_id: u32,
    pub flags: u32,
    pub pad: u32,
    pub processors: [u64; 5usize],
}

impl TryFrom<&sys::CUpti_ActivityUnifiedMemoryCounter3> for UnifiedMemoryCounterRecord {
    type Error = CuptirError;

    fn try_from(value: &sys::CUpti_ActivityUnifiedMemoryCounter3) -> Result<Self, Self::Error> {
        Ok(UnifiedMemoryCounterRecord {
            counter_kind: value.counterKind.try_into()?,
            value: value.value,
            start: value.start,
            end: value.end,
            address: value.address,
            src_id: value.srcId,
            dst_id: value.dstId,
            stream_id: value.streamId,
            process_id: value.processId,
            flags: value.flags,
            pad: value.pad,
            processors: value.processors,
        })
    }
}

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
    UnifiedMemoryCounter(UnifiedMemoryCounterRecord),
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
                    name: callback_name(
                        sys::CUpti_CallbackDomain::CUPTI_CB_DOMAIN_DRIVER_API,
                        api_record.cbid,
                    )?,
                    props: api_record.into(),
                }))
            }
            sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_RUNTIME => {
                let api_record = unsafe { &*(record_ptr as *const sys::CUpti_ActivityAPI) };
                Ok(Record::RuntimeApi(RuntimeApiRecord {
                    name: callback_name(
                        sys::CUpti_CallbackDomain::CUPTI_CB_DOMAIN_RUNTIME_API,
                        api_record.cbid,
                    )?,
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
            sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_MEMCPY2 => {
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

    unified_memory_counter_configs: HashSet<UnifiedMemoryCounterConfig>,

    flush_period: Option<NonZero<u32>>,
}

#[derive(Debug, Default)]
pub struct Context {
    enabled_kinds: Vec<Kind>,
    enabled_driver_functions: Vec<driver::Function>,
    enabled_runtime_functions: Vec<runtime::Function>,

    unified_memory_counter_configs: Vec<sys::CUpti_ActivityUnifiedMemoryCounterConfig>,
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
    /// `submitted` fields for [activity::KernelRecord] event records.
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
        configs: impl IntoIterator<Item = UnifiedMemoryCounterConfig>,
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
                set_record_buffer_handler(record_buffer_handler)?;
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
        } else {
            Ok(None)
        }
    }
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
    }
}
