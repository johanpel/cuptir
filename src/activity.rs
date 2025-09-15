use std::{
    alloc::{Layout, alloc, dealloc},
    sync::OnceLock,
};

use cudarc::cupti::{result as cupti, sys};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use strum::FromRepr;
use tracing::{trace, warn};

use crate::{callback::callback_name, error::CuptirError, try_demangle_from_ffi, try_str_from_ffi};

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
            cupti::activity::get_next_record(
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

/// CUPTI activity kind.
///
/// This matches CUpti_ActivityKind, but contains no sentinel values.
#[derive(Clone, Copy, Debug, FromRepr, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[repr(u32)]
#[rustfmt::skip]
pub enum Kind {
    Memset                      = sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_MEMSET as u32,
    Memcpy                      = sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_MEMCPY as u32,
    Kernel                      = sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_KERNEL as u32,
    Driver                      = sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_DRIVER as u32,
    Runtime                     = sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_RUNTIME as u32,
    Event                       = sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_EVENT as u32,
    Metric                      = sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_METRIC as u32,
    Device                      = sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_DEVICE as u32,
    Context                     = sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_CONTEXT as u32,
    ConcurrentKernel            = sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL as u32,
    Name                        = sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_NAME as u32,
    Marker                      = sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_MARKER as u32,
    MarkerData                  = sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_MARKER_DATA as u32,
    SourceLocator               = sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_SOURCE_LOCATOR as u32,
    GlobalAccess                = sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_GLOBAL_ACCESS as u32,
    Branch                      = sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_BRANCH as u32,
    Overhead                    = sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_OVERHEAD as u32,
    CdpKernel                   = sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_CDP_KERNEL as u32,
    Preemption                  = sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_PREEMPTION as u32,
    Environment                 = sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_ENVIRONMENT as u32,
    EventInstance               = sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_EVENT_INSTANCE as u32,
    Memcpy2                     = sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_MEMCPY2 as u32,
    MetricInstance              = sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_METRIC_INSTANCE as u32,
    InstructionExecution        = sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_INSTRUCTION_EXECUTION as u32,
    UnifiedMemoryCounter        = sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_UNIFIED_MEMORY_COUNTER as u32,
    Function                    = sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_FUNCTION as u32,
    Module                      = sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_MODULE as u32,
    DeviceAttribute             = sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_DEVICE_ATTRIBUTE as u32,
    SharedAccess                = sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_SHARED_ACCESS as u32,
    PcSampling                  = sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_PC_SAMPLING as u32,
    PcSamplingRecordInfo        = sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_PC_SAMPLING_RECORD_INFO as u32,
    InstructionCorrelation      = sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_INSTRUCTION_CORRELATION as u32,
    OpenaccData                 = sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_OPENACC_DATA as u32,
    OpenaccLaunch               = sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_OPENACC_LAUNCH as u32,
    OpenaccOther                = sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_OPENACC_OTHER as u32,
    CudaEvent                   = sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_CUDA_EVENT as u32,
    Stream                      = sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_STREAM as u32,
    Synchronization             = sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_SYNCHRONIZATION as u32,
    ExternalCorrelation         = sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION as u32,
    Nvlink                      = sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_NVLINK as u32,
    InstantaneousEvent          = sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_INSTANTANEOUS_EVENT as u32,
    InstantaneousEventInstance  = sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_INSTANTANEOUS_EVENT_INSTANCE as u32,
    InstantaneousMetric         = sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_INSTANTANEOUS_METRIC as u32,
    InstantaneousMetricInstance = sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_INSTANTANEOUS_METRIC_INSTANCE as u32,
    Memory                      = sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_MEMORY as u32,
    Pcie                        = sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_PCIE as u32,
    Openmp                      = sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_OPENMP as u32,
    InternalLaunchApi           = sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_INTERNAL_LAUNCH_API as u32,
    Memory2                     = sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_MEMORY2 as u32,
    MemoryPool                  = sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_MEMORY_POOL as u32,
    GraphTrace                  = sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_GRAPH_TRACE as u32,
    Jit                         = sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_JIT as u32,
    DeviceGraphTrace            = sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_DEVICE_GRAPH_TRACE as u32,
    MemDecompress               = sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_MEM_DECOMPRESS as u32,
    // ConfidentialComputeRotation = 55,
}

impl From<Kind> for sys::CUpti_ActivityKind {
    #[rustfmt::skip]
    fn from(value: Kind) -> Self {
        use sys::CUpti_ActivityKind as c;
        match value {
            Kind::Memcpy                      => c::CUPTI_ACTIVITY_KIND_MEMCPY,
            Kind::Memset                      => c::CUPTI_ACTIVITY_KIND_MEMSET,
            Kind::Kernel                      => c::CUPTI_ACTIVITY_KIND_KERNEL,
            Kind::Driver                      => c::CUPTI_ACTIVITY_KIND_DRIVER,
            Kind::Runtime                     => c::CUPTI_ACTIVITY_KIND_RUNTIME,
            Kind::Event                       => c::CUPTI_ACTIVITY_KIND_EVENT,
            Kind::Metric                      => c::CUPTI_ACTIVITY_KIND_METRIC,
            Kind::Device                      => c::CUPTI_ACTIVITY_KIND_DEVICE,
            Kind::Context                     => c::CUPTI_ACTIVITY_KIND_CONTEXT,
            Kind::ConcurrentKernel            => c::CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL,
            Kind::Name                        => c::CUPTI_ACTIVITY_KIND_NAME,
            Kind::Marker                      => c::CUPTI_ACTIVITY_KIND_MARKER,
            Kind::MarkerData                  => c::CUPTI_ACTIVITY_KIND_MARKER_DATA,
            Kind::SourceLocator               => c::CUPTI_ACTIVITY_KIND_SOURCE_LOCATOR,
            Kind::GlobalAccess                => c::CUPTI_ACTIVITY_KIND_GLOBAL_ACCESS,
            Kind::Branch                      => c::CUPTI_ACTIVITY_KIND_BRANCH,
            Kind::Overhead                    => c::CUPTI_ACTIVITY_KIND_OVERHEAD,
            Kind::CdpKernel                   => c::CUPTI_ACTIVITY_KIND_CDP_KERNEL,
            Kind::Preemption                  => c::CUPTI_ACTIVITY_KIND_PREEMPTION,
            Kind::Environment                 => c::CUPTI_ACTIVITY_KIND_ENVIRONMENT,
            Kind::EventInstance               => c::CUPTI_ACTIVITY_KIND_EVENT_INSTANCE,
            Kind::Memcpy2                     => c::CUPTI_ACTIVITY_KIND_MEMCPY2,
            Kind::MetricInstance              => c::CUPTI_ACTIVITY_KIND_METRIC_INSTANCE,
            Kind::InstructionExecution        => c::CUPTI_ACTIVITY_KIND_INSTRUCTION_EXECUTION,
            Kind::UnifiedMemoryCounter        => c::CUPTI_ACTIVITY_KIND_UNIFIED_MEMORY_COUNTER,
            Kind::Function                    => c::CUPTI_ACTIVITY_KIND_FUNCTION,
            Kind::Module                      => c::CUPTI_ACTIVITY_KIND_MODULE,
            Kind::DeviceAttribute             => c::CUPTI_ACTIVITY_KIND_DEVICE_ATTRIBUTE,
            Kind::SharedAccess                => c::CUPTI_ACTIVITY_KIND_SHARED_ACCESS,
            Kind::PcSampling                  => c::CUPTI_ACTIVITY_KIND_PC_SAMPLING,
            Kind::PcSamplingRecordInfo        => c::CUPTI_ACTIVITY_KIND_PC_SAMPLING_RECORD_INFO,
            Kind::InstructionCorrelation      => c::CUPTI_ACTIVITY_KIND_INSTRUCTION_CORRELATION,
            Kind::OpenaccData                 => c::CUPTI_ACTIVITY_KIND_OPENACC_DATA,
            Kind::OpenaccLaunch               => c::CUPTI_ACTIVITY_KIND_OPENACC_LAUNCH,
            Kind::OpenaccOther                => c::CUPTI_ACTIVITY_KIND_OPENACC_OTHER,
            Kind::CudaEvent                   => c::CUPTI_ACTIVITY_KIND_CUDA_EVENT,
            Kind::Stream                      => c::CUPTI_ACTIVITY_KIND_STREAM,
            Kind::Synchronization             => c::CUPTI_ACTIVITY_KIND_SYNCHRONIZATION,
            Kind::ExternalCorrelation         => c::CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION,
            Kind::Nvlink                      => c::CUPTI_ACTIVITY_KIND_NVLINK,
            Kind::InstantaneousEvent          => c::CUPTI_ACTIVITY_KIND_INSTANTANEOUS_EVENT,
            Kind::InstantaneousEventInstance  => c::CUPTI_ACTIVITY_KIND_INSTANTANEOUS_EVENT_INSTANCE,
            Kind::InstantaneousMetric         => c::CUPTI_ACTIVITY_KIND_INSTANTANEOUS_METRIC,
            Kind::InstantaneousMetricInstance => c::CUPTI_ACTIVITY_KIND_INSTANTANEOUS_METRIC_INSTANCE,
            Kind::Memory                      => c::CUPTI_ACTIVITY_KIND_MEMORY,
            Kind::Pcie                        => c::CUPTI_ACTIVITY_KIND_PCIE,
            Kind::Openmp                      => c::CUPTI_ACTIVITY_KIND_OPENMP,
            Kind::InternalLaunchApi           => c::CUPTI_ACTIVITY_KIND_INTERNAL_LAUNCH_API,
            Kind::Memory2                     => c::CUPTI_ACTIVITY_KIND_MEMORY2,
            Kind::MemoryPool                  => c::CUPTI_ACTIVITY_KIND_MEMORY_POOL,
            Kind::GraphTrace                  => c::CUPTI_ACTIVITY_KIND_GRAPH_TRACE,
            Kind::Jit                         => c::CUPTI_ACTIVITY_KIND_JIT,
            Kind::DeviceGraphTrace            => c::CUPTI_ACTIVITY_KIND_DEVICE_GRAPH_TRACE,
            Kind::MemDecompress               => c::CUPTI_ACTIVITY_KIND_MEM_DECOMPRESS,
            // Kind::ConfidentialComputeRotation => c::CUPTI_ACTIVITY_KIND_CONFIDENTIAL_COMPUTE_ROTATION
        }
    }
}

impl TryFrom<sys::CUpti_ActivityKind> for Kind {
    type Error = CuptirError;

    fn try_from(value: sys::CUpti_ActivityKind) -> Result<Self, Self::Error> {
        match value {
            sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_INVALID
            | sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_COUNT
            | sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_FORCE_INT => {
                Err(CuptirError::SentinelEnum(value as u32))
            }
            other => Self::from_repr(other as u32).ok_or(CuptirError::Corrupted),
        }
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

#[derive(Clone, Copy, Debug, FromRepr, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[repr(u32)]
pub enum FuncShmemLimitConfig {
    Default = sys::CUpti_FuncShmemLimitConfig::CUPTI_FUNC_SHMEM_LIMIT_DEFAULT as u32,
    Optin = sys::CUpti_FuncShmemLimitConfig::CUPTI_FUNC_SHMEM_LIMIT_OPTIN as u32,
}

impl TryFrom<sys::CUpti_FuncShmemLimitConfig> for FuncShmemLimitConfig {
    type Error = CuptirError;

    fn try_from(value: sys::CUpti_FuncShmemLimitConfig) -> Result<Self, Self::Error> {
        match value {
            sys::CUpti_FuncShmemLimitConfig::CUPTI_FUNC_SHMEM_LIMIT_DEFAULT => {
                Ok(FuncShmemLimitConfig::Default)
            }
            sys::CUpti_FuncShmemLimitConfig::CUPTI_FUNC_SHMEM_LIMIT_OPTIN => {
                Ok(FuncShmemLimitConfig::Optin)
            }
            sys::CUpti_FuncShmemLimitConfig::CUPTI_FUNC_SHMEM_LIMIT_FORCE_INT => {
                Err(CuptirError::SentinelEnum(value as u32))
            }
        }
    }
}

#[derive(Clone, Copy, Debug, FromRepr, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[repr(u32)]
pub enum PartitionedGlobalCacheConfig {
    NotSupported = sys::CUpti_ActivityPartitionedGlobalCacheConfig::CUPTI_ACTIVITY_PARTITIONED_GLOBAL_CACHE_CONFIG_NOT_SUPPORTED as u32,
    Off = sys::CUpti_ActivityPartitionedGlobalCacheConfig::CUPTI_ACTIVITY_PARTITIONED_GLOBAL_CACHE_CONFIG_OFF as u32,
    On = sys::CUpti_ActivityPartitionedGlobalCacheConfig::CUPTI_ACTIVITY_PARTITIONED_GLOBAL_CACHE_CONFIG_ON as u32,
}

impl PartitionedGlobalCacheConfig {
    fn try_from_sys(
        value: sys::CUpti_ActivityPartitionedGlobalCacheConfig,
    ) -> Result<Option<Self>, CuptirError> {
        use sys::CUpti_ActivityPartitionedGlobalCacheConfig as c;
        Ok(match value {
            c::CUPTI_ACTIVITY_PARTITIONED_GLOBAL_CACHE_CONFIG_UNKNOWN => None,
            c::CUPTI_ACTIVITY_PARTITIONED_GLOBAL_CACHE_CONFIG_NOT_SUPPORTED => {
                Some(PartitionedGlobalCacheConfig::NotSupported)
            }
            c::CUPTI_ACTIVITY_PARTITIONED_GLOBAL_CACHE_CONFIG_OFF => {
                Some(PartitionedGlobalCacheConfig::Off)
            }
            c::CUPTI_ACTIVITY_PARTITIONED_GLOBAL_CACHE_CONFIG_ON => {
                Some(PartitionedGlobalCacheConfig::On)
            }
            c::CUPTI_ACTIVITY_PARTITIONED_GLOBAL_CACHE_CONFIG_FORCE_INT => {
                Err(CuptirError::SentinelEnum(value as u32))?
            }
        })
    }
}

/// A kernel record
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
pub struct KernelRecord {
    // TODO: this union:
    // pub cache_config: sys::CUpti_ActivityKernel9__bindgen_ty_1,
    pub shared_memory_config: u8,
    pub registers_per_thread: u16,
    pub partitioned_global_cache_requested: Option<PartitionedGlobalCacheConfig>,
    pub partitioned_global_cache_executed: Option<PartitionedGlobalCacheConfig>,
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
            partitioned_global_cache_requested: PartitionedGlobalCacheConfig::try_from_sys(
                value.partitionedGlobalCacheRequested,
            )?,
            partitioned_global_cache_executed: PartitionedGlobalCacheConfig::try_from_sys(
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[repr(u32)]
pub enum ChannelType {
    Compute = sys::CUpti_ChannelType::CUPTI_CHANNEL_TYPE_COMPUTE as u32,
    AsyncMemcpy = sys::CUpti_ChannelType::CUPTI_CHANNEL_TYPE_ASYNC_MEMCPY as u32,
    Decomp = sys::CUpti_ChannelType::CUPTI_CHANNEL_TYPE_DECOMP as u32,
}

impl TryFrom<sys::CUpti_ChannelType> for ChannelType {
    type Error = CuptirError;

    fn try_from(value: sys::CUpti_ChannelType) -> Result<Self, Self::Error> {
        match value {
            sys::CUpti_ChannelType::CUPTI_CHANNEL_TYPE_COMPUTE => Ok(Self::Compute),
            sys::CUpti_ChannelType::CUPTI_CHANNEL_TYPE_ASYNC_MEMCPY => Ok(Self::AsyncMemcpy),
            sys::CUpti_ChannelType::CUPTI_CHANNEL_TYPE_DECOMP => Ok(Self::Decomp),
            sys::CUpti_ChannelType::CUPTI_CHANNEL_TYPE_INVALID
            | sys::CUpti_ChannelType::CUPTI_CHANNEL_TYPE_FORCE_INT => {
                Err(CuptirError::SentinelEnum(value as u32))
            }
        }
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[repr(u32)]
pub enum MemoryOperationType {
    Allocation =
        sys::CUpti_ActivityMemoryOperationType::CUPTI_ACTIVITY_MEMORY_OPERATION_TYPE_ALLOCATION
            as u32,
    Release =
        sys::CUpti_ActivityMemoryOperationType::CUPTI_ACTIVITY_MEMORY_OPERATION_TYPE_RELEASE as u32,
}

impl TryFrom<sys::CUpti_ActivityMemoryOperationType> for MemoryOperationType {
    type Error = CuptirError;

    fn try_from(value: sys::CUpti_ActivityMemoryOperationType) -> Result<Self, Self::Error> {
        use sys::CUpti_ActivityMemoryOperationType as c;
        match value {
            c::CUPTI_ACTIVITY_MEMORY_OPERATION_TYPE_ALLOCATION => {
                Ok(MemoryOperationType::Allocation)
            }
            c::CUPTI_ACTIVITY_MEMORY_OPERATION_TYPE_RELEASE => Ok(MemoryOperationType::Release),
            c::CUPTI_ACTIVITY_MEMORY_OPERATION_TYPE_FORCE_INT
            | c::CUPTI_ACTIVITY_MEMORY_OPERATION_TYPE_INVALID => {
                Err(CuptirError::SentinelEnum(value as u32))
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[repr(u32)]
pub enum MemoryKind {
    Pageable = sys::CUpti_ActivityMemoryKind::CUPTI_ACTIVITY_MEMORY_KIND_PAGEABLE as u32,
    Pinned = sys::CUpti_ActivityMemoryKind::CUPTI_ACTIVITY_MEMORY_KIND_PINNED as u32,
    Device = sys::CUpti_ActivityMemoryKind::CUPTI_ACTIVITY_MEMORY_KIND_DEVICE as u32,
    Array = sys::CUpti_ActivityMemoryKind::CUPTI_ACTIVITY_MEMORY_KIND_ARRAY as u32,
    Managed = sys::CUpti_ActivityMemoryKind::CUPTI_ACTIVITY_MEMORY_KIND_MANAGED as u32,
    DeviceStatic = sys::CUpti_ActivityMemoryKind::CUPTI_ACTIVITY_MEMORY_KIND_DEVICE_STATIC as u32,
    ManagedStatic = sys::CUpti_ActivityMemoryKind::CUPTI_ACTIVITY_MEMORY_KIND_MANAGED_STATIC as u32,
}

impl MemoryKind {
    fn try_from_sys(value: sys::CUpti_ActivityMemoryKind) -> Result<Option<Self>, CuptirError> {
        use sys::CUpti_ActivityMemoryKind as c;
        Ok(match value {
            c::CUPTI_ACTIVITY_MEMORY_KIND_UNKNOWN => None,
            c::CUPTI_ACTIVITY_MEMORY_KIND_PAGEABLE => Some(MemoryKind::Pageable),
            c::CUPTI_ACTIVITY_MEMORY_KIND_PINNED => Some(MemoryKind::Pinned),
            c::CUPTI_ACTIVITY_MEMORY_KIND_DEVICE => Some(MemoryKind::Device),
            c::CUPTI_ACTIVITY_MEMORY_KIND_ARRAY => Some(MemoryKind::Array),
            c::CUPTI_ACTIVITY_MEMORY_KIND_MANAGED => Some(MemoryKind::Managed),
            c::CUPTI_ACTIVITY_MEMORY_KIND_DEVICE_STATIC => Some(MemoryKind::DeviceStatic),
            c::CUPTI_ACTIVITY_MEMORY_KIND_MANAGED_STATIC => Some(MemoryKind::ManagedStatic),
            c::CUPTI_ACTIVITY_MEMORY_KIND_FORCE_INT => {
                Err(CuptirError::SentinelEnum(value as u32))?
            }
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
pub struct MemoryRecord {
    pub memory_operation_type: MemoryOperationType,
    pub memory_kind: Option<MemoryKind>,
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
            memory_kind: MemoryKind::try_from_sys(value.memoryKind)?,
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[repr(u32)]
pub enum MemoryPoolOperationType {
    Created = sys::CUpti_ActivityMemoryPoolOperationType::CUPTI_ACTIVITY_MEMORY_POOL_OPERATION_TYPE_CREATED as u32,
    Destroyed = sys::CUpti_ActivityMemoryPoolOperationType::CUPTI_ACTIVITY_MEMORY_POOL_OPERATION_TYPE_DESTROYED as u32,
    Trimmed = sys::CUpti_ActivityMemoryPoolOperationType::CUPTI_ACTIVITY_MEMORY_POOL_OPERATION_TYPE_TRIMMED as u32,
}

impl TryFrom<sys::CUpti_ActivityMemoryPoolOperationType> for MemoryPoolOperationType {
    type Error = CuptirError;

    fn try_from(value: sys::CUpti_ActivityMemoryPoolOperationType) -> Result<Self, Self::Error> {
        use sys::CUpti_ActivityMemoryPoolOperationType as c;
        match value {
            c::CUPTI_ACTIVITY_MEMORY_POOL_OPERATION_TYPE_CREATED => {
                Ok(MemoryPoolOperationType::Created)
            }
            c::CUPTI_ACTIVITY_MEMORY_POOL_OPERATION_TYPE_DESTROYED => {
                Ok(MemoryPoolOperationType::Destroyed)
            }
            c::CUPTI_ACTIVITY_MEMORY_POOL_OPERATION_TYPE_TRIMMED => {
                Ok(MemoryPoolOperationType::Trimmed)
            }
            c::CUPTI_ACTIVITY_MEMORY_POOL_OPERATION_TYPE_FORCE_INT
            | c::CUPTI_ACTIVITY_MEMORY_POOL_OPERATION_TYPE_INVALID => {
                Err(CuptirError::SentinelEnum(value as u32))
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[repr(u32)]
pub enum MemoryPoolType {
    Local = sys::CUpti_ActivityMemoryPoolType::CUPTI_ACTIVITY_MEMORY_POOL_TYPE_LOCAL as u32,
    Imported = sys::CUpti_ActivityMemoryPoolType::CUPTI_ACTIVITY_MEMORY_POOL_TYPE_IMPORTED as u32,
}

impl TryFrom<sys::CUpti_ActivityMemoryPoolType> for MemoryPoolType {
    type Error = CuptirError;

    fn try_from(value: sys::CUpti_ActivityMemoryPoolType) -> Result<Self, Self::Error> {
        match value {
            sys::CUpti_ActivityMemoryPoolType::CUPTI_ACTIVITY_MEMORY_POOL_TYPE_LOCAL => {
                Ok(MemoryPoolType::Local)
            }
            sys::CUpti_ActivityMemoryPoolType::CUPTI_ACTIVITY_MEMORY_POOL_TYPE_IMPORTED => {
                Ok(MemoryPoolType::Imported)
            }
            sys::CUpti_ActivityMemoryPoolType::CUPTI_ACTIVITY_MEMORY_POOL_TYPE_FORCE_INT
            | sys::CUpti_ActivityMemoryPoolType::CUPTI_ACTIVITY_MEMORY_POOL_TYPE_INVALID => {
                Err(CuptirError::SentinelEnum(value as u32))
            }
        }
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
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
pub enum Record {
    DriverApi(DriverApiRecord),
    RuntimeApi(RuntimeApiRecord),
    InternalLaunchApi(InternalLaunchApiRecord),
    Kernel(KernelRecord),
    Memcpy(MemcpyRecord),
    Memory(MemoryRecord),
    MemoryPool(MemoryPoolRecord),
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
