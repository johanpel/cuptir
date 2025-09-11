use std::{
    alloc::{Layout, alloc, dealloc},
    ffi::CStr,
    ptr::NonNull,
    sync::OnceLock,
};

use cudarc::cupti::{result as cupti, sys};
use strum::FromRepr;
use tracing::trace;

use crate::error::CuptirError;

// Same constants as used in cuda/extras/CUPTI/samples/common/helper_cupti_activity.h
const CUPTI_BUFFER_SIZE: usize = 16 * 1024 * 1024;
const CUPTI_BUFFER_ALIGN: usize = 8;

/// Type of function used to handle a [Record].
pub type RecordHandler =
    dyn Fn(Record) -> Result<(), Box<dyn std::error::Error + Send + Sync>> + Send + Sync;

/// Because the buffer complete callback doesn't have a way of passing custom data,
/// we need something global to hold on to the Rust callback for record processing.
pub(crate) static RECORD_HANDLER: OnceLock<Box<RecordHandler>> = OnceLock::new();

/// Sets the activity record handler. This can be only called once.
pub(crate) fn set_record_handler(
    activity_record_handler: Box<RecordHandler>,
) -> Result<(), CuptirError> {
    RECORD_HANDLER
        .set(activity_record_handler)
        .map_err(|_| CuptirError::AcivityRecordHandler("can only be set once".into()))
}

/// Calls the global record handler function if it is installed.
fn handle_record(record: Record) -> Result<(), CuptirError> {
    if let Some(handler) = RECORD_HANDLER.get() {
        handler(record).map_err(|e| CuptirError::AcivityRecordHandler(e.to_string()))
    } else {
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, FromRepr, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum Kind {
    Memcpy = 1,
    Memset = 2,
    Kernel = 3,
    Driver = 4,
    Runtime = 5,
    Event = 6,
    Metric = 7,
    Device = 8,
    Context = 9,
    ConcurrentKernel = 10,
    Name = 11,
    Marker = 12,
    MarkerData = 13,
    SourceLocator = 14,
    GlobalAccess = 15,
    Branch = 16,
    Overhead = 17,
    CdpKernel = 18,
    Preemption = 19,
    Environment = 20,
    EventInstance = 21,
    Memcpy2 = 22,
    MetricInstance = 23,
    InstructionExecution = 24,
    UnifiedMemoryCounter = 25,
    Function = 26,
    Module = 27,
    DeviceAttribute = 28,
    SharedAccess = 29,
    PcSampling = 30,
    PcSamplingRecordInfo = 31,
    InstructionCorrelation = 32,
    OpenaccData = 33,
    OpenaccLaunch = 34,
    OpenaccOther = 35,
    CudaEvent = 36,
    Stream = 37,
    Synchronization = 38,
    ExternalCorrelation = 39,
    Nvlink = 40,
    InstantaneousEvent = 41,
    InstantaneousEventInstance = 42,
    InstantaneousMetric = 43,
    InstantaneousMetricInstance = 44,
    Memory = 45,
    Pcie = 46,
    Openmp = 47,
    InternalLaunchApi = 48,
    Memory2 = 49,
    MemoryPool = 50,
    GraphTrace = 51,
    Jit = 52,
    DeviceGraphTrace = 53,
    MemDecompress = 54,
    ConfidentialComputeRotation = 55,
}

impl From<Kind> for sys::CUpti_ActivityKind {
    fn from(value: Kind) -> Self {
        unsafe { std::mem::transmute(value) }
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

/// A time span
#[derive(Debug)]
pub struct Span {
    pub start: Timestamp,
    pub end: Timestamp,
}

// Obtain the name of a callback within a specific domain.
fn callback_name(domain: sys::CUpti_CallbackDomain, id: u32) -> Result<String, CuptirError> {
    let mut name_ptr: *const ::std::os::raw::c_char = std::ptr::null_mut();
    unsafe {
        sys::cuptiGetCallbackName(domain, id, &mut name_ptr);
    }
    let name = if !name_ptr.is_null() {
        unsafe { CStr::from_ptr(name_ptr) }
            .to_string_lossy()
            .into_owned()
    } else {
        "<unnamed>".to_string()
    };
    Ok(name)
}

pub type ProcessId = u32;
pub type ThreadId = u32;
pub type CorrelationId = u32;

#[derive(Debug)]
pub struct ApiProps {
    pub span: Span,
    pub process_id: ProcessId,
    pub thread_id: ThreadId,
    pub correlation_id: CorrelationId,
    pub return_value: u32,
}

impl From<&sys::CUpti_ActivityAPI> for ApiProps {
    fn from(value: &sys::CUpti_ActivityAPI) -> Self {
        Self {
            span: Span {
                start: value.start,
                end: value.end,
            },
            process_id: value.processId,
            thread_id: value.threadId,
            correlation_id: value.correlationId,
            return_value: value.returnValue,
        }
    }
}

/// A CUDA driver API record
#[derive(Debug)]
pub struct DriverApiRecord {
    // TODO: use enum instead of name
    pub name: String,
    pub props: ApiProps,
}

/// A CUDA runtime API record
#[derive(Debug)]
pub struct RuntimeApiRecord {
    // TODO: use enum instead of name
    pub name: String,
    pub props: ApiProps,
}

/// An internal launch API record
#[derive(Debug)]
pub struct InternalLaunchApiRecord {
    pub props: ApiProps,
}

/// A kernel record
#[derive(Debug)]
pub struct KernelRecord {
    pub shared_memory_config: u8,
    pub registers_per_thread: u16,
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
    pub name: String,
    pub queued: Option<Timestamp>,
    pub submitted: Option<Timestamp>,
    pub launch_type: u8,
    pub is_shared_memory_carveout_requested: u8,
    pub shared_memory_carveout_requested: u8,
    pub padding: u8,
    pub shared_memory_executed: u32,
    pub graph_node_id: u64,
    pub graph_id: u32,
    pub channel_id: u32,
    pub cluster_x: u32,
    pub cluster_y: u32,
    pub cluster_z: u32,
    pub cluster_scheduling_policy: u32,
    pub local_memory_total_v2: u64,
    pub max_potential_cluster_size: u32,
    pub max_active_clusters: u32,
    // pub cache_config: sys::CUpti_ActivityKernel9__bindgen_ty_1,
    // pub partitioned_global_cache_requested: CUpti_ActivityPartitionedGlobalCacheConfig,
    // pub partitioned_global_cache_executed: CUpti_ActivityPartitionedGlobalCacheConfig,
    // pub shmem_limit_config: CUpti_FuncShmemLimitConfig,
    // pub p_access_policy_window: *mut CUaccessPolicyWindow,
    pub channel_type: ChannelType,
}

#[derive(Debug)]
#[repr(u32)]
pub enum ChannelType {
    Compute = 1,
    AsyncMemcpy = 2,
    Decomp = 3,
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
#[derive(Debug)]
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

#[derive(Debug)]
#[repr(u32)]
pub enum MemoryOperationType {
    Allocation = 1,
    Release = 2,
}

impl TryFrom<sys::CUpti_ActivityMemoryOperationType> for MemoryOperationType {
    type Error = CuptirError;

    fn try_from(value: sys::CUpti_ActivityMemoryOperationType) -> Result<Self, Self::Error> {
        match value {
            sys::CUpti_ActivityMemoryOperationType::CUPTI_ACTIVITY_MEMORY_OPERATION_TYPE_ALLOCATION => Ok(MemoryOperationType::Allocation),
            sys::CUpti_ActivityMemoryOperationType::CUPTI_ACTIVITY_MEMORY_OPERATION_TYPE_RELEASE    => Ok(MemoryOperationType::Release),
            sys::CUpti_ActivityMemoryOperationType::CUPTI_ACTIVITY_MEMORY_OPERATION_TYPE_FORCE_INT  |
            sys::CUpti_ActivityMemoryOperationType::CUPTI_ACTIVITY_MEMORY_OPERATION_TYPE_INVALID    => Err(CuptirError::SentinelEnum(value as u32)),
        }
    }
}

#[derive(Debug)]
#[repr(u32)]
pub enum MemoryKind {
    Pageable = 1,
    Pinned = 2,
    Device = 3,
    Array = 4,
    Managed = 5,
    DeviceStatic = 6,
    ManagedStatic = 7,
}

impl MemoryKind {
    fn from_sys(value: sys::CUpti_ActivityMemoryKind) -> Result<Option<Self>, CuptirError> {
        match value {
            sys::CUpti_ActivityMemoryKind::CUPTI_ACTIVITY_MEMORY_KIND_UNKNOWN => Ok(None),
            sys::CUpti_ActivityMemoryKind::CUPTI_ACTIVITY_MEMORY_KIND_PAGEABLE => {
                Ok(Some(MemoryKind::Pageable))
            }
            sys::CUpti_ActivityMemoryKind::CUPTI_ACTIVITY_MEMORY_KIND_PINNED => {
                Ok(Some(MemoryKind::Pinned))
            }
            sys::CUpti_ActivityMemoryKind::CUPTI_ACTIVITY_MEMORY_KIND_DEVICE => {
                Ok(Some(MemoryKind::Device))
            }
            sys::CUpti_ActivityMemoryKind::CUPTI_ACTIVITY_MEMORY_KIND_ARRAY => {
                Ok(Some(MemoryKind::Array))
            }
            sys::CUpti_ActivityMemoryKind::CUPTI_ACTIVITY_MEMORY_KIND_MANAGED => {
                Ok(Some(MemoryKind::Managed))
            }
            sys::CUpti_ActivityMemoryKind::CUPTI_ACTIVITY_MEMORY_KIND_DEVICE_STATIC => {
                Ok(Some(MemoryKind::DeviceStatic))
            }
            sys::CUpti_ActivityMemoryKind::CUPTI_ACTIVITY_MEMORY_KIND_MANAGED_STATIC => {
                Ok(Some(MemoryKind::ManagedStatic))
            }
            sys::CUpti_ActivityMemoryKind::CUPTI_ACTIVITY_MEMORY_KIND_FORCE_INT => {
                Err(CuptirError::SentinelEnum(value as u32))
            }
        }
    }
}

#[derive(Debug)]
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
    pub source: Option<String>,
    // pub kind: CUpti_ActivityKind,
    // pub memory_pool_config: sys::CUpti_ActivityMemory4__bindgen_ty_1,
}

#[derive(Debug)]
#[repr(u32)]
pub enum MemoryPoolOperationType {
    Created = 1,
    Destroyed = 2,
    Trimmed = 3,
}

impl TryFrom<sys::CUpti_ActivityMemoryPoolOperationType> for MemoryPoolOperationType {
    type Error = CuptirError;

    fn try_from(value: sys::CUpti_ActivityMemoryPoolOperationType) -> Result<Self, Self::Error> {
        match value {
            sys::CUpti_ActivityMemoryPoolOperationType::CUPTI_ACTIVITY_MEMORY_POOL_OPERATION_TYPE_CREATED   => Ok(MemoryPoolOperationType::Created),
            sys::CUpti_ActivityMemoryPoolOperationType::CUPTI_ACTIVITY_MEMORY_POOL_OPERATION_TYPE_DESTROYED => Ok(MemoryPoolOperationType::Destroyed),
            sys::CUpti_ActivityMemoryPoolOperationType::CUPTI_ACTIVITY_MEMORY_POOL_OPERATION_TYPE_TRIMMED   => Ok(MemoryPoolOperationType::Trimmed),
            sys::CUpti_ActivityMemoryPoolOperationType::CUPTI_ACTIVITY_MEMORY_POOL_OPERATION_TYPE_FORCE_INT |
            sys::CUpti_ActivityMemoryPoolOperationType::CUPTI_ACTIVITY_MEMORY_POOL_OPERATION_TYPE_INVALID   => Err(CuptirError::SentinelEnum(value as u32)),
        }
    }
}

#[derive(Debug)]
#[repr(u32)]
pub enum MemoryPoolType {
    Local = 1,
    Imported = 2,
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

#[derive(Debug)]
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

#[derive(Debug)]
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
                let kernel_record = unsafe { &*(record_ptr as *const sys::CUpti_ActivityKernel9) };
                let kernel_name = unsafe { CStr::from_ptr(kernel_record.name) };
                let kernel_name_demangled =
                    cpp_demangle::Symbol::new(kernel_name.to_bytes())?.to_string();
                Ok(Record::Kernel(KernelRecord {
                    shared_memory_config: kernel_record.sharedMemoryConfig,
                    registers_per_thread: kernel_record.registersPerThread,
                    start: kernel_record.start,
                    end: kernel_record.end,
                    completed: kernel_record.completed,
                    device_id: kernel_record.deviceId,
                    context_id: kernel_record.contextId,
                    stream_id: kernel_record.streamId,
                    grid_x: kernel_record.gridX,
                    grid_y: kernel_record.gridY,
                    grid_z: kernel_record.gridZ,
                    block_x: kernel_record.blockX,
                    block_y: kernel_record.blockY,
                    block_z: kernel_record.blockZ,
                    static_shared_memory: kernel_record.staticSharedMemory,
                    dynamic_shared_memory: kernel_record.dynamicSharedMemory,
                    local_memory_per_thread: kernel_record.localMemoryPerThread,
                    local_memory_total: kernel_record.localMemoryTotal,
                    correlation_id: kernel_record.correlationId,
                    grid_id: kernel_record.gridId,
                    name: kernel_name_demangled,
                    queued: if kernel_record.queued == sys::CUPTI_TIMESTAMP_UNKNOWN as u64 {
                        None
                    } else {
                        Some(kernel_record.queued)
                    },
                    submitted: if kernel_record.submitted == sys::CUPTI_TIMESTAMP_UNKNOWN as u64 {
                        None
                    } else {
                        Some(kernel_record.submitted)
                    },
                    launch_type: kernel_record.launchType,
                    is_shared_memory_carveout_requested: kernel_record
                        .isSharedMemoryCarveoutRequested,
                    shared_memory_carveout_requested: kernel_record.sharedMemoryCarveoutRequested,
                    padding: kernel_record.padding,
                    shared_memory_executed: kernel_record.sharedMemoryExecuted,
                    graph_node_id: kernel_record.graphNodeId,
                    graph_id: kernel_record.graphId,
                    channel_id: kernel_record.channelID,
                    cluster_x: kernel_record.clusterX,
                    cluster_y: kernel_record.clusterY,
                    cluster_z: kernel_record.clusterZ,
                    cluster_scheduling_policy: kernel_record.clusterSchedulingPolicy,
                    local_memory_total_v2: kernel_record.localMemoryTotal_v2,
                    max_potential_cluster_size: kernel_record.maxPotentialClusterSize,
                    max_active_clusters: kernel_record.maxActiveClusters,
                    channel_type: kernel_record.channelType.try_into()?,
                }))
            }
            sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_MEMCPY2 => {
                let memcpy_record = unsafe { &*(record_ptr as *const sys::CUpti_ActivityMemcpy6) };
                Ok(Record::Memcpy(MemcpyRecord {
                    copy_kind: memcpy_record.copyKind,
                    src_kind: memcpy_record.srcKind,
                    dst_kind: memcpy_record.dstKind,
                    flags: memcpy_record.flags,
                    bytes: memcpy_record.bytes,
                    start: memcpy_record.start,
                    end: memcpy_record.end,
                    device_id: memcpy_record.deviceId,
                    context_id: memcpy_record.contextId,
                    stream_id: memcpy_record.streamId,
                    correlation_id: memcpy_record.correlationId,
                    runtime_correlation_id: memcpy_record.runtimeCorrelationId,
                    pad: memcpy_record.pad,
                    graph_node_id: memcpy_record.graphNodeId,
                    graph_id: memcpy_record.graphId,
                    channel_id: memcpy_record.channelID,
                    channel_type: memcpy_record.channelType.try_into()?,
                    pad2: memcpy_record.pad2,
                    copy_count: memcpy_record.copyCount,
                }))
            }
            sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_MEMORY2 => {
                let memory_record = unsafe { &*(record_ptr as *const sys::CUpti_ActivityMemory4) };
                let name = NonNull::new(memory_record.name as *mut _).map(|p| {
                    unsafe { CStr::from_ptr(p.as_ptr()) }
                        .to_string_lossy()
                        .into_owned()
                });
                let source = NonNull::new(memory_record.source as *mut _).map(|p| {
                    unsafe { CStr::from_ptr(p.as_ptr()) }
                        .to_string_lossy()
                        .into_owned()
                });
                Ok(Record::Memory(MemoryRecord {
                    memory_operation_type: memory_record.memoryOperationType.try_into()?,
                    memory_kind: MemoryKind::from_sys(memory_record.memoryKind)?,
                    correlation_id: memory_record.correlationId,
                    address: memory_record.address,
                    bytes: memory_record.bytes,
                    timestamp: memory_record.timestamp,
                    pc: memory_record.PC,
                    process_id: memory_record.processId,
                    device_id: memory_record.deviceId,
                    context_id: memory_record.contextId,
                    stream_id: memory_record.streamId,
                    name,
                    is_async: memory_record.isAsync,
                    pad1: memory_record.pad1,
                    source,
                }))
            }
            sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_MEMORY_POOL => {
                let memory_pool_record =
                    unsafe { &*(record_ptr as *const sys::CUpti_ActivityMemoryPool2) };
                Ok(Record::MemoryPool(MemoryPoolRecord {
                    memory_pool_operation_type: memory_pool_record
                        .memoryPoolOperationType
                        .try_into()?,
                    memory_pool_type: memory_pool_record.memoryPoolType.try_into()?,
                    correlation_id: memory_pool_record.correlationId,
                    process_id: memory_pool_record.processId,
                    device_id: memory_pool_record.deviceId,
                    min_bytes_to_keep: memory_pool_record.minBytesToKeep,
                    address: memory_pool_record.address,
                    size: memory_pool_record.size,
                    release_threshold: memory_pool_record.releaseThreshold,
                    timestamp: memory_pool_record.timestamp,
                    utilized_size: memory_pool_record.utilizedSize,
                }))
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
        // Safety: the memory allocated here is freed in buffer_complete_callback, which
        // cupti must call.
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

    if let Err(error) = process_buffer(buffer, valid_size) {
        trace!("error processing activity buffer: {error}");
    }

    let layout = Layout::from_size_align(CUPTI_BUFFER_SIZE, CUPTI_BUFFER_ALIGN).unwrap();
    unsafe {
        dealloc(buffer, layout);
    }
}

fn process_buffer(buffer_ptr: *mut u8, num_valid_bytes: usize) -> Result<(), CuptirError> {
    let mut record: *mut sys::CUpti_Activity = std::ptr::null_mut();

    loop {
        let result =
            unsafe { cupti::activity::get_next_record(buffer_ptr, num_valid_bytes, &mut record) };
        if let Err(error) = result {
            match error.0 {
                sys::CUptiResult::CUPTI_ERROR_MAX_LIMIT_REACHED => break,
                sys::CUptiResult::CUPTI_ERROR_INVALID_KIND => break,
                _ => result?,
            };
        } else {
            handle_record(unsafe { Record::try_from_record_ptr(record) }?)?;
        }
    }

    Ok(())
}
