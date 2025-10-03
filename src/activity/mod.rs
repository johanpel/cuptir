//! Safe wrappers around the CUPTI Activity API
use std::{
    alloc::{Layout, alloc_zeroed, dealloc},
    collections::HashSet,
    num::NonZero,
    sync::RwLock,
};

use cudarc::cupti::{
    result,
    sys::{self},
};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use tracing::{trace, warn};

use crate::{
    callback::callback_name,
    driver,
    enums::{ActivityAttribute, DriverFunc, RuntimeFunc},
    error::CuptirError,
    runtime,
};

pub mod device;
pub mod kernel;
pub mod memcpy;
pub mod memory;
pub mod memory_pool;
pub mod pcie;
pub mod uvm;

pub use crate::enums::ChannelType;
pub type Kind = crate::enums::ActivityKind;

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
    /// This resulting [`RecordBuffer`] takes ownership over the buffer and will
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
        unsafe { buffer_free(self.ptr, self.size) }
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

/// An iterator over activity [`Record`]s in a [`RecordBuffer`].
pub struct RecordBufferIterator {
    buffer: RecordBuffer,
    current_record_ptr: *mut sys::CUpti_Activity,
}

impl Iterator for RecordBufferIterator {
    type Item = Result<Record, CuptirError>;

    fn next(&mut self) -> Option<Self::Item> {
        // Safety: the buffer can only be constructed with a non-null ptr, plus CUPTI would return a
        // graceful error if it were null.
        let result: Result<(), result::CuptiError> = unsafe {
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

/// Type of callback function used to handle [`RecordBuffer`]s.
pub type RecordBufferHandlerFn =
    dyn Fn(RecordBuffer) -> Result<(), Box<dyn std::error::Error + Send + Sync>> + Send + Sync;

/// Globally accessible callback to handle activity record buffers.
///
/// Because the buffer complete callback doesn't have a way of passing custom data, e.g. a reference
/// to the Context, this is needed to get to the Rust callback for record processing. This is a
/// RwLock because this allows dropping e.g. captured [`std::sync::Arc`]s such that after the
/// [`Context`] drops, inner values can be taken out of the Arc.
pub(crate) static RECORD_BUFFER_HANDLER: RwLock<Option<Box<RecordBufferHandlerFn>>> =
    RwLock::new(None);

/// Sets the activity record handler.
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
        warn!("activity record buffer received, but no handler is installed");
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
pub type ContextId = u32;

/// Properties shared across of [`DriverApiRecord`] and [`RuntimeApiRecord`] activity records.
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

/// An activity record.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
pub enum Record {
    /// A record from a device, enabled through [`Kind::Device`].
    Device(device::Record),
    /// A record from the CUDA driver API, enabled through [`Kind::Driver`].
    DriverApi(DriverApiRecord),
    /// A record from the CUDA runtime API, enabled through [`Kind::Runtime`].
    RuntimeApi(RuntimeApiRecord),
    /// A record from the CUDA internal launch API, enabled through [`Kind::InternalLaunchApi`].
    InternalLaunchApi(InternalLaunchApiRecord),
    /// A record from a CUDA kernel, enabled through [`Kind::ConcurrentKernel`] or [`Kind::Kernel`].
    ///
    /// # Performance notes
    /// It is highly recommended to utilize [`Kind::ConcurrentKernel`].
    /// [`Kind::Kernel`] causes GPU kernel execution to be serialized.
    Kernel(kernel::Record),
    /// A record from a memory copy operation, enabled through [`Kind::Memcpy`].
    Memcpy(memcpy::Record),
    /// A record from a memory operation (allocation and release), enabled through
    /// [`Kind::Memory2`].
    Memory(memory::Record),
    /// A record from a memory pool operation (creation, trimmed, destroyed), enabled through
    /// [`Kind::MemoryPool`].
    MemoryPool(memory_pool::Record),
    /// A record from a unified memory operation (including page migrations), enabled through
    /// [`Kind::UnifiedMemoryCounter`].
    UnifiedMemoryCounter(uvm::CounterRecord),
    /// A PCIE record
    Pcie(pcie::Record),
}

impl Record {
    unsafe fn try_from_record_ptr(
        record_ptr: *mut sys::CUpti_Activity,
    ) -> Result<Self, CuptirError> {
        if record_ptr.is_null() {
            return Err(CuptirError::NullPointer);
        }

        // Safety: null check is done at the start of this function, so record
        // dereferences should be safe.
        let kind = unsafe { *record_ptr }.kind;
        match kind {
            sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_DEVICE => {
                let device_record = unsafe { &*(record_ptr as *const sys::CUpti_ActivityDevice5) };
                Ok(Record::Device(device_record.try_into()?))
            }
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
            sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_PCIE => {
                let pcie_record = unsafe { &*(record_ptr as *const sys::CUpti_ActivityPcie) };
                Ok(Record::Pcie(pcie_record.try_into()?))
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
    // TODO: consider providing buffers from a pool
    trace!("buffer requested");
    let layout = Layout::from_size_align(CUPTI_BUFFER_SIZE, CUPTI_BUFFER_ALIGN).unwrap();
    unsafe {
        // Safety: after usage by CUPTI, ownership of the memory allocated here is transferred back
        // to the RecordBuffer constructed in [buffer_complete_callback], and freed when the
        // RecordBuffer is dropped.
        let ptr = alloc_zeroed(layout);
        *buffer = ptr;
        *size = CUPTI_BUFFER_SIZE;
        *max_num_records = 0; // means: fill this with as many records as possible
    }
}

pub(crate) unsafe fn buffer_free(ptr: *mut u8, size: usize) {
    let layout = Layout::from_size_align(size, CUPTI_BUFFER_ALIGN).unwrap();
    unsafe {
        dealloc(ptr, layout);
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
    // Quickly return if the buffer is empty.
    if valid_size == 0 {
        return unsafe { buffer_free(buffer, size) };
    }
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
    latency_timestamps: bool,
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

    pub(crate) fn enable_unified_memory_counters(&mut self) -> Result<(), CuptirError> {
        if !self.unified_memory_counter_configs.is_empty() {
            trace!("enabling activity records for unified memory counters");
            unsafe {
                sys::cuptiActivityConfigureUnifiedMemoryCounter(
                    self.unified_memory_counter_configs.as_ptr() as *mut _,
                    self.unified_memory_counter_configs.len() as u32,
                )
            }
            .result()?;
            result::activity::enable(Kind::UnifiedMemoryCounter.into())?;
            self.enabled_kinds.push(Kind::UnifiedMemoryCounter);
            Ok(())
        } else {
            Err(CuptirError::Activity("enabling activity records for unified memory counters requires enabling the activity kind and supplying counter configurations".into()))
        }
    }

    pub(crate) fn enable_hardware_tracing(&mut self) -> Result<(), CuptirError> {
        if !self.latency_timestamps {
            trace!("enabling hardware tracing for activity records");
            result::activity::enable_hw_trace(1u8)?;
            Ok(())
        } else {
            // See: https://docs.nvidia.com/cupti/main/main.html#hardware-event-system-hes
            Err(CuptirError::Activity(
                "hardware tracing and latency timestamps are incompatible".into(),
            ))
        }
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        // Ideally we want to disable the unified memory counters here by setting enable to zero as
        // shown below, but this results in CUPTI_ERROR_INVALID_OPERATION. It is not clear from the
        // CUPTI docs why this isn't allowed.

        // if !self.unified_memory_counter_configs.is_empty() {
        //     trace!(
        //         "disabling
        //     activity record collection for unified memory counters"
        //     );
        //     self.unified_memory_counter_configs
        //         .iter_mut()
        //         .for_each(|counter| counter.enable = 0);
        //     if let Err(error) = unsafe {
        //         sys::cuptiActivityConfigureUnifiedMemoryCounter(
        //             self.unified_memory_counter_configs.as_ptr() as *mut _,
        //             self.unified_memory_counter_configs.len() as u32,
        //         )
        //     }
        //     .result()
        //     {
        //         warn!("unable to disable unified memory counters: {error}");
        //     }
        // }

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

        if let Err(error) = Self::flush_all(true) {
            warn!("unable to flush activity buffer: {error}");
        }

        // Unset the record buffer handler function. We would ideally tell CUPTI to stop
        // using the buffer completion callbacks by resetting them to nullptrs or
        // something, or through some more explicit API for it, but that does not exist,
        // so we will handle more records coming in somehow in [handle_record_buffer].
        trace!("resetting activity record buffer handler");
        if let Err(e) = set_record_buffer_handler(None) {
            warn!("unable to reset activity record buffer handler: {e}");
        }

        if let Err(e) = result::finalize() {
            warn!("unable to finalize cupti {e}");
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

    device_buffer_size: Option<usize>,
    device_buffer_pool_limit: Option<usize>,

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
    /// `submitted` fields for [`kernel::Record`] event records.
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

    /// Set the device buffer size for activity records.
    ///
    /// See [CUPTI docs](https://docs.nvidia.com/cupti/api/group__CUPTI__ACTIVITY__API.html?highlight=CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_POOL_LIMIT#_CPPv4N23CUpti_ActivityAttribute38CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_SIZEE).
    pub fn device_buffer_size(mut self, size: Option<usize>) -> Self {
        self.device_buffer_size = size;
        self
    }

    /// Set the device buffer pool limit.
    ///
    /// See [CUPTI docs](https://docs.nvidia.com/cupti/api/group__CUPTI__ACTIVITY__API.html?highlight=CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_POOL_LIMIT#_CPPv4N23CUpti_ActivityAttribute44CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_POOL_LIMITE).
    pub fn device_buffer_pool_limit(mut self, limit: Option<usize>) -> Self {
        self.device_buffer_pool_limit = limit;
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

                // We're going to provide zero-initialized buffers. Tell CUPTI so it doesn't have to
                // zero out buffers in its main thread (this is a perf consideration described in
                // CUPTI docs).
                unsafe {
                    result::activity::set_attribute(
                        ActivityAttribute::ZeroedOutActivityBuffer.into(),
                        &mut size_of::<u8>(),
                        &mut 1u8 as *mut _ as *mut std::ffi::c_void,
                    )
                }?;

                // Use a per-thread activity buffer. This is the default in CUDA 13 and onwards,
                // but not yet for CUDA 12, so ensure this is on.
                unsafe {
                    result::activity::set_attribute(
                        ActivityAttribute::PerThreadActivityBuffer.into(),
                        &mut size_of::<u8>(),
                        &mut 1u8 as *mut _ as *mut std::ffi::c_void,
                    )
                }?;

                if let Some(mut value) = self.device_buffer_size {
                    unsafe {
                        result::activity::set_attribute(
                            ActivityAttribute::PerThreadActivityBuffer.into(),
                            &mut size_of::<usize>(),
                            &mut value as *mut _ as *mut std::ffi::c_void,
                        )
                    }?;
                }

                if let Some(mut value) = self.device_buffer_pool_limit {
                    unsafe {
                        result::activity::set_attribute(
                            ActivityAttribute::PerThreadActivityBuffer.into(),
                            &mut size_of::<usize>(),
                            &mut value as *mut _ as *mut std::ffi::c_void,
                        )
                    }?;
                }

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
                latency_timestamps: self.latency_timestamps,
            }))
        } else if self.record_buffer_handler.is_some() {
            Err(CuptirError::Builder("An activity record buffer handler is installed but no activity kind or driver/runtime API functions activity is enabled".into()))
        } else {
            Ok(None)
        }
    }
}

#[cfg(test)]
pub(crate) mod test {
    use std::sync::{Arc, Mutex};

    use cudarc::driver::CudaContext;
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
        F: Fn(&mut Context, Arc<CudaContext>, Arc<Mutex<Vec<Record>>>) -> TestResult,
    {
        // Flush anything that may somehow have ended up in the buffer.
        Context::flush_all(true)?;

        // We can initialize the Activity API at any time, so ensure there is a CUDA
        // context attached to this thread.
        let cuda_context = cudarc::driver::CudaContext::new(0)?;

        let records: Arc<Mutex<Vec<Record>>> = Arc::new(Mutex::new(vec![]));
        let records_cb = Arc::clone(&records);

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

        func(&mut activity, cuda_context, Arc::clone(&records))?;

        drop(activity);
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
    pub fn record_nullptr_deref_errors_out() {
        assert!(unsafe { Record::try_from_record_ptr(std::ptr::null_mut()) }.is_err())
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
            |_, _, _| {
                // Do a driver thing.
                let mut ptr: cudarc::driver::sys::CUdeviceptr = 0;
                unsafe {
                    cudarc::driver::sys::cuMemAlloc_v2(&mut ptr, 1).result()?;
                    cudarc::driver::sys::cuMemFree_v2(ptr).result()?;
                }
                // Do a runtime thing. This causes a large amount driver records to be generated
                // probably as part of lazy runtime initialization.
                let mut ptr: *mut core::ffi::c_void = std::ptr::null_mut();
                unsafe {
                    cudarc::runtime::sys::cudaMalloc(&mut ptr, 1).result()?;
                    cudarc::runtime::sys::cudaFree(ptr).result()?;
                }

                Ok(())
            },
        )?;

        let mut num_cuda_malloc = 0;
        let mut num_cuda_free = 0;
        let mut num_cu_mem_alloc = 0;
        let mut num_cu_mem_free = 0;

        for rec in recs.iter() {
            match rec {
                Record::RuntimeApi(r) => {
                    if r.function == RuntimeFunc::cudaMalloc_v3020 {
                        assert_eq!(r.function_name().unwrap(), "cudaMalloc_v3020");
                        num_cuda_malloc += 1;
                    }
                    if r.function == RuntimeFunc::cudaFree_v3020 {
                        assert_eq!(r.function_name().unwrap(), "cudaFree_v3020");
                        num_cuda_free += 1;
                    }
                }
                Record::DriverApi(d) => {
                    if d.function == DriverFunc::cuMemAlloc_v2 {
                        assert_eq!(d.function_name().unwrap(), "cuMemAlloc_v2");
                        num_cu_mem_alloc += 1;
                    }
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
        assert_eq!(num_cuda_free, 1);
        assert_eq!(num_cu_mem_alloc, 1);
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
            |_, _, _| run_a_kernel(),
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
    fn memory_allocate_and_copy() -> TestResult {
        const SIZE: u64 = 1337;

        let mut recs = get_records(
            Builder::new().with_kinds([Kind::Memcpy, Kind::Memory2]),
            |_, cuda_context, _| {
                // Round-trip some bytes.
                let stream = cuda_context.default_stream();
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

        // Even when running with #[serial], disabling memory pool records in previous tests, and
        // not re-enabling in this test, somehow we can end up with those records when running this
        // together with other tests. Filter them out.
        recs = recs
            .into_iter()
            .filter(|rec| matches!(rec, Record::Memory(_) | Record::Memcpy(_)))
            .collect::<Vec<_>>();
        assert_eq!(recs.len(), 4);

        if let Record::Memory(alloc) = &recs[0] {
            assert_eq!(alloc.bytes, SIZE);
            assert_eq!(alloc.memory_kind, memory::Kind::Device);
            assert_eq!(
                alloc.memory_operation_type,
                memory::OperationType::Allocation
            );
        } else {
            panic!();
        }
        if let Record::Memcpy(h2d) = &recs[1] {
            assert_eq!(h2d.bytes, SIZE);
            assert_eq!(h2d.copy_kind, memcpy::Kind::Htod);
            assert!(h2d.is_async);
        } else {
            panic!();
        }
        if let Record::Memcpy(d2h) = &recs[2] {
            assert_eq!(d2h.bytes, SIZE);
            assert_eq!(d2h.copy_kind, memcpy::Kind::Dtoh);
            assert!(d2h.is_async);
        } else {
            panic!();
        }
        if let Record::Memory(free) = &recs[3] {
            assert_eq!(free.bytes, SIZE);
            assert_eq!(free.memory_kind, memory::Kind::Device);
            assert_eq!(free.memory_operation_type, memory::OperationType::Release);
        } else {
            panic!();
        }

        Ok(())
    }

    #[test]
    #[serial]
    fn memory_unified() -> TestResult {
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
            |activity, _, _| {
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

        // Without completely unraveling the implementation details of UVM by checking every record,
        // check whether we observe at least one page fault on host and gpu and at least one
        // migration in both directions, and that we went the same amount of times in either
        // direction, with the same amount of bytes.
        let mut num_page_faults_cpu = 0;
        let mut num_page_faults_gpu = 0;
        let mut num_migrations_h2d = 0;
        let mut num_h2d_bytes = 0;
        let mut num_migrations_d2h = 0;
        let mut num_d2h_bytes = 0;

        for rec in recs.into_iter() {
            if let Record::UnifiedMemoryCounter(counter_record) = rec {
                match counter_record {
                    uvm::CounterRecord::BytesTransferHtoD(transfer) => {
                        num_migrations_h2d += 1;
                        num_h2d_bytes += transfer.memory_region_bytes;
                    }
                    uvm::CounterRecord::BytesTransferDtoH(transfer) => {
                        num_migrations_d2h += 1;
                        num_d2h_bytes += transfer.memory_region_bytes;
                    }
                    uvm::CounterRecord::CpuPageFaultCount => num_page_faults_cpu += 1,
                    uvm::CounterRecord::GpuPageFault => num_page_faults_gpu += 1,
                    _ => (),
                }
            }
        }

        assert!(num_page_faults_cpu > 1);
        assert!(num_page_faults_gpu > 1);
        assert!(num_migrations_h2d > 1);
        assert!(num_migrations_d2h > 1);
        assert_eq!(num_page_faults_cpu, num_page_faults_gpu);
        assert_eq!(num_migrations_h2d, num_migrations_d2h);
        assert_eq!(num_h2d_bytes, num_d2h_bytes);

        Ok(())
    }

    #[test]
    #[serial]
    fn memory_pool() -> TestResult {
        // Constants must all be powers of 2.
        // A small pool size is going to be overridden by some minimum pool size in the
        // implementation, so we need to make it sufficiently large. There seems to be no
        // documentation on how this minimum is defined, so this test may fail if the implementation
        // changes (its 32 MiB on my machine :tm:).
        const POOL_SIZE: usize = 128 * 1024 * 1024;
        const NUM_ALLOCS: usize = 8;
        const ALLOC_SIZE: usize = POOL_SIZE / NUM_ALLOCS;

        let recs = get_records(
            // Even though we're not interested in the alloc/free which are captured using Memory2
            // records, it seems like CUPTI does not produce any memory pool records when only
            // MemoryPool is used.
            Builder::new().with_kinds([Kind::MemoryPool, Kind::Memory2]),
            |_, cuda, recs| {
                use cudarc::runtime::sys;

                let stream = cuda.default_stream();
                // work-around for duplicate definitions in cudarc
                let stream_rt = unsafe {
                    std::mem::transmute::<
                        *mut cudarc::driver::sys::CUstream_st,
                        *mut cudarc::runtime::sys::CUstream_st,
                    >(stream.cu_stream())
                };

                let mut memory_pools_supported: i32 = 0;
                unsafe {
                    sys::cudaDeviceGetAttribute(
                        &mut memory_pools_supported,
                        sys::cudaDeviceAttr::cudaDevAttrMemoryPoolsSupported,
                        cuda.ordinal() as i32,
                    )
                }
                .result()?;
                assert!(memory_pools_supported > 0);

                let props = sys::cudaMemPoolProps {
                    allocType: sys::cudaMemAllocationType::cudaMemAllocationTypePinned,
                    handleTypes: sys::cudaMemAllocationHandleType::cudaMemHandleTypeNone,
                    location: sys::cudaMemLocation {
                        type_: sys::cudaMemLocationType::cudaMemLocationTypeDevice,
                        id: cuda.cu_device(),
                    },
                    win32SecurityAttributes: std::ptr::null_mut(),
                    maxSize: POOL_SIZE,
                    usage: 0,
                    reserved: [0; 54],
                };

                // Note on CUDA implementaion details (speculative):
                //
                // The first cudaMemPoolCreate does not generate an activity record for creating the
                // pool. The second cudaMemPoolCreate only generates a record when the first pool
                // was actually used. This is likely an optimization in the implementation.

                // First create a dummy pool and use it, delete all records of it and then create a
                // second pool so we can focus the test on records from just this second pool.
                let mut dummy_pool: sys::cudaMemPool_t = std::ptr::null_mut();
                let mut dummy_ptr: *mut ::core::ffi::c_void = std::ptr::null_mut();
                unsafe {
                    sys::cudaMemPoolCreate(&mut dummy_pool, &props).result()?;
                    sys::cudaMallocFromPoolAsync(&mut dummy_ptr, 1, dummy_pool, stream_rt)
                        .result()?;
                    sys::cudaFreeAsync(dummy_ptr, stream_rt).result()?;
                    sys::cudaMemPoolDestroy(dummy_pool).result()?;
                }
                stream.synchronize()?;
                Context::flush_all(true)?;
                recs.lock().unwrap().clear();

                let mut pool: sys::cudaMemPool_t = std::ptr::null_mut();
                unsafe { sys::cudaMemPoolCreate(&mut pool, &props) }.result()?;
                let pool_size = POOL_SIZE;
                unsafe {
                    sys::cudaMemPoolSetAttribute(
                        pool,
                        sys::cudaMemPoolAttr::cudaMemPoolAttrReleaseThreshold,
                        &pool_size as *const usize as *mut core::ffi::c_void,
                    )
                }
                .result()?;

                // Allocate the entire pool.
                let mut ptrs = [std::ptr::null_mut(); NUM_ALLOCS];
                for ptr in ptrs.iter_mut().take(NUM_ALLOCS) {
                    unsafe { sys::cudaMallocFromPoolAsync(ptr, ALLOC_SIZE, pool, stream_rt) }
                        .result()?;
                }

                // Note on CUDA implementation details (speculative):
                // A stream syncrhonize will trigger a memory pool trim IF there were releases/frees
                // on the stream since last synchronize. This happens for EVERY pool regardless of
                // which ones you freed memory for. This is true EVEN when there is nothing to trim
                // , e.g. when setting a release threshold to the maximum pool size (which means it
                // should never trim).

                // This means the following synchronize will NOT produce a trim record, since we
                // haven't released anything yet.
                stream.synchronize()?;

                // Trim 1: Perform a manual trim and attempt to trim to half the pool size.
                unsafe { sys::cudaMemPoolTrimTo(pool, POOL_SIZE / 2) }.result()?;

                // Release half of the pool allocations.
                for ptr in ptrs.iter().take(NUM_ALLOCS / 2) {
                    unsafe { sys::cudaFreeAsync(*ptr, stream_rt) }.result()?;
                }

                // Trim 2: When we now synchronize, the first trim record is produced. Even though
                // there is a record, no trimming should take place since we've set the release
                // threshold to the pool size.
                stream.synchronize()?;

                // Trim 3: Now perform a manual trim and attempt to trim to a quarter of the pool
                // size.
                unsafe { sys::cudaMemPoolTrimTo(pool, POOL_SIZE / 4) }.result()?;

                // Release the other half of the pool allocations.
                for ptr in ptrs.iter().take(NUM_ALLOCS).skip(NUM_ALLOCS / 2) {
                    unsafe { sys::cudaFreeAsync(*ptr, stream_rt) }.result()?;
                }

                // Destroy the pool. This doesn't create a trim record.
                unsafe { sys::cudaMemPoolDestroy(pool) }.result()?;

                // Synchronize, this doesn't create a trim record after destroying the pool.
                stream.synchronize()?;

                Ok(())
            },
        )?;

        // If you're not running this test on its own, other memory pools may exist that produce
        // activity records upon synchronization, despite running this with #[serial]. Figure out
        // which pool is the one that belongs to this test by checking the creation event record's
        // address.
        let this_test_pool_address = recs
            .iter()
            .find_map(|rec| {
                if let Record::MemoryPool(r) = rec {
                    matches!(
                        r.memory_pool_operation_type,
                        memory_pool::OperationType::Created
                    )
                    .then_some(r.address)
                } else {
                    None
                }
            })
            .unwrap();

        let mut num_created = 0;
        let mut num_trimmed = 0;
        let mut num_destroyed = 0;

        for r in recs.into_iter().filter_map(|rec| {
            if let Record::MemoryPool(mem_pool_rec) = rec
                && mem_pool_rec.address == this_test_pool_address
            {
                Some(mem_pool_rec)
            } else {
                None
            }
        }) {
            match r.memory_pool_operation_type {
                crate::enums::ActivityMemoryPoolOperationType::Created => {
                    num_created += 1;
                    assert_eq!(r.memory_pool_type, memory_pool::PoolType::Local);
                    assert_eq!(r.utilized_size, Some(0));
                    assert_eq!(r.release_threshold, Some(0));
                    // We can't check the initial size, this is implementation-defined.
                }
                crate::enums::ActivityMemoryPoolOperationType::Trimmed => {
                    num_trimmed += 1;
                    if num_trimmed == 1 {
                        // First manual trim. We should be using the entire pool.
                        assert_eq!(r.memory_pool_type, memory_pool::PoolType::Local);
                        assert_eq!(r.release_threshold, Some(POOL_SIZE as u64));
                        assert_eq!(r.utilized_size, Some(POOL_SIZE as u64));
                        assert_eq!(r.size, Some(POOL_SIZE as u64));
                    } else if num_trimmed == 2 {
                        // Trim due to synchronize, we should see half the pool being used.
                        assert_eq!(r.memory_pool_type, memory_pool::PoolType::Local);
                        assert_eq!(r.release_threshold, Some(POOL_SIZE as u64));
                        assert_eq!(r.utilized_size, Some((POOL_SIZE / 2) as u64));
                        assert_eq!(r.size, Some(POOL_SIZE as u64));
                    } else if num_trimmed == 3 {
                        // Manual trim, we should see half the pool being used, and it having
                        // trimmed down to exactly its usage.
                        assert_eq!(r.memory_pool_type, memory_pool::PoolType::Local);
                        assert_eq!(r.release_threshold, Some(POOL_SIZE as u64));
                        assert_eq!(r.utilized_size, r.size);
                    } else {
                        panic!("unexpected fourth trim record")
                    }
                }
                crate::enums::ActivityMemoryPoolOperationType::Destroyed => {
                    num_destroyed += 1;
                }
            }
        }

        assert_eq!(num_created, 1);
        assert_eq!(num_trimmed, 3);
        assert_eq!(num_destroyed, 1);

        Ok(())
    }
}
