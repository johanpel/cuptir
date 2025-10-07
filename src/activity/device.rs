use cudarc::cupti::sys;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::{
    CuptirError,
    utils::{try_str_from_ffi, uuid_from_i8_slice},
};

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
pub struct Record {
    pub has_concurrent_kernels: bool,
    pub global_memory_bandwidth: u64,
    pub global_memory_size: u64,
    pub constant_memory_size: u32,
    pub l2_cache_size: u32,
    pub num_threads_per_warp: u32,
    pub core_clock_rate: u32,
    pub num_memcpy_engines: u32,
    pub num_multiprocessors: u32,
    pub max_ipc: u32,
    pub max_warps_per_multiprocessor: u32,
    pub max_blocks_per_multiprocessor: u32,
    pub max_shared_memory_per_multiprocessor: u32,
    pub max_registers_per_multiprocessor: u32,
    pub max_registers_per_block: u32,
    pub max_shared_memory_per_block: u32,
    pub max_threads_per_block: u32,
    pub max_block_dim_x: u32,
    pub max_block_dim_y: u32,
    pub max_block_dim_z: u32,
    pub max_grid_dim_x: u32,
    pub max_grid_dim_y: u32,
    pub max_grid_dim_z: u32,
    pub compute_capability_major: u32,
    pub compute_capability_minor: u32,
    pub id: u32,
    pub ecc_enabled: u32,
    pub uuid: uuid::Uuid,
    pub name: String,
    pub is_cuda_visible: u8,
    pub is_mig_enabled: u8,
    pub gpu_instance_id: u32,
    pub compute_instance_id: u32,
    pub mig_uuid: uuid::Uuid,
    pub is_numa_node: u32,
    pub numa_id: u32,
}

impl TryFrom<&sys::CUpti_ActivityDevice5> for Record {
    type Error = CuptirError;

    fn try_from(value: &sys::CUpti_ActivityDevice5) -> Result<Self, Self::Error> {
        Ok(Self {
            has_concurrent_kernels: (value.flags as u32)
                & (sys::CUpti_ActivityFlag::CUPTI_ACTIVITY_FLAG_DEVICE_CONCURRENT_KERNELS as u32)
                != 0,
            global_memory_bandwidth: value.globalMemoryBandwidth,
            global_memory_size: value.globalMemorySize,
            constant_memory_size: value.constantMemorySize,
            l2_cache_size: value.l2CacheSize,
            num_threads_per_warp: value.numThreadsPerWarp,
            core_clock_rate: value.coreClockRate,
            num_memcpy_engines: value.numMemcpyEngines,
            num_multiprocessors: value.numMultiprocessors,
            max_ipc: value.maxIPC,
            max_warps_per_multiprocessor: value.maxWarpsPerMultiprocessor,
            max_blocks_per_multiprocessor: value.maxBlocksPerMultiprocessor,
            max_shared_memory_per_multiprocessor: value.maxSharedMemoryPerMultiprocessor,
            max_registers_per_multiprocessor: value.maxRegistersPerMultiprocessor,
            max_registers_per_block: value.maxRegistersPerBlock,
            max_shared_memory_per_block: value.maxSharedMemoryPerBlock,
            max_threads_per_block: value.maxThreadsPerBlock,
            max_block_dim_x: value.maxBlockDimX,
            max_block_dim_y: value.maxBlockDimY,
            max_block_dim_z: value.maxBlockDimZ,
            max_grid_dim_x: value.maxGridDimX,
            max_grid_dim_y: value.maxGridDimY,
            max_grid_dim_z: value.maxGridDimZ,
            compute_capability_major: value.computeCapabilityMajor,
            compute_capability_minor: value.computeCapabilityMinor,
            id: value.id,
            ecc_enabled: value.eccEnabled,
            uuid: uuid_from_i8_slice(value.uuid.bytes),
            name: unsafe { try_str_from_ffi(value.name) }
                .unwrap_or("<null or non-utf8>")
                .to_owned(),
            is_cuda_visible: value.isCudaVisible,
            is_mig_enabled: value.isMigEnabled,
            gpu_instance_id: value.gpuInstanceId,
            compute_instance_id: value.computeInstanceId,
            mig_uuid: uuid_from_i8_slice(value.migUuid.bytes),
            is_numa_node: value.isNumaNode,
            numa_id: value.numaId,
        })
    }
}
