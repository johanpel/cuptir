//! Support for kernel activity records, obtained by enabling [`crate::activity::Kind::Kernel`] or
//! [`crate::activity::Kind::ConcurrentKernel`].

use cudarc::cupti::sys;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

pub use crate::enums::ChannelType;
pub use crate::enums::FuncShmemLimitConfig;
use crate::error::CuptirError;
use crate::utils::try_demangle_from_ffi;

pub type PartitionedGlobalCacheConfig = crate::enums::ActivityPartitionedGlobalCacheConfig;
pub type LaunchType = crate::enums::ActivityLaunchType;

/// Kernel activity record
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
pub struct Record {
    // TODO: this union:
    // pub cache_config: sys::CUpti_ActivityKernel9__bindgen_ty_1,
    pub shared_memory_config: u8,
    pub registers_per_thread: u16,
    pub partitioned_global_cache_requested: PartitionedGlobalCacheConfig,
    pub partitioned_global_cache_executed: PartitionedGlobalCacheConfig,
    pub start: super::Timestamp,
    pub end: super::Timestamp,
    pub completed: super::Timestamp,
    pub device_id: super::DeviceId,
    pub context_id: super::ContextId,
    pub stream_id: super::StreamId,
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
    pub correlation_id: super::CorrelationId,
    pub grid_id: i64,
    pub name: Option<String>,
    pub queued: Option<super::Timestamp>,
    pub submitted: Option<super::Timestamp>,
    pub launch_type: LaunchType,
    pub is_shared_memory_carveout_requested: bool,
    pub shared_memory_carveout_requested: u8,
    // pub padding: u8,
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

impl TryFrom<&sys::CUpti_ActivityKernel9> for Record {
    type Error = CuptirError;

    fn try_from(value: &sys::CUpti_ActivityKernel9) -> Result<Self, Self::Error> {
        Ok(Record {
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
            launch_type: (value.launchType as u32).try_into()?,
            is_shared_memory_carveout_requested: value.isSharedMemoryCarveoutRequested > 0,
            shared_memory_carveout_requested: value.sharedMemoryCarveoutRequested,
            // padding: value.padding,
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
