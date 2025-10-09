//! Support for memcpy activity records, obtained by enabling [`crate::activity::Kind::Memcpy`].

use cudarc::cupti::sys;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::error::CuptirError;

pub type Kind = crate::enums::ActivityMemcpyKind;

/// Memcpy activity record
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
pub struct Record {
    pub copy_kind: Kind,
    pub src_kind: super::memory::Kind,
    pub dst_kind: super::memory::Kind,
    pub is_async: bool,
    pub bytes: u64,
    pub start: super::Timestamp,
    pub end: super::Timestamp,
    pub device_id: super::DeviceId,
    pub context_id: super::ContextId,
    pub stream_id: super::StreamId,
    pub correlation_id: super::CorrelationId,
    pub runtime_correlation_id: super::CorrelationId,
    pub graph_node_id: u64,
    pub graph_id: u32,
    pub channel_id: u32,
    #[cfg(any(feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000"))]
    pub copy_count: u64,
    pub channel_type: super::ChannelType,
}

#[cfg(feature = "cuda-12060")]
impl TryFrom<&sys::CUpti_ActivityMemcpy5> for Record {
    type Error = CuptirError;

    fn try_from(value: &sys::CUpti_ActivityMemcpy5) -> Result<Self, Self::Error> {
        Ok(Self {
            copy_kind: Kind::try_from(value.copyKind as u32)?,
            src_kind: super::memory::Kind::try_from(value.srcKind as u32)?,
            dst_kind: super::memory::Kind::try_from(value.dstKind as u32)?,
            is_async: value.flags
                & sys::CUpti_ActivityFlag::CUPTI_ACTIVITY_FLAG_MEMCPY_ASYNC as u32 as u8
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
        })
    }
}

#[cfg(any(feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000"))]
impl TryFrom<&sys::CUpti_ActivityMemcpy6> for Record {
    type Error = CuptirError;

    fn try_from(value: &sys::CUpti_ActivityMemcpy6) -> Result<Self, Self::Error> {
        Ok(Self {
            copy_kind: Kind::try_from(value.copyKind as u32)?,
            src_kind: super::memory::Kind::try_from(value.srcKind as u32)?,
            dst_kind: super::memory::Kind::try_from(value.dstKind as u32)?,
            is_async: value.flags
                & sys::CUpti_ActivityFlag::CUPTI_ACTIVITY_FLAG_MEMCPY_ASYNC as u32 as u8
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
            copy_count: value.copyCount,
            channel_type: value.channelType.try_into()?,
        })
    }
}
