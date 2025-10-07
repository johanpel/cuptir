//! Support for pcie activity records, obtained by enabling [`crate::activity::Kind::Pcie`].

use cudarc::{cupti::sys, driver};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::{CuptirError, utils::uuid_from_i8_slice};

pub type DeviceType = crate::enums::PcieDeviceType;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
pub struct Node {
    pub domain: u32,
    pub pcie_generation: u16,
    pub link_rate: u16,
    pub link_width: u16,
    pub upstream_bus: u16,
}

impl From<&sys::CUpti_ActivityPcie> for Node {
    fn from(value: &sys::CUpti_ActivityPcie) -> Self {
        Self {
            domain: value.domain,
            pcie_generation: value.pcieGeneration,
            link_rate: value.linkRate,
            link_width: value.linkWidth,
            upstream_bus: value.upstreamBus,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
pub struct GpuRecord {
    pub device_id: driver::sys::CUdevice,
    pub node_props: Node,
    pub device_uuid: Uuid,
    pub peer_devices: [driver::sys::CUdevice; 32],
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
pub struct BridgeRecord {
    pub bridge_id: u32,
    pub node_props: Node,
    pub secondary_bus: u16,
    pub device_id: u16,
    pub vendor_id: u16,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
pub enum Record {
    Gpu(GpuRecord),
    Bridge(BridgeRecord),
}

impl TryFrom<&sys::CUpti_ActivityPcie> for Record {
    type Error = CuptirError;

    fn try_from(value: &sys::CUpti_ActivityPcie) -> Result<Self, Self::Error> {
        Ok(match value.type_.try_into()? {
            DeviceType::Gpu => Self::Gpu({
                let gpu_attr = unsafe { value.attr.gpuAttr.as_ref() };
                GpuRecord {
                    device_id: *unsafe { value.id.devId.as_ref() } as driver::sys::CUdevice,
                    node_props: value.into(),
                    device_uuid: uuid_from_i8_slice(gpu_attr.uuidDev.bytes),
                    peer_devices: gpu_attr.peerDev,
                }
            }),
            DeviceType::Bridge => Self::Bridge({
                let bridge_attr = unsafe { value.attr.bridgeAttr.as_ref() };
                BridgeRecord {
                    bridge_id: *unsafe { value.id.bridgeId.as_ref() },
                    node_props: value.into(),
                    secondary_bus: bridge_attr.secondaryBus,
                    device_id: bridge_attr.deviceId,
                    vendor_id: bridge_attr.vendorId,
                }
            }),
        })
    }
}
