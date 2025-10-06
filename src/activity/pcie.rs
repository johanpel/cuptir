use cudarc::cupti::sys;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::CuptirError;

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
    pub device_id: cudarc::driver::sys::CUdevice,
    pub node_props: Node,
    // TODO:
    // pub attr: CUpti_ActivityPcie__bindgen_ty_2,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
pub struct BridgeRecord {
    pub bridge_id: u32,
    pub node_props: Node,
    // TODO:
    // pub attr: CUpti_ActivityPcie__bindgen_ty_2,
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
            DeviceType::Gpu => Self::Gpu(GpuRecord {
                device_id: value.id.bindgen_union_field as i32,
                node_props: value.into(),
            }),
            DeviceType::Bridge => Self::Bridge(BridgeRecord {
                bridge_id: value.id.bindgen_union_field,
                node_props: value.into(),
            }),
        })
    }
}
