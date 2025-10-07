//! Support for device activity records, obtained by enabling
//! [`crate::activity::Kind::Environment`].

use cudarc::cupti::sys;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::CuptirError;

pub type ClocksThrottleReason = crate::enums::EnvironmentClocksThrottleReason;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
pub struct Speed {
    pub sm_clock: u32,
    pub memory_clock: u32,
    pub pcie_link_gen: u32,
    pub pcie_link_width: u32,
    pub clocks_throttle_reasons: ClocksThrottleReason,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
pub struct Temperature {
    pub gpu_temperature: u32,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
pub struct Power {
    pub power: u32,
    pub power_limit: u32,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
pub struct Cooling {
    pub fan_speed: u32,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
pub enum Data {
    Speed(Speed),
    Temperature(Temperature),
    Power(Power),
    Cooling(Cooling),
    Unknown,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
pub struct Record {
    pub device_id: super::DeviceId,
    pub timestamp: super::Timestamp,
    pub data: Data,
}

impl TryFrom<&sys::CUpti_ActivityEnvironment> for Record {
    type Error = CuptirError;

    fn try_from(value: &sys::CUpti_ActivityEnvironment) -> Result<Self, Self::Error> {
        use crate::enums::ActivityEnvironmentKind as Kind;
        let kind: Kind = value.environmentKind.try_into()?;
        Ok(Self {
            device_id: value.deviceId,
            timestamp: value.timestamp,
            data: match kind {
                Kind::Unknown => Data::Unknown,
                Kind::Speed => Data::Speed({
                    let speed = unsafe { value.data.speed };
                    Speed {
                        sm_clock: speed.smClock,
                        memory_clock: speed.memoryClock,
                        pcie_link_gen: speed.pcieLinkGen,
                        pcie_link_width: speed.pcieLinkWidth,
                        clocks_throttle_reasons: speed.clocksThrottleReasons.try_into()?,
                    }
                }),
                Kind::Temperature => Data::Temperature(Temperature {
                    gpu_temperature: unsafe { value.data.temperature }.gpuTemperature,
                }),
                Kind::Power => Data::Power({
                    let power = unsafe { value.data.power };
                    Power {
                        power: power.power,
                        power_limit: power.powerLimit,
                    }
                }),
                Kind::Cooling => Data::Cooling(Cooling {
                    fan_speed: unsafe { value.data.cooling }.fanSpeed,
                }),
            },
        })
    }
}
