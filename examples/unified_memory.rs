use cuptir::{
    Context,
    activity::{self, uvm},
    error::CuptirError,
};

fn setup() -> Result<Context, CuptirError> {
    cuptir::ContextBuilder::new()
        .with_activity(
            activity::Builder::new()
                .with_unified_memory_counter_configs(
                    // Enable all possible unified memory counters.
                    [
                        uvm::CounterKind::BytesTransferHtod,
                        uvm::CounterKind::BytesTransferDtoh,
                        uvm::CounterKind::BytesTransferDtod,
                        uvm::CounterKind::CpuPageFaultCount,
                        uvm::CounterKind::GpuPageFault,
                        uvm::CounterKind::RemoteMap,
                        uvm::CounterKind::Thrashing,
                        uvm::CounterKind::Throttling,
                    ]
                    .into_iter()
                    .map(|kind| uvm::CounterConfig {
                        scope: uvm::CounterScope::ProcessSingleDevice,
                        kind,
                        device_id: 0,
                        enable: true,
                    }),
                )
                .with_record_buffer_handler(move |buffer| {
                    buffer.into_iter().try_for_each(|record| {
                        match record {
                            Ok(rec) => println!("{rec:?}"),
                            Err(e) => tracing::warn!("erroneous record: {e}"),
                        }
                        Ok::<(), cuptir::error::CuptirError>(())
                    })?;
                    Ok(())
                }),
        )
        .build()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cuptir = setup()?;

    let context = cudarc::driver::CudaContext::new(0)?;
    let stream = context.default_stream();

    // This must be done after driver initialization.
    cuptir.activity_enable_unified_memory_counters()?;

    const SIZE: usize = 4096;

    // Cause host-to-device memory page migrations
    {
        let mut slice = unsafe { context.alloc_unified::<u8>(SIZE, true) }?;
        let host_slice = slice.as_mut_slice()?;
        // Write the data on the host.
        host_slice.fill(42);
        // Write the data on the device.
        stream.memset_zeros(&mut slice)?;
        stream.synchronize()?;
    }

    cuptir.activity_flush_all(false)?;
    println!("{}", "-".repeat(42));

    // Cause device-to-host memory page migrations
    {
        let mut slice = unsafe { context.alloc_unified::<u8>(SIZE, true) }?;
        // Set the the slice  to zeros on the device.
        stream.memset_zeros(&mut slice)?;
        stream.synchronize()?;
        // Write some data on the host.
        let host_slice = slice.as_mut_slice()?;
        host_slice.fill(42);
    }

    cuptir.activity_flush_all(true)?;

    Ok(())
}
