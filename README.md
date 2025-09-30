# cuptir - CUPTI for Rust

Experimental safe and easy-to-use Rust wrapper for the CUDA Profiling Tools Interface (CUPTI).

This builds on the [cudarc](https://github.com/coreylowman/cudarc) crate.
I may upstream some parts of this as a `cupti::safe` module there in the future.

# Usage

## Obtaining a kernel record from the Activity API:

```Rust
use cuptir::activity;

let context = cuptir::Context::builder()
    .with_activity(
        activity::Builder::new()
            .with_kinds([activity::Kind::ConcurrentKernel])
            .with_record_buffer_handler(|buffer| {
                buffer
                    .into_iter()
                    .for_each(|record| println!("{:#?}", record.unwrap()));
                Ok(())
            }),
    )
    .build()?;

    // CUDA usage goes here
```

Result (some fields omitted for brevity)

```rust
Kernel(
    Record {
        registers_per_thread: 22,
        start: 1759240421937784359,
        end: 1759240421937787047,
        completed: 1759240421937787047,
        device_id: 0,
        context_id: 1,
        stream_id: 7,
        block_x: 1024,
        block_y: 1,
        block_z: 1,
        local_memory_total: 31850496,
        correlation_id: 2,
        grid_id: 2,
        name: Some(
            "sin_kernel",
        ),
        /// ... and so forth
    },
)
...
```
