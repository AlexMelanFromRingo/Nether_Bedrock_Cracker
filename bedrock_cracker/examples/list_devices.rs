//! Example: List available GPU devices for OpenCL acceleration
//!
//! Run with: cargo run --features gpu -p bedrock_cracker --example list_devices

#[cfg(feature = "gpu")]
fn main() {
    use bedrock_cracker::gpu::list_gpu_devices;

    println!("GPU Acceleration Status");
    println!("========================");
    println!();

    match list_gpu_devices() {
        Ok(devices) => {
            if devices.is_empty() {
                println!("✗ No GPU devices found (empty list).");
            } else {
                println!("✓ GPU acceleration is available!");
                println!();
                println!("Found {} GPU device(s):", devices.len());
                println!();

                for device in &devices {
                    println!("  Device #{}", device.index);
                    println!("    Name:          {}", device.name);
                    println!("    Vendor:        {}", device.vendor);
                    println!("    Compute Units: {}", device.compute_units);
                    println!("    Memory:        {:.2} GB", device.memory as f64 / (1024.0 * 1024.0 * 1024.0));
                    println!();
                }
            }
        }
        Err(e) => {
            println!("✗ GPU acceleration is NOT available.");
            println!();
            println!("Error: {}", e);
            println!();
            println!("Possible solutions:");
            println!("  - Install OpenCL ICD loader: sudo apt install ocl-icd-libopencl1");
            println!("  - Install NVIDIA OpenCL: sudo apt install nvidia-opencl-icd");
            println!("  - Check /etc/OpenCL/vendors/ for .icd files");
        }
    }
}

#[cfg(not(feature = "gpu"))]
fn main() {
    println!("This example requires the 'gpu' feature.");
    println!("Run with: cargo run --features gpu -p bedrock_cracker --example list_devices");
}
