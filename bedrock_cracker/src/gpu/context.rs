use opencl3::command_queue::{CommandQueue, CL_QUEUE_PROFILING_ENABLE};
use opencl3::context::Context;
use opencl3::device::{get_all_devices, Device, CL_DEVICE_TYPE_GPU};
use opencl3::kernel::{ExecuteKernel, Kernel};
use opencl3::memory::{Buffer, CL_MEM_READ_ONLY, CL_MEM_READ_WRITE, CL_MEM_WRITE_ONLY};
use opencl3::program::Program;
use opencl3::types::{cl_ulong, cl_uint, CL_BLOCKING};
use std::ptr;

use super::error::GpuError;
use super::GpuCheckObject;

/// OpenCL kernel source embedded at compile time
const KERNEL_SOURCE: &str = include_str!("kernels/bedrock.cl");

/// Maximum number of candidates that can be found in a single batch
pub const MAX_CANDIDATES: usize = 1 << 16; // 64K candidates max per batch

/// GPU context holding OpenCL resources
pub struct GpuContext {
    pub(crate) context: Context,
    pub(crate) queue: CommandQueue,
    pub(crate) device: Device,
    pub(crate) kernel: Kernel,
    #[allow(dead_code)]
    pub(crate) kernel_batched: Kernel,
    pub(crate) checks_buffer: Option<Buffer<GpuCheckObject>>,
    pub(crate) candidates_buffer: Buffer<cl_ulong>,
    pub(crate) count_buffer: Buffer<cl_uint>,
}

impl GpuContext {
    /// Create a new GPU context for the specified device index
    pub fn new(device_index: usize) -> Result<Self, GpuError> {
        // Get all GPU devices
        let device_ids = get_all_devices(CL_DEVICE_TYPE_GPU)
            .map_err(|e| GpuError::DeviceNotFound(format!("Failed to enumerate GPU devices: {:?}", e)))?;

        if device_ids.is_empty() {
            return Err(GpuError::DeviceNotFound("No GPU devices found".to_string()));
        }

        if device_index >= device_ids.len() {
            return Err(GpuError::DeviceNotFound(format!(
                "Device index {} out of range (found {} devices)",
                device_index,
                device_ids.len()
            )));
        }

        let device_id = device_ids[device_index];
        let device = Device::new(device_id);

        // Create context
        let context = Context::from_device(&device)
            .map_err(|e| GpuError::ContextCreation(format!("{:?}", e)))?;

        // Create command queue with profiling enabled for benchmarking
        let queue = CommandQueue::create_default_with_properties(
            &context,
            CL_QUEUE_PROFILING_ENABLE,
            0,
        )
        .map_err(|e| GpuError::QueueCreation(format!("{:?}", e)))?;

        // Build program from kernel source
        let program = Program::create_and_build_from_source(&context, KERNEL_SOURCE, "")
            .map_err(|e| GpuError::KernelCompilation(format!("{:?}", e)))?;

        // Create kernels
        let kernel = Kernel::create(&program, "bedrock_filter")
            .map_err(|e| GpuError::KernelCreation(format!("{:?}", e)))?;

        let kernel_batched = Kernel::create(&program, "bedrock_filter_batched")
            .map_err(|e| GpuError::KernelCreation(format!("{:?}", e)))?;

        // Create output buffers
        let candidates_buffer = unsafe {
            Buffer::<cl_ulong>::create(&context, CL_MEM_WRITE_ONLY, MAX_CANDIDATES, ptr::null_mut())
                .map_err(|e| GpuError::BufferCreation(format!("{:?}", e)))?
        };

        let count_buffer = unsafe {
            Buffer::<cl_uint>::create(&context, CL_MEM_READ_WRITE, 1, ptr::null_mut())
                .map_err(|e| GpuError::BufferCreation(format!("{:?}", e)))?
        };

        Ok(Self {
            context,
            queue,
            device,
            kernel,
            kernel_batched,
            checks_buffer: None,
            candidates_buffer,
            count_buffer,
        })
    }

    /// Upload check objects to GPU memory
    pub fn upload_checks(&mut self, checks: &[GpuCheckObject]) -> Result<(), GpuError> {
        if checks.is_empty() {
            return Err(GpuError::InvalidInput("No checks provided".to_string()));
        }

        // Create buffer for checks
        let mut checks_buffer = unsafe {
            Buffer::<GpuCheckObject>::create(
                &self.context,
                CL_MEM_READ_ONLY,
                checks.len(),
                ptr::null_mut(),
            )
            .map_err(|e| GpuError::BufferCreation(format!("{:?}", e)))?
        };

        // Upload checks to GPU
        unsafe {
            self.queue
                .enqueue_write_buffer(&mut checks_buffer, CL_BLOCKING, 0, checks, &[])
                .map_err(|e| GpuError::BufferWrite(format!("{:?}", e)))?;
        }

        self.checks_buffer = Some(checks_buffer);
        Ok(())
    }

    /// Execute the bedrock filter kernel
    pub fn execute_filter(
        &mut self,
        start_seed: u64,
        batch_size: u32,
        num_checks: u32,
    ) -> Result<Vec<u64>, GpuError> {
        let checks_buffer = self
            .checks_buffer
            .as_ref()
            .ok_or_else(|| GpuError::InvalidInput("Checks not uploaded".to_string()))?;

        // Reset candidate count to 0
        let zero: cl_uint = 0;
        unsafe {
            self.queue
                .enqueue_write_buffer(&mut self.count_buffer, CL_BLOCKING, 0, &[zero], &[])
                .map_err(|e| GpuError::BufferWrite(format!("{:?}", e)))?;
        }

        // Calculate work group size
        // Get max work group size from device (more reliable than kernel query)
        let max_work_group_size = self.device.max_work_group_size().unwrap_or(256);

        let local_work_size = max_work_group_size.min(256);
        let global_work_size = ((batch_size as usize + local_work_size - 1) / local_work_size) * local_work_size;

        // Execute kernel
        let kernel_event = unsafe {
            ExecuteKernel::new(&self.kernel)
                .set_arg(&(start_seed as cl_ulong))
                .set_arg(&batch_size)
                .set_arg(checks_buffer)
                .set_arg(&num_checks)
                .set_arg(&self.candidates_buffer)
                .set_arg(&self.count_buffer)
                .set_global_work_size(global_work_size)
                .set_local_work_size(local_work_size)
                .enqueue_nd_range(&self.queue)
                .map_err(|e| GpuError::KernelExecution(format!("{:?}", e)))?
        };

        // Wait for kernel to complete
        kernel_event
            .wait()
            .map_err(|e| GpuError::KernelExecution(format!("{:?}", e)))?;

        // Read candidate count
        let mut count: [cl_uint; 1] = [0];
        unsafe {
            self.queue
                .enqueue_read_buffer(&self.count_buffer, CL_BLOCKING, 0, &mut count, &[])
                .map_err(|e| GpuError::BufferRead(format!("{:?}", e)))?;
        }

        let num_candidates = count[0] as usize;

        if num_candidates == 0 {
            return Ok(Vec::new());
        }

        // Clamp to MAX_CANDIDATES to prevent buffer overflow
        let num_candidates = num_candidates.min(MAX_CANDIDATES);

        // Read candidates
        let mut candidates = vec![0u64; num_candidates];
        unsafe {
            self.queue
                .enqueue_read_buffer(
                    &self.candidates_buffer,
                    CL_BLOCKING,
                    0,
                    &mut candidates,
                    &[],
                )
                .map_err(|e| GpuError::BufferRead(format!("{:?}", e)))?;
        }

        Ok(candidates)
    }

    /// Get the device name
    pub fn device_name(&self) -> String {
        self.device
            .name()
            .unwrap_or_else(|_| "Unknown".to_string())
    }

    /// Get the device vendor
    pub fn device_vendor(&self) -> String {
        self.device
            .vendor()
            .unwrap_or_else(|_| "Unknown".to_string())
    }

    /// Get the number of compute units
    pub fn compute_units(&self) -> u32 {
        self.device
            .max_compute_units()
            .unwrap_or(0)
    }

    /// Get device global memory size in bytes
    pub fn global_memory(&self) -> u64 {
        self.device
            .global_mem_size()
            .unwrap_or(0)
    }
}
