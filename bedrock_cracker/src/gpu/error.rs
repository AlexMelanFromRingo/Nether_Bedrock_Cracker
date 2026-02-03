use std::fmt;

/// Errors that can occur during GPU operations
#[derive(Debug, Clone)]
pub enum GpuError {
    /// No GPU device found or invalid device index
    DeviceNotFound(String),
    /// Failed to create OpenCL context
    ContextCreation(String),
    /// Failed to create command queue
    QueueCreation(String),
    /// Failed to compile OpenCL kernel
    KernelCompilation(String),
    /// Failed to create kernel
    KernelCreation(String),
    /// Failed to create buffer
    BufferCreation(String),
    /// Failed to write to buffer
    BufferWrite(String),
    /// Failed to read from buffer
    BufferRead(String),
    /// Failed to execute kernel
    KernelExecution(String),
    /// Invalid input data
    InvalidInput(String),
}

impl fmt::Display for GpuError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GpuError::DeviceNotFound(msg) => write!(f, "GPU device not found: {}", msg),
            GpuError::ContextCreation(msg) => write!(f, "Failed to create OpenCL context: {}", msg),
            GpuError::QueueCreation(msg) => write!(f, "Failed to create command queue: {}", msg),
            GpuError::KernelCompilation(msg) => write!(f, "Failed to compile OpenCL kernel: {}", msg),
            GpuError::KernelCreation(msg) => write!(f, "Failed to create kernel: {}", msg),
            GpuError::BufferCreation(msg) => write!(f, "Failed to create buffer: {}", msg),
            GpuError::BufferWrite(msg) => write!(f, "Failed to write to buffer: {}", msg),
            GpuError::BufferRead(msg) => write!(f, "Failed to read from buffer: {}", msg),
            GpuError::KernelExecution(msg) => write!(f, "Failed to execute kernel: {}", msg),
            GpuError::InvalidInput(msg) => write!(f, "Invalid input: {}", msg),
        }
    }
}

impl std::error::Error for GpuError {}
