//! GPU acceleration module for bedrock pattern cracking using OpenCL.
//!
//! This module provides GPU-accelerated seed search capabilities that can
//! speed up the bedrock pattern cracking process by 20-100x compared to CPU.
//!
//! # Example
//!
//! ```no_run
//! use bedrock_cracker::gpu::{list_gpu_devices, search_bedrock_pattern_gpu, GpuSearchConfig};
//! use bedrock_cracker::raw_data::block::Block;
//! use bedrock_cracker::raw_data::block_type::BlockType;
//! use bedrock_cracker::raw_data::modes::{BedrockGeneration, OutputMode};
//! use std::sync::mpsc;
//!
//! // List available GPU devices
//! let devices = list_gpu_devices().unwrap();
//! for device in &devices {
//!     println!("Found GPU: {} ({} compute units)", device.name, device.compute_units);
//! }
//!
//! // Create blocks for search
//! let blocks = vec![
//!     Block::new(0, 123, 0, BlockType::BEDROCK),
//!     Block::new(1, 123, 0, BlockType::OTHER),
//! ];
//!
//! // Setup channel for results
//! let (sender, receiver) = mpsc::channel();
//!
//! // Run GPU search
//! let config = GpuSearchConfig::default();
//! search_bedrock_pattern_gpu(
//!     &blocks,
//!     config,
//!     BedrockGeneration::Normal,
//!     OutputMode::WorldSeed,
//!     sender,
//! ).unwrap();
//! ```

mod context;
mod error;
mod search;

use java_random::JAVA_LCG;
use opencl3::device::{get_all_devices, Device, CL_DEVICE_TYPE_GPU};

pub use context::GpuContext;
pub use error::GpuError;
pub use search::{search_bedrock_pattern_gpu, search_bedrock_pattern_gpu_range, GpuSearchConfig};

use crate::block_data::CheckObject;
use crate::MASK48;

/// Information about an available GPU device
#[derive(Debug, Clone)]
pub struct GpuDevice {
    /// Device name
    pub name: String,
    /// Vendor name
    pub vendor: String,
    /// Number of compute units (shader cores / streaming multiprocessors)
    pub compute_units: u32,
    /// Global memory size in bytes
    pub memory: u64,
    /// Device index (for use with GpuContext::new)
    pub index: usize,
}

/// List all available GPU devices that support OpenCL
pub fn list_gpu_devices() -> Result<Vec<GpuDevice>, GpuError> {
    let device_ids = get_all_devices(CL_DEVICE_TYPE_GPU)
        .map_err(|e| GpuError::DeviceNotFound(format!("Failed to enumerate GPU devices: {:?}", e)))?;

    let mut devices = Vec::new();

    for (index, device_id) in device_ids.into_iter().enumerate() {
        let device = Device::new(device_id);

        let name = device.name().unwrap_or_else(|_| "Unknown".to_string());
        let vendor = device.vendor().unwrap_or_else(|_| "Unknown".to_string());
        let compute_units = device.max_compute_units().unwrap_or(0);
        let memory = device.global_mem_size().unwrap_or(0);

        devices.push(GpuDevice {
            name,
            vendor,
            compute_units,
            memory,
            index,
        });
    }

    Ok(devices)
}

/// Check if GPU acceleration is available
pub fn is_gpu_available() -> bool {
    list_gpu_devices()
        .map(|devices| !devices.is_empty())
        .unwrap_or(false)
}

/// GPU-compatible check object structure
/// Must be #[repr(C)] and match the OpenCL struct layout exactly
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct GpuCheckObject {
    pub pos_hash: u64,
    pub condition: u64,
    pub offset: u64,
    pub _padding: u64,
}

impl GpuCheckObject {
    /// Create a GpuCheckObject from a CPU CheckObject
    pub fn from_check_object(check: &CheckObject) -> Self {
        Self {
            pos_hash: check.pos_hash(),
            condition: check.condition(),
            offset: check.offset(),
            _padding: 0,
        }
    }

    /// CPU-side check function for verification (matches OpenCL kernel logic)
    #[inline(always)]
    pub fn check_cpu(&self, upper_bits: u64) -> bool {
        ((upper_bits ^ self.pos_hash)
            .wrapping_mul(JAVA_LCG.multiplier)
            .wrapping_add(self.offset)
            & MASK48)
            < self.condition
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_list_devices() {
        // This test just ensures the function doesn't panic
        // It may return empty list if no GPU is available
        let result = list_gpu_devices();
        assert!(result.is_ok() || matches!(result, Err(GpuError::DeviceNotFound(_))));
    }

    #[test]
    fn test_gpu_check_object_size() {
        // Verify the struct is 32 bytes (4 x 8-byte fields)
        assert_eq!(std::mem::size_of::<GpuCheckObject>(), 32);
    }
}
