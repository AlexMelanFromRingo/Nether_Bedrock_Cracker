use java_random::{Random, JAVA_LCG};
use next_long_reverser::get_next_long;

use crate::block_data::{BlockFilter, CheckObject};
use crate::raw_data::block::Block;
use crate::raw_data::modes::{BedrockGeneration, OutputMode};
use crate::raw_data::sender::Sender;
use crate::{CrackProgress, FLOOR_HASH, ROOF_HASH};

use super::context::GpuContext;
use super::error::GpuError;
use super::GpuCheckObject;

/// Batch size for GPU processing (number of seeds per batch)
const BATCH_SIZE: u32 = 1 << 20; // 1M seeds

/// Total number of upper bits to search (36 bits = 68.7 billion iterations)
const TOTAL_UPPER_BITS: u64 = 1 << 36;

/// Configuration for GPU search
#[derive(Clone, Debug)]
pub struct GpuSearchConfig {
    /// Device index to use
    pub device_index: usize,
    /// Batch size (number of seeds per GPU dispatch)
    pub batch_size: u32,
}

impl Default for GpuSearchConfig {
    fn default() -> Self {
        Self {
            device_index: 0,
            batch_size: BATCH_SIZE,
        }
    }
}

/// Prepare GPU check objects from block data
/// Uses create_check(12) to match CPU's top layer behavior
fn prepare_gpu_checks(blocks: &[Block], mode: BedrockGeneration) -> (Vec<GpuCheckObject>, Vec<BlockFilter>, bool) {
    let mut floor_blocks = Vec::new();
    let mut roof_blocks = Vec::new();

    for block in blocks {
        let filter = BlockFilter::from(block, mode);
        if block.y < 64 {
            floor_blocks.push(filter);
        } else {
            roof_blocks.push(filter);
        }
    }

    // Determine which surface provides better filtering
    let floor_power = crate::block_data::get_filter_power(&floor_blocks);
    let roof_power = crate::block_data::get_filter_power(&roof_blocks);

    let is_floor_primary = floor_power < roof_power;

    let (primary_filter, secondary_filter) = if is_floor_primary {
        (floor_blocks, roof_blocks)
    } else {
        (roof_blocks, floor_blocks)
    };

    // Create check objects for GPU using 12 lower bits (matches CPU top layer)
    // This is more permissive and will pass candidates that MIGHT be valid
    let gpu_checks: Vec<GpuCheckObject> = primary_filter
        .iter()
        .cloned()
        .map(|mut filter| {
            let check = filter.create_check(12);
            GpuCheckObject::from_check_object(&check)
        })
        .collect();

    (gpu_checks, secondary_filter, is_floor_primary)
}

/// Create final check objects for exact 48-bit verification
fn create_final_checks(blocks: &[Block], mode: BedrockGeneration, is_floor_primary: bool) -> (Vec<CheckObject>, Vec<CheckObject>) {
    let mut floor_checks = Vec::new();
    let mut roof_checks = Vec::new();

    for block in blocks {
        let mut filter = BlockFilter::from(block, mode);
        let check = filter.create_check(0); // Exact check for full 48-bit seeds
        if block.y < 64 {
            floor_checks.push(check);
        } else {
            roof_checks.push(check);
        }
    }

    if is_floor_primary {
        (floor_checks, roof_checks)
    } else {
        (roof_checks, floor_checks)
    }
}

/// CPU finalization for candidates found by GPU
/// candidate is upper 36 bits - we need to test all 4096 lower bit combinations
fn finalize_candidate<S: Sender>(
    upper_bits: u64,
    primary_checks: &[CheckObject],
    secondary_checks: &[CheckObject],
    primary_hash: u64,
    secondary_hash: u64,
    output: OutputMode,
    sender: &S,
) {
    // Test all 4096 combinations of lower 12 bits
    for lower in 0u64..4096 {
        let full_seed = upper_bits | lower;

        // Verify this exact seed passes all primary surface checks
        let primary_passes = primary_checks.iter().all(|check| !check.check(full_seed));
        if !primary_passes {
            continue;
        }

        // This seed passed all primary checks - now do CrossComparison
        cross_comparison(
            full_seed,
            secondary_checks,
            primary_hash,
            secondary_hash,
            output,
            sender,
        );
    }
}

/// CrossComparison - same logic as CPU version
fn cross_comparison<S: Sender>(
    seed: u64,
    secondary_checks: &[CheckObject],
    primary_hash: u64,
    secondary_hash: u64,
    output: OutputMode,
    sender: &S,
) {
    // Reverse next_long to get potential internal seeds
    for internal_seed in reverse_next_long(seed) {
        // Get common bedrock seed
        let bedrock_seed = internal_seed ^ primary_hash;

        // Check against secondary surface
        let mut secondary_seed = bedrock_seed ^ secondary_hash;
        secondary_seed = next_long(secondary_seed);

        // Verify all secondary checks pass
        let secondary_passes = secondary_checks.iter().all(|check| !check.check(secondary_seed));
        if !secondary_passes {
            continue;
        }

        // Reverse to structure seed
        for structure_seed in reverse_next_long(bedrock_seed) {
            if output == OutputMode::WorldSeed {
                // Reverse to world seed
                for prev_seed in reverse_next_long(structure_seed) {
                    let world_seed = next_long(prev_seed);
                    sender.send(CrackProgress::Seed(world_seed));
                }
            } else {
                sender.send(CrackProgress::Seed(structure_seed));
            }
        }
    }
}

fn reverse_next_long(seed: u64) -> Vec<u64> {
    get_next_long(seed)
        .into_iter()
        .map(|seed| seed ^ JAVA_LCG.multiplier)
        .collect()
}

fn next_long(seed: u64) -> u64 {
    Random::with_seed(seed).next_long() as u64
}

/// Main GPU search function
pub fn search_bedrock_pattern_gpu<S: Sender + 'static>(
    blocks: &[Block],
    config: GpuSearchConfig,
    mode: BedrockGeneration,
    output: OutputMode,
    sender: S,
) -> Result<(), GpuError> {
    // Prepare GPU checks (permissive, for upper 36 bits)
    let (gpu_checks, _secondary_filters, is_floor_primary) = prepare_gpu_checks(blocks, mode);

    if gpu_checks.is_empty() {
        return Err(GpuError::InvalidInput("No valid blocks for GPU filtering".to_string()));
    }

    // Create final checks for CPU verification (exact 48-bit)
    let (primary_checks, secondary_checks) = create_final_checks(blocks, mode, is_floor_primary);

    let (primary_hash, secondary_hash) = if is_floor_primary {
        (FLOOR_HASH, ROOF_HASH)
    } else {
        (ROOF_HASH, FLOOR_HASH)
    };

    // Initialize GPU context
    let mut ctx = GpuContext::new(config.device_index)?;

    // Upload checks to GPU
    ctx.upload_checks(&gpu_checks)?;

    let num_checks = gpu_checks.len() as u32;
    let batch_size = config.batch_size;
    let total_batches = TOTAL_UPPER_BITS / batch_size as u64;

    // Process batches
    for batch_idx in 0..total_batches {
        // Calculate start seed for this batch (already shifted by 12)
        let start_seed = batch_idx * (batch_size as u64) << 12;

        // Execute GPU filter
        let candidates = ctx.execute_filter(start_seed, batch_size, num_checks)?;

        // CPU finalization for each candidate
        // Each candidate is upper 36 bits that passed the permissive GPU filter
        for candidate in candidates {
            finalize_candidate(
                candidate,
                &primary_checks,
                &secondary_checks,
                primary_hash,
                secondary_hash,
                output,
                &sender,
            );
        }

        // Report progress (seeds processed = batch_size * 4096 lower bits)
        let seeds_processed = (batch_size as u64) << 12;
        if !sender.send(CrackProgress::Progress(seeds_processed)) {
            // Receiver dropped, stop processing
            return Ok(());
        }
    }

    Ok(())
}

/// Search with a specific range (for multi-GPU or distributed processing)
pub fn search_bedrock_pattern_gpu_range<S: Sender + 'static>(
    blocks: &[Block],
    config: GpuSearchConfig,
    mode: BedrockGeneration,
    output: OutputMode,
    start_upper_bits: u64,
    end_upper_bits: u64,
    sender: S,
) -> Result<(), GpuError> {
    // Prepare GPU checks
    let (gpu_checks, _secondary_filters, is_floor_primary) = prepare_gpu_checks(blocks, mode);

    if gpu_checks.is_empty() {
        return Err(GpuError::InvalidInput("No valid blocks for GPU filtering".to_string()));
    }

    // Create final checks for CPU verification
    let (primary_checks, secondary_checks) = create_final_checks(blocks, mode, is_floor_primary);

    let (primary_hash, secondary_hash) = if is_floor_primary {
        (FLOOR_HASH, ROOF_HASH)
    } else {
        (ROOF_HASH, FLOOR_HASH)
    };

    // Initialize GPU context
    let mut ctx = GpuContext::new(config.device_index)?;

    // Upload checks to GPU
    ctx.upload_checks(&gpu_checks)?;

    let num_checks = gpu_checks.len() as u32;
    let batch_size = config.batch_size;

    // Calculate batch range
    let start_batch = start_upper_bits / batch_size as u64;
    let end_batch = (end_upper_bits + batch_size as u64 - 1) / batch_size as u64;

    // Process batches
    for batch_idx in start_batch..end_batch {
        let batch_start = batch_idx * (batch_size as u64);

        // Adjust batch size for last partial batch
        let current_batch_size = if batch_start + batch_size as u64 > end_upper_bits {
            (end_upper_bits - batch_start) as u32
        } else {
            batch_size
        };

        // Calculate start seed (shifted by 12)
        let start_seed = batch_start << 12;

        // Execute GPU filter
        let candidates = ctx.execute_filter(start_seed, current_batch_size, num_checks)?;

        // CPU finalization for each candidate
        for candidate in candidates {
            finalize_candidate(
                candidate,
                &primary_checks,
                &secondary_checks,
                primary_hash,
                secondary_hash,
                output,
                &sender,
            );
        }

        // Report progress
        let seeds_processed = (current_batch_size as u64) << 12;
        if !sender.send(CrackProgress::Progress(seeds_processed)) {
            return Ok(());
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::raw_data::block_type::BlockType;
    use std::sync::mpsc;

    const WORLD_SEED: u64 = 765906787396911863;

    const TEST_BLOCKS: [Block; 24] = [
        Block::new(18, 123, -117, BlockType::OTHER),
        Block::new(18, 123, -118, BlockType::OTHER),
        Block::new(18, 123, -119, BlockType::OTHER),
        Block::new(33, 126, -99, BlockType::OTHER),
        Block::new(35, 126, -99, BlockType::OTHER),
        Block::new(38, 126, -99, BlockType::OTHER),
        Block::new(19, 123, -117, BlockType::BEDROCK),
        Block::new(19, 123, -118, BlockType::BEDROCK),
        Block::new(19, 123, -119, BlockType::BEDROCK),
        Block::new(25, 126, -112, BlockType::BEDROCK),
        Block::new(25, 126, -113, BlockType::BEDROCK),
        Block::new(25, 126, -114, BlockType::BEDROCK),
        Block::new(11, 1, -111, BlockType::OTHER),
        Block::new(11, 1, -110, BlockType::OTHER),
        Block::new(11, 1, -109, BlockType::OTHER),
        Block::new(14, 4, -97, BlockType::OTHER),
        Block::new(14, 4, -96, BlockType::OTHER),
        Block::new(14, 4, -94, BlockType::OTHER),
        Block::new(10, 1, -111, BlockType::BEDROCK),
        Block::new(10, 1, -110, BlockType::BEDROCK),
        Block::new(10, 1, -109, BlockType::BEDROCK),
        Block::new(11, 4, -97, BlockType::BEDROCK),
        Block::new(11, 4, -96, BlockType::BEDROCK),
        Block::new(11, 4, -94, BlockType::BEDROCK),
    ];

    #[test]
    fn test_prepare_gpu_checks() {
        let (checks, _secondary, _is_floor_primary) = prepare_gpu_checks(&TEST_BLOCKS, BedrockGeneration::Normal);
        assert!(!checks.is_empty(), "GPU checks should not be empty");
    }

    #[test]
    fn test_gpu_check_object_conversion() {
        let block = Block::new(18, 123, -117, BlockType::OTHER);
        let mut filter = BlockFilter::from(&block, BedrockGeneration::Normal);
        let check = filter.create_check(12);
        let gpu_check = GpuCheckObject::from_check_object(&check);

        // Verify the GPU check matches the CPU check behavior
        let test_seed: u64 = 0x123456789000; // Multiple of 4096
        let cpu_result = check.check(test_seed);
        let gpu_result = gpu_check.check_cpu(test_seed);
        assert_eq!(cpu_result, gpu_result, "GPU and CPU check should produce same result");
    }

    // Integration test - only run if GPU is available
    #[test]
    #[ignore] // Run with `cargo test --features gpu -- --ignored`
    fn test_gpu_search_finds_known_seed() {
        let (sender, receiver) = mpsc::channel();

        let config = GpuSearchConfig {
            device_index: 0,
            batch_size: 1 << 16,
        };

        let result = search_bedrock_pattern_gpu(
            &TEST_BLOCKS,
            config,
            BedrockGeneration::Normal,
            OutputMode::WorldSeed,
            sender,
        );

        assert!(result.is_ok(), "GPU search should succeed");

        // Collect all found seeds
        let mut found_seeds = Vec::new();
        while let Ok(progress) = receiver.try_recv() {
            if let CrackProgress::Seed(seed) = progress {
                found_seeds.push(seed);
            }
        }

        assert!(
            found_seeds.contains(&WORLD_SEED),
            "Should find the known world seed, found: {:?}",
            found_seeds
        );
    }
}
