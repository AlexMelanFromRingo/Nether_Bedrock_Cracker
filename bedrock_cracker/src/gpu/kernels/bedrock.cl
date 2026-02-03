// Nether Bedrock Cracker - OpenCL Kernel
// This kernel filters potential seed candidates using the bedrock pattern check

#define JAVA_LCG_MULT 0x5DEECE66DUL
#define MASK48 0xFFFFFFFFFFFFUL

// CheckObject structure - must match Rust's CheckObject with #[repr(C)]
typedef struct {
    ulong pos_hash;
    ulong condition;
    ulong offset;
    ulong _padding;  // Alignment padding for 32-byte struct
} CheckObject;

// Inline function to check if a seed passes a single block check
// Returns true if the seed should be REJECTED (matches original CPU logic)
inline bool check_single(ulong upper_bits, __constant CheckObject* check) {
    ulong result = upper_bits ^ check->pos_hash;
    result = result * JAVA_LCG_MULT;
    result = result + check->offset;
    result = result & MASK48;
    return result < check->condition;
}

// Main kernel for filtering bedrock patterns
// Each work item processes one potential seed (upper 36 bits)
__kernel void bedrock_filter(
    ulong start_seed,                    // Starting seed for this batch (upper 36 bits, already shifted by 12)
    uint batch_size,                     // Number of seeds to process in this batch
    __constant CheckObject* checks,      // Array of block checks to apply
    uint num_checks,                     // Number of checks in the array
    __global ulong* candidates,          // Output buffer for candidates that pass all checks
    __global uint* candidate_count       // Atomic counter for number of candidates found
) {
    uint gid = get_global_id(0);

    if (gid >= batch_size) {
        return;
    }

    // Calculate the upper bits for this work item
    // Each seed is spaced by 4096 (1 << 12) in the search space
    ulong upper_bits = start_seed + ((ulong)gid << 12);

    // Apply all checks - if any check returns true, the seed is rejected
    for (uint i = 0; i < num_checks; i++) {
        if (check_single(upper_bits, &checks[i])) {
            return;  // Seed rejected, exit early
        }
    }

    // Seed passed all checks - add to candidates list
    // Use atomic increment to get a unique index
    uint idx = atomic_inc(candidate_count);

    // Store the candidate (with bounds check for safety)
    // Max candidates buffer size is checked on host side
    candidates[idx] = upper_bits;
}

// Alternative kernel that processes multiple checks per work item
// This can be more efficient for GPUs with high memory latency
__kernel void bedrock_filter_batched(
    ulong start_seed,
    uint batch_size,
    __constant CheckObject* checks,
    uint num_checks,
    __global ulong* candidates,
    __global uint* candidate_count,
    uint seeds_per_workitem              // Number of seeds each work item processes
) {
    uint gid = get_global_id(0);
    uint base_idx = gid * seeds_per_workitem;

    for (uint s = 0; s < seeds_per_workitem; s++) {
        uint seed_idx = base_idx + s;
        if (seed_idx >= batch_size) {
            return;
        }

        ulong upper_bits = start_seed + ((ulong)seed_idx << 12);

        bool rejected = false;
        for (uint i = 0; i < num_checks; i++) {
            if (check_single(upper_bits, &checks[i])) {
                rejected = true;
                break;
            }
        }

        if (!rejected) {
            uint idx = atomic_inc(candidate_count);
            candidates[idx] = upper_bits;
        }
    }
}
