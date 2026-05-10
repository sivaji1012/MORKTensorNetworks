"""
MORKTensorNetworks — Standalone Julia package

Ports "From Path Algebra in MORK to Tensor Logic on GPUs"
(Goertzel, October 2025) as a standalone package on top of MORK + PathMap.

Spec: docs/specs/mork_tensor_networks_spec.md

Package layout (mirrors paper sections):
  §1-3  core/  — Semirings + PathAlgebra (path algebra → tensor logic)
  §5    gpu/   — SemiringKernels + GPULayout (GPU-friendly execution)
  §2    shard/ — ShardZipper (partition/capture/materialize/compute/patch/adapt)
                 CrossShardJoin (halo/batched/reshard strategies)
  §5.4  decomp/ — TuckerDecomposition (sparse→dense densification)
  §6    hrt/   — HRT pyramid + PredictiveCodingTrainer

Depends on: MORK, PathMap, KernelAbstractions (GPU vendor-neutral)
"""
module MORKTensorNetworks

using MORK
using PathMap
using LinearAlgebra
using SparseArrays

# Load order follows dependency graph (consumers after providers):
# Semirings → SemiringKernels → PathAlgebra
# TuckerDecomposition → GPULayout
# CrossShardJoin (standalone) → ShardZipper (uses MORK.Space)
# HRT → PredictiveCodingTrainer

include("core/Semirings.jl")           # §3.5: AbstractSemiring + 6 instances
include("gpu/SemiringKernels.jl")      # §5.2: GPU kernels (needs Semirings)
include("core/PathAlgebra.jl")         # §1,§3: path ops (needs Semirings + SemiringKernels)
include("decomp/TuckerDecomposition.jl") # §5.4: Tucker (standalone)
include("gpu/GPULayout.jl")            # §5.1: CSR/BCSR layout (needs Semirings + Tucker)
include("shard/CrossShardJoin.jl")     # §5.6: join strategies (needs Semirings)
include("shard/ShardZipper.jl")        # §2: 6-step workflow (needs MORK + PathMap)
include("hrt/HRT.jl")                  # §6: HRT pyramid (standalone)
include("hrt/PredictiveCodingTrainer.jl") # §6.4: PC training (needs HRT)
include("ecan/ECANTensorBridge.jl")    # §7.3: ECAN STI spreading + Hebbian + attention fund

# Re-export public API from submodules
using .Semirings
using .SemiringKernels
using .PathAlgebra
using .TuckerDecomposition
using .GPULayout
using .CrossShardJoin
using .HRT
using .PredictiveCodingTrainer
# ECANTensorBridge is top-level (no submodule wrapper)

# ShardZipper is top-level (no submodule wrapper)
# Semirings
export AbstractSemiring, BooleanSemiring, SumProductSemiring
export MaxPlusSemiring, MinPlusSemiring, PLNSemiring, CostSemiring
export szero, sone, oplus, otimes, semiring_matmul, semiring_matvec, semiring_reduce
# PathAlgebra
export path_compose, path_union, path_intersect, path_restrict
export path_project, path_reachability, path_viterbi, path_count, path_universal
# TuckerDecomposition
export tucker_decompose_2d, tucker_reconstruct_2d, should_densify
export tucker_decompose_3d, tucker_reconstruct_3d
export BCSRMatrix, dense_to_bcsr, bcsr_to_dense
export ECANState, ecan_sti_spread!, ecan_hebbian_update!
export ecan_collect_rent!, ecan_distribute_wages!
export ecan_build_weight_matrix, ecan_sti_vector
export should_densify
# CrossShardJoin
export HaloStrategy, BatchedBoundaryStrategy, ReshardStrategy, cross_shard_join, select_join_strategy
# HRT
export HRTParams, HRTState, hrt_init, hrt_forward!, hrt_down_project!, hrt_cross_attn!
export hrt_gated_fuse!, hrt_reconstruction_loss
# PredictiveCodingTrainer
export PCTrainerConfig, pc_train_step!, pc_inner_loop!, hebbian_update!
# ShardZipper (top-level)
export Shard, PatchRecord, partition_trie, capture_shard, materialize!
export compute!, patch_and_reattach!, should_adapt, run_shard!, run_all_shards!
# MORK re-exports for tests
export new_space, space_add_all_sexpr!, space_metta_calculus!, space_dump_all_sexpr

end # module
