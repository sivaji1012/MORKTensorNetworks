"""
ECANTensorBridge.jl — ECAN Attention as Tensor Operations

MORK-Tensor-Networks §5+ECAN, TODO §7.3.

Maps ECAN's attention dynamics onto the semiring tensor algebra:

  §7.3.1  STI spreading = sparse (max,+) matmul on GPU
          STI_new[x] = max_y (W[x,y] + STI[y])
          where W = Hebbian link weight matrix

  §7.3.2  Hebbian link weights = sparse adjacency matrix W[src,dst]
          Updated via Hebbian rule: ΔW[x,y] = η × STI[x] × STI[y]

  §7.3.3  Attention fund (rent/wage) = tensor reduction
          Rent:  total_sti_deducted = Σ_x max(0, STI[x] - af_threshold) × rent_rate
          Wage:  budget distributed proportionally to STI

Reference algorithm: t1_SpreadingActivation.metta (tradeSti, Hebbian learning)
"""

using ..Semirings: MaxPlusSemiring, SumProductSemiring, otimes, oplus, szero, sone
using SparseArrays
using LinearAlgebra

export ECANState, ecan_sti_spread!, ecan_hebbian_update!
export ecan_collect_rent!, ecan_distribute_wages!
export ecan_build_weight_matrix, ecan_sti_vector

# ─── §7.3.2: Hebbian Link Weight Matrix ──────────────────────────────────────

"""
    ECANState

Holds the ECAN attention state as tensors:
  sti        — Short-Term Importance vector (n_atoms)
  lti        — Long-Term Importance vector (n_atoms)
  W          — Hebbian link weight sparse matrix (n_atoms × n_atoms)
  atom_ids   — mapping from index → atom identifier
"""
mutable struct ECANState
    sti      :: Vector{Float32}   # STI per atom
    lti      :: Vector{Float32}   # LTI per atom
    W        :: Matrix{Float32}   # Hebbian weight matrix (dense for small spaces)
    atom_ids :: Vector{Any}       # atom_ids[i] = identifier for index i
end

"""
    ECANState(n) → ECANState

Create empty ECAN state for n atoms. STI and LTI initialised to 0.
W initialised to -Inf (no Hebbian links yet — (max,+) identity).
"""
function ECANState(n::Int)
    ECANState(
        zeros(Float32, n),
        zeros(Float32, n),
        fill(-Inf32, n, n),   # (max,+) zero = -Inf
        Any[i for i in 1:n]
    )
end

"""
    ecan_build_weight_matrix(links, n) → Matrix{Float32}

§7.3.2: Build Hebbian weight matrix W from link list.
links = Vector of (src_idx, dst_idx, weight) triples.
W[src, dst] = weight; -Inf where no link exists (max,+ zero).
"""
function ecan_build_weight_matrix(links::Vector{<:Tuple}, n::Int) :: Matrix{Float32}
    W = fill(-Inf32, n, n)
    for (src, dst, w) in links
        W[src, dst] = Float32(w)
    end
    W
end

"""
    ecan_sti_vector(state) → Vector{Float32}

Return current STI as a column vector for tensor operations.
"""
ecan_sti_vector(s::ECANState) = copy(s.sti)

# ─── §7.3.1: STI Spreading as (max,+) matmul ─────────────────────────────────

"""
    ecan_sti_spread!(state; decay=0.9f0, max_spread=1.0f0) → state

§7.3.1: Batch STI spreading as sparse (max,+) matrix-vector product.

Formula (from paper + t1_SpreadingActivation.metta tradeSti):
    STI_new[x] = max_y (W[x,y] + STI[y])

This is a (max,+) semiring matvec — exactly Viterbi/best-path scoring
on the ECAN attention graph. Each atom receives the maximum incoming
attention boost along any single Hebbian link.

Post-spread: apply decay and clamp to [0, max_spread].
"""
function ecan_sti_spread!(state::ECANState;
                          decay::Float32   = 0.9f0,
                          max_spread::Float32 = 1.0f0) :: ECANState
    n   = length(state.sti)
    sti = state.sti
    W   = state.W

    # (max,+) matvec: new_sti[x] = max_y (W[x,y] + sti[y])
    new_sti = fill(-Inf32, n)
    for x in 1:n
        best = -Inf32
        for y in 1:n
            w = W[x, y]
            w == -Inf32 && continue
            v = w + sti[y]
            v > best && (best = v)
        end
        # If no incoming links, keep self value (decayed)
        new_sti[x] = best == -Inf32 ? sti[x] * decay : best
    end

    # Apply decay and clamp
    @. new_sti = clamp(new_sti * decay, 0.0f0, max_spread)
    state.sti  = new_sti
    state
end

# ─── §7.3.2: Hebbian Weight Update ───────────────────────────────────────────

"""
    ecan_hebbian_update!(state; η=0.01f0, decay=0.99f0) → state

§7.3.2: Update Hebbian link weights for co-active atom pairs.

Formula (from t1_SpreadingActivation.metta Hebbian learning):
    ΔW[x,y] = η × STI[x] × STI[y]
    W[x,y]  ← clamp(decay × W[x,y] + ΔW[x,y], -10, 10)

Only updates existing links (W[x,y] > -Inf).
Uses (sum,product) = standard multiplication for co-activation.
"""
function ecan_hebbian_update!(state::ECANState;
                               η::Float32    = 0.01f0,
                               decay::Float32 = 0.99f0) :: ECANState
    n   = length(state.sti)
    sti = state.sti
    W   = state.W

    for x in 1:n, y in 1:n
        W[x, y] == -Inf32 && continue   # no link — skip
        Δ = η * sti[x] * sti[y]
        W[x, y] = clamp(decay * W[x, y] + Δ, -10.0f0, 10.0f0)
    end
    state
end

# ─── §7.3.3: Attention Fund — Rent/Wage as Tensor Reduction ──────────────────

"""
    ecan_collect_rent!(state; af_threshold=0.5f0, rent_rate=0.1f0) → Float32

§7.3.3: Collect rent from all atoms above the Attention Focus (AF) threshold.
Rent is a tensor reduction (sum of clipped STI above threshold):

    rent[x] = max(0, STI[x] - af_threshold) × rent_rate
    total_rent = Σ_x rent[x]

Deducts rent from each atom's STI in-place. Returns total collected.
"""
function ecan_collect_rent!(state::ECANState;
                             af_threshold::Float32 = 0.5f0,
                             rent_rate::Float32    = 0.1f0) :: Float32
    total = 0.0f0
    for x in eachindex(state.sti)
        excess = max(0.0f0, state.sti[x] - af_threshold)
        rent   = excess * rent_rate
        state.sti[x] -= rent
        total += rent
    end
    total
end

"""
    ecan_distribute_wages!(state, budget) → state

§7.3.3: Distribute wage budget proportionally to current STI.
Atoms with higher STI receive more of the budget (STI-weighted reduction).

    weight[x] = max(0, STI[x])
    wage[x]   = budget × weight[x] / Σ_y weight[y]

This is a softmax-style distribution — a standard GPU reduction pattern.
"""
function ecan_distribute_wages!(state::ECANState, budget::Float32) :: ECANState
    weights = max.(0.0f0, state.sti)
    total   = sum(weights)
    total == 0.0f0 && return state  # no positive STI — nothing to distribute
    @. state.sti += budget * weights / total
    state
end

export ECANState, ecan_sti_spread!, ecan_hebbian_update!
export ecan_collect_rent!, ecan_distribute_wages!
export ecan_build_weight_matrix, ecan_sti_vector
