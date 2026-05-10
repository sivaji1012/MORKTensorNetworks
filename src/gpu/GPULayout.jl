"""
    GPULayout.jl — GPU data layout optimization for MORK tensor operations

MORK-Tensor-Networks paper §3, §4: efficient data layout decisions for
GPU dispatch. Routes operations to sparse or dense kernels based on
matrix properties.

Components:
  - CSR/BCSR conversion from dense matrices
  - Densification routing via Tucker decomposition
  - Quantifier-as-reduction mapping
  - Layout analysis and recommendation
"""
module GPULayout

using ..Semirings: AbstractSemiring, szero
using ..TuckerDecomposition: should_densify, tucker_decompose_2d, tucker_reconstruct_2d

export CSRMatrix, dense_to_csr, csr_to_dense,
       BCSRMatrix, dense_to_bcsr, bcsr_to_dense,
       analyze_layout, recommend_strategy,
       LayoutStrategy, SparseCSR, DenseTucker, DenseDirect

# ─── CSR Matrix ──────────────────────────────────────────────────────────────

"""
    CSRMatrix{T}

Compressed Sparse Row format for GPU-friendly sparse operations.
"""
struct CSRMatrix{T}
    rowptr::Vector{Int32}
    colval::Vector{Int32}
    nzval::Vector{T}
    m::Int32  # rows
    n::Int32  # cols
end

"""Convert dense matrix to CSR."""
function dense_to_csr(A::AbstractMatrix{T}) where T
    m, n = size(A)
    rowptr = Int32[1]
    colval = Int32[]
    nzval = T[]

    z = zero(T)
    for i in 1:m
        for j in 1:n
            v = A[i, j]
            if v != z && isfinite(v)
                push!(colval, Int32(j))
                push!(nzval, v)
            end
        end
        push!(rowptr, Int32(length(colval) + 1))
    end
    return CSRMatrix{T}(rowptr, colval, nzval, Int32(m), Int32(n))
end

"""Convert CSR back to dense."""
function csr_to_dense(csr::CSRMatrix{T}) where T
    A = zeros(T, csr.m, csr.n)
    for i in 1:csr.m
        for idx in csr.rowptr[i]:csr.rowptr[i+1]-1
            A[i, csr.colval[idx]] = csr.nzval[idx]
        end
    end
    return A
end

"""Number of nonzeros."""
nnz(csr::CSRMatrix) = length(csr.nzval)

# ─── BCSR Matrix (§5.1) ──────────────────────────────────────────────────────

"""
    BCSRMatrix{T}

Block Compressed Sparse Row format. §5.1: for structured sparsity patterns
where nonzeros cluster in dense blocks of size (block_r × block_c).
GPU-friendly: each block maps to a warp-local dense computation.

  block_rowptr[i] — start of block-row i in block_colval / block_data
  block_colval[k] — block-column index of block k
  block_data[k]   — dense (block_r × block_c) array for block k
"""
struct BCSRMatrix{T}
    block_rowptr :: Vector{Int32}   # length = n_block_rows + 1
    block_colval :: Vector{Int32}   # length = n_blocks
    block_data   :: Vector{Matrix{T}} # each entry = block_r × block_c dense matrix
    m :: Int32   # full matrix rows
    n :: Int32   # full matrix cols
    block_r :: Int32
    block_c :: Int32
end

"""
    dense_to_bcsr(A, block_r, block_c) → BCSRMatrix

§5.1: Convert dense matrix A to BCSR with block size (block_r × block_c).
Blocks where all entries are zero are dropped. Partial blocks at boundary
are zero-padded to full block size.
"""
function dense_to_bcsr(A::AbstractMatrix{T}, block_r::Int, block_c::Int) where T
    m, n = size(A)
    n_br = cld(m, block_r)  # number of block-rows
    n_bc = cld(n, block_c)  # number of block-cols

    block_rowptr = Int32[1]
    block_colval = Int32[]
    block_data   = Matrix{T}[]

    z = zero(T)
    for bi in 1:n_br
        r_start = (bi - 1) * block_r + 1
        r_end   = min(bi * block_r, m)
        for bj in 1:n_bc
            c_start = (bj - 1) * block_c + 1
            c_end   = min(bj * block_c, n)
            block = zeros(T, block_r, block_c)
            has_nz = false
            for ii in r_start:r_end, jj in c_start:c_end
                v = A[ii, jj]
                if v != z && isfinite(v)
                    block[ii - r_start + 1, jj - c_start + 1] = v
                    has_nz = true
                end
            end
            if has_nz
                push!(block_colval, Int32(bj))
                push!(block_data, block)
            end
        end
        push!(block_rowptr, Int32(length(block_colval) + 1))
    end
    BCSRMatrix{T}(block_rowptr, block_colval, block_data,
                  Int32(m), Int32(n), Int32(block_r), Int32(block_c))
end

"""
    bcsr_to_dense(bcsr) → Matrix

Reconstruct dense matrix from BCSR. Inverse of dense_to_bcsr.
"""
function bcsr_to_dense(bcsr::BCSRMatrix{T}) where T
    A = zeros(T, bcsr.m, bcsr.n)
    n_br = length(bcsr.block_rowptr) - 1
    for bi in 1:n_br
        r_start = (bi - 1) * Int(bcsr.block_r) + 1
        for idx in bcsr.block_rowptr[bi]:bcsr.block_rowptr[bi+1]-1
            bj      = Int(bcsr.block_colval[idx])
            c_start = (bj - 1) * Int(bcsr.block_c) + 1
            blk     = bcsr.block_data[idx]
            for ii in 1:size(blk,1), jj in 1:size(blk,2)
                ri = r_start + ii - 1
                ci = c_start + jj - 1
                ri <= bcsr.m && ci <= bcsr.n || continue
                A[ri, ci] = blk[ii, jj]
            end
        end
    end
    A
end

n_blocks(bcsr::BCSRMatrix) = length(bcsr.block_colval)

"""Fill ratio."""
fill_ratio(csr::CSRMatrix) = nnz(csr) / (Float64(csr.m) * csr.n)

# ─── Layout Strategy ─────────────────────────────────────────────────────────

abstract type LayoutStrategy end

"""Use sparse CSR format (SpMV/SpGEMM kernels)."""
struct SparseCSR <: LayoutStrategy end

"""Use Tucker-decomposed dense format (dense einsum kernels)."""
struct DenseTucker <: LayoutStrategy
    rank::Int
end

"""Use direct dense format (standard matmul)."""
struct DenseDirect <: LayoutStrategy end

# ─── Layout Analysis ─────────────────────────────────────────────────────────

"""
    analyze_layout(A) → (fill_ratio, effective_rank, recommended_strategy)

Analyze a matrix and recommend the best GPU layout strategy.
"""
function analyze_layout(A::AbstractMatrix)
    m, n = size(A)
    nnz_count = count(x -> x != 0 && isfinite(x), A)
    fill = nnz_count / (m * n)

    # Effective rank via SVD (if small enough)
    effective_rank = min(m, n)
    if m <= 512 && n <= 512
        try
            sv = svdvals(Float64.(A))
            total = sum(sv .^ 2)
            if total > 0
                cum = cumsum(sv .^ 2) ./ total
                r = findfirst(>=(0.9), cum)
                effective_rank = r === nothing ? length(sv) : r
            end
        catch
            # SVD failed — assume full rank
        end
    end

    strategy = recommend_strategy(fill, effective_rank, min(m, n))
    return (fill_ratio=fill, effective_rank=effective_rank, strategy=strategy)
end

"""
    recommend_strategy(fill, eff_rank, min_dim) → LayoutStrategy

Heuristic routing:
  - fill < 0.1: SparseCSR (sparse kernels efficient)
  - fill 0.1-0.5 and low rank: DenseTucker (compress then dense)
  - fill > 0.5: DenseDirect (already dense)
"""
function recommend_strategy(fill::Float64, eff_rank::Int, min_dim::Int)
    if fill < 0.1
        return SparseCSR()
    elseif fill <= 0.5 && eff_rank <= min_dim ÷ 2
        return DenseTucker(eff_rank)
    else
        return DenseDirect()
    end
end

# ─── Quantifier-as-Reduction Mapping ─────────────────────────────────────────

"""
    existential_reduce(v) → Bool

Existential quantification as GPU reduction: ∃x P(x) ≡ (sum(v) > 0).
Native GPU primitive: sum + threshold.
"""
existential_reduce(v::AbstractVector) = sum(v) > 0

"""
    universal_reduce(v) → Bool

Universal quantification as GPU reduction: ∀x P(x) ≡ (min(v) > 0).
Native GPU primitive: min + threshold.
"""
universal_reduce(v::AbstractVector) = minimum(v) > 0

end # module
