using Test
using MORKTensorNetworks

@testset "MORKTensorNetworks" begin

    @testset "Semirings — §3.5" begin
        sr = BooleanSemiring()
        @test oplus(sr, false, true)  == true
        @test otimes(sr, true, false) == false
        @test szero(sr) == false
        @test sone(sr)  == true

        sr2 = MaxPlusSemiring()
        @test oplus(sr2, 3.0f0, 5.0f0) == 5.0f0
        @test otimes(sr2, 2.0f0, 3.0f0) == 5.0f0
        @test szero(sr2) == -Inf32
        @test sone(sr2)  == 0.0f0

        sr3 = SumProductSemiring()
        @test oplus(sr3, 2.0f0, 3.0f0)  == 5.0f0
        @test otimes(sr3, 2.0f0, 3.0f0) == 6.0f0
    end

    @testset "PathAlgebra — §1, §3" begin
        sr = SumProductSemiring()
        # Sister(a,b)=1 → S[1,2]=1; Parent(b,c)=1 → P[2,3]=1 → Aunt(a,c)=1
        S = Float32[0 1 0; 0 0 0; 0 0 0]
        P = Float32[0 0 0; 0 0 1; 0 0 0]
        A = path_compose(sr, S, P)
        @test A[1, 3] > 0.0f0   # a is aunt of c
        @test A[2, 3] == 0.0f0  # no path from row 2

        # Union: S has (1,2), P has (2,3) → union has both
        U = path_union(sr, S, P)
        @test U[1,2] > 0   # from S
        @test U[2,3] > 0   # from P

        I = path_intersect(sr, S, S)
        @test I[1,2] > 0  # S∩S = S

        # Viterbi best-path
        W = Float32[0 1 0; 0 0 2; 0 0 0]
        Score = path_viterbi(W)
        @test Score[1,3] ≈ 3.0f0   # best 2-hop: 1+2=3
    end

    @testset "ShardZipper — §2 end-to-end" begin
        s = new_space()
        space_add_all_sexpr!(s, "(edge a b) (edge b c) (edge a c)")

        # Step 1: partition
        prefixes = partition_trie(s, 10)
        @test !isempty(prefixes)

        # Steps 2-3: capture + materialize
        shard = capture_shard(s, UInt8[])
        materialize!(shard, s)
        @test !isempty(shard.node_keys)

        # Step 4: compute (trivial kernel — mark all edges as inserts)
        compute!(shard, sh -> begin
            for (i, key) in enumerate(sh.node_keys)
                push!(sh.patch_log, PatchRecord(:insert, key, 1.0f0))
            end
        end)
        @test !isempty(shard.patch_log)

        # Step 5: patch & reattach
        n = patch_and_reattach!(shard, s)
        @test n > 0

        # Step 6: adapt check
        @test should_adapt(shard, 2) || !should_adapt(shard, 10000)
    end

    @testset "TuckerDecomposition — §5.4" begin
        A = rand(Float32, 6, 6)
        C, M, N = tucker_decompose_2d(A, 3)
        @test size(M) == (6, 3)
        @test size(N) == (6, 3)
        @test size(C) == (3, 3)
        A_recon = tucker_reconstruct_2d(C, M, N)
        @test size(A_recon) == (6, 6)
    end

    @testset "CrossShardJoin — §5.6 strategies" begin
        sr = BooleanSemiring()
        A  = Float32[1 0; 1 1]
        B  = Float32[0 1; 1 0]
        strat = HaloStrategy(1)
        @test strat isa HaloStrategy
        @test strat.halo_width == 1
    end


    @testset "BCSR — §5.1 Block CSR" begin
        A = Float32[1 0 2 0;
                    0 0 0 3;
                    4 0 5 0;
                    0 6 0 7]
        bcsr = dense_to_bcsr(A, 2, 2)
        @test bcsr isa BCSRMatrix{Float32}
        @test bcsr.block_r == 2
        @test bcsr.block_c == 2
        A2 = bcsr_to_dense(bcsr)
        @test A2 ≈ A

        # Block size that evenly divides
        B = rand(Float32, 4, 6)
        bcsr2 = dense_to_bcsr(B, 2, 3)
        @test bcsr_to_dense(bcsr2) ≈ B
    end

    @testset "Tucker 3D — §5.4 A_ijk ≈ C×M×N×P" begin
        A = rand(Float32, 6, 5, 4)
        C, M, N, P, err = tucker_decompose_3d(A, 3, 3, 3)
        @test size(C) == (3, 3, 3)
        @test size(M) == (6, 3)
        @test size(N) == (5, 3)
        @test size(P) == (4, 3)
        @test err < 0.5   # rough reconstruction quality

        A_recon = tucker_reconstruct_3d(C, M, N, P)
        @test size(A_recon) == (6, 5, 4)

        # Full rank → near-perfect reconstruction
        A2 = rand(Float32, 4, 4, 4)
        C2, M2, N2, P2, err2 = tucker_decompose_3d(A2, 4, 4, 4)
        @test err2 < 0.05
    end

    @testset "ECAN tensor bridge — §7.3" begin
        n = 4
        state = ECANState(n)
        state.sti = Float32[0.8, 0.3, 0.1, 0.6]

        # §7.3.2: Build Hebbian weight matrix
        links = [(1,2,0.5f0), (2,3,0.4f0), (3,4,0.3f0), (4,1,0.2f0)]
        W = ecan_build_weight_matrix(links, n)
        state.W = W
        @test W[1,2] == 0.5f0
        @test W[2,4] == -Inf32  # no link

        # §7.3.1: STI spreading as (max,+) matmul
        old_sti = copy(state.sti)
        ecan_sti_spread!(state; decay=0.9f0)
        @test length(state.sti) == n
        @test all(isfinite, state.sti)
        @test all(state.sti .>= 0.0f0)

        # §7.3.2: Hebbian update — save original before update
        state.sti = Float32[0.8, 0.3, 0.1, 0.6]
        state.W   = copy(W)
        w12_before = state.W[1,2]
        ecan_hebbian_update!(state; η=0.1f0, decay=1.0f0)  # no decay → pure Hebbian
        @test state.W[1,2] > w12_before  # co-active pair strengthened

        # §7.3.3: Rent collection
        state.sti = Float32[0.8, 0.3, 0.1, 0.6]
        total_rent = ecan_collect_rent!(state; af_threshold=0.5f0, rent_rate=0.1f0)
        @test total_rent > 0.0f0
        @test state.sti[1] < 0.8f0   # high STI atom paid rent
        @test state.sti[2] == 0.3f0  # below threshold — no rent

        # §7.3.3: Wage distribution
        state.sti = Float32[0.5, 0.2, 0.1, 0.4]
        ecan_distribute_wages!(state, 0.1f0)
        @test sum(state.sti) > 0.5f0 + 0.2f0 + 0.1f0 + 0.4f0 - 1e-5  # budget added
    end

end  # MORKTensorNetworks
