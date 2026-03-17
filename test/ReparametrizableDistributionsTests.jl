@testmodule begin


    @testset "advance!!" begin
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        v, pos = advance!!(x, 0)
        @test v == 1.0
        @test pos == 1

        vs, pos = advance!!(x, 1, 3)
        @test vs == [2.0, 3.0, 4.0]
        @test pos == 4
    end

    @testset "log_abs_tanh" begin
        for x in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
            @test ReparametrizableDistributions.log_abs_tanh(x) ≈ log(abs(tanh(x))) atol=1e-10
            @test ReparametrizableDistributions.log_abs_tanh(-x) ≈ log(abs(tanh(-x))) atol=1e-10
        end
        @test isfinite(ReparametrizableDistributions.log_abs_tanh(100.0))
        @test ReparametrizableDistributions.log_abs_tanh(100.0) ≈ 0.0 atol=1e-10
    end

    @testset "NormalTransform" begin
        rng = MersenneTwister(42)
        t = NormalTransform(5)
        x = randn(rng, 5)
        lp, pos = fused_logdensity(t, x)
        @test pos == 5
        @test lp ≈ sum(logpdf.(Normal(), x))
    end

    @testset "LKJCholeskyTransform n=1" begin
        rng = MersenneTwister(123)
        t = LKJCholeskyTransform(1)
        x = randn(rng, 1)
        lp, pos = fused_logdensity(t, x)
        @test pos == 1
        @test t.L[1, 1] ≈ exp(x[1])
        @test lp ≈ logpdf(Normal(), x[1])
    end

    @testset "LKJCholeskyTransform n=2" begin
        rng = MersenneTwister(123)
        t = LKJCholeskyTransform(2)
        x = randn(rng, 3)
        lp, pos = fused_logdensity(t, x)
        @test pos == 3
        @test t.L[1, 1] > 0
        @test t.L[2, 2] > 0
        @test isfinite(lp)
        Sigma = Matrix(t.L) * Matrix(t.L)'
        @test isposdef(Sigma)
    end

    @testset "LKJCholeskyTransform n=3" begin
        rng = MersenneTwister(123)
        t = LKJCholeskyTransform(3)
        np = ReparametrizableDistributions.nparams(t)
        @test np == 6
        x = randn(rng, np)
        lp, pos = fused_logdensity(t, x)
        @test pos == np
        @test isfinite(lp)
        Sigma = Matrix(t.L) * Matrix(t.L)'
        @test isposdef(Sigma)
    end

    @testset "LKJCholeskyTransform eta parameter" begin
        rng = MersenneTwister(123)
        t1 = LKJCholeskyTransform(3; eta=1.0)
        t2 = LKJCholeskyTransform(3; eta=2.0)
        x = randn(rng, 6)
        lp1, _ = fused_logdensity(t1, x)
        lp2, _ = fused_logdensity(t2, x)
        @test lp1 != lp2
    end

    @testset "LKJCholeskyTransform init accumulation" begin
        rng = MersenneTwister(123)
        t = LKJCholeskyTransform(2)
        x = [0.0, 0.0, randn(rng, 3)...]
        lp1, pos1 = fused_logdensity(t, x; init=(10.0, 2))
        @test pos1 == 5
        lp2, _ = fused_logdensity(t, view(x, 3:5); init=(10.0, 0))
        @test lp1 ≈ lp2
    end

    @testset "LKJCholeskyTransform finite gradient" begin
        rng = MersenneTwister(123)
        n = 3
        np = n + n * (n - 1) ÷ 2
        function f(x)
            t = LKJCholeskyTransform(n)
            lp, _ = fused_logdensity(t, x)
            lp
        end
        x = randn(rng, np)
        grad = FiniteDifferences.grad(central_fdm(5, 1), f, x)[1]
        @test all(isfinite, grad)
    end

    @testset "AffineNormalTransform scalar non-centered" begin
        rng = MersenneTwister(99)
        mu, sigma = 3.0, 2.0
        t = AffineNormalTransform(mu, sigma; centered=0.0)
        z = randn(rng)
        lp, pos = fused_logdensity(t, [z])
        @test pos == 1
        @test lp ≈ logpdf(Normal(), z)
        @test constrain(t, z) ≈ mu + sigma * z
    end

    @testset "AffineNormalTransform scalar centered" begin
        rng = MersenneTwister(99)
        mu, sigma = 3.0, 2.0
        t = AffineNormalTransform(mu, sigma; centered=1.0)
        z = randn(rng)
        lp, pos = fused_logdensity(t, [z])
        @test pos == 1
        @test lp ≈ logpdf(Normal(mu, sigma), z)
        @test constrain(t, z) ≈ z
    end

    @testset "AffineNormalTransform scalar partial centering" begin
        rng = MersenneTwister(99)
        mu, sigma = 3.0, 2.0
        c = 0.5
        t = AffineNormalTransform(mu, sigma; centered=c)
        z = randn(rng)
        lp, pos = fused_logdensity(t, [z])
        @test pos == 1
        @test lp ≈ logpdf(Normal(c * mu, sigma^c), z)
        @test constrain(t, z) ≈ mu + sigma^(1-c) * (z - c*mu)
    end

    @testset "AffineNormalTransform centering endpoints" begin
        mu, sigma = 5.0, 3.0
        z_std = 1.5
        for c in [0.0, 0.25, 0.5, 0.75, 1.0]
            t = AffineNormalTransform(mu, sigma; centered=c)
            z = c * mu + sigma^c * z_std
            x = constrain(t, z)
            @test x ≈ mu + sigma * z_std
        end
    end

    @testset "AffineNormalTransform vector" begin
        rng = MersenneTwister(99)
        mu = [1.0, 2.0, 3.0]
        sigma = [0.5, 1.0, 2.0]
        t = AffineNormalTransform(mu, sigma; centered=0.0)
        z = randn(rng, 3)
        lp, pos = fused_logdensity(t, z)
        @test pos == 3
        @test lp ≈ sum(logpdf.(Normal(), z))
        x = constrain(t, z)
        @test x ≈ mu .+ sigma .* z
    end

    @testset "AffineNormalTransform per-element centering" begin
        rng = MersenneTwister(99)
        mu = [1.0, 2.0]
        sigma = [0.5, 2.0]
        c = [0.0, 1.0]
        t = AffineNormalTransform(mu, sigma, c)
        z = randn(rng, 2)
        lp, pos = fused_logdensity(t, z)
        @test pos == 2
        @test lp ≈ logpdf(Normal(), z[1]) + logpdf(Normal(2.0, 2.0), z[2])
    end

    @testset "AffineNormalTransform finite gradient" begin
        rng = MersenneTwister(99)
        mu, sigma = 3.0, 2.0
        for c in [0.0, 0.3, 0.7, 1.0]
            f(x) = begin
                t = AffineNormalTransform(mu, sigma; centered=c)
                lp, _ = fused_logdensity(t, x)
                lp
            end
            z = randn(rng, 1)
            grad = FiniteDifferences.grad(central_fdm(5, 1), f, z)[1]
            @test all(isfinite, grad)
        end
    end

    @testset "AffineNormalTransform eight schools" begin
        rng = MersenneTwister(99)
        mu, tau = 0.0, 5.0
        J = 8
        t = AffineNormalTransform(fill(mu, J), fill(tau, J); centered=0.0)
        z = randn(rng, J)
        lp, pos = fused_logdensity(t, z)
        @test pos == J
        @test lp ≈ sum(logpdf.(Normal(), z))
        theta = constrain(t, z)
        @test theta ≈ mu .+ tau .* z
    end

    @testset "CorrelatedEffectsTransform non-centered" begin
        rng = MersenneTwister(77)
        n_dims = 3
        n_groups = 4
        lkj = LKJCholeskyTransform(n_dims)
        np_chol = ReparametrizableDistributions.nparams(lkj)
        x_chol = randn(rng, np_chol)
        fused_logdensity(lkj, x_chol)
        L = lkj.L
        scales = scales_from_cholesky(L, n_dims)

        effects = zeros(n_groups, n_dims)
        t = CorrelatedEffectsTransform(L, scales, effects; centered=0.0)
        z = randn(rng, n_groups * n_dims)
        lp, pos = fused_logdensity(t, z)
        @test pos == n_groups * n_dims

        @test lp ≈ sum(logpdf.(Normal(), z))
        for j in 1:n_groups
            zj = z[(j-1)*n_dims+1:j*n_dims]
            @test effects[j, :] ≈ Matrix(L) * zj
        end
    end

    @testset "CorrelatedEffectsTransform exact distribution" begin
        rng = MersenneTwister(77)
        n_dims = 3
        n_groups = 1
        lkj = LKJCholeskyTransform(n_dims)
        np_chol = ReparametrizableDistributions.nparams(lkj)
        x_chol = randn(rng, np_chol)
        fused_logdensity(lkj, x_chol)
        L = lkj.L
        scales = scales_from_cholesky(L, n_dims)

        for c in [0.0, 0.3, 0.5, 0.7, 1.0]
            effects = zeros(n_groups, n_dims)
            t = CorrelatedEffectsTransform(L, scales, effects; centered=c)
            eta = [1.0, -0.5, 0.8]
            z_input = [scales[i]^c * eta[i] for i in 1:n_dims]
            fused_logdensity(t, z_input)
            @test effects[1, :] ≈ Matrix(L) * eta
        end
    end

    @testset "CorrelatedEffectsTransform log density depends on c_i" begin
        rng = MersenneTwister(77)
        n_dims = 3
        n_groups = 1
        lkj = LKJCholeskyTransform(n_dims)
        x_chol = randn(rng, ReparametrizableDistributions.nparams(lkj))
        fused_logdensity(lkj, x_chol)
        L = lkj.L
        scales = scales_from_cholesky(L, n_dims)

        z = randn(rng, n_dims)
        c_base = [0.3, 0.5, 0.7]

        effects = zeros(n_groups, n_dims)
        t = CorrelatedEffectsTransform(L, scales, effects, c_base)
        lp_base, _ = fused_logdensity(t, z)

        c_alt = [0.3, 0.8, 0.7]
        effects2 = zeros(n_groups, n_dims)
        t2 = CorrelatedEffectsTransform(L, scales, effects2, c_alt)
        lp_alt, _ = fused_logdensity(t2, z)

        expected_diff = logpdf(Normal(0, scales[2]^c_alt[2]), z[2]) -
                       logpdf(Normal(0, scales[2]^c_base[2]), z[2])
        @test lp_alt - lp_base ≈ expected_diff
    end

    @testset "scales_from_cholesky" begin
        rng = MersenneTwister(77)
        n = 3
        lkj = LKJCholeskyTransform(n)
        x_chol = randn(rng, ReparametrizableDistributions.nparams(lkj))
        fused_logdensity(lkj, x_chol)
        L = lkj.L
        scales = scales_from_cholesky(L, n)

        for i in 1:n
            @test scales[i] ≈ sqrt(sum(L[i, k]^2 for k in 1:i))
            @test scales[i] > 0
        end
    end

    @testset "CorrelatedEffectsTransform per-dimension centering" begin
        rng = MersenneTwister(77)
        n_dims = 2
        n_groups = 3
        lkj = LKJCholeskyTransform(n_dims)
        x_chol = randn(rng, ReparametrizableDistributions.nparams(lkj))
        fused_logdensity(lkj, x_chol)
        L = lkj.L
        scales = scales_from_cholesky(L, n_dims)

        c = [0.0, 1.0]
        effects = zeros(n_groups, n_dims)
        t = CorrelatedEffectsTransform(L, scales, effects, c)
        z = randn(rng, n_groups * n_dims)
        lp, pos = fused_logdensity(t, z)

        expected_lp = 0.0
        for j in 1:n_groups
            zj = z[(j-1)*n_dims+1:j*n_dims]
            expected_lp += logpdf(Normal(0, 1), zj[1])
            expected_lp += logpdf(Normal(0, scales[2]), zj[2])
        end
        @test lp ≈ expected_lp
    end

    @testset "CorrelatedEffectsTransform finite gradient" begin
        rng = MersenneTwister(77)
        n_dims = 3
        n_groups = 2
        lkj = LKJCholeskyTransform(n_dims)
        np_chol = ReparametrizableDistributions.nparams(lkj)
        x_chol = randn(rng, np_chol)
        fused_logdensity(lkj, x_chol)
        L_mat = Matrix(lkj.L)
        scales = scales_from_cholesky(lkj.L, n_dims)

        for c in [0.0, 0.5, 1.0]
            function f(z)
                L_copy = LowerTriangular(copy(L_mat))
                effects = zeros(n_groups, n_dims)
                t = CorrelatedEffectsTransform(L_copy, scales, effects; centered=c)
                lp, _ = fused_logdensity(t, z)
                lp
            end
            z = randn(rng, n_groups * n_dims)
            grad = FiniteDifferences.grad(central_fdm(5, 1), f, z)[1]
            @test all(isfinite, grad)
        end
    end

    @testset "CorrelatedEffectsTransform init accumulation" begin
        rng = MersenneTwister(77)
        n_dims = 2
        n_groups = 1
        lkj = LKJCholeskyTransform(n_dims)
        x_chol = randn(rng, ReparametrizableDistributions.nparams(lkj))
        fused_logdensity(lkj, x_chol)
        L = lkj.L
        scales = scales_from_cholesky(L, n_dims)

        x = [0.0, 0.0, randn(rng, n_dims)...]
        effects = zeros(n_groups, n_dims)
        t = CorrelatedEffectsTransform(L, scales, effects; centered=0.5)
        lp1, pos1 = fused_logdensity(t, x; init=(5.0, 2))
        @test pos1 == 4

        effects2 = zeros(n_groups, n_dims)
        t2 = CorrelatedEffectsTransform(L, scales, effects2; centered=0.5)
        lp2, _ = fused_logdensity(t2, view(x, 3:4); init=(5.0, 0))
        @test lp1 ≈ lp2
    end

end # @testmodule
