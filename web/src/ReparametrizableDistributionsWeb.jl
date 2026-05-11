module ReparametrizableDistributionsWeb
using HTMXObjects
using ReparametrizableDistributions
using LinearAlgebra
using TestModules, Random

include("test/runtests.jl")


@htmx struct AppContext

    __page__(content) = htmx(h.main(class="container")(
        h.nav(h.ul(h.li(h.a(href="/")("ReparametrizableDistributions")))),
        content
    ); pico_version="2")

    # Bundle of available FusedTransform variants. Adding a new variant is a
    # single `@struct foo = begin label, description, build(dim), extra_text(t) end`
    # inside this bundle — no list edit, no dispatcher branch. The indexed `pick`
    # IP looks variants up by bare name (no `Symbol("foo_$name")` munging).
    @struct variants = begin
        @struct normal = begin
            label       = "NormalTransform"
            description = "Identity transform + standard normal prior"
            build(dim::Int) = NormalTransform(dim)
            extra_text(transform) = ""
        end

        @struct lkj_cholesky = begin
            label       = "LKJCholeskyTransform"
            description = "Cholesky factor reparametrization + LKJ prior"
            build(dim::Int) = LKJCholeskyTransform(dim)
            extra_text(transform) = "\nCholesky factor L:\n$(round.(Matrix(transform.L); digits=4))"
        end

        names = (:normal, :lkj_cholesky)

        pick(name::Symbol) = getproperty(__self__, name)
    end

    @get index() = h.div(
        h.h1("FusedTransforms"),
        h.p("Available transforms:"),
        h.ul([
            let v = __self__.variants.pick(name)
                h.li(h.a(href="/demo/$name?dim=3")(v.label), " - ", v.description)
            end
            for name in __self__.variants.names
        ]...),
        h.p(h.a(href="/tests")("Tests")),
    )

    @get demo(name::Symbol; dim::Int=3) = let v         = __self__.variants.pick(name),
                                              transform = v.build(dim),
                                              np        = nparams(transform),
                                              x         = randn(np),
                                              (lp, pos) = fused_logdensity(transform, x)
        local text = """Transform: $(v.label)(dim=$dim)
Unconstrained params ($np): $(round.(x; digits=4))
Log density: $(round(lp; digits=6))
Params consumed: $pos$(v.extra_text(transform))"""
        h.div(
            h.h1("$(v.label) Demo"),
            h.p("Dimension: $dim"),
            h.pre(text),
            h.p(
                h.a(href="/demo/$name?dim=$dim")("Resample"),
                " | ",
                [h.span(h.a(href="/demo/$name?dim=$d")("dim=$d"), " ") for d in [2,3,4,5]]...
            ),
            h.p(h.a(href="/")("Back to index"))
        )
    end

    @include tests = TestRoutes(; __req__, test_module=@__MODULE__)
    @include structure = StructureRoutes(; root=AppContext)
end

function __init__()
    route!(AppContext())
end

end
