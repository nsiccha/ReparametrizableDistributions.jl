module ReparametrizableDistributionsWeb
using HTMXObjects
using ReparametrizableDistributions
using LinearAlgebra
using TestModules, Random

include("test/runtests.jl")


transforms_info = [
    (name="NormalTransform", description="Identity transform + standard normal prior", factory=n -> NormalTransform(parse(Int, n))),
    (name="LKJCholeskyTransform", description="Cholesky factor reparametrization + LKJ prior", factory=n -> LKJCholeskyTransform(parse(Int, n))),
]

function demo_transform(t, np)
    x = randn(np)
    lp, pos = fused_logdensity(t, x)
    (; x, lp, pos)
end

function render_demo(name, dim)
    idx = findfirst(t -> t.name == name, transforms_info)
    if isnothing(idx)
        valid = join((t.name for t in transforms_info), ", ")
        throw(ArgumentError("Unknown transform name: $(repr(name)). Valid names: $valid"))
    end
    info = transforms_info[idx]
    t = info.factory(dim)
    np = nparams(t)
    result = demo_transform(t, np)
    plain_text = """Transform: $name(dim=$dim)
Unconstrained params ($np): $(round.(result.x; digits=4))
Log density: $(round(result.lp; digits=6))
Params consumed: $(result.pos)"""
    if t isa LKJCholeskyTransform
        plain_text *= "\nCholesky factor L:\n$(round.(Matrix(t.L); digits=4))"
    end
    plain_text
end

@htmx struct AppContext

    __page__(content) = htmx(h.body(h.main(class="container")(
        h.nav(h.ul(h.li(h.a(href="/")("ReparametrizableDistributions")))),
        content
    )); pico_version="2")

    @get index = h.div(
        h.h1("FusedTransforms"),
        h.p("Available transforms:"),
        h.ul([
            h.li(
                h.a(href="/demo/$(t.name)?dim=3")(t.name),
                " - ", t.description
            )
            for t in transforms_info
        ]...),
        h.p(h.a(href="/tests")("Tests")),
    )

    @get demo(name, dim="3") = begin
        local text = render_demo(name, dim)
        h.div(
            h.h1("$name Demo"),
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
