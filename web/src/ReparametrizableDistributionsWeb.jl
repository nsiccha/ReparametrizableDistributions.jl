module ReparametrizableDistributionsWeb
using HTMXObjects
using ReparametrizableDistributions
using LinearAlgebra
using TestModules, Random
# Treebars activates HTMXObjects's HTMXObjectsTreebarsExt, which gives
# `RecordingRoutes` its live polling_fetchindex progress tree.
using Treebars
using HTMXObjects: RecordingRoutes

include("test/runtests.jl")

# --- AppData: cached gallery + recording wiring ---
@dynamicstruct struct RdAppData
    gallery_dir = joinpath(dirname(dirname(@__DIR__)), "web", "gallery")
    gallery     = Gallery(gallery_dir)

    # `record!` writes here; the docs site under
    # docs/src/public/live-rd/ picks them up in production.
    recording_dir   = joinpath(dirname(dirname(@__DIR__)), "docs", "src", "public", "live-rd")
    recording_base  = get(ENV, "RECORD_BASE_PREFIX", "/ReparametrizableDistributions.jl/dev/live-rd")
    recording_paths = let ids = [it.id for it in gallery.items]
        ["/", "/gallery", ["/entries/$id" for id in ids]...]
    end
end

const APPDATA = RdAppData()

@htmx struct AppContext
    __appdata__ = APPDATA

    __page__(content) = htmx(h.main(class="container")(
        h.nav(h.ul(
            h.li(h.a(href=__self__/"")("ReparametrizableDistributions")),
            h.li(h.a(href=__self__/"gallery")("Gallery")),
            h.li(h.a(href=__self__/"demo/normal?dim=3")("Legacy demo")),
            h.li(h.a(href=__self__/"tests")("Tests")),
        )),
        content,
    ); pico_version="2")

    # Bundle of available FusedTransform variants for the legacy /demo route.
    # Adding a new variant is a single `@struct foo = begin label, description,
    # build(dim), extra_text(t) end` inside this bundle.
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
        h.h1("ReparametrizableDistributions"),
        h.p("Fused bijector + log-prior-density transforms."),
        h.ul(
            h.li(h.a(href=__self__/"gallery")("Gallery"),
                 " — live cards for every transform in ",
                 h.code("web/gallery/"), "."),
            h.li(h.a(href=__self__/"demo/normal?dim=3")("Legacy /demo"),
                 " — old per-variant resampling page."),
            h.li(h.a(href=__self__/"record_gallery")("Record gallery"),
                 " — drive ", h.code("record!"), " to refresh ",
                 h.code("docs/src/public/live-rd/"), "."),
            h.li(h.a(href=__self__/"tests")("Tests")),
            h.li(h.a(href=__self__/"structure")("Structure")),
        ),
    )

    # === Gallery surface ===
    # Use `__appdata__.gallery` rather than destructuring, so the route
    # name `gallery` doesn't clash with a property of the same name
    # (htmxo-gallery §6 pitfall).
    @get gallery() = h.div(; data_layout="wide")(
        h.h1("Transform Gallery"),
        h.p("Each card builds a ", h.code("FusedTransform"), ", draws a fresh ",
            "vector of unconstrained parameters, and runs ",
            h.code("fused_logdensity"), ". Reload to resample."),
        gallery_grid(__appdata__.gallery.items;
                     section_titles = __appdata__.gallery.section_titles,
                     card_renderer  = rd_gallery_card),
    )

    @include entries(id::String) = begin
        item        = find_item(__appdata__.gallery, id)
        transform   = Base.include(@__MODULE__, item.path)
        np          = nparams(transform)
        x           = randn(np)
        (lp, pos)   = fused_logdensity(transform, x)

        @get index() = h.article(
            h.header(h.h2(item.title)),
            isempty(item.description) ? h.span() : h.p(item.description),
            h.p(h.strong("nparams: "), string(np)),
            h.p(h.strong("Unconstrained x: "), h.code(string(round.(x; digits=4)))),
            h.p(h.strong("Log density: "), h.code(string(round(lp; digits=6)))),
            extra_block(transform),
            h.details(
                h.summary("Source"),
                h.pre(h.code(item.code_string; class="language-julia")),
            ),
        )
    end

    # === Legacy resampling demo (kept for back-compat) ===
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
                h.a(href=__self__/"demo/$name?dim=$dim")("Resample"),
                " | ",
                [h.span(h.a(href=__self__/"demo/$name?dim=$d")("dim=$d"), " ") for d in [2,3,4,5]]...
            ),
            h.p(h.a(href=__self__/"")("Back to index")),
        )
    end

    @include record_gallery = RecordingRoutes(;
        app_type    = AppContext,
        paths       = __appdata__.recording_paths,
        record_dir  = __appdata__.recording_dir,
        record_base = __appdata__.recording_base,
        label       = "Recording ReparametrizableDistributions gallery",
    )

    @include tests     = TestRoutes(; __req__, test_module=@__MODULE__)
    @include structure = StructureRoutes(; root=AppContext)
end

# Renderer for the index gallery grid. Loads the spec from the item path
# and shows transform metadata. The full per-id details live at
# `/entries/<id>` (provided by the `entries(id)` include above).
function rd_gallery_card(item)
    transform = try
        Base.include(@__MODULE__, item.path)
    catch err
        return h.article(
            h.header(h.h4(item.title)),
            h.p(item.description),
            h.p("Error loading: $(sprint(showerror, err))"; style="color:var(--htmxo-error)"),
        )
    end
    np = nparams(transform)
    h.article(
        h.header(h.h4(h.a(item.title; href="/entries/$(item.id)"))),
        isempty(item.description) ? h.span() : h.p(item.description),
        h.p(h.strong("nparams: "), string(np), " · ",
            h.strong("type: "), h.code(string(typeof(transform).name.name))),
        h.details(
            h.summary("Source"),
            h.pre(h.code(item.code_string; class="language-julia")),
        ),
    )
end

# Per-transform extra info shown on /entries/<id>. Default is empty.
extra_block(::Any) = h.span()
extra_block(t::LKJCholeskyTransform) = h.div(
    h.p(h.strong("Constrained Cholesky factor L:")),
    h.pre(string(round.(Matrix(t.L); digits=4))),
)
extra_block(t::CorrelatedEffectsTransform) = h.div(
    h.p(h.strong("Constrained effects (rows = groups):")),
    h.pre(string(round.(t.effects; digits=4))),
)

function __init__()
    route!(AppContext())
end

end
