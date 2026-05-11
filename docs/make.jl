using Documenter, DocumenterVitepress, ReparametrizableDistributions
import HTMXObjects

# Sync the canonical `htmxo-embed.ts` (+ companion CSS) into our theme dir
# before DocumenterVitepress runs. The theme's `index.ts` imports from it.
HTMXObjects.vitepress_theme_install(joinpath(@__DIR__, "src", ".vitepress", "theme"))

makedocs(
    sitename = "ReparametrizableDistributions.jl",
    modules  = [ReparametrizableDistributions],
    format   = DocumenterVitepress.MarkdownVitepress(
        repo = "github.com/nsiccha/ReparametrizableDistributions.jl",
        devurl = "dev",
        devbranch = "dev",
    ),
    pages = [
        "Home"    => "index.md",
        "Gallery" => "gallery.md",
        "API"     => "api.md",
    ],
    checkdocs = :none,
    warnonly = true,
)

# Ensure a root index.html redirect exists
let redirect = joinpath(@__DIR__, "build", "index.html")
    isfile(redirect) || write(redirect, """
    <!DOCTYPE html>
    <html><head>
    <meta http-equiv="refresh" content="0; url=dev/">
    </head><body>Redirecting to <a href="dev/">dev</a>...</body></html>
    """)
end

DocumenterVitepress.deploydocs(
    repo = "github.com/nsiccha/ReparametrizableDistributions.jl",
    devbranch = "dev",
    push_preview = true,
)
