using Documenter, DocumenterVitepress, ReparametrizableDistributions

# The HTMXObjects embed runtime (`htmxo-embed.ts` + companion CSS) is committed
# in `src/.vitepress/theme/`; see the note in its `index.ts` for re-syncing.

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
