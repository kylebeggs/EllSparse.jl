using EllSparse
using Documenter
using DocumenterVitepress

DocMeta.setdocmeta!(EllSparse, :DocTestSetup, :(using EllSparse); recursive = true)

makedocs(;
    modules = [EllSparse],
    authors = "Kyle Beggs (beggskw@gmail.com) and contributors",
    sitename = "EllSparse.jl",
    repo = Documenter.Remotes.GitHub("kylebeggs", "EllSparse.jl"),
    format = DocumenterVitepress.MarkdownVitepress(;
        repo = "https://github.com/kylebeggs/EllSparse.jl",
        devbranch = "main",
        devurl = "dev",
        build_vitepress = (!haskey(ENV, "VITEPRESS_DEV")),
    ),
    pages = [
        "Home" => "index.md",
        "cuSPARSE" => "cusparse.md",
        "API" => "api.md",
    ],
    checkdocs = :exports,
)

DocumenterVitepress.deploydocs(;
    repo = "github.com/kylebeggs/EllSparse.jl",
    target = joinpath(@__DIR__, "build"),
    branch = "gh-pages",
    devbranch = "main",
    push_preview = true,
)
