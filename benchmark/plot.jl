#=
Turns benchmark/results/*.csv into the docs figures (docs/src/assets/bench_*.svg) and
prints the markdown tables for docs/src/benchmarks.md.

    julia --project=benchmark --startup-file=no benchmark/plot.jl
=#

using CairoMakie
using CSV
using DataFrames
using Printf

const RESULTS = joinpath(@__DIR__, "results")
const ASSETS = joinpath(@__DIR__, "..", "docs", "src", "assets")

# Figure panels. Colors are validated categorical slots assigned in fixed order within a
# panel; EllSparse and the stdlib baseline keep the same color in every CPU panel. Each
# format also has its own marker everywhere, the secondary encoding, so identity never
# rests on color alone.
const SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]
const PANELS = [
    (backend = "cpu", key = "julia", subtitle = "pure-Julia packages",
        formats = ["EllSparse", "SparseArrays CSC", "SparseMatricesCSR", "ThreadedSparseArrays",
            "GenericSparseArrays CSR", "GenericSparseArrays CSC"]),
    (backend = "cpu", key = "libs", subtitle = "wrapped C libraries and references",
        formats = ["EllSparse", "SparseArrays CSC", "MKLSparse", "SuiteSparseGraphBLAS",
            "CompressedSparseBlocks", "CSR loop (reference)", "dense BLAS"]),
    (backend = "gpu", key = "", subtitle = "",
        formats = ["EllSparse", "CUDA.jl CSR", "CUDA.jl CSC", "CUDA.jl COO",
            "GenericSparseArrays CSR", "GenericSparseArrays CSC"]),
]
const MARKER = Dict(
    "EllSparse" => :circle, "SparseArrays CSC" => :rect, "SparseMatricesCSR" => :diamond,
    "ThreadedSparseArrays" => :utriangle, "GenericSparseArrays CSR" => :dtriangle,
    "GenericSparseArrays CSC" => :ltriangle, "MKLSparse" => :star5, "SuiteSparseGraphBLAS" => :hexagon,
    "CompressedSparseBlocks" => :pentagon, "CSR loop (reference)" => :cross, "dense BLAS" => :xcross,
    "CUDA.jl CSR" => :rect, "CUDA.jl CSC" => :diamond, "CUDA.jl COO" => :utriangle,
)
style(panel, f) = (SLOTS[findfirst(==(f), panel.formats)], MARKER[f])
const FAMILY_ORDER = ["lap2d", "lap3d", "st27", "knn", "ragged"]

const OPS = [
    (key = "spmv", file = "spmv", title = "Forward product A·x", pick = df -> df.op .== "spmv"),
    (key = "spmv_adj", file = "adjoint", title = "Adjoint product Aᵀ·x", pick = df -> df.op .== "spmv_adj"),
    (key = "spmm8", file = "spmm8", title = "Block product A·X, 8 columns", pick = df -> (df.op .== "spmm") .& (df.nrhs .== 8)),
    (key = "spmm32", file = "spmm32", title = "Block product A·X, 32 columns", pick = df -> (df.op .== "spmm") .& (df.nrhs .== 32))
]

const INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"

function load(backend)
    paths = filter(isfile, map(f -> joinpath(RESULTS, f), ["$backend.csv", "$(backend)_mkl.csv"]))
    isempty(paths) && return nothing
    df = reduce(vcat, map(p -> CSV.read(p, DataFrame), paths); cols = :union)
    df.gbs = df.useful_bytes ./ df.time_s ./ 1.0e9
    meta = CSV.read(joinpath(RESULTS, "$(backend)_meta.csv"), DataFrame)[1, :]
    return (df, meta)
end

function figure(df, meta, op, panel)
    backend = panel.backend
    sub = df[op.pick(df) .& in.(df.format, Ref(Set(panel.formats))), :]
    fams = filter(in(Set(sub.family)), FAMILY_ORDER)
    fmts = filter(in(Set(sub.format)), panel.formats)
    fig = Figure(; size = (1100, 640), backgroundcolor = SURFACE, fontsize = 13,
        figure_padding = 12)
    title = isempty(panel.subtitle) ? op.title : "$(op.title), $(panel.subtitle)"
    Label(fig[0, 1:3], "$title — $(meta.device)"; color = INK, fontsize = 16,
        font = :bold, halign = :left)
    axes = map(enumerate(fams)) do (i, fam)
        fsub = sub[sub.family .== fam, :]
        ax = Axis(fig[fldmod1(i, 3)...]; xscale = log10, title = first(fsub.label),
            titlecolor = INK, titlealign = :left, titlefont = :regular,
            xlabel = "stored nonzeros", ylabel = "useful bandwidth (GB/s)",
            xgridcolor = GRID, ygridcolor = GRID, backgroundcolor = SURFACE,
            xlabelcolor = INK2, ylabelcolor = INK2, xticklabelcolor = INK2, yticklabelcolor = INK2,
            leftspinecolor = GRID, bottomspinecolor = GRID, topspinevisible = false,
            rightspinevisible = false, xtickcolor = GRID, ytickcolor = GRID)
        hlines!(ax, [meta.triad_gbs]; color = INK2, linestyle = :dash, linewidth = 1)
        vlines!(ax, [meta.llc_bytes / 12]; color = INK2, linestyle = :dot, linewidth = 1)
        foreach(fmts) do fmt
            d = sort(fsub[fsub.format .== fmt, :], :nnz)
            isempty(d) && return
            (color, marker) = style(panel, fmt)
            scatterlines!(ax, d.nnz, d.gbs; color, marker, markersize = 8, linewidth = 2,
                strokecolor = SURFACE, strokewidth = 1)
        end
        ylims!(ax; low = 0)
        ax
    end
    linkyaxes!(axes...)
    elems = map(fmts) do f
        (color, marker) = style(panel, f)
        [LineElement(; color, linewidth = 2), MarkerElement(; color, marker, markersize = 8)]
    end
    refs = [LineElement(; color = INK2, linestyle = :dash), LineElement(; color = INK2, linestyle = :dot)]
    cache = backend == "gpu" ? "L2" : "L3"
    Legend(fig[2, 3], [elems; refs],
        [fmts; "triad bandwidth ($(round(Int, meta.triad_gbs)) GB/s)"; "matrix values+indices = $cache size"];
        framevisible = false, labelcolor = INK, halign = :left, valign = :center,
        backgroundcolor = SURFACE)
    return fig
end

# Markdown table: useful GB/s per format at each family's largest size.
function table(df, op, formats)
    sub = df[op.pick(df), :]
    fmts = filter(in(Set(sub.format)), formats)
    fams = filter(in(Set(sub.family)), FAMILY_ORDER)
    io = IOBuffer()
    println(io, "| Matrix (rows) | ", join(fmts, " | "), " |")
    println(io, "|:--|", repeat("--:|", length(fmts)))
    foreach(fams) do fam
        fsub = sub[sub.family .== fam, :]
        big = fsub[fsub.nnz .== maximum(fsub.nnz), :]
        cells = map(fmts) do f
            r = big[big.format .== f, :]
            isempty(r) ? "—" : @sprintf("%.1f", r.gbs[1])
        end
        m = first(big.m)
        println(io, "| $(first(big.label)) ($(m ≥ 10^6 ? @sprintf("%.1fM", m / 1e6) : @sprintf("%.0fk", m / 1e3))) | ", join(cells, " | "), " |")
    end
    return String(take!(io))
end

function determinism(df)
    sub = df[df.op .== "spmv_adj", :]
    return combine(groupby(sub, :format), :deterministic => (d -> all(skipmissing(d))) => :always_repeatable)
end

mkpath(ASSETS)
foreach(("cpu", "gpu")) do backend
    loaded = load(backend)
    isnothing(loaded) && return println("no $backend results, skipping")
    (df, meta) = loaded
    println("\n## $backend — $(meta.device), triad $(round(meta.triad_gbs; digits = 1)) GB/s\n")
    panels = filter(p -> p.backend == backend, PANELS)
    foreach(OPS) do op
        any(op.pick(df)) || return
        println("### $(op.title)\n")
        foreach(panels) do panel
            name = join(filter(!isempty, ["bench", backend, op.file, panel.key]), "_")
            path = joinpath(ASSETS, "$name.svg")
            save(path, figure(df, meta, op, panel))
            println("→ $(relpath(path, joinpath(@__DIR__, "..")))\n")
            println(table(df, op, panel.formats))
        end
    end
    println("adjoint repeatability (20 runs, bitwise):")
    show(stdout, determinism(df); summary = false, eltypes = false)
    println()
end
