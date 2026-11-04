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

# Fixed color + marker per format (validated categorical slots, light surface). The
# marker is the secondary encoding, so identity never rests on color alone.
const STYLE = Dict(
    "SELL-1" => ("#2a78d6", :circle),
    "SELL-32" => ("#eb6834", :rect),
    "CSC (stdlib)" => ("#1baf7a", :diamond),
    "CSR (threaded)" => ("#eda100", :utriangle),
    "MKL CSR" => ("#e87ba4", :dtriangle),
    "dense BLAS" => ("#008300", :xcross),
    "EllSparse SELL-32" => ("#2a78d6", :circle),
    "cuSPARSE SELL" => ("#eb6834", :rect),
    "cuSPARSE CSR" => ("#1baf7a", :diamond),
)
const ORDER = ["SELL-1", "SELL-32", "CSC (stdlib)", "CSR (threaded)", "MKL CSR", "dense BLAS",
    "EllSparse SELL-32", "cuSPARSE SELL", "cuSPARSE CSR"]
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

function figure(df, meta, op, backend)
    sub = df[op.pick(df), :]
    fams = filter(in(Set(sub.family)), FAMILY_ORDER)
    fmts = filter(in(Set(sub.format)), ORDER)
    fig = Figure(; size = (1100, 640), backgroundcolor = SURFACE, fontsize = 13,
        figure_padding = 12)
    Label(fig[0, 1:3], "$(op.title) — $(meta.device)"; color = INK, fontsize = 16,
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
            (color, marker) = STYLE[fmt]
            scatterlines!(ax, d.nnz, d.gbs; color, marker, markersize = 8, linewidth = 2,
                strokecolor = SURFACE, strokewidth = 1)
        end
        ylims!(ax; low = 0)
        ax
    end
    linkyaxes!(axes...)
    elems = map(f -> [LineElement(; color = STYLE[f][1], linewidth = 2),
        MarkerElement(; color = STYLE[f][1], marker = STYLE[f][2], markersize = 8)], fmts)
    refs = [LineElement(; color = INK2, linestyle = :dash), LineElement(; color = INK2, linestyle = :dot)]
    cache = backend == "gpu" ? "L2" : "L3"
    Legend(fig[2, 3], [elems; refs],
        [fmts; "triad bandwidth ($(round(Int, meta.triad_gbs)) GB/s)"; "matrix values+indices = $cache size"];
        framevisible = false, labelcolor = INK, halign = :left, valign = :center,
        backgroundcolor = SURFACE)
    return fig
end

# Markdown table: useful GB/s per format at each family's largest size, plus speedup of
# the EllSparse layout over the best non-EllSparse format.
function table(df, op, backend)
    sub = df[op.pick(df), :]
    fmts = filter(in(Set(sub.format)), ORDER)
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
    foreach(OPS) do op
        any(op.pick(df)) || return
        path = joinpath(ASSETS, "bench_$(backend)_$(op.file).svg")
        save(path, figure(df, meta, op, backend))
        println("### $(op.title)  → $(relpath(path, joinpath(@__DIR__, "..")))\n")
        println(table(df, op, backend))
    end
    println("adjoint repeatability (20 runs, bitwise):")
    show(stdout, determinism(df); summary = false, eltypes = false)
    println()
end
