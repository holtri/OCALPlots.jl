
function get_grid(min, max, resolution, axis_overhang)
    grid_range = range(min - axis_overhang, max + axis_overhang, length = resolution)
    grid_data = hcat([[x,y] for x in grid_range for y in grid_range]...)
    return (grid_range, grid_data)
end

function split_2d_array(x, idx)
    (x[1, idx], x[2, idx])
end

function get_legend_text(l, dc)
    if l == :inlier && dc == :inlier
        return :TN
    elseif l == :inlier && dc == :outlier
        return :FP
    elseif l == :outlier && dc == :inlier
        return :FN
    else
        return :TP
    end
end

@recipe function plot_svdd(m::SVDD.SVDDClassifier, labels::Vector{Symbol}; grid_resolution = 100, axis_overhang = 0.2)
    grid_range, grid_data = get_grid(extrema(m.data)..., grid_resolution, axis_overhang)
    grid_scores = reshape(SVDD.predict(m, grid_data), grid_resolution, grid_resolution)
    data_class = SVDD.classify.(SVDD.predict(m, m.data))
    title := "Decision Boundary"
    @series begin
        seriestype := :contourf
        seriescolor --> :greens
        levels := range(0, maximum(grid_scores), length=10)
        grid_range, grid_range, grid_scores
    end

    colors = (inlier = :blue, outlier = :red)
    shapes = (inlier = :circle, outlier = :star8)

    markeralpha --> 0.7
    markersize --> 5

    for l in [:inlier, :outlier]
        markercolor := colors[l]
        for dc in [:inlier, :outlier]
            markershape := shapes[dc]
            @series begin
                seriestype := :scatter
                label := get_legend_text(l, dc)
                split_2d_array(m.data, (labels.==l) .& (data_class .== dc))
            end
       end
    end

    @series begin
        seriestype := :contour
        levels := [0]
        linewidth := 2
        color := :black
        cbar:= false
        grid_range, grid_range, grid_scores
    end
end

@recipe function plot_qs(m::SVDD.SVDDClassifier, qs::OneClassActiveLearning.ModelBasedPQs, history=Int[]; grid_resolution = 100, axis_overhang = 0.2)
    grid_range, grid_data = get_grid(extrema(m.data)..., grid_resolution, axis_overhang)
    grid_scores = reshape(OneClassActiveLearning.qs_score(qs, grid_data, labelmap(fill(:U, size(grid_range,2)))), grid_resolution, grid_resolution)

    data_pools = fill(:U, size(m.data,2))
    query_object = OneClassActiveLearning.get_query_object(qs, m.data, data_pools, collect(1:size(m.data,2)), history)
    title := "Query Scores (Selected ID: $query_object)"

    cbar := false
    @series begin
        seriestype := :contourf
        seriescolor --> :bjy
        levels := range(0, maximum(grid_scores), length=10)
        grid_range, grid_range, grid_scores
    end

    colors = (in_history = :green, not_in_history = :white)
    sub_idx_history = [i ∈ history for i in 1:size(m.data, 2)]

    seriestype := :scatter
    markersize := 5
    markeralpha := 0.7
    @series begin
        label := "history"
        markercolor := :orange
        markershape := :square
        split_2d_array(m.data, sub_idx_history)
    end

    @series begin
        label := "non-history"
        markercolor := :lightgrey
        OCALPlots.split_2d_array(m.data, .!sub_idx_history)
    end

    @series begin
        label := "query-selection"
        markercolor := :black
        markershape := :star5
        markersize := 7
        split_2d_array(m.data, 1:size(m.data, 2) .== query_object)
    end
end


@recipe function plot_svdd(m::SVDD.SVDDClassifier, poolmap::Dict{Symbol, Vector{Int}})
    seriestype := :scatter
    markeralpha --> 0.7
    colors = (U = :grey, Lin = :blue, Lout = :red)
    shapes = (U = :circle, Lin = :square, Lout = :square)
    title := "Pools"

    for (k,v) in poolmap
        @series begin
            label := k
            markercolor := colors[k]
            markershape := shapes[k]
            OCALPlots.split_2d_array(m.data, [i ∈ v for i in 1:size(m.data,2)])
        end
    end
end

@recipe function f(m::SVDD.SVDDClassifier, qs::OneClassActiveLearning.ModelBasedPQs, history, labels::Vector{Symbol}, poolmap::Dict{Symbol, Vector{Int}})
    layout := @layout [a{0.5w, 1.0h} b{0.5w, 0.5h}
                       _ c{0.5w, 0.5h} ]
    legend --> :topright
    @series begin
       subplot := 1
       (m, labels)
    end

    @series begin
        subplot := 2
       (m, qs, history)
    end

    @series begin
        subplot := 3
       (m, poolmap)
    end
end
