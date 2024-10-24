using LinearAlgebra, CSV, DataFrames, Statistics, StatsPlots, YFinance

```
Projection onto halfspace:
```
function proj_halfspace(b, a, β)
    γ = dot(a,b) - β
    if γ ≤ 0
        return b
    else
        return b - γ/dot(a,a) * a
    end
end

"""
Projection onto the unit simplex.
    projsplx(b) -> x
Variant of `projsplx`.
"""
function proj_simplex(b::Vector; τ=1.0)
    x = copy(b)
    projsplx!(x, τ)
    return x
end;

function projsplx!(b::Vector{T}, τ::T) where T

    n = length(b)
    bget = false

    idx = sortperm(b, rev=true)
    tsum = zero(T)

    @inbounds for i = 1:n-1
        tsum += b[idx[i]]
        tmax = (tsum - τ)/i
        if tmax ≥ b[idx[i+1]]
            bget = true
            break
        end
    end

    if !bget
        tmax = (tsum + b[idx[n]] - τ) / n
    end

    @inbounds for i = 1:n
        b[i] = max(b[i] - tmax, 0)
    end
end;

function proj_intersection(x::Vector, p1::Function, p2::Function; maxits=100)
    err = []
    x0 = copy(x)
    ϵ *= norm(x0, Inf)
    for i ∈ 1:maxits
        x = p1(x)
        x = p2(x)
        push!(err, norm(x-x0, Inf))
        err[end] ≤ ϵ && break
        x0 = copy(x)
    end
    return x, err
end;

# Define stocks in S&P500, in technology, services, finance, energy sectors
stocks = 
    ["AAPL", "META", "GOOG", "MSFT", "NVDA", # technology
     "AMZN", "COST", "EBAY", "TGT", "WMT",   # services
     "BMO", "BNS", "HBCP", "RY", "TD",       # finance
     "BP", "CVX", "IMO", "TTE", "XOM"        # energy
    ]

tmpdir = mktempdir()

for s in stocks
    url = "https://query1.finance.yahoo.com/v7/finance/download/$(s)?period1=1647815267&period2=1679351267&interval=1d&events=history&includeAdjustedClose=true"
    Base.download(url, joinpath(tmpdir,"$(s).csv"))
end

df = map(stocks) do stock
    f = joinpath(tmpdir,"$(stock).csv")
    dfs = DataFrame(CSV.File(f, select=[1,6])) # keep only date and closing price
    rename!(dfs, "Adj Close" => stock)
    select!(dfs, :Date, "$stock" .=> p->(p.-p[1])/p[1])
    rename!(dfs, "$(stock)_function" => "$stock")
end |> z->innerjoin(z..., on=:Date)

function meancov(df)
    r = mean.(eachcol(df[:,2:end]))
    Σ = cov(Matrix(df[:,2:end]))
    return r, Σ
end;

r, Σ = meancov(df)
df_plot = DataFrame(stock=stocks, r=r)
sort!(df_plot, :r)
@df df_plot bar(:stock, :r, legend=false)
xlabel!("Stock"); ylabel!("Return"); title!("Returns of Stocks")

#####################################################################################################################################################################################################
"""
Projected gradient method for the quadratic optimization problem

    minimize_{x}  1/2 x'Qx  subj to  x ∈ C.

The function `proj(b)` compute the projection of the vector `b` onto the convex set 𝒞.
"""
function pgrad(Q, proj::Function; maxits=100, ϵ=1e-5)
    x = proj(zeros(size(Q, 1)))
    err = []
    L = eigmax(Q)
    x0 = copy(x)

    for i in 1:maxits
        gk = Q * x
        x = proj(x - (1/L) * gk)
        push!(err, norm(x - x0, Inf))
        err[end] ≤ ϵ && break
        x0 = copy(x)
    end

    return x, err
end;


# Complete this function
function efficient_portfolio(r, Σ, μ)
    n = length(r)
    p1 = b -> proj_halfspace(b, -r, -μ)
    p2 = b -> proj_simplex(b)
    p = b -> proj_intersection(b, p1, p2, maxits=10000, ϵ=1e-8)[1]
    x = pgrad(Σ, p, maxits=1000)
end;

xp1 = efficient_portfolio(r, Σ, .1)
pie(stocks, xp1, title="Portfolio Allocation")



###########################################################################################################################################################################################################################
# Define target returns
target_returns = range(0, 0.1, length=10)

# Initialize arrays to store results
portfolio_risks = Float64[]
portfolio_returns = Float64[]

xp1, xp1_itn_errors = efficient_portfolio(r, Σ, .1)

# Loop over target returns
for μ in target_returns
    x = efficient_portfolio(r, Σ, μ)[1]
    risks = sqrt(x' * Σ * x)
    returns = r' * x
    push!(portfolio_risks, risks)
    push!(portfolio_returns, returns)
end

plot(portfolio_returns, portfolio_risks, label="Efficient Frontier", 
    xlabel="Return", ylabel="Risk", title="Portfolio Risk vs Return")