tests = [
    "uniloss",
    "multiloss",
    "prediction",
    "risks"
]


for t in tests
    tfile = string(t, ".jl")
    println("  * $tfile ...")
    include(tfile)
end
