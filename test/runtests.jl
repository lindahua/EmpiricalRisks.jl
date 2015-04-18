tests = [
    "uniloss",
    "multiloss",
    "prediction",
    "unirisks"
]


for t in tests
    tfile = string(t, ".jl")
    println("  * $tfile ...")
    include(tfile)
end
