tests = [
    "uniloss",
    "multiloss",
    "prediction",
    "link",
    "risks",
    "regularizers"
]


for t in tests
    tfile = string(t, ".jl")
    println("  * $tfile ...")
    include(tfile)
end
