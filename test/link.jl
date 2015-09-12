using EmpiricalRisks
using Base.Test

## Auxiliary functions

sigmoid(x::FloatingPoint) = 1. / (1. + exp(-x))

## Data preparation

d = 5
k = 3
n = 12

X = randn(d, n)

## Links

# LogitLink

a = randn()
w = randn(d)
wa = [w; a]
b = 2.5

links = (LogitLink(LinearPred(d)), LogitLink(AffinePred(d, b)))
for link in links
    @test inputlen(link) == inputlen(link.predmodel) 
    @test inputsize(link) == inputsize(link.predmodel)
    @test outputlen(link) == outputlen(link.predmodel)
    @test outputsize(link) == outputsize(link.predmodel)
    @test paramlen(link) == paramlen(link.predmodel)
    @test paramsize(link) == paramsize(link.predmodel)
end
@test isvalidparam(links[1], w)  == isvalidparam(links[1].predmodel, w)
@test isvalidparam(links[2], wa) == isvalidparam(links[2].predmodel, wa)

for i = 1:n
    x_i = X[:,i]
    @test_approx_eq predict(links[1], w, x_i)  sigmoid(predict(links[1].predmodel, w,  x_i))
    @test_approx_eq predict(links[2], wa, x_i) sigmoid(predict(links[2].predmodel, wa, x_i))
end
verify_multipred(links[1], w, X)
verify_multipred(links[2], wa, X)
