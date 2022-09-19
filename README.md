# ForwardDiffSparseSolve

[![Build Status](https://travis-ci.com/magerton/ForwardDiffSparseSolve.jl.svg?branch=main)](https://travis-ci.com/magerton/ForwardDiffSparseSolve.jl)
[![Build Status](https://ci.appveyor.com/api/projects/status/github/magerton/ForwardDiffSparseSolve.jl?svg=true)](https://ci.appveyor.com/project/magerton/ForwardDiffSparseSolve-jl)

This package is designed to allow use of [`ForwardDiff.jl`](https://github.com/JuliaDiff/ForwardDiff.jl) with left division of a Dual vector by a sparse matrix (See <https://github.com/JuliaDiff/ForwardDiff.jl/issues/363>), eg $y = A^{-1}b$ where `A<:SparseMatrixCSC{<:Dual}` and `b<:AbstractVector{<:Dual}`. The package is largely an update of [`DualMatrixTools.jl`](https://github.com/briochemc/DualMatrixTools.jl) by @briochemc, simply updated to use the `Dual` numbers from [`ForwardDiff`](https://github.com/JuliaDiff/ForwardDiff.jl). It's also inspired by the solution in <https://github.com/JuliaDiff/ForwardDiff.jl/issues/363#issuecomment-585252302>
