# ForwardDiffMatrixTools

[![Build Status](https://travis-ci.com/magerton/ForwardDiffMatrixTools.jl.svg?branch=main)](https://travis-ci.com/magerton/ForwardDiffMatrixTools.jl)
[![Build Status](https://ci.appveyor.com/api/projects/status/github/magerton/ForwardDiffMatrixTools.jl?svg=true)](https://ci.appveyor.com/project/magerton/ForwardDiffMatrixTools-jl)

This package is designed to allow use of [`ForwardDiff.jl`](https://github.com/JuliaDiff/ForwardDiff.jl) with left division by a sparse matrix (See <https://github.com/JuliaDiff/ForwardDiff.jl/issues/363>). The package is largely an update of [`DualMatrixTools.jl`](https://github.com/briochemc/DualMatrixTools.jl), simply updated to use the `Dual` numbers from [`ForwardDiff`](https://github.com/JuliaDiff/ForwardDiff.jl). It's also inspired by the solution in <https://github.com/JuliaDiff/ForwardDiff.jl/issues/363#issuecomment-585252302>
