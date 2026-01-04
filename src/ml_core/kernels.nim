## Tensor Computation Kernels
##
## This module provides actual computation implementations for tensor operations.
## Currently supports CPU backend; GPU backends can be added in the future.
##
## Usage:
##   import ml_core/kernels
##   let result = addKernel(a, b)
##
## Or import specific backend:
##   import ml_core/kernels/cpu
##   let result = addKernel(a, b)

import kernels/cpu

export cpu
