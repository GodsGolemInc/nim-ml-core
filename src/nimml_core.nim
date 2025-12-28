## nim-ml-core: Core ML types for the heterogeneous distributed ML framework
##
## This module provides platform-agnostic ML primitives:
## - DType: Data types with promotion rules
## - Shape: Multi-dimensional shapes with broadcasting
## - TensorRef: Content-addressed tensor references
## - OpSpec: Operation specifications (v0.0.3)
## - Graph IR: Computation graph representation (v0.0.4)

import nimml_core/dtype
import nimml_core/shape
import nimml_core/tensor

export dtype
export shape
export tensor
