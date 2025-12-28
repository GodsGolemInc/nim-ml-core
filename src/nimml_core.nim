## nim-ml-core: Core ML types for the heterogeneous distributed ML framework
##
## This module provides platform-agnostic ML primitives:
## - DType: Data types with promotion rules
## - Shape: Multi-dimensional shapes with broadcasting
## - TensorRef: Content-addressed tensor references
## - OpSpec: Operation specifications
## - Graph IR: Computation graph representation

import nimml_core/dtype
import nimml_core/shape
import nimml_core/tensor
import nimml_core/ops
import nimml_core/ir

export dtype
export shape
export tensor
export ops
export ir
