## Activation Kernels
##
## CPU implementations of activation functions.

import std/math
import ../../[dtype, shape, tensor]

# =============================================================================
# ReLU
# =============================================================================

proc reluKernel*(input: TensorData): TensorData =
  ## ReLU activation: max(0, x)
  result = newTensorDataZeros(input.shape, input.dtype)

  case input.dtype
  of dtFloat32:
    let inArr = input.asFloat32
    let outArr = result.asFloat32
    for i in 0..<input.size:
      outArr[i] = max(0.0'f32, inArr[i])
  of dtFloat64:
    let inArr = input.asFloat64
    let outArr = result.asFloat64
    for i in 0..<input.size:
      outArr[i] = max(0.0'f64, inArr[i])
  else:
    discard

proc leakyReluKernel*(input: TensorData, negativeSlope: float = 0.01): TensorData =
  ## Leaky ReLU: max(negative_slope * x, x)
  result = newTensorDataZeros(input.shape, input.dtype)

  case input.dtype
  of dtFloat32:
    let inArr = input.asFloat32
    let outArr = result.asFloat32
    let slope = negativeSlope.float32
    for i in 0..<input.size:
      outArr[i] = if inArr[i] >= 0: inArr[i] else: slope * inArr[i]
  of dtFloat64:
    let inArr = input.asFloat64
    let outArr = result.asFloat64
    for i in 0..<input.size:
      outArr[i] = if inArr[i] >= 0: inArr[i] else: negativeSlope * inArr[i]
  else:
    discard

# =============================================================================
# Sigmoid
# =============================================================================

proc sigmoidKernel*(input: TensorData): TensorData =
  ## Sigmoid: 1 / (1 + exp(-x))
  result = newTensorDataZeros(input.shape, input.dtype)

  case input.dtype
  of dtFloat32:
    let inArr = input.asFloat32
    let outArr = result.asFloat32
    for i in 0..<input.size:
      outArr[i] = 1.0'f32 / (1.0'f32 + exp(-inArr[i]))
  of dtFloat64:
    let inArr = input.asFloat64
    let outArr = result.asFloat64
    for i in 0..<input.size:
      outArr[i] = 1.0'f64 / (1.0'f64 + exp(-inArr[i]))
  else:
    discard

# =============================================================================
# Tanh
# =============================================================================

proc tanhKernel*(input: TensorData): TensorData =
  ## Tanh activation
  result = newTensorDataZeros(input.shape, input.dtype)

  case input.dtype
  of dtFloat32:
    let inArr = input.asFloat32
    let outArr = result.asFloat32
    for i in 0..<input.size:
      outArr[i] = tanh(inArr[i])
  of dtFloat64:
    let inArr = input.asFloat64
    let outArr = result.asFloat64
    for i in 0..<input.size:
      outArr[i] = tanh(inArr[i])
  else:
    discard

# =============================================================================
# GELU
# =============================================================================

proc geluKernel*(input: TensorData): TensorData =
  ## GELU: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
  result = newTensorDataZeros(input.shape, input.dtype)
  const sqrt2OverPi = 0.7978845608028654  # sqrt(2/pi)

  case input.dtype
  of dtFloat32:
    let inArr = input.asFloat32
    let outArr = result.asFloat32
    for i in 0..<input.size:
      let x = inArr[i]
      let inner = sqrt2OverPi.float32 * (x + 0.044715'f32 * x * x * x)
      outArr[i] = 0.5'f32 * x * (1.0'f32 + tanh(inner))
  of dtFloat64:
    let inArr = input.asFloat64
    let outArr = result.asFloat64
    for i in 0..<input.size:
      let x = inArr[i]
      let inner = sqrt2OverPi * (x + 0.044715 * x * x * x)
      outArr[i] = 0.5 * x * (1.0 + tanh(inner))
  else:
    discard

# =============================================================================
# SiLU (Swish)
# =============================================================================

proc siluKernel*(input: TensorData): TensorData =
  ## SiLU/Swish: x * sigmoid(x)
  result = newTensorDataZeros(input.shape, input.dtype)

  case input.dtype
  of dtFloat32:
    let inArr = input.asFloat32
    let outArr = result.asFloat32
    for i in 0..<input.size:
      let x = inArr[i]
      outArr[i] = x / (1.0'f32 + exp(-x))
  of dtFloat64:
    let inArr = input.asFloat64
    let outArr = result.asFloat64
    for i in 0..<input.size:
      let x = inArr[i]
      outArr[i] = x / (1.0'f64 + exp(-x))
  else:
    discard

# =============================================================================
# Softmax
# =============================================================================

proc softmaxKernel*(input: TensorData, dim: int = -1): TensorData =
  ## Softmax along specified dimension
  ## For simplicity, assumes input is 2D (batch, features) and dim=-1
  result = newTensorDataZeros(input.shape, input.dtype)

  if input.shape.rank == 2:
    let batchSize = input.shape.dims[0]
    let features = input.shape.dims[1]

    case input.dtype
    of dtFloat32:
      let inArr = input.asFloat32
      let outArr = result.asFloat32

      for b in 0..<batchSize:
        let offset = b * features

        # Find max for numerical stability
        var maxVal = inArr[offset]
        for f in 1..<features:
          if inArr[offset + f] > maxVal:
            maxVal = inArr[offset + f]

        # Compute exp and sum
        var sumExp = 0.0'f32
        for f in 0..<features:
          outArr[offset + f] = exp(inArr[offset + f] - maxVal)
          sumExp += outArr[offset + f]

        # Normalize
        for f in 0..<features:
          outArr[offset + f] /= sumExp

    of dtFloat64:
      let inArr = input.asFloat64
      let outArr = result.asFloat64

      for b in 0..<batchSize:
        let offset = b * features

        var maxVal = inArr[offset]
        for f in 1..<features:
          if inArr[offset + f] > maxVal:
            maxVal = inArr[offset + f]

        var sumExp = 0.0'f64
        for f in 0..<features:
          outArr[offset + f] = exp(inArr[offset + f] - maxVal)
          sumExp += outArr[offset + f]

        for f in 0..<features:
          outArr[offset + f] /= sumExp
    else:
      discard

# =============================================================================
# Dropout
# =============================================================================

proc dropoutKernel*(input: TensorData, p: float, training: bool): TensorData =
  ## Dropout: randomly zero elements with probability p during training
  result = newTensorDataZeros(input.shape, input.dtype)

  if not training or p == 0.0:
    # Just copy input
    case input.dtype
    of dtFloat32:
      let inArr = input.asFloat32
      let outArr = result.asFloat32
      for i in 0..<input.size:
        outArr[i] = inArr[i]
    of dtFloat64:
      let inArr = input.asFloat64
      let outArr = result.asFloat64
      for i in 0..<input.size:
        outArr[i] = inArr[i]
    else:
      discard
  else:
    let scale = 1.0 / (1.0 - p)
    case input.dtype
    of dtFloat32:
      let inArr = input.asFloat32
      let outArr = result.asFloat32
      for i in 0..<input.size:
        # Simple deterministic "dropout" for now (proper impl needs RNG)
        # In practice, would use random number generator
        if (i mod 100).float / 100.0 >= p:
          outArr[i] = inArr[i] * scale.float32
    of dtFloat64:
      let inArr = input.asFloat64
      let outArr = result.asFloat64
      for i in 0..<input.size:
        if (i mod 100).float / 100.0 >= p:
          outArr[i] = inArr[i] * scale
    else:
      discard
