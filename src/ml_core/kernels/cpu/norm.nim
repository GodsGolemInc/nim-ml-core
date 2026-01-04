## Normalization Kernels
##
## CPU implementations of normalization operations.

import std/math
import ../../[dtype, shape, tensor]

# =============================================================================
# BatchNorm
# =============================================================================

proc batchNorm2dKernel*(input: TensorData, gamma: TensorData, beta: TensorData,
                        runningMean: TensorData, runningVar: TensorData,
                        eps: float, training: bool, momentum: float): tuple[output: TensorData, mean: TensorData, variance: TensorData] =
  ## Batch Normalization 2D
  ## Input: (N, C, H, W)
  ## Output: (N, C, H, W)
  assert input.shape.rank == 4, "batchNorm2d expects 4D input"

  let batchSize = input.shape.dims[0]
  let numChannels = input.shape.dims[1]
  let height = input.shape.dims[2]
  let width = input.shape.dims[3]
  let spatialSize = height * width
  let batchSpatial = batchSize * spatialSize

  result.output = newTensorDataZeros(input.shape, input.dtype)
  result.mean = newTensorDataZeros(newShape(numChannels), input.dtype)
  result.variance = newTensorDataZeros(newShape(numChannels), input.dtype)

  case input.dtype
  of dtFloat32:
    let inArr = input.asFloat32
    let outArr = result.output.asFloat32
    let meanArr = result.mean.asFloat32
    let varArr = result.variance.asFloat32

    # Compute mean and variance per channel
    for c in 0..<numChannels:
      var mean = 0.0'f32
      for n in 0..<batchSize:
        for h in 0..<height:
          for w in 0..<width:
            let idx = n * numChannels * spatialSize + c * spatialSize + h * width + w
            mean += inArr[idx]
      mean /= batchSpatial.float32
      meanArr[c] = mean

      var variance = 0.0'f32
      for n in 0..<batchSize:
        for h in 0..<height:
          for w in 0..<width:
            let idx = n * numChannels * spatialSize + c * spatialSize + h * width + w
            let diff = inArr[idx] - mean
            variance += diff * diff
      variance /= batchSpatial.float32
      varArr[c] = variance

    # Normalize and apply affine transformation
    let gammaArr = if gamma.isNil or gamma.size == 0: nil else: gamma.asFloat32
    let betaArr = if beta.isNil or beta.size == 0: nil else: beta.asFloat32

    for c in 0..<numChannels:
      let mean = meanArr[c]
      let variance = varArr[c]
      let invStd = 1.0'f32 / sqrt(variance + eps.float32)
      let g = if gammaArr.isNil: 1.0'f32 else: gammaArr[c]
      let b = if betaArr.isNil: 0.0'f32 else: betaArr[c]

      for n in 0..<batchSize:
        for h in 0..<height:
          for w in 0..<width:
            let idx = n * numChannels * spatialSize + c * spatialSize + h * width + w
            outArr[idx] = g * (inArr[idx] - mean) * invStd + b

  of dtFloat64:
    let inArr = input.asFloat64
    let outArr = result.output.asFloat64
    let meanArr = result.mean.asFloat64
    let varArr = result.variance.asFloat64

    for c in 0..<numChannels:
      var mean = 0.0'f64
      for n in 0..<batchSize:
        for h in 0..<height:
          for w in 0..<width:
            let idx = n * numChannels * spatialSize + c * spatialSize + h * width + w
            mean += inArr[idx]
      mean /= batchSpatial.float64
      meanArr[c] = mean

      var variance = 0.0'f64
      for n in 0..<batchSize:
        for h in 0..<height:
          for w in 0..<width:
            let idx = n * numChannels * spatialSize + c * spatialSize + h * width + w
            let diff = inArr[idx] - mean
            variance += diff * diff
      variance /= batchSpatial.float64
      varArr[c] = variance

    let gammaArr = if gamma.isNil or gamma.size == 0: nil else: gamma.asFloat64
    let betaArr = if beta.isNil or beta.size == 0: nil else: beta.asFloat64

    for c in 0..<numChannels:
      let mean = meanArr[c]
      let variance = varArr[c]
      let invStd = 1.0'f64 / sqrt(variance + eps)
      let g = if gammaArr.isNil: 1.0'f64 else: gammaArr[c]
      let b = if betaArr.isNil: 0.0'f64 else: betaArr[c]

      for n in 0..<batchSize:
        for h in 0..<height:
          for w in 0..<width:
            let idx = n * numChannels * spatialSize + c * spatialSize + h * width + w
            outArr[idx] = g * (inArr[idx] - mean) * invStd + b
  else:
    discard

# =============================================================================
# LayerNorm
# =============================================================================

proc layerNormKernel*(input: TensorData, normalizedShape: seq[int],
                      gamma: TensorData, beta: TensorData,
                      eps: float): TensorData =
  ## Layer Normalization
  ## Normalizes over the last len(normalizedShape) dimensions
  let inputRank = input.shape.rank
  let normDims = normalizedShape.len

  # Calculate normalization size
  var normSize = 1
  for i in 0..<normDims:
    normSize *= normalizedShape[i]

  # Calculate batch size (remaining dimensions)
  var batchSize = 1
  for i in 0..<(inputRank - normDims):
    batchSize *= input.shape.dims[i]

  result = newTensorDataZeros(input.shape, input.dtype)

  case input.dtype
  of dtFloat32:
    let inArr = input.asFloat32
    let outArr = result.asFloat32
    let gammaArr = if gamma.isNil or gamma.size == 0: nil else: gamma.asFloat32
    let betaArr = if beta.isNil or beta.size == 0: nil else: beta.asFloat32

    for b in 0..<batchSize:
      let offset = b * normSize

      # Compute mean
      var mean = 0.0'f32
      for i in 0..<normSize:
        mean += inArr[offset + i]
      mean /= normSize.float32

      # Compute variance
      var variance = 0.0'f32
      for i in 0..<normSize:
        let diff = inArr[offset + i] - mean
        variance += diff * diff
      variance /= normSize.float32

      # Normalize and apply affine
      let invStd = 1.0'f32 / sqrt(variance + eps.float32)
      for i in 0..<normSize:
        let g = if gammaArr.isNil: 1.0'f32 else: gammaArr[i]
        let be = if betaArr.isNil: 0.0'f32 else: betaArr[i]
        outArr[offset + i] = g * (inArr[offset + i] - mean) * invStd + be

  of dtFloat64:
    let inArr = input.asFloat64
    let outArr = result.asFloat64
    let gammaArr = if gamma.isNil or gamma.size == 0: nil else: gamma.asFloat64
    let betaArr = if beta.isNil or beta.size == 0: nil else: beta.asFloat64

    for b in 0..<batchSize:
      let offset = b * normSize

      var mean = 0.0'f64
      for i in 0..<normSize:
        mean += inArr[offset + i]
      mean /= normSize.float64

      var variance = 0.0'f64
      for i in 0..<normSize:
        let diff = inArr[offset + i] - mean
        variance += diff * diff
      variance /= normSize.float64

      let invStd = 1.0'f64 / sqrt(variance + eps)
      for i in 0..<normSize:
        let g = if gammaArr.isNil: 1.0'f64 else: gammaArr[i]
        let be = if betaArr.isNil: 0.0'f64 else: betaArr[i]
        outArr[offset + i] = g * (inArr[offset + i] - mean) * invStd + be
  else:
    discard

# =============================================================================
# RMSNorm
# =============================================================================

proc rmsNormKernel*(input: TensorData, normalizedShape: seq[int],
                    gamma: TensorData, eps: float): TensorData =
  ## RMS Normalization (no mean subtraction)
  let inputRank = input.shape.rank
  let normDims = normalizedShape.len

  var normSize = 1
  for i in 0..<normDims:
    normSize *= normalizedShape[i]

  var batchSize = 1
  for i in 0..<(inputRank - normDims):
    batchSize *= input.shape.dims[i]

  result = newTensorDataZeros(input.shape, input.dtype)

  case input.dtype
  of dtFloat32:
    let inArr = input.asFloat32
    let outArr = result.asFloat32
    let gammaArr = if gamma.isNil or gamma.size == 0: nil else: gamma.asFloat32

    for b in 0..<batchSize:
      let offset = b * normSize

      # Compute RMS
      var sumSq = 0.0'f32
      for i in 0..<normSize:
        sumSq += inArr[offset + i] * inArr[offset + i]
      let rms = sqrt(sumSq / normSize.float32 + eps.float32)

      # Normalize
      for i in 0..<normSize:
        let g = if gammaArr.isNil: 1.0'f32 else: gammaArr[i]
        outArr[offset + i] = g * inArr[offset + i] / rms

  of dtFloat64:
    let inArr = input.asFloat64
    let outArr = result.asFloat64
    let gammaArr = if gamma.isNil or gamma.size == 0: nil else: gamma.asFloat64

    for b in 0..<batchSize:
      let offset = b * normSize

      var sumSq = 0.0'f64
      for i in 0..<normSize:
        sumSq += inArr[offset + i] * inArr[offset + i]
      let rms = sqrt(sumSq / normSize.float64 + eps)

      for i in 0..<normSize:
        let g = if gammaArr.isNil: 1.0'f64 else: gammaArr[i]
        outArr[offset + i] = g * inArr[offset + i] / rms
  else:
    discard

# =============================================================================
# GroupNorm
# =============================================================================

proc groupNormKernel*(input: TensorData, numGroups: int,
                      gamma: TensorData, beta: TensorData,
                      eps: float): TensorData =
  ## Group Normalization
  ## Input: (N, C, ...)
  assert input.shape.rank >= 2, "groupNorm expects at least 2D input"

  let batchSize = input.shape.dims[0]
  let numChannels = input.shape.dims[1]
  let channelsPerGroup = numChannels div numGroups

  # Spatial size (product of remaining dims)
  var spatialSize = 1
  for i in 2..<input.shape.rank:
    spatialSize *= input.shape.dims[i]

  let groupSize = channelsPerGroup * spatialSize

  result = newTensorDataZeros(input.shape, input.dtype)

  case input.dtype
  of dtFloat32:
    let inArr = input.asFloat32
    let outArr = result.asFloat32
    let gammaArr = if gamma.isNil or gamma.size == 0: nil else: gamma.asFloat32
    let betaArr = if beta.isNil or beta.size == 0: nil else: beta.asFloat32

    for n in 0..<batchSize:
      for g in 0..<numGroups:
        let groupStart = g * channelsPerGroup

        # Compute mean and variance for this group
        var mean = 0.0'f32
        for c in groupStart..<(groupStart + channelsPerGroup):
          for s in 0..<spatialSize:
            let idx = n * numChannels * spatialSize + c * spatialSize + s
            mean += inArr[idx]
        mean /= groupSize.float32

        var variance = 0.0'f32
        for c in groupStart..<(groupStart + channelsPerGroup):
          for s in 0..<spatialSize:
            let idx = n * numChannels * spatialSize + c * spatialSize + s
            let diff = inArr[idx] - mean
            variance += diff * diff
        variance /= groupSize.float32

        let invStd = 1.0'f32 / sqrt(variance + eps.float32)

        # Normalize
        for c in groupStart..<(groupStart + channelsPerGroup):
          let ga = if gammaArr.isNil: 1.0'f32 else: gammaArr[c]
          let be = if betaArr.isNil: 0.0'f32 else: betaArr[c]
          for s in 0..<spatialSize:
            let idx = n * numChannels * spatialSize + c * spatialSize + s
            outArr[idx] = ga * (inArr[idx] - mean) * invStd + be

  of dtFloat64:
    let inArr = input.asFloat64
    let outArr = result.asFloat64
    let gammaArr = if gamma.isNil or gamma.size == 0: nil else: gamma.asFloat64
    let betaArr = if beta.isNil or beta.size == 0: nil else: beta.asFloat64

    for n in 0..<batchSize:
      for g in 0..<numGroups:
        let groupStart = g * channelsPerGroup

        var mean = 0.0'f64
        for c in groupStart..<(groupStart + channelsPerGroup):
          for s in 0..<spatialSize:
            let idx = n * numChannels * spatialSize + c * spatialSize + s
            mean += inArr[idx]
        mean /= groupSize.float64

        var variance = 0.0'f64
        for c in groupStart..<(groupStart + channelsPerGroup):
          for s in 0..<spatialSize:
            let idx = n * numChannels * spatialSize + c * spatialSize + s
            let diff = inArr[idx] - mean
            variance += diff * diff
        variance /= groupSize.float64

        let invStd = 1.0'f64 / sqrt(variance + eps)

        for c in groupStart..<(groupStart + channelsPerGroup):
          let ga = if gammaArr.isNil: 1.0'f64 else: gammaArr[c]
          let be = if betaArr.isNil: 0.0'f64 else: betaArr[c]
          for s in 0..<spatialSize:
            let idx = n * numChannels * spatialSize + c * spatialSize + s
            outArr[idx] = ga * (inArr[idx] - mean) * invStd + be
  else:
    discard
