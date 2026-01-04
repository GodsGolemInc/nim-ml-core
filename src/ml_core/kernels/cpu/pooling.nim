## Pooling Kernels
##
## CPU implementations of pooling operations.

import std/math
import ../../[dtype, shape, tensor]

# =============================================================================
# MaxPool2d
# =============================================================================

proc maxPool2dKernel*(input: TensorData, kernelH, kernelW: int,
                      strideH, strideW: int,
                      padH, padW: int,
                      dilationH, dilationW: int): tuple[output: TensorData, indices: TensorData] =
  ## Max Pooling 2D
  ## Input: (N, C, H, W)
  ## Output: (N, C, H_out, W_out)
  assert input.shape.rank == 4, "maxPool2d expects 4D input"

  let batchSize = input.shape.dims[0]
  let channels = input.shape.dims[1]
  let inH = input.shape.dims[2]
  let inW = input.shape.dims[3]

  let effKernelH = dilationH * (kernelH - 1) + 1
  let effKernelW = dilationW * (kernelW - 1) + 1

  let outH = (inH + 2 * padH - effKernelH) div strideH + 1
  let outW = (inW + 2 * padW - effKernelW) div strideW + 1

  result.output = newTensorDataZeros(newShape(batchSize, channels, outH, outW), input.dtype)
  result.indices = newTensorDataZeros(newShape(batchSize, channels, outH, outW), dtInt64)

  case input.dtype
  of dtFloat32:
    let inArr = input.asFloat32
    let outArr = result.output.asFloat32
    let idxArr = result.indices.asInt64

    for n in 0..<batchSize:
      for c in 0..<channels:
        for oh in 0..<outH:
          for ow in 0..<outW:
            var maxVal = -Inf.float32
            var maxIdx = 0

            for kh in 0..<kernelH:
              for kw in 0..<kernelW:
                let ih = oh * strideH - padH + kh * dilationH
                let iw = ow * strideW - padW + kw * dilationW

                if ih >= 0 and ih < inH and iw >= 0 and iw < inW:
                  let inIdx = n * channels * inH * inW + c * inH * inW + ih * inW + iw
                  if inArr[inIdx] > maxVal:
                    maxVal = inArr[inIdx]
                    maxIdx = inIdx

            let outIdx = n * channels * outH * outW + c * outH * outW + oh * outW + ow
            outArr[outIdx] = maxVal
            idxArr[outIdx] = maxIdx.int64

  of dtFloat64:
    let inArr = input.asFloat64
    let outArr = result.output.asFloat64
    let idxArr = result.indices.asInt64

    for n in 0..<batchSize:
      for c in 0..<channels:
        for oh in 0..<outH:
          for ow in 0..<outW:
            var maxVal = -Inf.float64
            var maxIdx = 0

            for kh in 0..<kernelH:
              for kw in 0..<kernelW:
                let ih = oh * strideH - padH + kh * dilationH
                let iw = ow * strideW - padW + kw * dilationW

                if ih >= 0 and ih < inH and iw >= 0 and iw < inW:
                  let inIdx = n * channels * inH * inW + c * inH * inW + ih * inW + iw
                  if inArr[inIdx] > maxVal:
                    maxVal = inArr[inIdx]
                    maxIdx = inIdx

            let outIdx = n * channels * outH * outW + c * outH * outW + oh * outW + ow
            outArr[outIdx] = maxVal
            idxArr[outIdx] = maxIdx.int64
  else:
    discard

# =============================================================================
# AvgPool2d
# =============================================================================

proc avgPool2dKernel*(input: TensorData, kernelH, kernelW: int,
                      strideH, strideW: int,
                      padH, padW: int,
                      countIncludePad: bool): TensorData =
  ## Average Pooling 2D
  ## Input: (N, C, H, W)
  ## Output: (N, C, H_out, W_out)
  assert input.shape.rank == 4, "avgPool2d expects 4D input"

  let batchSize = input.shape.dims[0]
  let channels = input.shape.dims[1]
  let inH = input.shape.dims[2]
  let inW = input.shape.dims[3]

  let outH = (inH + 2 * padH - kernelH) div strideH + 1
  let outW = (inW + 2 * padW - kernelW) div strideW + 1

  result = newTensorDataZeros(newShape(batchSize, channels, outH, outW), input.dtype)

  case input.dtype
  of dtFloat32:
    let inArr = input.asFloat32
    let outArr = result.asFloat32

    for n in 0..<batchSize:
      for c in 0..<channels:
        for oh in 0..<outH:
          for ow in 0..<outW:
            var sum = 0.0'f32
            var count = 0

            for kh in 0..<kernelH:
              for kw in 0..<kernelW:
                let ih = oh * strideH - padH + kh
                let iw = ow * strideW - padW + kw

                if ih >= 0 and ih < inH and iw >= 0 and iw < inW:
                  let inIdx = n * channels * inH * inW + c * inH * inW + ih * inW + iw
                  sum += inArr[inIdx]
                  count += 1
                elif countIncludePad:
                  count += 1

            let outIdx = n * channels * outH * outW + c * outH * outW + oh * outW + ow
            if countIncludePad:
              outArr[outIdx] = sum / (kernelH * kernelW).float32
            else:
              outArr[outIdx] = if count > 0: sum / count.float32 else: 0.0'f32

  of dtFloat64:
    let inArr = input.asFloat64
    let outArr = result.asFloat64

    for n in 0..<batchSize:
      for c in 0..<channels:
        for oh in 0..<outH:
          for ow in 0..<outW:
            var sum = 0.0'f64
            var count = 0

            for kh in 0..<kernelH:
              for kw in 0..<kernelW:
                let ih = oh * strideH - padH + kh
                let iw = ow * strideW - padW + kw

                if ih >= 0 and ih < inH and iw >= 0 and iw < inW:
                  let inIdx = n * channels * inH * inW + c * inH * inW + ih * inW + iw
                  sum += inArr[inIdx]
                  count += 1
                elif countIncludePad:
                  count += 1

            let outIdx = n * channels * outH * outW + c * outH * outW + oh * outW + ow
            if countIncludePad:
              outArr[outIdx] = sum / (kernelH * kernelW).float64
            else:
              outArr[outIdx] = if count > 0: sum / count.float64 else: 0.0'f64
  else:
    discard

# =============================================================================
# AdaptiveAvgPool2d
# =============================================================================

proc adaptiveAvgPool2dKernel*(input: TensorData, outputH, outputW: int): TensorData =
  ## Adaptive Average Pooling 2D
  ## Automatically calculates kernel size and stride to produce target output size
  assert input.shape.rank == 4, "adaptiveAvgPool2d expects 4D input"

  let batchSize = input.shape.dims[0]
  let channels = input.shape.dims[1]
  let inH = input.shape.dims[2]
  let inW = input.shape.dims[3]

  result = newTensorDataZeros(newShape(batchSize, channels, outputH, outputW), input.dtype)

  case input.dtype
  of dtFloat32:
    let inArr = input.asFloat32
    let outArr = result.asFloat32

    for n in 0..<batchSize:
      for c in 0..<channels:
        for oh in 0..<outputH:
          for ow in 0..<outputW:
            # Calculate input region for this output
            let ihStart = (oh * inH) div outputH
            let ihEnd = ((oh + 1) * inH + outputH - 1) div outputH
            let iwStart = (ow * inW) div outputW
            let iwEnd = ((ow + 1) * inW + outputW - 1) div outputW

            var sum = 0.0'f32
            var count = 0
            for ih in ihStart..<ihEnd:
              for iw in iwStart..<iwEnd:
                let inIdx = n * channels * inH * inW + c * inH * inW + ih * inW + iw
                sum += inArr[inIdx]
                count += 1

            let outIdx = n * channels * outputH * outputW + c * outputH * outputW + oh * outputW + ow
            outArr[outIdx] = if count > 0: sum / count.float32 else: 0.0'f32

  of dtFloat64:
    let inArr = input.asFloat64
    let outArr = result.asFloat64

    for n in 0..<batchSize:
      for c in 0..<channels:
        for oh in 0..<outputH:
          for ow in 0..<outputW:
            let ihStart = (oh * inH) div outputH
            let ihEnd = ((oh + 1) * inH + outputH - 1) div outputH
            let iwStart = (ow * inW) div outputW
            let iwEnd = ((ow + 1) * inW + outputW - 1) div outputW

            var sum = 0.0'f64
            var count = 0
            for ih in ihStart..<ihEnd:
              for iw in iwStart..<iwEnd:
                let inIdx = n * channels * inH * inW + c * inH * inW + ih * inW + iw
                sum += inArr[inIdx]
                count += 1

            let outIdx = n * channels * outputH * outputW + c * outputH * outputW + oh * outputW + ow
            outArr[outIdx] = if count > 0: sum / count.float64 else: 0.0'f64
  else:
    discard

# =============================================================================
# MaxPool1d
# =============================================================================

proc maxPool1dKernel*(input: TensorData, kernelSize: int,
                      stride: int, padding: int, dilation: int): tuple[output: TensorData, indices: TensorData] =
  ## Max Pooling 1D
  ## Input: (N, C, L)
  ## Output: (N, C, L_out)
  assert input.shape.rank == 3, "maxPool1d expects 3D input"

  let batchSize = input.shape.dims[0]
  let channels = input.shape.dims[1]
  let inLen = input.shape.dims[2]

  let effKernel = dilation * (kernelSize - 1) + 1
  let outLen = (inLen + 2 * padding - effKernel) div stride + 1

  result.output = newTensorDataZeros(newShape(batchSize, channels, outLen), input.dtype)
  result.indices = newTensorDataZeros(newShape(batchSize, channels, outLen), dtInt64)

  case input.dtype
  of dtFloat32:
    let inArr = input.asFloat32
    let outArr = result.output.asFloat32
    let idxArr = result.indices.asInt64

    for n in 0..<batchSize:
      for c in 0..<channels:
        for ol in 0..<outLen:
          var maxVal = -Inf.float32
          var maxIdx = 0

          for k in 0..<kernelSize:
            let il = ol * stride - padding + k * dilation
            if il >= 0 and il < inLen:
              let inIdx = n * channels * inLen + c * inLen + il
              if inArr[inIdx] > maxVal:
                maxVal = inArr[inIdx]
                maxIdx = inIdx

          let outIdx = n * channels * outLen + c * outLen + ol
          outArr[outIdx] = maxVal
          idxArr[outIdx] = maxIdx.int64

  of dtFloat64:
    let inArr = input.asFloat64
    let outArr = result.output.asFloat64
    let idxArr = result.indices.asInt64

    for n in 0..<batchSize:
      for c in 0..<channels:
        for ol in 0..<outLen:
          var maxVal = -Inf.float64
          var maxIdx = 0

          for k in 0..<kernelSize:
            let il = ol * stride - padding + k * dilation
            if il >= 0 and il < inLen:
              let inIdx = n * channels * inLen + c * inLen + il
              if inArr[inIdx] > maxVal:
                maxVal = inArr[inIdx]
                maxIdx = inIdx

          let outIdx = n * channels * outLen + c * outLen + ol
          outArr[outIdx] = maxVal
          idxArr[outIdx] = maxIdx.int64
  else:
    discard

# =============================================================================
# GlobalAvgPool
# =============================================================================

proc globalAvgPool2dKernel*(input: TensorData): TensorData =
  ## Global Average Pooling 2D
  ## Input: (N, C, H, W)
  ## Output: (N, C, 1, 1)
  adaptiveAvgPool2dKernel(input, 1, 1)
