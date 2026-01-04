## Convolution Kernels
##
## CPU implementations of convolution operations using im2col + matmul.

import std/math
import ../../[dtype, shape, tensor]
import matmul

# =============================================================================
# Im2col / Col2im
# =============================================================================

proc im2colKernel*(input: TensorData, kernelH, kernelW: int,
                   strideH, strideW: int,
                   padH, padW: int,
                   dilationH, dilationW: int): TensorData =
  ## Convert image patches to columns for efficient convolution
  ## Input: (C, H, W)
  ## Output: (C * kH * kW, outH * outW)
  assert input.shape.rank == 3, "im2col expects 3D input (C, H, W)"

  let inC = input.shape.dims[0]
  let inH = input.shape.dims[1]
  let inW = input.shape.dims[2]

  let effKernelH = dilationH * (kernelH - 1) + 1
  let effKernelW = dilationW * (kernelW - 1) + 1

  let outH = (inH + 2 * padH - effKernelH) div strideH + 1
  let outW = (inW + 2 * padW - effKernelW) div strideW + 1

  let colH = inC * kernelH * kernelW
  let colW = outH * outW

  result = newTensorDataZeros(newShape(colH, colW), input.dtype)

  case input.dtype
  of dtFloat32:
    let inArr = input.asFloat32
    let outArr = result.asFloat32

    for c in 0..<inC:
      for kh in 0..<kernelH:
        for kw in 0..<kernelW:
          let colRow = c * kernelH * kernelW + kh * kernelW + kw
          for oh in 0..<outH:
            for ow in 0..<outW:
              let ih = oh * strideH - padH + kh * dilationH
              let iw = ow * strideW - padW + kw * dilationW
              let colCol = oh * outW + ow

              if ih >= 0 and ih < inH and iw >= 0 and iw < inW:
                let inIdx = c * inH * inW + ih * inW + iw
                outArr[colRow * colW + colCol] = inArr[inIdx]
              # else: already 0 (padding)

  of dtFloat64:
    let inArr = input.asFloat64
    let outArr = result.asFloat64

    for c in 0..<inC:
      for kh in 0..<kernelH:
        for kw in 0..<kernelW:
          let colRow = c * kernelH * kernelW + kh * kernelW + kw
          for oh in 0..<outH:
            for ow in 0..<outW:
              let ih = oh * strideH - padH + kh * dilationH
              let iw = ow * strideW - padW + kw * dilationW
              let colCol = oh * outW + ow

              if ih >= 0 and ih < inH and iw >= 0 and iw < inW:
                let inIdx = c * inH * inW + ih * inW + iw
                outArr[colRow * colW + colCol] = inArr[inIdx]
  else:
    discard

proc col2imKernel*(colData: TensorData, inputShape: Shape,
                   kernelH, kernelW: int,
                   strideH, strideW: int,
                   padH, padW: int,
                   dilationH, dilationW: int): TensorData =
  ## Convert columns back to image (for backward pass)
  ## ColData: (C * kH * kW, outH * outW)
  ## Output: (C, H, W)
  assert inputShape.rank == 3, "col2im expects 3D output shape"

  let inC = inputShape.dims[0]
  let inH = inputShape.dims[1]
  let inW = inputShape.dims[2]

  let effKernelH = dilationH * (kernelH - 1) + 1
  let effKernelW = dilationW * (kernelW - 1) + 1

  let outH = (inH + 2 * padH - effKernelH) div strideH + 1
  let outW = (inW + 2 * padW - effKernelW) div strideW + 1

  let colW = outH * outW

  result = newTensorDataZeros(inputShape, colData.dtype)

  case colData.dtype
  of dtFloat32:
    let colArr = colData.asFloat32
    let outArr = result.asFloat32

    for c in 0..<inC:
      for kh in 0..<kernelH:
        for kw in 0..<kernelW:
          let colRow = c * kernelH * kernelW + kh * kernelW + kw
          for oh in 0..<outH:
            for ow in 0..<outW:
              let ih = oh * strideH - padH + kh * dilationH
              let iw = ow * strideW - padW + kw * dilationW
              let colCol = oh * outW + ow

              if ih >= 0 and ih < inH and iw >= 0 and iw < inW:
                let outIdx = c * inH * inW + ih * inW + iw
                outArr[outIdx] += colArr[colRow * colW + colCol]

  of dtFloat64:
    let colArr = colData.asFloat64
    let outArr = result.asFloat64

    for c in 0..<inC:
      for kh in 0..<kernelH:
        for kw in 0..<kernelW:
          let colRow = c * kernelH * kernelW + kh * kernelW + kw
          for oh in 0..<outH:
            for ow in 0..<outW:
              let ih = oh * strideH - padH + kh * dilationH
              let iw = ow * strideW - padW + kw * dilationW
              let colCol = oh * outW + ow

              if ih >= 0 and ih < inH and iw >= 0 and iw < inW:
                let outIdx = c * inH * inW + ih * inW + iw
                outArr[outIdx] += colArr[colRow * colW + colCol]
  else:
    discard

# =============================================================================
# Conv2d Forward
# =============================================================================

proc conv2dKernel*(input: TensorData, weight: TensorData, bias: TensorData,
                   strideH, strideW: int,
                   padH, padW: int,
                   dilationH, dilationW: int,
                   groups: int): TensorData =
  ## 2D Convolution using im2col + matmul
  ## Input: (N, C_in, H, W)
  ## Weight: (C_out, C_in/groups, kH, kW)
  ## Bias: (C_out,) or nil
  ## Output: (N, C_out, H_out, W_out)
  assert input.shape.rank == 4, "conv2d expects 4D input"
  assert weight.shape.rank == 4, "conv2d expects 4D weight"

  let batchSize = input.shape.dims[0]
  let inChannels = input.shape.dims[1]
  let inH = input.shape.dims[2]
  let inW = input.shape.dims[3]

  let outChannels = weight.shape.dims[0]
  let kernelH = weight.shape.dims[2]
  let kernelW = weight.shape.dims[3]

  let effKernelH = dilationH * (kernelH - 1) + 1
  let effKernelW = dilationW * (kernelW - 1) + 1

  let outH = (inH + 2 * padH - effKernelH) div strideH + 1
  let outW = (inW + 2 * padW - effKernelW) div strideW + 1

  result = newTensorDataZeros(newShape(batchSize, outChannels, outH, outW), input.dtype)

  # Reshape weight to (C_out, C_in/groups * kH * kW)
  let weightRows = outChannels
  let weightCols = (inChannels div groups) * kernelH * kernelW
  let weightReshaped = newTensorData(newShape(weightRows, weightCols), weight.dtype)

  case weight.dtype
  of dtFloat32:
    let wArr = weight.asFloat32
    let wrArr = weightReshaped.asFloat32
    for i in 0..<weight.size:
      wrArr[i] = wArr[i]
  of dtFloat64:
    let wArr = weight.asFloat64
    let wrArr = weightReshaped.asFloat64
    for i in 0..<weight.size:
      wrArr[i] = wArr[i]
  else:
    discard

  # Process each batch
  for n in 0..<batchSize:
    # Extract single image (C_in, H, W)
    let imgShape = newShape(inChannels, inH, inW)
    let imgData = newTensorDataZeros(imgShape, input.dtype)

    case input.dtype
    of dtFloat32:
      let inArr = input.asFloat32
      let imgArr = imgData.asFloat32
      let imgSize = inChannels * inH * inW
      for i in 0..<imgSize:
        imgArr[i] = inArr[n * imgSize + i]
    of dtFloat64:
      let inArr = input.asFloat64
      let imgArr = imgData.asFloat64
      let imgSize = inChannels * inH * inW
      for i in 0..<imgSize:
        imgArr[i] = inArr[n * imgSize + i]
    else:
      discard

    # im2col
    let colData = im2colKernel(imgData, kernelH, kernelW,
                               strideH, strideW, padH, padW,
                               dilationH, dilationW)

    # matmul: (C_out, C_in*kH*kW) x (C_in*kH*kW, outH*outW) = (C_out, outH*outW)
    let outFlat = mmKernel(weightReshaped, colData)

    # Add bias if present
    if not bias.isNil and bias.size > 0:
      case input.dtype
      of dtFloat32:
        let outArr = outFlat.asFloat32
        let biasArr = bias.asFloat32
        for c in 0..<outChannels:
          for i in 0..<(outH * outW):
            outArr[c * outH * outW + i] += biasArr[c]
      of dtFloat64:
        let outArr = outFlat.asFloat64
        let biasArr = bias.asFloat64
        for c in 0..<outChannels:
          for i in 0..<(outH * outW):
            outArr[c * outH * outW + i] += biasArr[c]
      else:
        discard

    # Copy to result
    case input.dtype
    of dtFloat32:
      let outArr = outFlat.asFloat32
      let resArr = result.asFloat32
      let outSize = outChannels * outH * outW
      for i in 0..<outSize:
        resArr[n * outSize + i] = outArr[i]
    of dtFloat64:
      let outArr = outFlat.asFloat64
      let resArr = result.asFloat64
      let outSize = outChannels * outH * outW
      for i in 0..<outSize:
        resArr[n * outSize + i] = outArr[i]
    else:
      discard

# =============================================================================
# Conv1d Forward
# =============================================================================

proc conv1dKernel*(input: TensorData, weight: TensorData, bias: TensorData,
                   stride: int, padding: int, dilation: int, groups: int): TensorData =
  ## 1D Convolution
  ## Input: (N, C_in, L)
  ## Weight: (C_out, C_in/groups, K)
  ## Output: (N, C_out, L_out)
  assert input.shape.rank == 3, "conv1d expects 3D input"
  assert weight.shape.rank == 3, "conv1d expects 3D weight"

  let batchSize = input.shape.dims[0]
  let inChannels = input.shape.dims[1]
  let inLen = input.shape.dims[2]

  let outChannels = weight.shape.dims[0]
  let kernelSize = weight.shape.dims[2]

  let effKernel = dilation * (kernelSize - 1) + 1
  let outLen = (inLen + 2 * padding - effKernel) div stride + 1

  result = newTensorDataZeros(newShape(batchSize, outChannels, outLen), input.dtype)

  case input.dtype
  of dtFloat32:
    let inArr = input.asFloat32
    let wArr = weight.asFloat32
    let outArr = result.asFloat32

    for n in 0..<batchSize:
      for oc in 0..<outChannels:
        for ol in 0..<outLen:
          var sum = 0.0'f32
          for ic in 0..<(inChannels div groups):
            for k in 0..<kernelSize:
              let il = ol * stride - padding + k * dilation
              if il >= 0 and il < inLen:
                let inIdx = n * inChannels * inLen + ic * inLen + il
                let wIdx = oc * (inChannels div groups) * kernelSize + ic * kernelSize + k
                sum += inArr[inIdx] * wArr[wIdx]

          let outIdx = n * outChannels * outLen + oc * outLen + ol
          outArr[outIdx] = sum

          if not bias.isNil and bias.size > 0:
            outArr[outIdx] += bias.asFloat32[oc]

  of dtFloat64:
    let inArr = input.asFloat64
    let wArr = weight.asFloat64
    let outArr = result.asFloat64

    for n in 0..<batchSize:
      for oc in 0..<outChannels:
        for ol in 0..<outLen:
          var sum = 0.0'f64
          for ic in 0..<(inChannels div groups):
            for k in 0..<kernelSize:
              let il = ol * stride - padding + k * dilation
              if il >= 0 and il < inLen:
                let inIdx = n * inChannels * inLen + ic * inLen + il
                let wIdx = oc * (inChannels div groups) * kernelSize + ic * kernelSize + k
                sum += inArr[inIdx] * wArr[wIdx]

          let outIdx = n * outChannels * outLen + oc * outLen + ol
          outArr[outIdx] = sum

          if not bias.isNil and bias.size > 0:
            outArr[outIdx] += bias.asFloat64[oc]
  else:
    discard
