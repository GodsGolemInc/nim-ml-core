## Operations module for ML computation specifications
##
## Defines OpKind enum with 50+ operations and OpSpec for
## describing computation steps in the ML framework.

import std/[options, tables, json, hashes, sequtils, strutils]
import dtype, shape, tensor

type
  OpCategory* = enum
    ## Category of operations
    ocUnary       # Single input
    ocBinary      # Two inputs
    ocReduction   # Reduce dimensions
    ocMatrix      # Matrix operations
    ocNeural      # Neural network ops
    ocNormalization
    ocActivation
    ocLoss
    ocCollective  # Distributed ops
    ocMemory      # Memory/reshape ops
    ocComparison
    ocLogical
    ocMisc

  OpKind* = enum
    ## All supported operation kinds
    # Unary ops
    opNeg = "neg"
    opAbs = "abs"
    opExp = "exp"
    opLog = "log"
    opLog2 = "log2"
    opLog10 = "log10"
    opSqrt = "sqrt"
    opRsqrt = "rsqrt"
    opSquare = "square"
    opSin = "sin"
    opCos = "cos"
    opTan = "tan"
    opSinh = "sinh"
    opCosh = "cosh"
    opTanh = "tanh"
    opFloor = "floor"
    opCeil = "ceil"
    opRound = "round"
    opSign = "sign"
    opReciprocal = "reciprocal"
    opErf = "erf"

    # Binary ops
    opAdd = "add"
    opSub = "sub"
    opMul = "mul"
    opDiv = "div"
    opPow = "pow"
    opMod = "mod"
    opMax = "max"
    opMin = "min"

    # Comparison ops
    opEq = "eq"
    opNe = "ne"
    opLt = "lt"
    opLe = "le"
    opGt = "gt"
    opGe = "ge"

    # Logical ops
    opAnd = "and"
    opOr = "or"
    opXor = "xor"
    opNot = "not"

    # Reduction ops
    opSum = "sum"
    opMean = "mean"
    opProd = "prod"
    opMax2 = "reduce_max"
    opMin2 = "reduce_min"
    opArgMax = "argmax"
    opArgMin = "argmin"
    opAll = "all"
    opAny = "any"
    opVariance = "variance"
    opStd = "std"

    # Matrix ops
    opMatMul = "matmul"
    opBatchMatMul = "batch_matmul"
    opTranspose = "transpose"
    opDot = "dot"
    opOuter = "outer"
    opEinsum = "einsum"

    # Activation ops
    opRelu = "relu"
    opLeakyRelu = "leaky_relu"
    opGelu = "gelu"
    opSilu = "silu"
    opMish = "mish"
    opSigmoid = "sigmoid"
    opSoftmax = "softmax"
    opLogSoftmax = "log_softmax"
    opSoftplus = "softplus"
    opHardswish = "hardswish"

    # Normalization ops
    opBatchNorm = "batch_norm"
    opLayerNorm = "layer_norm"
    opGroupNorm = "group_norm"
    opInstanceNorm = "instance_norm"
    opRmsNorm = "rms_norm"

    # Neural network ops
    opConv2d = "conv2d"
    opConv1d = "conv1d"
    opConv3d = "conv3d"
    opConvTranspose2d = "conv_transpose2d"
    opMaxPool2d = "max_pool2d"
    opAvgPool2d = "avg_pool2d"
    opAdaptiveAvgPool2d = "adaptive_avg_pool2d"
    opDropout = "dropout"
    opLinear = "linear"
    opEmbedding = "embedding"
    opAttention = "attention"
    opMultiHeadAttention = "multi_head_attention"

    # Loss ops
    opMseLoss = "mse_loss"
    opCrossEntropyLoss = "cross_entropy_loss"
    opBceLoss = "bce_loss"
    opNllLoss = "nll_loss"
    opL1Loss = "l1_loss"
    opHuberLoss = "huber_loss"
    opKlDivLoss = "kl_div_loss"

    # Memory/reshape ops
    opReshape = "reshape"
    opView = "view"
    opFlatten = "flatten"
    opSqueeze = "squeeze"
    opUnsqueeze = "unsqueeze"
    opPermute = "permute"
    opContiguous = "contiguous"
    opClone = "clone"
    opCat = "cat"
    opStack = "stack"
    opSplit = "split"
    opChunk = "chunk"
    opSlice = "slice"
    opGather = "gather"
    opScatter = "scatter"
    opWhere = "where"
    opPad = "pad"

    # Collective ops (distributed)
    opAllReduce = "all_reduce"
    opAllGather = "all_gather"
    opReduceScatter = "reduce_scatter"
    opBroadcast = "broadcast"
    opAllToAll = "all_to_all"

    # Misc ops
    opCast = "cast"
    opFill = "fill"
    opZeros = "zeros"
    opOnes = "ones"
    opRand = "rand"
    opRandn = "randn"
    opArange = "arange"
    opLinspace = "linspace"
    opClamp = "clamp"
    opMaskedFill = "masked_fill"
    opTril = "tril"
    opTriu = "triu"
    opDiag = "diag"
    opEye = "eye"

  OpSpec* = object
    ## Specification for a single operation
    id*: string
    kind*: OpKind
    inputs*: seq[TensorRef]
    output*: TensorRef
    attrs*: JsonNode
    dtype*: DType
    device*: string  # Target device hint

  OpError* = object of CatchableError

# OpKind utilities

proc category*(op: OpKind): OpCategory =
  ## Get the category of an operation
  case op
  of opNeg .. opErf: ocUnary
  of opAdd .. opMin: ocBinary
  of opEq .. opGe: ocComparison
  of opAnd .. opNot: ocLogical
  of opSum .. opStd: ocReduction
  of opMatMul .. opEinsum: ocMatrix
  of opRelu .. opHardswish: ocActivation
  of opBatchNorm .. opRmsNorm: ocNormalization
  of opConv2d .. opMultiHeadAttention: ocNeural
  of opMseLoss .. opKlDivLoss: ocLoss
  of opReshape .. opPad: ocMemory
  of opAllReduce .. opAllToAll: ocCollective
  of opCast .. opEye: ocMisc

proc `$`*(op: OpKind): string =
  ## String representation
  case op
  of opNeg: "neg"
  of opAdd: "add"
  of opMatMul: "matmul"
  else: $ord(op)  # Fallback

proc numInputs*(op: OpKind): int =
  ## Expected number of inputs for an operation
  case op.category
  of ocUnary: 1
  of ocBinary, ocComparison: 2
  of ocLogical:
    if op == opNot: 1 else: 2
  of ocReduction: 1
  of ocMatrix:
    case op
    of opMatMul, opBatchMatMul, opDot, opOuter: 2
    of opTranspose, opEinsum: 1  # Einsum can have variable
    else: 1
  of ocActivation, ocNormalization: 1
  of ocNeural: -1  # Variable
  of ocLoss: 2
  of ocMemory: -1  # Variable
  of ocCollective: -1  # Variable
  of ocMisc: -1  # Variable

proc isElementwise*(op: OpKind): bool =
  ## Check if operation is elementwise
  op.category in {ocUnary, ocBinary, ocComparison, ocLogical, ocActivation}

proc isInplace*(op: OpKind): bool =
  ## Check if operation can be done in-place
  op.isElementwise

proc requiresGrad*(op: OpKind): bool =
  ## Check if operation supports gradient computation
  op.category notin {ocLogical, ocComparison}

# OpSpec creation

proc newOpSpec*(kind: OpKind, inputs: seq[TensorRef],
                attrs: JsonNode = newJObject()): OpSpec =
  ## Create a new operation specification
  result = OpSpec(
    id: "",
    kind: kind,
    inputs: inputs,
    output: nil,
    attrs: attrs,
    dtype: dtFloat32,
    device: ""
  )

proc newOpSpec*(kind: OpKind, input: TensorRef,
                attrs: JsonNode = newJObject()): OpSpec =
  ## Create a new operation specification with single input
  newOpSpec(kind, @[input], attrs)

proc newOpSpec*(kind: OpKind, a, b: TensorRef,
                attrs: JsonNode = newJObject()): OpSpec =
  ## Create a new operation specification with two inputs
  newOpSpec(kind, @[a, b], attrs)

proc setOutput*(spec: var OpSpec, output: TensorRef) =
  ## Set the output tensor reference
  spec.output = output

proc setId*(spec: var OpSpec, id: string) =
  ## Set the operation ID
  spec.id = id

proc setDevice*(spec: var OpSpec, device: string) =
  ## Set the target device
  spec.device = device

proc setDtype*(spec: var OpSpec, dtype: DType) =
  ## Set the output dtype
  spec.dtype = dtype

# Attribute helpers

proc getAttr*[T](spec: OpSpec, key: string, default: T): T =
  ## Get attribute with default value
  if spec.attrs.hasKey(key):
    when T is int:
      return spec.attrs[key].getInt
    elif T is float:
      return spec.attrs[key].getFloat
    elif T is bool:
      return spec.attrs[key].getBool
    elif T is string:
      return spec.attrs[key].getStr
    elif T is seq[int]:
      result = @[]
      for item in spec.attrs[key]:
        result.add(item.getInt)
    else:
      return default
  else:
    return default

proc setAttr*(spec: var OpSpec, key: string, value: int) =
  ## Set integer attribute
  spec.attrs[key] = %value

proc setAttr*(spec: var OpSpec, key: string, value: float) =
  ## Set float attribute
  spec.attrs[key] = %value

proc setAttr*(spec: var OpSpec, key: string, value: bool) =
  ## Set boolean attribute
  spec.attrs[key] = %value

proc setAttr*(spec: var OpSpec, key: string, value: string) =
  ## Set string attribute
  spec.attrs[key] = %value

proc setAttr*(spec: var OpSpec, key: string, value: seq[int]) =
  ## Set integer sequence attribute
  spec.attrs[key] = %value

# Output shape inference

proc inferOutputShape*(spec: OpSpec): Shape =
  ## Infer the output shape based on operation and inputs.
  ## This is a simplified version - full implementation would handle all ops.
  if spec.inputs.len == 0:
    raise newException(OpError, "Cannot infer shape without inputs")

  case spec.kind
  # Elementwise unary ops preserve shape
  of opNeg .. opErf, opRelu .. opHardswish:
    return spec.inputs[0].shape

  # Elementwise binary ops broadcast
  of opAdd .. opMin, opEq .. opGe, opAnd .. opXor:
    if spec.inputs.len < 2:
      raise newException(OpError, "Binary op requires 2 inputs")
    return broadcast(spec.inputs[0].shape, spec.inputs[1].shape)

  of opNot:
    return spec.inputs[0].shape

  # Reduction ops
  of opSum, opMean, opProd, opMax2, opMin2, opVariance, opStd:
    let axis = spec.getAttr("axis", -1)
    let keepdims = spec.getAttr("keepdims", false)
    let inShape = spec.inputs[0].shape

    if axis < 0:
      # Reduce all dimensions
      if keepdims:
        return newShape(newSeq[int](inShape.rank).mapIt(1))
      else:
        return newShape()  # Scalar
    else:
      var newDims: seq[int] = @[]
      for i, d in inShape.dims:
        if i == axis:
          if keepdims:
            newDims.add(1)
        else:
          newDims.add(d)
      return newShape(newDims)

  of opArgMax, opArgMin:
    return spec.inputs[0].shape  # Simplified

  of opAll, opAny:
    return newShape()  # Scalar

  # Matrix ops
  of opMatMul, opBatchMatMul:
    if spec.inputs.len < 2:
      raise newException(OpError, "MatMul requires 2 inputs")
    return matmulShape(spec.inputs[0].shape, spec.inputs[1].shape)

  of opTranspose:
    return spec.inputs[0].shape.transpose()

  of opDot:
    return newShape()  # Scalar

  of opOuter:
    let m = spec.inputs[0].shape.size
    let n = spec.inputs[1].shape.size
    return newShape(m, n)

  # Reshape ops
  of opReshape, opView:
    let newShapeAttr = spec.getAttr("shape", newSeq[int]())
    return newShape(newShapeAttr)

  of opFlatten:
    let startDim = spec.getAttr("start_dim", 0)
    let endDim = spec.getAttr("end_dim", -1)
    return spec.inputs[0].shape.flatten(startDim, endDim)

  of opSqueeze:
    let dim = spec.getAttr("dim", -1)
    return spec.inputs[0].shape.squeeze(dim)

  of opUnsqueeze:
    let dim = spec.getAttr("dim", 0)
    return spec.inputs[0].shape.unsqueeze(dim)

  of opPermute:
    let perm = spec.getAttr("dims", newSeq[int]())
    return spec.inputs[0].shape.transpose(perm)

  # Default: preserve input shape
  else:
    return spec.inputs[0].shape

proc inferOutputDtype*(spec: OpSpec): DType =
  ## Infer the output dtype based on operation and inputs.
  if spec.inputs.len == 0:
    return dtFloat32

  case spec.kind
  # Comparison ops return bool
  of opEq .. opGe:
    return dtBool

  # Logical ops return bool
  of opAnd .. opNot:
    return dtBool

  # ArgMax/ArgMin return int64
  of opArgMax, opArgMin:
    return dtInt64

  # All/Any return bool
  of opAll, opAny:
    return dtBool

  # Cast op uses specified dtype
  of opCast:
    if spec.attrs.hasKey("dtype"):
      let dtypeStr = spec.attrs["dtype"].getStr
      # Try matching by DType name
      for dt in DType:
        if $dt == dtypeStr or ("dt" & dtypeStr.capitalizeAscii) == $dt:
          return dt
      # Fallback: try with dt prefix
      try:
        return parseEnum[DType]("dt" & dtypeStr.capitalizeAscii)
      except ValueError:
        return spec.inputs[0].dtype
    return spec.inputs[0].dtype

  # Binary ops promote types
  of opAdd .. opMin:
    if spec.inputs.len >= 2:
      return promote(spec.inputs[0].dtype, spec.inputs[1].dtype)
    return spec.inputs[0].dtype

  # Default: preserve input dtype
  else:
    return spec.inputs[0].dtype

# Validation

proc validate*(spec: OpSpec): bool =
  ## Validate the operation specification
  let expectedInputs = spec.kind.numInputs
  if expectedInputs > 0 and spec.inputs.len != expectedInputs:
    return false

  # Check inputs are not nil
  for input in spec.inputs:
    if input.isNil:
      return false

  true

proc hash*(spec: OpSpec): Hash =
  ## Hash for use in Nim's hash tables
  var h: Hash = 0
  h = h !& hash(spec.id)
  h = h !& hash(ord(spec.kind))
  for input in spec.inputs:
    h = h !& hash(input)
  result = !$h
