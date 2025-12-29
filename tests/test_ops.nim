## Tests for ops module

import unittest
import std/[json, sequtils]
import ../src/ml_core/dtype
import ../src/ml_core/shape
import ../src/ml_core/tensor
import ../src/ml_core/ops

suite "OpKind":
  test "category classification":
    check opNeg.category == ocUnary
    check opAbs.category == ocUnary
    check opAdd.category == ocBinary
    check opMul.category == ocBinary
    check opEq.category == ocComparison
    check opAnd.category == ocLogical
    check opSum.category == ocReduction
    check opMatMul.category == ocMatrix
    check opRelu.category == ocActivation
    check opBatchNorm.category == ocNormalization
    check opConv2d.category == ocNeural
    check opMseLoss.category == ocLoss
    check opReshape.category == ocMemory
    check opAllReduce.category == ocCollective
    check opCast.category == ocMisc

  test "numInputs":
    check opNeg.numInputs == 1
    check opAdd.numInputs == 2
    check opNot.numInputs == 1
    check opAnd.numInputs == 2
    check opSum.numInputs == 1
    check opMatMul.numInputs == 2
    check opMseLoss.numInputs == 2

  test "isElementwise":
    check opAdd.isElementwise == true
    check opRelu.isElementwise == true
    check opMatMul.isElementwise == false
    check opSum.isElementwise == false

  test "requiresGrad":
    check opAdd.requiresGrad == true
    check opMatMul.requiresGrad == true
    check opEq.requiresGrad == false
    check opAnd.requiresGrad == false

suite "OpSpec":
  setup:
    let td1 = newTensorData(newShape(2, 3), dtFloat32)
    let td2 = newTensorData(newShape(2, 3), dtFloat32)
    let tr1 = newTensorRef(td1)
    let tr2 = newTensorRef(td2)

  test "create unary op":
    let spec = newOpSpec(opNeg, tr1)
    check spec.kind == opNeg
    check spec.inputs.len == 1
    check spec.inputs[0] == tr1

  test "create binary op":
    let spec = newOpSpec(opAdd, tr1, tr2)
    check spec.kind == opAdd
    check spec.inputs.len == 2
    check spec.inputs[0] == tr1
    check spec.inputs[1] == tr2

  test "set output":
    var spec = newOpSpec(opAdd, tr1, tr2)
    let output = newTensorRef(newShape(2, 3), dtFloat32)
    spec.setOutput(output)
    check spec.output == output

  test "set id and device":
    var spec = newOpSpec(opMatMul, tr1, tr2)
    spec.setId("op_001")
    spec.setDevice("cuda:0")
    check spec.id == "op_001"
    check spec.device == "cuda:0"

  test "attributes":
    var spec = newOpSpec(opConv2d, tr1, %*{
      "kernel_size": [3, 3],
      "stride": [1, 1],
      "padding": [1, 1]
    })

    check spec.getAttr("stride", newSeq[int]()) == @[1, 1]

  test "set attribute":
    var spec = newOpSpec(opDropout, tr1)
    spec.setAttr("p", 0.5)
    spec.setAttr("training", true)

    check spec.getAttr("p", 0.0) == 0.5
    check spec.getAttr("training", false) == true

  test "validate":
    let validSpec = newOpSpec(opAdd, tr1, tr2)
    check validSpec.validate == true

    let invalidSpec = newOpSpec(opAdd, tr1)  # Missing second input
    check invalidSpec.validate == false

suite "Shape Inference":
  setup:
    let tr1 = newTensorRef(newShape(2, 3), dtFloat32)
    let tr2 = newTensorRef(newShape(2, 3), dtFloat32)
    let tr3 = newTensorRef(newShape(3, 4), dtFloat32)

  test "unary op preserves shape":
    let spec = newOpSpec(opNeg, tr1)
    check inferOutputShape(spec) == newShape(2, 3)

  test "binary op broadcasts":
    let spec = newOpSpec(opAdd, tr1, tr2)
    check inferOutputShape(spec) == newShape(2, 3)

    let tr_scalar = newTensorRef(newShape(1), dtFloat32)
    let spec2 = newOpSpec(opMul, tr1, tr_scalar)
    check inferOutputShape(spec2) == newShape(2, 3)

  test "matmul shape":
    let spec = newOpSpec(opMatMul, tr1, tr3)
    check inferOutputShape(spec) == newShape(2, 4)

  test "transpose shape":
    let spec = newOpSpec(opTranspose, tr1)
    check inferOutputShape(spec) == newShape(3, 2)

  test "reduction shape":
    var spec = newOpSpec(opSum, tr1, %*{"axis": 1, "keepdims": false})
    check inferOutputShape(spec) == newShape(2)

    spec = newOpSpec(opMean, tr1, %*{"axis": 0, "keepdims": true})
    check inferOutputShape(spec) == newShape(1, 3)

    spec = newOpSpec(opSum, tr1, %*{"axis": -1, "keepdims": false})
    check inferOutputShape(spec) == newShape()  # Scalar

  test "squeeze shape":
    let tr = newTensorRef(newShape(1, 3, 1, 4), dtFloat32)
    var spec = newOpSpec(opSqueeze, tr, %*{"dim": 0})
    check inferOutputShape(spec) == newShape(3, 1, 4)

  test "unsqueeze shape":
    var spec = newOpSpec(opUnsqueeze, tr1, %*{"dim": 0})
    check inferOutputShape(spec) == newShape(1, 2, 3)

  test "flatten shape":
    let tr = newTensorRef(newShape(2, 3, 4), dtFloat32)
    var spec = newOpSpec(opFlatten, tr, %*{"start_dim": 1, "end_dim": 2})
    check inferOutputShape(spec) == newShape(2, 12)

suite "DType Inference":
  setup:
    let trFloat = newTensorRef(newShape(2, 3), dtFloat32)
    let trInt = newTensorRef(newShape(2, 3), dtInt32)

  test "unary op preserves dtype":
    let spec = newOpSpec(opNeg, trFloat)
    check inferOutputDtype(spec) == dtFloat32

  test "comparison returns bool":
    let spec = newOpSpec(opEq, trFloat, trFloat)
    check inferOutputDtype(spec) == dtBool

  test "logical returns bool":
    let trBool = newTensorRef(newShape(2, 3), dtBool)
    let spec = newOpSpec(opAnd, trBool, trBool)
    check inferOutputDtype(spec) == dtBool

  test "argmax returns int64":
    let spec = newOpSpec(opArgMax, trFloat)
    check inferOutputDtype(spec) == dtInt64

  test "binary promotes types":
    let spec = newOpSpec(opAdd, trFloat, trInt)
    check inferOutputDtype(spec) == dtFloat32  # Float wins

  test "cast changes dtype":
    var spec = newOpSpec(opCast, trFloat, %*{"dtype": "float64"})
    check inferOutputDtype(spec) == dtFloat64

suite "OpKind Additional Coverage":
  test "string representation":
    check $opNeg == "neg"
    check $opAdd == "add"
    check $opMatMul == "matmul"
    # Test fallback to ordinal
    check ($opSub).len > 0

  test "isInplace":
    check opAdd.isInplace == true
    check opRelu.isInplace == true
    check opMatMul.isInplace == false

  test "numInputs matrix ops":
    check opDot.numInputs == 2
    check opOuter.numInputs == 2
    check opBatchMatMul.numInputs == 2
    check opTranspose.numInputs == 1
    check opEinsum.numInputs == 1

  test "numInputs neural ops":
    check opConv2d.numInputs == -1
    check opLinear.numInputs == -1

  test "numInputs memory ops":
    check opReshape.numInputs == -1

suite "OpSpec Additional Coverage":
  setup:
    let td1 = newTensorData(newShape(2, 3), dtFloat32)
    let td2 = newTensorData(newShape(2, 3), dtFloat32)
    let tr1 = newTensorRef(td1)
    let tr2 = newTensorRef(td2)

  test "setDtype":
    var spec = newOpSpec(opNeg, tr1)
    spec.setDtype(dtFloat64)
    check spec.dtype == dtFloat64

  test "getAttr string":
    var spec = newOpSpec(opDropout, tr1, %*{"mode": "train"})
    check spec.getAttr("mode", "") == "train"
    check spec.getAttr("nonexistent", "default") == "default"

  test "setAttr string":
    var spec = newOpSpec(opDropout, tr1)
    spec.setAttr("name", "dropout_1")
    check spec.getAttr("name", "") == "dropout_1"

  test "setAttr seq[int]":
    var spec = newOpSpec(opConv2d, tr1)
    spec.setAttr("kernel_size", @[3, 3])
    check spec.getAttr("kernel_size", newSeq[int]()) == @[3, 3]

  test "create with seq inputs":
    let spec = newOpSpec(opAdd, @[tr1, tr2])
    check spec.inputs.len == 2

  test "hash":
    var spec1 = newOpSpec(opAdd, tr1, tr2)
    spec1.setId("op1")
    var spec2 = newOpSpec(opAdd, tr1, tr2)
    spec2.setId("op1")
    check hash(spec1) == hash(spec2)

    spec2.setId("op2")
    check hash(spec1) != hash(spec2)

  test "validate nil input":
    let nilTr: TensorRef = nil
    let spec = newOpSpec(opNeg, nilTr)
    check spec.validate == false

  test "validate variable input count":
    let spec = newOpSpec(opConv2d, tr1)  # Neural op with -1 expected inputs
    check spec.validate == true  # No specific count required

suite "Shape Inference Additional Coverage":
  setup:
    let tr1 = newTensorRef(newShape(2, 3), dtFloat32)
    let tr2 = newTensorRef(newShape(2, 3), dtFloat32)
    let tr3 = newTensorRef(newShape(3, 4), dtFloat32)
    let trBool = newTensorRef(newShape(2, 3), dtBool)

  test "opNot shape":
    let spec = newOpSpec(opNot, trBool)
    check inferOutputShape(spec) == newShape(2, 3)

  test "opDot shape":
    let v1 = newTensorRef(newShape(5), dtFloat32)
    let v2 = newTensorRef(newShape(5), dtFloat32)
    let spec = newOpSpec(opDot, v1, v2)
    check inferOutputShape(spec) == newShape()  # Scalar

  test "opOuter shape":
    let v1 = newTensorRef(newShape(3), dtFloat32)
    let v2 = newTensorRef(newShape(4), dtFloat32)
    let spec = newOpSpec(opOuter, v1, v2)
    check inferOutputShape(spec) == newShape(3, 4)

  test "opAll opAny shape":
    let specAll = newOpSpec(opAll, trBool)
    check inferOutputShape(specAll) == newShape()
    let specAny = newOpSpec(opAny, trBool)
    check inferOutputShape(specAny) == newShape()

  test "opArgMax opArgMin shape":
    let specMax = newOpSpec(opArgMax, tr1)
    check inferOutputShape(specMax) == newShape(2, 3)
    let specMin = newOpSpec(opArgMin, tr1)
    check inferOutputShape(specMin) == newShape(2, 3)

  test "opReshape shape":
    var spec = newOpSpec(opReshape, tr1, %*{"shape": [6]})
    check inferOutputShape(spec) == newShape(6)

  test "opView shape":
    var spec = newOpSpec(opView, tr1, %*{"shape": [3, 2]})
    check inferOutputShape(spec) == newShape(3, 2)

  test "opPermute shape":
    let tr = newTensorRef(newShape(2, 3, 4), dtFloat32)
    var spec = newOpSpec(opPermute, tr, %*{"dims": [2, 0, 1]})
    check inferOutputShape(spec) == newShape(4, 2, 3)

  test "reduction with keepdims true":
    var spec = newOpSpec(opSum, tr1, %*{"axis": 1, "keepdims": true})
    check inferOutputShape(spec) == newShape(2, 1)

  test "default case preserves shape":
    # opClone is in the default branch
    let spec = newOpSpec(opClone, tr1)
    check inferOutputShape(spec) == newShape(2, 3)

  test "no inputs raises":
    let spec = OpSpec(kind: opNeg, inputs: @[], attrs: newJObject())
    expect OpError:
      discard inferOutputShape(spec)

  test "binary op missing input raises":
    let spec = newOpSpec(opAdd, tr1)  # Only 1 input
    expect OpError:
      discard inferOutputShape(spec)

  test "matmul missing input raises":
    let spec = newOpSpec(opMatMul, tr1)  # Only 1 input
    expect OpError:
      discard inferOutputShape(spec)

suite "DType Inference Additional Coverage":
  setup:
    let trFloat = newTensorRef(newShape(2, 3), dtFloat32)
    let trBool = newTensorRef(newShape(2, 3), dtBool)

  test "opNot returns bool":
    let spec = newOpSpec(opNot, trBool)
    check inferOutputDtype(spec) == dtBool

  test "opAll opAny returns bool":
    let specAll = newOpSpec(opAll, trBool)
    check inferOutputDtype(specAll) == dtBool
    let specAny = newOpSpec(opAny, trBool)
    check inferOutputDtype(specAny) == dtBool

  test "binary single input fallback":
    let spec = newOpSpec(opAdd, trFloat)  # Only 1 input
    check inferOutputDtype(spec) == dtFloat32

  test "no inputs returns default":
    let spec = OpSpec(kind: opNeg, inputs: @[], attrs: newJObject())
    check inferOutputDtype(spec) == dtFloat32

  test "cast unknown dtype fallback":
    var spec = newOpSpec(opCast, trFloat, %*{"dtype": "unknown_type"})
    check inferOutputDtype(spec) == dtFloat32  # Falls back to input dtype

  test "cast without dtype attr":
    var spec = newOpSpec(opCast, trFloat)
    check inferOutputDtype(spec) == dtFloat32
