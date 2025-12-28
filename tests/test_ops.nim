## Tests for ops module

import unittest
import std/[json, sequtils]
import ../src/nimml_core/dtype
import ../src/nimml_core/shape
import ../src/nimml_core/tensor
import ../src/nimml_core/ops

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
