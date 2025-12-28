## Tests for IR module

import unittest
import std/[json, options, strutils, tables]
import ../src/nimml_core/dtype
import ../src/nimml_core/shape
import ../src/nimml_core/tensor
import ../src/nimml_core/ops
import ../src/nimml_core/ir

suite "Node Creation":
  test "input node":
    let tr = newTensorRef(newShape(2, 3), dtFloat32)
    let node = newInputNode("x", tr)
    check node.id == "x"
    check node.kind == nkInput
    check node.inputs.len == 0
    check node.tensorRef == tr

  test "const node":
    let tr = newTensorRef(newShape(3, 3), dtFloat32)
    let node = newConstNode("w", tr)
    check node.id == "w"
    check node.kind == nkConst

  test "op node":
    let tr1 = newTensorRef(newShape(2, 3), dtFloat32)
    let tr2 = newTensorRef(newShape(3, 4), dtFloat32)
    let outRef = newTensorRef(newShape(2, 4), dtFloat32)
    let opSpec = newOpSpec(opMatMul, tr1, tr2)
    let node = newOpNode("matmul_0", opSpec, @["x", "w"], outRef)

    check node.id == "matmul_0"
    check node.kind == nkOp
    check node.inputs == @["x", "w"]
    check node.opSpec.isSome
    check node.opSpec.get.kind == opMatMul

  test "output node":
    let tr = newTensorRef(newShape(2, 4), dtFloat32)
    let node = newOutputNode("out", "matmul_0", tr)
    check node.id == "out"
    check node.kind == nkOutput
    check node.inputs == @["matmul_0"]

suite "Graph Creation":
  test "empty graph":
    let g = newGraph("test")
    check g.name == "test"
    check g.nodeCount == 0
    check g.opCount == 0

  test "add nodes":
    let g = newGraph("test")
    let tr = newTensorRef(newShape(2, 3), dtFloat32)

    let inputNode = newInputNode("x", tr)
    g.addNode(inputNode)

    check g.nodeCount == 1
    check g.hasNode("x")
    check g.inputs == @["x"]

  test "node connections":
    let g = newGraph("test")
    let tr1 = newTensorRef(newShape(2, 3), dtFloat32)
    let tr2 = newTensorRef(newShape(2, 3), dtFloat32)

    g.addNode(newInputNode("x", tr1))
    g.addNode(newOpNode("neg", newOpSpec(opNeg, tr1), @["x"], tr2))

    check g.nodes["x"].outputs == @["neg"]
    check g.nodes["neg"].inputs == @["x"]

  test "duplicate node raises":
    let g = newGraph("test")
    let tr = newTensorRef(newShape(2, 3), dtFloat32)

    g.addNode(newInputNode("x", tr))

    expect GraphError:
      g.addNode(newInputNode("x", tr))

  test "remove node":
    let g = newGraph("test")
    let tr1 = newTensorRef(newShape(2, 3), dtFloat32)
    let tr2 = newTensorRef(newShape(2, 3), dtFloat32)

    g.addNode(newInputNode("x", tr1))
    g.addNode(newOpNode("neg", newOpSpec(opNeg, tr1), @["x"], tr2))

    g.removeNode("neg")

    check g.nodeCount == 1
    check not g.hasNode("neg")
    check g.nodes["x"].outputs.len == 0

suite "Graph Traversal":
  setup:
    # Build a simple graph: x -> neg -> relu -> out
    let g = newGraph("test")
    let tr = newTensorRef(newShape(2, 3), dtFloat32)
    g.addNode(newInputNode("x", tr))
    g.addNode(newOpNode("neg", newOpSpec(opNeg, tr), @["x"], tr))
    g.addNode(newOpNode("relu", newOpSpec(opRelu, tr), @["neg"], tr))
    g.addNode(newOutputNode("out", "relu", tr))

  test "get inputs":
    let inputs = g.getInputs("neg")
    check inputs.len == 1
    check inputs[0].id == "x"

  test "get outputs":
    let outputs = g.getOutputs("neg")
    check outputs.len == 1
    check outputs[0].id == "relu"

  test "predecessors":
    let preds = g.predecessors("out")
    check "x" in preds
    check "neg" in preds
    check "relu" in preds

  test "successors":
    let succs = g.successors("x")
    check "neg" in succs
    check "relu" in succs
    check "out" in succs

suite "Topological Sort":
  test "linear graph":
    let g = newGraph("test")
    let tr = newTensorRef(newShape(2, 3), dtFloat32)
    g.addNode(newInputNode("x", tr))
    g.addNode(newOpNode("a", newOpSpec(opNeg, tr), @["x"], tr))
    g.addNode(newOpNode("b", newOpSpec(opRelu, tr), @["a"], tr))
    g.addNode(newOutputNode("out", "b", tr))

    let order = g.topologicalSort()
    check order.find("x") < order.find("a")
    check order.find("a") < order.find("b")
    check order.find("b") < order.find("out")

  test "diamond graph":
    let g = newGraph("test")
    let tr = newTensorRef(newShape(2, 3), dtFloat32)
    g.addNode(newInputNode("x", tr))
    g.addNode(newOpNode("a", newOpSpec(opNeg, tr), @["x"], tr))
    g.addNode(newOpNode("b", newOpSpec(opRelu, tr), @["x"], tr))
    g.addNode(newOpNode("c", newOpSpec(opAdd, tr, tr), @["a", "b"], tr))

    let order = g.topologicalSort()
    check order.find("x") < order.find("a")
    check order.find("x") < order.find("b")
    check order.find("a") < order.find("c")
    check order.find("b") < order.find("c")

  test "reverse topological sort":
    let g = newGraph("test")
    let tr = newTensorRef(newShape(2, 3), dtFloat32)
    g.addNode(newInputNode("x", tr))
    g.addNode(newOpNode("a", newOpSpec(opNeg, tr), @["x"], tr))
    g.addNode(newOutputNode("out", "a", tr))

    let order = g.reverseTopologicalSort()
    check order.find("out") < order.find("a")
    check order.find("a") < order.find("x")

suite "Graph Validation":
  test "valid graph":
    let g = newGraph("test")
    let tr = newTensorRef(newShape(2, 3), dtFloat32)
    g.addNode(newInputNode("x", tr))
    g.addNode(newOpNode("a", newOpSpec(opNeg, tr), @["x"], tr))
    g.addNode(newOutputNode("out", "a", tr))

    check g.validate == true
    check g.hasCycles == false

  test "cycle detection":
    let g = newGraph("test")
    let tr = newTensorRef(newShape(2, 3), dtFloat32)

    # Create a cycle manually
    let nodeA = Node(
      id: "a", kind: nkOp,
      inputs: @["b"], outputs: @["b"],
      tensorRef: tr
    )
    let nodeB = Node(
      id: "b", kind: nkOp,
      inputs: @["a"], outputs: @["a"],
      tensorRef: tr
    )
    g.nodes["a"] = nodeA
    g.nodes["b"] = nodeB

    check g.hasCycles == true

  test "validation errors":
    let g = newGraph("test")
    let tr = newTensorRef(newShape(2, 3), dtFloat32)
    g.addNode(newOpNode("a", newOpSpec(opNeg, tr), @["nonexistent"], tr))

    let errors = g.validateInputs()
    check errors.len > 0

suite "Graph Transformations":
  test "clone":
    let g = newGraph("original")
    let tr = newTensorRef(newShape(2, 3), dtFloat32)
    g.addNode(newInputNode("x", tr))
    g.addNode(newOpNode("a", newOpSpec(opNeg, tr), @["x"], tr))

    let g2 = g.clone()
    check g2.name == "original"
    check g2.nodeCount == 2
    check g2.hasNode("x")
    check g2.hasNode("a")

    # Modify original shouldn't affect clone
    g.removeNode("a")
    check g2.hasNode("a")

  test "subgraph":
    let g = newGraph("test")
    let tr = newTensorRef(newShape(2, 3), dtFloat32)
    g.addNode(newInputNode("x", tr))
    g.addNode(newOpNode("a", newOpSpec(opNeg, tr), @["x"], tr))
    g.addNode(newOpNode("b", newOpSpec(opRelu, tr), @["a"], tr))
    g.addNode(newOpNode("c", newOpSpec(opSigmoid, tr), @["b"], tr))

    let sub = g.subgraph(@["a", "b"])
    check sub.nodeCount == 2
    check sub.hasNode("a")
    check sub.hasNode("b")
    check not sub.hasNode("x")
    check not sub.hasNode("c")

suite "Serialization":
  test "toJson":
    let g = newGraph("test")
    let tr = newTensorRef(newShape(2, 3), dtFloat32)
    g.addNode(newInputNode("x", tr))
    g.addNode(newOpNode("neg", newOpSpec(opNeg, tr), @["x"], tr))

    let j = g.toJson()
    check j["name"].getStr == "test"
    check j["nodes"].len == 2
    check j["inputs"].len == 1

  test "toDot":
    let g = newGraph("test")
    let tr = newTensorRef(newShape(2, 3), dtFloat32)
    g.addNode(newInputNode("x", tr))
    g.addNode(newOpNode("neg", newOpSpec(opNeg, tr), @["x"], tr))

    let dot = g.toDot()
    check "digraph" in dot
    check "x" in dot
    check "neg" in dot
    check "->" in dot

suite "GraphBuilder":
  test "build simple graph":
    let b = newGraphBuilder("mlp")

    let x = b.addInput(newTensorRef(newShape(2, 784), dtFloat32), "input")
    let w1 = b.addConst(newTensorRef(newShape(784, 256), dtFloat32), "weight1")
    let h = b.addOp(opMatMul, @[x, w1], newShape(2, 256), name = "hidden")
    let act = b.addOp(opRelu, @[h], newShape(2, 256), name = "activation")
    discard b.addOutput(act, "output")

    let g = b.build()
    check g.name == "mlp"
    check g.nodeCount == 5
    check g.opCount == 2
    check g.inputs == @["input"]
    check g.outputs == @["output"]

  test "auto-generated node ids":
    let b = newGraphBuilder()

    let x = b.addInput(newTensorRef(newShape(10), dtFloat32))
    let y = b.addOp(opNeg, @[x], newShape(10))

    check x.startsWith("input_")
    check y.startsWith("neg_")
