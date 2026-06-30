#!/usr/bin/env python3
"""Convertit un modèle ONNX pour le runtime du Jetson (onnxruntime 1.10 : IR<=8, opset<=15).

Le Jetson Nano (JetPack/Python 3.6) est figé sur onnxruntime 1.10, qui refuse
les modèles IR>8 / opset>15 et ne connaît pas LayerNormalization (op natif opset17+).
Ce script décompose chaque LayerNormalization en primitives (ReduceMean/Sub/Mul/
Add/Sqrt/Div), retire les attributs introduits après l'opset 15 (s'ils valent le
défaut), abaisse opset+IR, puis VÉRIFIE l'équivalence numérique avec l'original.

À lancer sur une machine de DEV (onnx + onnxruntime installés), PAS sur le Jetson :
    python scripts/convert_onnx_for_jetson.py models/v18/best.onnx models/v18/best_jetson.onnx
"""
import argparse
import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, numpy_helper


def decompose_layernorm(graph):
    """Remplace chaque node LayerNormalization par sa décomposition primitive."""
    new_nodes, consts = [], []
    for n in graph.node:
        if n.op_type != "LayerNormalization":
            new_nodes.append(n)
            continue
        X, Scale = n.input[0], n.input[1]
        B = n.input[2] if len(n.input) > 2 else None
        Y = n.output[0]
        axis, eps = -1, 1e-5
        for a in n.attribute:
            if a.name == "axis":
                axis = a.i
            if a.name == "epsilon":
                eps = a.f
        p = n.name or Y
        eps_n = f"{p}_eps"
        consts.append(numpy_helper.from_array(np.array(eps, np.float32), eps_n))
        t = {s: f"{p}_{s}" for s in ["mean", "d", "dd", "var", "ve", "std", "norm", "scaled"]}
        new_nodes += [
            helper.make_node("ReduceMean", [X], [t["mean"]], axes=[axis], keepdims=1, name=f"{p}_rm1"),
            helper.make_node("Sub", [X, t["mean"]], [t["d"]], name=f"{p}_sub"),
            helper.make_node("Mul", [t["d"], t["d"]], [t["dd"]], name=f"{p}_sq"),
            helper.make_node("ReduceMean", [t["dd"]], [t["var"]], axes=[axis], keepdims=1, name=f"{p}_rm2"),
            helper.make_node("Add", [t["var"], eps_n], [t["ve"]], name=f"{p}_addeps"),
            helper.make_node("Sqrt", [t["ve"]], [t["std"]], name=f"{p}_sqrt"),
            helper.make_node("Div", [t["d"], t["std"]], [t["norm"]], name=f"{p}_div"),
            helper.make_node("Mul", [t["norm"], Scale], [t["scaled"]], name=f"{p}_mul"),
            helper.make_node("Add", [t["scaled"], B], [Y], name=f"{p}_addb") if B else
            helper.make_node("Identity", [t["scaled"]], [Y], name=f"{p}_id"),
        ]
    del graph.node[:]
    graph.node.extend(new_nodes)
    graph.initializer.extend(consts)


def strip_recent_attrs(graph):
    """Retire les attributs introduits après l'opset 15 quand ils valent le défaut."""
    removed = []
    for n in graph.node:
        if n.op_type == "ScatterND":
            for i in range(len(n.attribute) - 1, -1, -1):
                a = n.attribute[i]
                if a.name == "reduction":
                    val = a.s.decode() if a.s else "none"
                    assert val == "none", f"ScatterND reduction={val} non retirable trivialement"
                    del n.attribute[i]
                    removed.append("ScatterND.reduction")
    return removed


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--opset", type=int, default=15)
    ap.add_argument("--ir", type=int, default=8)
    ap.add_argument("--input-dim", type=int, default=23,
                    help="dim. d'entrée du modèle : 20 rayons + 3 features dérivées (asym, front, min)")
    args = ap.parse_args()

    m = onnx.load(args.input)
    decompose_layernorm(m.graph)
    print("attributs >opset15 retirés :", strip_recent_attrs(m.graph))
    for op in m.opset_import:
        if op.domain in ("", "ai.onnx"):
            op.version = args.opset
    m.ir_version = args.ir
    onnx.checker.check_model(m)
    onnx.save(m, args.output)

    s0 = ort.InferenceSession(args.input, providers=["CPUExecutionProvider"])
    s1 = ort.InferenceSession(args.output, providers=["CPUExecutionProvider"])
    name = s0.get_inputs()[0].name
    maxd = max(
        float(np.abs(s0.run(None, {name: x})[0] - s1.run(None, {name: x})[0]).max())
        for x in [np.random.randn(1, args.input_dim).astype(np.float32) * 2 for _ in range(300)]
    )
    print(f"écrit {args.output} (opset{args.opset}, ir{args.ir}) | max diff {maxd:.2e} | équivalent {maxd < 1e-4}")


if __name__ == "__main__":
    main()
