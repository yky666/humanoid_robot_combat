#!/usr/bin/env python3
"""Extract the actor-only graph from the training export and validate its contract."""

import argparse
from pathlib import Path

import onnx


def tensor_shape(value_info: onnx.ValueInfoProto) -> list[int | str]:
    dims = value_info.type.tensor_type.shape.dim
    return [dim.dim_value if dim.HasField("dim_value") else dim.dim_param for dim in dims]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()

    model = onnx.load(args.source)
    onnx.checker.check_model(model)
    inputs = {item.name: tensor_shape(item) for item in model.graph.input}
    outputs = {item.name: tensor_shape(item) for item in model.graph.output}
    if inputs.get("obs") != [1, 140] or outputs.get("actions") != [1, 25]:
        raise RuntimeError(f"unexpected policy contract: inputs={inputs}, outputs={outputs}")

    args.destination.parent.mkdir(parents=True, exist_ok=True)
    onnx.utils.extract_model(str(args.source), str(args.destination), ["obs"], ["actions"])
    actor = onnx.load(args.destination)
    onnx.checker.check_model(actor)
    actor_inputs = {item.name: tensor_shape(item) for item in actor.graph.input}
    actor_outputs = {item.name: tensor_shape(item) for item in actor.graph.output}
    if actor_inputs != {"obs": [1, 140]} or actor_outputs != {"actions": [1, 25]}:
        raise RuntimeError(f"unexpected actor-only contract: inputs={actor_inputs}, outputs={actor_outputs}")
    print(f"actor-only graph: {actor_inputs} -> {actor_outputs}; nodes={len(actor.graph.node)}")


if __name__ == "__main__":
    main()
