"""Bounded state-weight expressions for the fixed learning operator.

The learner owns clipping, detaching and normalized NLL. Generated source
cannot read likelihoods, labels, model objects, data files or protected scores.
"""
from __future__ import annotations

import ast
import hashlib
import math
import os
from typing import Callable

FIELDS = frozenset(["eef_" + c for c in ("x", "y", "z")] +
                   ["quat_" + str(i) for i in range(4)] +
                   ["gripper_" + str(i) for i in range(2)] +
                   ["object_" + str(i) for i in range(44)] +
                   ["object_valid_" + str(i) for i in range(44)])
UNARY = {ast.UAdd: 'positive', ast.USub: 'negative'}
BINARY = {ast.Add: 'add', ast.Sub: 'subtract', ast.Mult: 'multiply', ast.Div: 'divide'}
CALLS = {'abs': 1, 'square': 1, 'clamp': 3}


def _require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def parse(source: str) -> ast.expr:
    _require(type(source) is str and 0 < len(source.encode()) <= 2048, 'expression source bytes')
    try:
        tree = ast.parse(source)
    except (SyntaxError, RecursionError) as exc:
        raise ValueError('weight syntax') from exc
    _require(len(list(ast.walk(tree))) <= 160, 'weight AST nodes')
    _require(len(tree.body) == 1 and type(tree.body[0]) is ast.FunctionDef, 'one weight function')
    fn = tree.body[0]
    a = fn.args
    _require(fn.name == 'weights' and [v.arg for v in a.args] == ['z'], 'weights(z) ABI')
    _require(not (fn.decorator_list or fn.returns or a.defaults or a.kw_defaults
                  or a.posonlyargs or a.kwonlyargs or a.vararg or a.kwarg
                  or getattr(fn, 'type_params', []) or fn.type_comment
                  or any(v.annotation or v.type_comment for v in a.args)), 'plain function signature')
    _require(len(fn.body) == 1 and type(fn.body[0]) is ast.Return, 'one return expression')
    expr = fn.body[0].value
    _require(expr is not None, 'weight return')

    def visit(n: ast.expr, depth: int = 0) -> None:
        _require(depth <= 24, 'weight expression depth')
        if type(n) is ast.Constant:
            _require(type(n.value) is float and math.isfinite(n.value)
                     and abs(n.value) <= 16., 'finite float literal')
        elif type(n) is ast.Subscript:
            _require(type(n.value) is ast.Name and n.value.id == 'z'
                     and type(n.slice) is ast.Constant and type(n.slice.value) is str
                     and n.slice.value in FIELDS, 'explicit state selector')
        elif type(n) is ast.UnaryOp:
            _require(type(n.op) in UNARY, 'weight unary operation')
            visit(n.operand, depth+1)
        elif type(n) is ast.BinOp:
            _require(type(n.op) in BINARY, 'weight arithmetic')
            visit(n.left, depth+1); visit(n.right, depth+1)
        elif type(n) is ast.Call:
            _require(type(n.func) is ast.Name and n.func.id in CALLS
                     and len(n.args) == CALLS[n.func.id] and not n.keywords, 'weight operation binding')
            for value in n.args:
                visit(value, depth+1)
            if n.func.id == 'clamp':
                _require(all(type(v) is ast.Constant and type(v.value) is float for v in n.args[1:]),
                         'literal clamp limits')
                _require(n.args[1].value <= n.args[2].value, 'ordered clamp limits')
        else:
            raise ValueError('weight expression grammar')
    visit(expr)
    return expr


def describe(source: str) -> dict:
    expr = parse(source)
    fields = sorted({n.slice.value for n in ast.walk(expr) if type(n) is ast.Subscript})
    return {'source_sha256': hashlib.sha256(source.encode()).hexdigest(),
            'expression_sha256': hashlib.sha256(ast.dump(expr, include_attributes=False).encode()).hexdigest(),
            'selected_features': fields, 'syntactically_state_dependent': bool(fields),
            'actual_nonconstant_weights_proven': False, 'learning_benefit_proven': False}


def evaluate(source: str, z: dict, operations: dict[str, Callable], checked: Callable):
    """Interpret the reviewed AST with fixed, shape-bounded host operations.

    This does not compile or exec source. The caller supplies no model objects
    in z and must detach/clamp the returned weights in the fixed learner.
    """
    if not __debug__:
        raise RuntimeError('optimized Python prohibited')
    _require(type(z) is dict and set(z) == FIELDS, 'exact observable feature set')
    names = set(UNARY.values()) | set(BINARY.values()) | set(CALLS)
    _require(set(operations) == names, 'fixed weight operations')
    for value in z.values():
        checked(value)
    expr = parse(source)

    def run(n):
        if type(n) is ast.Constant:
            result = n.value
        elif type(n) is ast.Subscript:
            result = z[n.slice.value]
        elif type(n) is ast.UnaryOp:
            result = operations[UNARY[type(n.op)]](run(n.operand))
        elif type(n) is ast.BinOp:
            result = operations[BINARY[type(n.op)]](run(n.left), run(n.right))
        else:
            result = operations[n.func.id](*(run(v) for v in n.args))
        return checked(result)
    return run(expr)

