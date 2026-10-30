"""Scientific data, update and checkpoint contracts."""
from __future__ import annotations
import ast, hashlib, json, math
from expression import parse

def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)

class LocalBudgetStop(ValueError):
    """Known stage-local exhaustion, distinct from whole-study interruption."""

def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()

def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()

def strict_json(raw: str) -> object:
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key")
            result[key] = value
        return result

    def invalid(value):
        raise ValueError("nonfinite JSON " + value)

    def finite_float(value):
        number = float(value)
        require(math.isfinite(number), "nonfinite JSON number")
        return number

    return json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid, parse_float=finite_float)

def validate_update(value: dict, space: str) -> dict:
    require(type(value) is dict and set(value) == {"weight_source", "mix_source", "radius", "K"},
            "exact update fields")
    require(space in {"anchor", "neighborhood"}, "known edit space")
    for key, limit in [("weight_source", 2048), ("mix_source", 1024)]:
        require(type(value[key]) is str and 0 < len(value[key].encode()) <= limit, key + " bytes")
        parse(value[key])
    require(type(value["radius"]) in {int, float} and value["radius"] in {0.05, 0.1, 0.2, 0.4}
            and type(value["K"]) is int and value["K"] == 5, "fixed target family")
    if space == "anchor":
        expr = parse(value["mix_source"])
        require(type(expr) is ast.Constant and type(expr.value) is float and expr.value == 0.,
                "anchor space requires literal zero mix")
    return value

def same_binding(saved: dict, expected: dict) -> bool:
    """JSON protocol normalizes historical tuples to lists; never drop fields."""
    require(type(saved) is dict and type(expected) is dict, "binding dictionaries")
    return canonical(saved) == canonical(expected)

def verify_checkpoint_metadata(saved: dict, item: dict, externals: dict) -> str:
    require(saved["schema"] == 1 and same_binding(saved["binding"], item["binding"]), "exact checkpoint binding")
    if item.get("reference_kind") == "registered_external_manifest":
        require(item == externals[item["external_id"]], "unchanged registered external reference")
        require(saved["counters"]["updates"] == item["selected_update"], "original manifest selected update")
        return "external readonly weights: complete binding/hash and manifest update, no invented full counters"
    require("reference_kind" not in item and saved["counters"] == item["counters"], "full generated checkpoint counters")
    return "new endpoint: complete binding/hash and all counters"
