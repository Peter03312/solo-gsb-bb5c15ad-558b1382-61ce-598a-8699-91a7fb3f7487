"""POST /import-preview 的端到端 / 单元测试。运行：python -m pytest -q"""

import json

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.minimizer import build_validator
from app.repairer import (
    FIXABLE_KEYWORDS,
    RepairSearcher,
    _canonical,
    _max_length_fixes,
    _min_length_fixes,
    _required_fixes,
)

client = TestClient(app)


def post(schema, instance, **extra):
    body = {"schema": schema, "instance": instance}
    body.update(extra)
    return client.post("/import-preview", json=body)


# ---- HTTP 层：请求校验 -------------------------------------------------------


def test_invalid_schema_returns_422():
    resp = post({"type": "not-a-type"}, {})
    assert resp.status_code == 422
    assert resp.json()["error"] == "invalid_schema"


def test_broken_ref_returns_422():
    # check_schema 通过，但实际校验无法运行（坏 $ref）：不可用 schema。
    resp = post({"$ref": "#/$defs/missing"}, {})
    assert resp.status_code == 422
    assert resp.json()["error"] == "invalid_schema"


def test_missing_instance_returns_422():
    resp = client.post("/import-preview", json={"schema": {}})
    assert resp.status_code == 422


def test_bad_max_evaluations_returns_422():
    assert post({}, 1, max_evaluations=0).status_code == 422
    assert post({}, 1, max_evaluations=-3).status_code == 422


# ---- 原实例合法：不作修改 -----------------------------------------------------


def test_valid_instance_is_returned_unchanged():
    resp = post({"type": "integer", "minimum": 1}, 5)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "valid"
    assert data["candidate_instance"] == 5
    assert data["changes"] == []
    assert data["change_count"] == 0
    assert data["remaining_error_count"] == 0
    assert data["remaining_errors"] == []
    assert data["unrepaired_reason"] is None
    # 仅消耗一次全量校验。
    assert data["evaluations"] == 1


def test_boolean_true_schema_is_valid():
    data = post(True, "anything").json()
    assert data["status"] == "valid"
    assert data["candidate_instance"] == "anything"


# ---- required + default -----------------------------------------------------


def test_required_filled_from_default():
    schema = {
        "type": "object",
        "required": ["a"],
        "properties": {"a": {"type": "integer", "default": 42}},
    }
    data = post(schema, {}).json()
    assert data["status"] == "repaired"
    assert data["candidate_instance"] == {"a": 42}
    assert data["remaining_error_count"] == 0
    change = data["changes"][0]
    assert change["pointer"] == "/a"
    assert change["keyword"] == "required"
    assert change["op"] == "add"
    assert change["from"] is None
    assert change["to"] == 42
    assert data["change_count"] == 1


def test_required_without_default_is_not_invented():
    schema = {"type": "object", "required": ["a"]}
    data = post(schema, {}).json()
    assert data["status"] == "unrepaired"
    assert data["unrepaired_reason"] == "no_feasible_change"
    assert data["candidate_instance"] == {}
    assert data["changes"] == []
    assert data["remaining_errors"][0]["keyword"] == "required"
    assert data["stopped_by_budget"] is False


def test_existing_property_is_never_overwritten_by_required():
    # 属性存在但值非法（type 错误，不可修复）：required 不应提出覆盖改动。
    schema = {
        "type": "object",
        "required": ["a"],
        "properties": {"a": {"type": "integer", "default": 42}},
    }
    data = post(schema, {"a": "bad"}).json()
    assert data["status"] == "unrepaired"
    assert data["candidate_instance"] == {"a": "bad"}


def test_mutable_default_is_deep_copied_and_independent():
    schema = {
        "type": "object",
        "required": ["a", "b"],
        "properties": {"a": {"default": []}, "b": {"default": []}},
    }
    data = post(schema, {}).json()
    assert data["status"] == "repaired"
    inst = data["candidate_instance"]
    assert inst == {"a": [], "b": []}
    inst["a"].append(1)
    assert inst["b"] == []  # 两处默认值不共享列表对象。
    assert data["original_instance"] == {}


# ---- enum / const ------------------------------------------------------------


def test_enum_replaced_by_first_allowed_value():
    data = post({"enum": ["a", "b", "c"]}, "z").json()
    assert data["status"] == "repaired"
    assert data["candidate_instance"] == "a"
    assert data["changes"][0]["keyword"] == "enum"
    assert data["changes"][0]["op"] == "replace"


def test_enum_respects_sibling_type_by_declaration_order():
    data = post({"type": "string", "enum": ["a", "b"]}, 99).json()
    # 枚举首个同声明序值 "a" 同时满足 type：直接修复成功。
    assert data["status"] == "repaired"
    assert isinstance(data["candidate_instance"], str)


def test_enum_none_allowed_is_fixable():
    data = post({"enum": [None, 1]}, "x").json()
    assert data["status"] == "repaired"
    assert data["candidate_instance"] is None


def test_const_replaced_with_constant():
    data = post({"const": {"k": 1}}, {"k": 2}).json()
    assert data["status"] == "repaired"
    assert data["candidate_instance"] == {"k": 1}


def test_enum_fix_may_be_blocked_by_other_constraint():
    # 枚举值全是整数，但 type 要求 string：枚举替换后仍剩 1 个 type 错误，
    # type 不构造改动 -> 无法完全修复；但错误数从 2 降到 1，最佳候选保留。
    data = post({"type": "string", "enum": [1, 2]}, 9).json()
    assert data["status"] == "unrepaired"
    assert data["candidate_instance"] == 1
    assert data["remaining_error_count"] == 1
    assert data["remaining_errors"][0]["keyword"] == "type"


# ---- minimum / maximum -------------------------------------------------------


def test_minimum_clamps_to_boundary():
    data = post({"type": "integer", "minimum": 5}, -10).json()
    assert data["status"] == "repaired"
    assert data["candidate_instance"] == 5


def test_maximum_clamps_to_boundary():
    data = post({"type": "number", "maximum": 10}, 99.5).json()
    assert data["status"] == "repaired"
    assert data["candidate_instance"] == 10


def test_integer_value_vs_float_boundary_keeps_integer():
    # 边界 2.5：候选 2.5 会被 type:integer 拦截，应改用保持整型的 3。
    data = post({"type": "integer", "minimum": 2.5}, 1).json()
    assert data["status"] == "repaired"
    assert data["candidate_instance"] == 3
    assert isinstance(data["candidate_instance"], int)


def test_exclusive_boundaries_are_not_fixable():
    # 需求范围只含 minimum/maximum：exclusiveMinimum 不构造改动。
    data = post({"type": "integer", "exclusiveMinimum": 5}, 1).json()
    assert data["status"] == "unrepaired"
    assert data["changes"] == []
    assert data["remaining_errors"][0]["keyword"] == "exclusiveMinimum"


# ---- minLength / maxLength ---------------------------------------------------


def test_min_length_pads_with_last_char():
    data = post({"type": "string", "minLength": 4}, "ab").json()
    assert data["status"] == "repaired"
    assert data["candidate_instance"] == "abbb"


def test_min_length_empty_string_uses_fixed_pad():
    data = post({"type": "string", "minLength": 3}, "").json()
    assert data["status"] == "repaired"
    assert data["candidate_instance"] == "xxx"


def test_max_length_truncates():
    data = post({"type": "string", "maxLength": 3}, "abcdef").json()
    assert data["status"] == "repaired"
    assert data["candidate_instance"] == "abc"


# ---- 多错误有界搜索：择优、去重、链式 ----------------------------------------


MULTI_SCHEMA = {
    "type": "object",
    "required": ["name", "age", "role"],
    "properties": {
        "name": {"type": "string", "minLength": 3, "default": "anon"},
        "age": {"type": "integer", "minimum": 0, "default": 18},
        "role": {"enum": ["a", "b"], "default": "a"},
    },
}
MULTI_INSTANCE = {"name": "ab", "age": -5, "role": "z"}


def test_multiple_errors_all_repaired_with_changes_per_pointer():
    data = post(MULTI_SCHEMA, MULTI_INSTANCE).json()
    assert data["status"] == "repaired"
    assert data["candidate_instance"] == {"name": "abb", "age": 0, "role": "a"}
    assert data["remaining_error_count"] == 0
    pointers = sorted(c["pointer"] for c in data["changes"])
    assert pointers == ["/age", "/name", "/role"]
    keywords = {c["pointer"]: c["keyword"] for c in data["changes"]}
    assert keywords == {"/name": "minLength", "/age": "minimum", "/role": "enum"}
    # change_count 按 JSON Pointer 位置去重计数。
    assert data["change_count"] == 3
    # 每个候选都重新全量校验：最终实例确实合法。
    assert not list(build_validator(MULTI_SCHEMA).iter_errors(data["candidate_instance"]))


def test_search_prefers_fewer_errors_then_fewer_locations():
    # x 可经 enum 修复；t 的 type 错误不可修复。最佳候选只改 /x 一处。
    schema = {
        "type": "object",
        "properties": {"x": {"enum": [1]}, "t": {"type": "integer"}},
    }
    data = post(schema, {"x": 9, "t": "bad"}).json()
    assert data["status"] == "unrepaired"
    assert data["candidate_instance"] == {"x": 1, "t": "bad"}
    assert data["change_count"] == 1
    assert [c["pointer"] for c in data["changes"]] == ["/x"]
    assert data["remaining_error_count"] == 1
    assert data["remaining_errors"][0]["keyword"] == "type"
    assert data["remaining_errors"][0]["pointer"] == "/t"


def test_chained_fix_same_location_counts_as_one_change_location():
    # 补入的 default 自身违反 maxLength，随后在同一位置截断：
    # 最终只需 1 个改动位置（change 链有 2 步）。
    schema = {
        "type": "object",
        "required": ["s"],
        "properties": {"s": {"type": "string", "maxLength": 2, "default": "abcdef"}},
    }
    data = post(schema, {}).json()
    assert data["status"] == "repaired"
    assert data["candidate_instance"] == {"s": "ab"}
    assert [c["keyword"] for c in data["changes"]] == ["required", "maxLength"]
    assert data["change_count"] == 1


def test_chained_fix_required_then_enum():
    schema = {
        "type": "object",
        "required": ["r"],
        "properties": {"r": {"enum": ["a", "b"], "default": "zzz"}},
    }
    data = post(schema, {}).json()
    assert data["status"] == "repaired"
    assert data["candidate_instance"] == {"r": "a"}


def test_array_items_paths():
    schema = {
        "type": "array",
        "items": {
            "type": "object",
            "required": ["id"],
            "properties": {"id": {"default": 0}, "n": {"minimum": 1}},
        },
    }
    data = post(schema, [{"n": 0}, {"n": -3}]).json()
    assert data["status"] == "repaired"
    assert data["candidate_instance"] == [{"n": 1, "id": 0}, {"n": 1, "id": 0}]
    pointers = {c["pointer"] for c in data["changes"]}
    assert pointers == {"/0/id", "/0/n", "/1/id", "/1/n"}


def test_nested_local_ref_default():
    schema = {
        "$defs": {"sub": {"type": "object", "required": ["id"],
                          "properties": {"id": {"default": 7}}}},
        "type": "object",
        "properties": {"sub": {"$ref": "#/$defs/sub"}},
    }
    data = post(schema, {"sub": {}}).json()
    assert data["status"] == "repaired"
    assert data["candidate_instance"] == {"sub": {"id": 7}}
    assert data["changes"][0]["pointer"] == "/sub/id"


def test_duplicate_proposals_via_allof_are_deduplicated():
    # 同一改动经 allOf 与 properties 两条路径提出：只评估一次。
    schema = {
        "allOf": [{"properties": {"x": {"enum": [1]}}}],
        "properties": {"x": {"enum": [1]}},
    }
    data = post(schema, {"x": 2}).json()
    assert data["status"] == "repaired"
    assert data["candidate_instance"] == {"x": 1}
    assert data["evaluations"] == 2  # 初始校验 + 唯一候选


def test_candidate_dedup_by_canonical_json():
    assert _canonical({"a": 1, "b": 2}) == _canonical({"b": 2, "a": 1})


# ---- 预算 --------------------------------------------------------------------


def test_budget_one_keeps_original_and_reports_exhausted():
    schema = {
        "type": "object",
        "required": ["a", "b"],
        "properties": {"a": {"default": 1}, "b": {"default": 2}},
    }
    data = post(schema, {}, max_evaluations=1).json()
    assert data["status"] == "unrepaired"
    assert data["unrepaired_reason"] == "budget_exhausted"
    assert data["stopped_by_budget"] is True
    assert data["evaluations"] == 1
    assert data["candidate_instance"] == {}
    assert data["remaining_error_count"] == 2


def test_budget_mid_search_keeps_best_candidate():
    schema = {
        "type": "object",
        "required": ["a", "b", "c"],
        "properties": {
            "a": {"default": 1},
            "b": {"default": 2},
            "c": {"default": 3},
            "x": {"enum": [9]},
        },
    }
    data = post(schema, {"x": 1}, max_evaluations=3).json()
    assert data["status"] == "unrepaired"
    assert data["unrepaired_reason"] == "budget_exhausted"
    # 已确认的单点修复（{"a":1,...} 固定序首个）保留为最佳候选。
    assert data["candidate_instance"] == {"x": 1, "a": 1}
    assert data["change_count"] == 1
    assert data["remaining_error_count"] == 3


# ---- 确定性 ------------------------------------------------------------------


def test_determinism_same_request_same_response():
    first = post(MULTI_SCHEMA, MULTI_INSTANCE).json()
    second = post(MULTI_SCHEMA, MULTI_INSTANCE).json()
    assert first == second


# ---- 布尔 false schema -------------------------------------------------------


def test_boolean_false_schema_is_unrepairable():
    data = post(False, 1).json()
    assert data["status"] == "unrepaired"
    assert data["unrepaired_reason"] == "no_feasible_change"
    assert data["candidate_instance"] == 1
    assert data["remaining_errors"][0]["keyword"] is None


# ---- 直接测试 RepairSearcher / 单点改动构造 -----------------------------------


def test_required_fix_builder_skips_present_and_defaultless():
    schema = {
        "type": "object",
        "required": ["a", "b"],
        "properties": {"a": {"default": 1}, "b": {}},
    }
    validator = build_validator(schema)
    [err] = validator.iter_errors({"b": 2})
    fixes = _required_fixes(err, {"b": 2})
    assert len(fixes) == 1
    assert fixes[0].path == ("a",)
    assert fixes[0].to_value == 1
    assert fixes[0].op == "add"


def test_length_fix_builders_noop_when_not_violating():
    validator = build_validator({"type": "string", "minLength": 2})
    # 长度已满足时没有 minLength 错误，构造器自然不会被调用；直接校验守卫。
    [err] = validator.iter_errors("a")
    assert _min_length_fixes(err, "a")[0].to_value == "aa"
    validator2 = build_validator({"type": "string", "maxLength": 2})
    [err2] = validator2.iter_errors("abcd")
    assert _max_length_fixes(err2, "abcd")[0].to_value == "ab"


def test_fixable_keyword_set_matches_spec():
    assert FIXABLE_KEYWORDS == {
        "required",
        "enum",
        "const",
        "minimum",
        "maximum",
        "minLength",
        "maxLength",
    }


def test_repair_searcher_direct_usage():
    validator = build_validator({"type": "integer", "minimum": 5})
    searcher = RepairSearcher(validator, 1, max_evaluations=10)
    result = searcher.run()
    assert result.status == "repaired"
    assert result.candidate_instance == 5
    assert result.evaluations == 2
    payload = result.to_dict()
    assert payload["remaining_errors"] == []
    assert payload["max_evaluations"] == 10


@pytest.mark.parametrize(
    "schema,instance",
    [
        # type / pattern / uniqueItems / minItems 均不在可修复集合。
        ({"type": "integer"}, "x"),
        ({"type": "string", "pattern": "^z+$"}, "abc"),
        ({"type": "array", "uniqueItems": True}, [1, 1]),
        ({"type": "array", "minItems": 3}, [1]),
    ],
)
def test_other_keywords_only_participate_in_validation(schema, instance):
    data = post(schema, instance).json()
    assert data["status"] == "unrepaired"
    assert data["candidate_instance"] == instance
    assert data["changes"] == []
    assert data["remaining_error_count"] >= 1
