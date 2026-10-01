"""POST /repair（数据导入预览自动修复）的端到端 / 单元测试。运行：python -m pytest -q"""

from fastapi.testclient import TestClient

from app.main import app
from app.minimizer import build_validator
from app.repairer import Repairer

client = TestClient(app)


def post(schema, instance, **extra):
    body = {"schema": schema, "instance": instance}
    body.update(extra)
    return client.post("/repair", json=body)


# ---- HTTP 层 ---------------------------------------------------------------


def test_valid_instance_unchanged():
    schema = {"type": "object", "required": ["id"], "properties": {"id": {"type": "integer"}}}
    resp = post(schema, {"id": 3})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "valid"
    assert data["reason"] is None
    assert data["candidate_instance"] == {"id": 3}
    assert data["changes"] == []
    assert data["remaining_errors"] == []
    assert data["original_valid"] is True
    assert data["evaluations"] == 1
    assert data["stopped_by_budget"] is False


def test_invalid_schema_returns_422():
    resp = post({"type": "not-a-type"}, {})
    assert resp.status_code == 422
    assert resp.json()["error"] == "invalid_schema"


def test_unusable_schema_bad_ref_returns_422():
    resp = post({"$ref": "#/definitions/missing"}, {})
    assert resp.status_code == 422
    assert resp.json()["error"] == "invalid_schema"


def test_bad_max_evaluations_returns_422():
    assert post({"type": "integer"}, "x", max_evaluations=0).status_code == 422
    assert post({"type": "integer"}, "x", max_evaluations=10**9).status_code == 422


def test_healthz_still_ok():
    assert client.get("/healthz").json() == {"status": "ok"}


# ---- required + default -----------------------------------------------------


def test_required_missing_without_default_is_not_fabricated():
    schema = {"type": "object", "required": ["id"], "properties": {"id": {"type": "integer"}}}
    data = post(schema, {"x": 1}).json()
    assert data["status"] == "unrepaired"
    assert data["reason"] == "no_applicable_edits"
    # 不凭空补必填值：候选保持原实例不变。
    assert data["candidate_instance"] == {"x": 1}
    assert data["changes"] == []
    assert data["remaining_error_count"] == 1
    assert data["remaining_errors"][0]["keyword"] == "required"


def test_required_add_default_and_changes_payload():
    schema = {
        "type": "object",
        "required": ["id"],
        "properties": {"id": {"type": "integer", "default": 7}},
    }
    data = post(schema, {"keep": "me"}).json()
    assert data["status"] == "repaired"
    assert data["candidate_instance"] == {"keep": "me", "id": 7}
    assert data["changed_location_count"] == 1
    change = data["changes"][0]
    assert change["pointer"] == "/id"
    assert change["keyword"] == "required"
    assert change["action"] == "add-default"
    assert change["added"] is True
    assert change["from"] is None
    assert change["to"] == 7


def test_extra_properties_are_kept():
    schema = {"type": "object", "required": ["id"], "properties": {"id": {"default": 1}}}
    data = post(schema, {"junk": "x", "n": 2}).json()
    assert data["candidate_instance"] == {"junk": "x", "n": 2, "id": 1}


def test_nested_default_added_then_its_own_enum_fixed():
    schema = {
        "type": "object",
        "required": ["meta"],
        "properties": {
            "meta": {
                "type": "object",
                "default": {},
                "required": ["kind"],
                "properties": {"kind": {"enum": ["a", "b"], "default": "a"}},
            }
        },
    }
    data = post(schema, {}).json()
    assert data["status"] == "repaired"
    assert data["candidate_instance"] == {"meta": {"kind": "a"}}
    pointers = [c["pointer"] for c in data["changes"]]
    assert pointers == ["/meta", "/meta/kind"]
    assert all(c["added"] for c in data["changes"])


def test_invalid_default_is_still_fully_validated():
    # default 类型不符：补进去后被 type 拦住（type 不在可修复关键字内）。
    # 候选与原实例都只剩 1 个错误，但原实例改动位置更少（0 个），
    # 故按 (剩余错误数, 改动位置数, 固定顺序) 保留原实例。
    schema = {"required": ["v"], "properties": {"v": {"type": "integer", "default": "oops"}}}
    data = post(schema, {}).json()
    assert data["status"] == "unrepaired"
    assert data["reason"] == "unrepairable"
    assert data["candidate_instance"] == {}
    assert data["remaining_errors"][0]["keyword"] == "required"


# ---- enum / const -----------------------------------------------------------


def test_enum_prefers_same_type_value():
    data = post({"enum": ["a", "b", "c"]}, "z").json()
    assert data["status"] == "repaired"
    assert data["candidate_instance"] == "a"
    assert data["changes"][0]["action"] == "set-enum"


def test_enum_falls_back_to_second_candidate():
    # 同类型首选 "cde"（长 3）被 maxLength 2 拦截，次选 "ab" 通过。
    schema = {"properties": {"v": {"type": "string", "enum": ["cde", "ab"], "maxLength": 2}}}
    data = post(schema, {"v": "z"}).json()
    assert data["status"] == "repaired"
    assert data["candidate_instance"] == {"v": "ab"}


def test_enum_bool_is_distinct_from_integer():
    assert post({"enum": [1]}, True).json()["candidate_instance"] == 1
    assert post({"enum": [True]}, 1).json()["candidate_instance"] is True


def test_const_repair():
    data = post({"const": 42}, 7).json()
    assert data["status"] == "repaired"
    assert data["candidate_instance"] == 42
    assert data["changes"][0]["keyword"] == "const"


# ---- minimum / maximum ------------------------------------------------------


def test_minimum_clamps_to_boundary():
    data = post({"type": "integer", "minimum": 5}, 1).json()
    assert data["status"] == "repaired"
    assert data["candidate_instance"] == 5
    ch = data["changes"][0]
    assert ch["keyword"] == "minimum"
    assert ch["from"] == 1 and ch["to"] == 5


def test_maximum_non_integer_boundary_uses_floor():
    data = post({"type": "integer", "maximum": 2.5}, 9).json()
    assert data["candidate_instance"] == 2
    assert isinstance(data["candidate_instance"], int)


def test_minimum_non_integer_boundary_uses_ceil():
    data = post({"type": "integer", "minimum": 0.5}, 0).json()
    assert data["candidate_instance"] == 1


def test_cascade_default_then_minimum_is_single_location():
    schema = {"required": ["v"], "properties": {"v": {"type": "integer", "minimum": 10, "default": 1}}}
    data = post(schema, {}).json()
    assert data["status"] == "repaired"
    assert data["candidate_instance"] == {"v": 10}
    # 同一位置两步编辑只算一个改动位置。
    assert data["changed_location_count"] == 1
    assert [c["pointer"] for c in data["changes"]] == ["/v"]
    assert data["changes"][0]["keyword"] == "minimum"


def test_draft4_exclusive_minimum_true_makes_boundary_unrepairable():
    schema = {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "type": "integer",
        "minimum": 5,
        "exclusiveMinimum": True,
    }
    data = post(schema, 1).json()
    assert data["status"] == "unrepaired"
    assert data["reason"] == "unrepairable"
    assert [e["keyword"] for e in data["remaining_errors"]] == ["minimum"]


# ---- minLength / maxLength --------------------------------------------------


def test_max_length_truncates():
    data = post({"type": "string", "maxLength": 3}, "abcdef").json()
    assert data["status"] == "repaired"
    assert data["candidate_instance"] == "abc"
    assert data["changes"][0]["action"] == "truncate-to-max-length"


def test_min_length_pads_with_last_char():
    data = post({"type": "string", "minLength": 5}, "ab").json()
    assert data["candidate_instance"] == "abbbb"


def test_min_length_empty_string_pads_with_x():
    data = post({"type": "string", "minLength": 4}, "").json()
    assert data["candidate_instance"] == "xxxx"


def test_padding_preserves_pattern():
    schema = {"type": "string", "minLength": 5, "pattern": "^[0-9]+$"}
    data = post(schema, "12").json()
    assert data["status"] == "repaired"
    assert data["candidate_instance"] == "12222"


# ---- 其他约束：不构造改动，但参与完整校验 ------------------------------------


def test_unsupported_keywords_still_count_as_remaining_errors():
    assert post({"type": "integer"}, "str").json()["reason"] == "no_applicable_edits"
    assert post({"type": "integer", "multipleOf": 3}, 10).json()["status"] == "unrepaired"
    data = post(
        {"type": "integer", "exclusiveMinimum": 5}, 1
    ).json()
    assert data["status"] == "unrepaired"
    assert data["remaining_errors"][0]["keyword"] == "exclusiveMinimum"


def test_boolean_schemas():
    data_true = post(True, {"anything": 1}).json()
    assert data_true["status"] == "valid"
    data_false = post(False, 1).json()
    assert data_false["status"] == "unrepaired"
    assert data_false["reason"] == "no_applicable_edits"
    assert data_false["candidate_instance"] == 1


# ---- 多错误有界搜索 ---------------------------------------------------------


MULTI_SCHEMA = {
    "type": "object",
    "required": ["id", "name", "role"],
    "properties": {
        "id": {"type": "integer", "minimum": 1, "maximum": 100, "default": 1},
        "name": {"type": "string", "minLength": 3, "maxLength": 10, "default": "abc"},
        "role": {"enum": ["admin", "user"], "default": "user"},
    },
}
MULTI_INSTANCE = {"id": -5, "name": "this-name-is-way-too-long", "role": "root"}


def test_multiple_errors_repaired_at_distinct_locations():
    data = post(MULTI_SCHEMA, MULTI_INSTANCE).json()
    assert data["status"] == "repaired"
    assert data["remaining_error_count"] == 0
    assert data["candidate_instance"] == {"id": 1, "name": "this-name-", "role": "admin"}
    assert data["changed_location_count"] == 3
    assert {c["pointer"] for c in data["changes"]} == {"/id", "/name", "/role"}
    # 每个成功候选都重新做了全量校验；至少 1（原）+ 3 次。
    assert data["evaluations"] >= 4


def test_array_elements_repaired_independently():
    schema = {
        "type": "array",
        "items": {
            "type": "object",
            "required": ["id"],
            "properties": {"id": {"type": "integer", "minimum": 1, "default": 1}},
        },
    }
    data = post(schema, [{"id": 0}, {}]).json()
    assert data["status"] == "repaired"
    assert data["candidate_instance"] == [{"id": 1}, {"id": 1}]


def test_budget_exhaustion_keeps_best_candidate():
    data = post(MULTI_SCHEMA, MULTI_INSTANCE, max_evaluations=3).json()
    assert data["status"] == "unrepaired"
    assert data["reason"] == "budget_exhausted"
    assert data["stopped_by_budget"] is True
    assert data["evaluations"] == 3
    # 已评估过一个单位置候选：最佳候选只剩 2 个错误、1 个改动位置。
    assert data["remaining_error_count"] == 2
    assert data["changed_location_count"] == 1


def test_budget_of_one_leaves_original_as_best():
    data = post(MULTI_SCHEMA, MULTI_INSTANCE, max_evaluations=1).json()
    assert data["status"] == "unrepaired"
    assert data["reason"] == "budget_exhausted"
    assert data["candidate_instance"] == MULTI_INSTANCE
    assert data["remaining_error_count"] == 3


def test_unrepairable_when_candidates_cannot_fully_pass():
    # 补默认值 "oops" 后永远违 type；其余路径都探索完仍无法通过。
    schema = {"required": ["v"], "properties": {"v": {"type": "integer", "default": "oops"}}}
    data = post(schema, {}).json()
    assert data["status"] == "unrepaired"
    assert data["reason"] == "unrepairable"


def test_duplicate_candidates_collapse():
    # allOf 两次声明同一 required，jsonschema 会重复报错；剩余错误需去重。
    schema = {"allOf": [{"required": ["a"]}, {"required": ["a"]}]}
    data = post(schema, {}).json()
    assert data["remaining_error_count"] == 1


def test_determinism():
    first = post(MULTI_SCHEMA, MULTI_INSTANCE).json()
    second = post(MULTI_SCHEMA, MULTI_INSTANCE).json()
    assert first["candidate_instance"] == second["candidate_instance"]
    assert first["changes"] == second["changes"]
    assert first["evaluations"] == second["evaluations"]


# ---- Repairer 直测 ----------------------------------------------------------


def test_every_candidate_is_fully_validated():
    validator = build_validator(MULTI_SCHEMA)
    repairer = Repairer(MULTI_SCHEMA, validator, MULTI_INSTANCE, 1000)
    result = repairer.repair()
    assert result.status == "repaired"
    # 成功候选必然通过全量校验（再独立验证一次）。
    assert not list(validator.iter_errors(result.candidate_instance))
