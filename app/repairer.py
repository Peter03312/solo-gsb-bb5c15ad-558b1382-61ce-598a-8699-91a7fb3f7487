"""数据导入预览的确定性有界修复搜索：``POST /import-preview`` 的内核。

调用方提交 JSON Schema、待导入实例与校验次数上限，服务在预算内做
**有界最佳优先搜索（bounded best-first search）**：

1. 首次全量校验原实例：合法则原样返回（``status="valid"``），不做任何修改；
2. 从当前候选的 *全部* 校验错误出发，只为以下关键字构造 **单点改动**
   （每次只改一个 JSON Pointer 位置）：

   - ``required``：仅当缺失属性在对应属性子 schema 中声明了 ``default``
     时，用深拷贝的默认值 *补属性*——绝不凭空捏造必填值；
   - ``enum`` / ``const``：依次取枚举/常量声明值替换；
   - ``minimum`` / ``maximum``：钳到边界值（整型实例遇小数边界时额外提供
     保持整型的最近值，避免被 ``type: integer`` 抢先拦截）；
   - ``minLength`` / ``maxLength``：按现有字符末字符（空串用 ``"x"``）
     填充到下界 / 截断到上界；

   其他约束（``type`` / ``pattern`` / ``uniqueItems`` / ``format`` …）
   **不构造改动**，但仍参与每个候选的完整校验，体现在剩余错误中。
3. 每个候选都用真实校验器 **重新全量校验**：统计剩余错误总数；全部通过
   （零错误）才标记 ``status="repaired"``。候选按规范 JSON 序列化全局去重；
4. 搜索顺序为 ``(剩余错误数, 改动位置数, 发现序号)``，固定有序、无随机源；
   因此第一个被发现的零错误候选必然是错误数最少（0）且改动位置数最少的
   候选，同票时由固定生成顺序裁决。
5. 预算耗尽或搜索空间穷尽（无可行改动）时，保留搜索过程中见到的 **最佳
   候选**（同样按上述顺序比较），``status="unrepaired"`` 并给出原因与
   候选剩余错误。
"""

from __future__ import annotations

import copy
import heapq
import json
import math
from dataclasses import dataclass, field
from typing import Any

from jsonschema.exceptions import ValidationError

from .minimizer import BudgetExhausted, get_at, pointer_from_path, set_at
from .mutator import _brief, _is_int, _is_num, _json_equal

# 状态值（对外暴露在响应 status 字段中）。
STATUS_VALID = "valid"
STATUS_REPAIRED = "repaired"
STATUS_UNREPAIRED = "unrepaired"

# unrepaired_reason 取值。
REASON_BUDGET = "budget_exhausted"
REASON_NO_CHANGE = "no_feasible_change"

# 允许构造单点改动的关键字白名单；白名单外的约束只参与完整校验。
FIXABLE_KEYWORDS = frozenset(
    {"required", "enum", "const", "minimum", "maximum", "minLength", "maxLength"}
)


def _canonical(value: Any) -> str:
    """规范 JSON 序列化：键排序、无空白，用作候选去重键。"""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


# ---- 变更 / 错误 / 状态 ------------------------------------------------------


@dataclass(frozen=True)
class Change:
    """一次单点改动：在 ``pointer`` 处把 ``from_value`` 改为 ``to_value``。"""

    pointer: str
    keyword: str
    op: str  # "add"（补缺失属性）/ "replace"（替换既有标量）
    from_value: Any
    to_value: Any
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "pointer": self.pointer,
            "keyword": self.keyword,
            "op": self.op,
            "from": self.from_value,
            "to": self.to_value,
            "detail": self.detail,
        }


@dataclass
class _State:
    """搜索中的一个候选：实例实例、累积改动链与全量校验错误。"""

    instance: Any
    changes: tuple[Change, ...]
    locations: frozenset[str]
    errors: list[ValidationError]

    @property
    def error_count(self) -> int:
        return len(self.errors)

    @property
    def change_count(self) -> int:
        """改动位置数：按 JSON Pointer 去重后的位置个数。"""
        return len(self.locations)


def _error_dict(err: ValidationError) -> dict[str, Any]:
    """把校验错误整理为响应中的剩余错误结构。"""
    return {
        "keyword": err.validator,
        "pointer": pointer_from_path(tuple(err.absolute_path)),
        "message": err.message,
    }


@dataclass
class RepairResult:
    status: str
    original_instance: Any
    candidate_instance: Any
    changes: list[Change] = field(default_factory=list)
    remaining_errors: list[ValidationError] = field(default_factory=list)
    evaluations: int = 0
    max_evaluations: int = 0
    stopped_by_budget: bool = False
    unrepaired_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "unrepaired_reason": self.unrepaired_reason,
            "original_instance": self.original_instance,
            "candidate_instance": self.candidate_instance,
            "changes": [c.to_dict() for c in self.changes],
            "change_count": len({c.pointer for c in self.changes}),
            "remaining_error_count": len(self.remaining_errors),
            "remaining_errors": [_error_dict(e) for e in self.remaining_errors],
            "evaluations": self.evaluations,
            "max_evaluations": self.max_evaluations,
            "stopped_by_budget": self.stopped_by_budget,
        }


# ---- 不可变对象补属性 ---------------------------------------------------------


def _with_property(root: Any, path: tuple, key: str, value: Any) -> Any:
    """不可变式地在 ``path`` 指向的对象上新增/覆盖属性 ``key``。"""
    if not path:
        new_obj = dict(root)
        new_obj[key] = value
        return new_obj
    head, rest = path[0], path[1:]
    child = _with_property(root[head], rest, key, value)
    if isinstance(root, dict):
        new_root = dict(root)
        new_root[head] = child
        return new_root
    new_root = list(root)
    new_root[head] = child
    return new_root


# ---- 单点改动构造 -------------------------------------------------------------


@dataclass(frozen=True)
class _Fix:
    """一个待尝试的单点改动（尚未应用、尚未校验）。"""

    path: tuple
    pointer: str
    keyword: str
    op: str
    from_value: Any
    to_value: Any
    detail: str


def _required_fixes(err: ValidationError, inst: Any) -> list[_Fix]:
    """required：只为“确实缺失且属性子 schema 带 default”的键构造补属性改动。"""
    path = tuple(err.absolute_path)
    obj = get_at(inst, path)
    if not isinstance(obj, dict):
        return []
    required = err.validator_value if isinstance(err.validator_value, list) else []
    props = err.schema.get("properties") if isinstance(err.schema, dict) else None
    props = props if isinstance(props, dict) else {}
    fixes: list[_Fix] = []
    for key in required:  # 按 schema 声明顺序，保证确定性。
        if not isinstance(key, str) or key in obj:
            continue  # 只处理缺失属性；不删除、不覆盖既有值。
        sub = props.get(key)
        if not isinstance(sub, dict) or "default" not in sub:
            continue  # 没有 default：绝不凭空补必填值。
        value = copy.deepcopy(sub["default"])
        fixes.append(
            _Fix(
                path=path + (key,),
                pointer=pointer_from_path(path + (key,)),
                keyword="required",
                op="add",
                from_value=None,
                to_value=value,
                detail=f"add missing required property {key!r} from schema default",
            )
        )
    return fixes


def _enum_fixes(err: ValidationError, inst: Any) -> list[_Fix]:
    path = tuple(err.absolute_path)
    current = get_at(inst, path)
    allowed = err.validator_value if isinstance(err.validator_value, list) else []
    fixes: list[_Fix] = []
    offered: list[Any] = []
    for value in allowed:  # 按枚举声明顺序逐个尝试。
        if any(_json_equal(value, seen) for seen in offered):
            continue  # 容忍 enum 自身重复。
        offered.append(value)
        if _json_equal(value, current):
            continue
        fixes.append(
            _Fix(
                path=path,
                pointer=pointer_from_path(path),
                keyword="enum",
                op="replace",
                from_value=current,
                to_value=copy.deepcopy(value),
                detail=f"set {_brief(current)} to enum value {_brief(value)}",
            )
        )
    return fixes


def _const_fixes(err: ValidationError, inst: Any) -> list[_Fix]:
    path = tuple(err.absolute_path)
    current = get_at(inst, path)
    value = err.validator_value
    if _json_equal(value, current):
        return []
    return [
        _Fix(
            path=path,
            pointer=pointer_from_path(path),
            keyword="const",
            op="replace",
            from_value=current,
            to_value=copy.deepcopy(value),
            detail=f"set {_brief(current)} to const value {_brief(value)}",
        )
    ]


def _minimum_fixes(err: ValidationError, inst: Any) -> list[_Fix]:
    path = tuple(err.absolute_path)
    current = get_at(inst, path)
    bound = err.validator_value
    if not _is_num(bound) or not _is_num(current):
        return []
    values: list[Any] = [bound]
    # 整型实例遇小数边界：额外给一个保持整型的最近值，避免被 type 抢先拦截。
    if _is_int(current) and not float(bound).is_integer():
        nearest = math.ceil(bound)
        if not any(_json_equal(nearest, v) for v in values):
            values.append(nearest)
    return [
        _Fix(
            path=path,
            pointer=pointer_from_path(path),
            keyword="minimum",
            op="replace",
            from_value=current,
            to_value=value,
            detail=f"set {_brief(current)} to minimum boundary {_brief(value)}",
        )
        for value in values
        if not _json_equal(value, current)
    ]


def _maximum_fixes(err: ValidationError, inst: Any) -> list[_Fix]:
    path = tuple(err.absolute_path)
    current = get_at(inst, path)
    bound = err.validator_value
    if not _is_num(bound) or not _is_num(current):
        return []
    values: list[Any] = [bound]
    if _is_int(current) and not float(bound).is_integer():
        nearest = math.floor(bound)
        if not any(_json_equal(nearest, v) for v in values):
            values.append(nearest)
    return [
        _Fix(
            path=path,
            pointer=pointer_from_path(path),
            keyword="maximum",
            op="replace",
            from_value=current,
            to_value=value,
            detail=f"set {_brief(current)} to maximum boundary {_brief(value)}",
        )
        for value in values
        if not _json_equal(value, current)
    ]


def _min_length_fixes(err: ValidationError, inst: Any) -> list[_Fix]:
    path = tuple(err.absolute_path)
    current = get_at(inst, path)
    bound = err.validator_value
    if not _is_int(bound) or not isinstance(current, str) or len(current) >= bound:
        return []
    fill = current[-1] if current else "x"  # 空串没有可重复字符，退到固定 "x"。
    value = current + fill * (bound - len(current))
    return [
        _Fix(
            path=path,
            pointer=pointer_from_path(path),
            keyword="minLength",
            op="replace",
            from_value=current,
            to_value=value,
            detail=f"extend string {len(current)} -> {len(value)} chars (minLength {bound})",
        )
    ]


def _max_length_fixes(err: ValidationError, inst: Any) -> list[_Fix]:
    path = tuple(err.absolute_path)
    current = get_at(inst, path)
    bound = err.validator_value
    if not _is_int(bound) or not isinstance(current, str) or len(current) <= bound:
        return []
    value = current[:bound]
    return [
        _Fix(
            path=path,
            pointer=pointer_from_path(path),
            keyword="maxLength",
            op="replace",
            from_value=current,
            to_value=value,
            detail=f"truncate string {len(current)} -> {len(value)} chars (maxLength {bound})",
        )
    ]


_FIX_BUILDERS = {
    "required": _required_fixes,
    "enum": _enum_fixes,
    "const": _const_fixes,
    "minimum": _minimum_fixes,
    "maximum": _maximum_fixes,
    "minLength": _min_length_fixes,
    "maxLength": _max_length_fixes,
}


def _fixes_for_state(state: _State) -> list[_Fix]:
    """从状态的全部错误构造单点改动；白名单外关键字只参与校验、不构造。"""
    fixes: list[_Fix] = []
    seen_keys: set[tuple[str, str]] = set()
    for err in state.errors:
        builder = _FIX_BUILDERS.get(err.validator)
        if builder is None:
            continue
        for fix in builder(err, state.instance):
            # 同一位置改成同一值只构造一次（可能由多处 allOf 错误同时提出）。
            key = (fix.pointer, _canonical(fix.to_value))
            if key in seen_keys:
                continue
            seen_keys.add(key)
            fixes.append(fix)
    return fixes


def _apply_fix(state: _State, fix: _Fix) -> Any:
    if fix.op == "add":
        # path 形如 (...对象路径, 新键名)。
        return _with_property(state.instance, fix.path[:-1], fix.path[-1], fix.to_value)
    return set_at(state.instance, fix.path, fix.to_value)


# ---- 有界最佳优先搜索 ---------------------------------------------------------


class RepairSearcher:
    """对单个 (schema, instance) 对执行有界修复搜索。"""

    def __init__(self, validator, instance: Any, max_evaluations: int):
        self.validator = validator
        self.original = instance
        self.max_evaluations = max_evaluations
        self.evaluations = 0
        self.stopped_by_budget = False

    def evaluate(self, instance: Any) -> list[ValidationError]:
        """计一次评估并返回 *全部* 校验错误（无错误为空列表）；预算耗尽抛异常。"""
        if self.evaluations >= self.max_evaluations:
            self.stopped_by_budget = True
            raise BudgetExhausted()
        self.evaluations += 1
        # 完整校验：所有约束（含不可修复的）都参与，调用方按错误总数排序。
        return list(self.validator.iter_errors(instance))

    def run(self) -> RepairResult:
        # 第 1 次评估：原实例。坏 $ref、非法正则等导致校验无法运行时，
        # 异常向上抛给接口层，按不可用 schema 返回 422。
        initial_errors = self.evaluate(self.original)
        initial = _State(self.original, (), frozenset(), initial_errors)

        if not initial_errors:
            # 原实例合法：不作修改，直接返回 valid。
            return self._result(STATUS_VALID, initial, None)

        # 堆项固定为 (剩余错误数, 改动位置数, 发现序号, 状态)；
        # 发现序号严格递增且唯一，堆比较永远不会落到不可比较的状态对象。
        best = initial
        counter = 0
        heap: list[tuple[int, int, int, _State]] = [(0, 0, counter, initial)]
        visited = {_canonical(initial.instance)}

        def is_better(candidate: _State, current: _State) -> bool:
            return (candidate.error_count, candidate.change_count) < (
                current.error_count,
                current.change_count,
            )

        while heap:
            _, _, _, state = heapq.heappop(heap)
            try:
                fixes = _fixes_for_state(state)
                for fix in fixes:
                    candidate_instance = _apply_fix(state, fix)
                    key = _canonical(candidate_instance)
                    if key in visited:
                        continue  # 候选全局去重，防止环路与重复评估。
                    visited.add(key)

                    try:
                        errors = self.evaluate(candidate_instance)
                    except BudgetExhausted:
                        raise
                    except Exception:
                        # 候选触发了校验器无法运行的 schema 分支（坏 $ref 等）：
                        # 该候选不可用，跳过继续尝试其他改动。
                        continue

                    changes = state.changes + (
                        Change(
                            pointer=fix.pointer,
                            keyword=fix.keyword,
                            op=fix.op,
                            from_value=fix.from_value,
                            to_value=fix.to_value,
                            detail=fix.detail,
                        ),
                    )
                    child = _State(
                        candidate_instance,
                        changes,
                        frozenset(c.pointer for c in changes),
                        errors,
                    )
                    if is_better(child, best):
                        best = child
                    if not errors:
                        # 最佳优先序保证第一个零错误候选即改动位置数最少者。
                        return self._result(STATUS_REPAIRED, child, None)
                    counter += 1
                    heapq.heappush(
                        heap,
                        (child.error_count, child.change_count, counter, child),
                    )
            except BudgetExhausted:
                break

        reason = REASON_BUDGET if self.stopped_by_budget else REASON_NO_CHANGE
        return self._result(STATUS_UNREPAIRED, best, reason)

    def _result(
        self, status: str, state: _State, reason: str | None
    ) -> RepairResult:
        return RepairResult(
            status=status,
            original_instance=self.original,
            candidate_instance=state.instance,
            changes=list(state.changes),
            remaining_errors=state.errors,
            evaluations=self.evaluations,
            max_evaluations=self.max_evaluations,
            stopped_by_budget=self.stopped_by_budget,
            unrepaired_reason=reason,
        )
