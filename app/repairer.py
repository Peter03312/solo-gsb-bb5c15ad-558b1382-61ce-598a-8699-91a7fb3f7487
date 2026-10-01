"""数据导入预览的有界自动修复器：``POST /repair`` 的搜索内核。

给定 JSON Schema 与一个待导入实例，在 **校验次数上限** 内做确定性的有界搜索，
尝试用尽量少的单点改动把实例修到通过校验：

1. 仅为下列关键字构造 *单点改动*（其余约束照常参与完整校验）：

   - ``required``：缺失属性在其声明子 schema 中带 ``default`` 时补默认值，
     **不凭空捏造必填值**（没有 default 的缺失属性无法修复）；
   - ``enum`` / ``const``：把违例值替换为取值集合中的成员（enum 至多取
     "同类型首个 + 任意首个" 两个确定性候选，避免大枚举拖垮预算）；
   - ``minimum`` / ``maximum``：夹紧到边界值本身（整型值且边界非整数时，
     取仍满足约束的最近整数 ceil/floor）；
   - ``minLength`` / ``maxLength``：重复末字符（空串补 ``"x"``）填充到下界，
     或截断到上界；

2. 多处错误采用 **按改动位置数分层的有界最佳优先搜索（BFS）**：第 k 层的
   候选恰好触及 k 个 JSON Pointer 位置；每个候选都重新执行一次全量校验，
   只有 *全部通过* 才标记成功；候选实例按规范 JSON 序列化去重；
3. 预算耗尽或不存在可行改动时，保留搜索过程中见到的最佳候选
   （剩余错误最少、改动位置最少，次序相同时按固定的生成顺序）并明确标记未修复；
4. 原实例已经合法时不做任何修改，直接返回 ``valid``。

一次评估 = 对一个候选实例执行一次完整 JSON Schema 校验（含对原实例的首次
校验）。全程无随机源，相同输入必然得到相同结果。
"""

from __future__ import annotations

import copy
import math
import re
from dataclasses import dataclass, field
from typing import Any

from .minimizer import get_at, pointer_from_path, set_at
from .mutator import _canonical, _is_int, _is_num, _json_equal, _resolve_local_ref, _types_of

# 修复状态。
STATUS_VALID = "valid"          # 原实例已合法，未做修改。
STATUS_REPAIRED = "repaired"    # 存在全部通过校验的候选实例。
STATUS_UNREPAIRED = "unrepaired"  # 预算耗尽或无可行改动，仅保留最佳候选。

# 未修复原因。
REASON_BUDGET = "budget_exhausted"      # 评估预算耗尽，尚未找到合法候选。
REASON_NO_EDITS = "no_applicable_edits"  # 没有任何可构造的单点改动。
REASON_UNREACHABLE = "unrepairable"      # 可行改动均尝试过，仍无法全部通过。

# 单个违例值最多尝试的 enum 候选数（同类型优先），用于给搜索规模设界。
_MAX_ENUM_CANDIDATES = 2


def _brief(value: Any, limit: int = 24) -> str:
    text = repr(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _peek(root: Any, path: tuple) -> tuple[Any, bool]:
    """读取 ``path`` 处的值；路径不存在时返回 ``(None, False)``。"""
    node = root
    for p in path:
        try:
            if isinstance(node, dict):
                if p not in node:
                    return None, False
                node = node[p]
            elif isinstance(node, list):
                if not isinstance(p, int) or p < 0 or p >= len(node):
                    return None, False
                node = node[p]
            else:
                return None, False
        except (KeyError, IndexError, TypeError):
            return None, False
    return node, True


@dataclass(frozen=True)
class _Edit:
    """一次单点改动。

    ``key`` 非空表示在 ``path`` 处的对象上新增缺失属性；否则替换 ``path``
    处的现有值。
    """

    path: tuple
    key: Any
    keyword: str
    action: str
    new_value: Any
    detail: str

    @property
    def target_path(self) -> tuple:
        return self.path if self.key is None else self.path + (self.key,)

    def apply(self, instance: Any) -> Any:
        value = copy.deepcopy(self.new_value)
        if self.key is not None:
            parent = get_at(instance, self.path)
            return set_at(instance, self.path, {**parent, self.key: value})
        return set_at(instance, self.path, value)


@dataclass(frozen=True)
class _Err:
    """全量校验后保留的单条错误（按 path/keyword/message 去重）。"""

    path: tuple
    keyword: str
    message: str


@dataclass
class _State:
    """搜索中的一个候选状态。"""

    instance: Any
    edits: tuple[_Edit, ...]
    errors: tuple[_Err, ...]
    # 去重前的原始校验错误对象（仅用于读取 validator_value 构造改动；
    # 每个状态在入堆前只执行过一次全量校验，这里不额外消耗评估）。
    raw_errors: tuple = ()
    # 截至该状态，改动链触及过的 JSON Pointer 位置集合（同一位置的多次
    # 编辑只计一个"改动位置"）。
    locations: frozenset = frozenset()


class _SchemaIndex:
    """schema 节点在实例路径上的同步索引（带本地 ``$ref`` / ``allOf`` 展开）。

    遍历史终 *实例驱动*：沿 ``properties`` / ``patternProperties`` /
    ``additionalProperties`` / ``prefixItems`` / ``items`` / ``allOf`` 与
    本地 ``$ref`` 与当前实例结构同步下降。新增默认值后出现的新路径按需用
    最新候选重新遍历（按节点 id 去重，已记录的路径不受影响）。
    """

    def __init__(self, schema: Any):
        self.root = schema
        # path -> 应用于该路径的去重 schema 节点（遍历顺序）。
        self._applied: dict[tuple, list] = {}

    def nodes_at(self, path: tuple, instance: Any) -> list:
        if path not in self._applied:
            self._walk(self.root, instance, (), set())
        return self._applied.get(path, [])

    def required_at(self, path: tuple, instance: Any, present: set) -> list[str]:
        """该路径上所有应用子 schema 声明的、当前仍缺失的必填属性（有序并集）。"""
        ordered: list[str] = []
        for node in self.nodes_at(path, instance):
            required = node.get("required")
            if not isinstance(required, list):
                continue
            for key in required:
                if (
                    isinstance(key, str)
                    and key not in present
                    and key not in ordered
                ):
                    ordered.append(key)
        return ordered

    def default_for(self, path: tuple, key: str, instance: Any) -> tuple[bool, Any]:
        """首个声明 ``properties.<key>.default`` 的应用子 schema 给出的默认值。"""
        for node in self.nodes_at(path, instance):
            props = node.get("properties")
            if isinstance(props, dict) and isinstance(props.get(key), dict):
                sub = props[key]
                if "default" in sub:
                    return True, copy.deepcopy(sub["default"])
        return False, None

    def _walk(self, node: Any, inst: Any, path: tuple, visited: set) -> None:
        if not isinstance(node, dict):
            return  # 布尔 schema 没有可遍历的约束。
        ident = (id(node), path)
        if ident in visited:
            return  # $ref 环 / 递归 schema：同一节点同一路径只展开一次。
        visited.add(ident)

        recorded = self._applied.setdefault(path, [])
        if not any(existing is node for existing in recorded):
            recorded.append(node)

        ref = node.get("$ref")
        if isinstance(ref, str) and ref.startswith("#"):
            resolved = _resolve_local_ref(self.root, ref)
            if isinstance(resolved, dict):
                self._walk(resolved, inst, path, visited)

        all_of = node.get("allOf")
        if isinstance(all_of, list):
            for sub in all_of:
                self._walk(sub, inst, path, visited)

        if isinstance(inst, dict):
            props = node.get("properties")
            if isinstance(props, dict):
                for key, sub in props.items():
                    if isinstance(key, str) and key in inst:
                        self._walk(sub, inst[key], path + (key,), visited)

            compiled = []
            patterns = node.get("patternProperties")
            if isinstance(patterns, dict):
                for pattern, sub in patterns.items():
                    try:
                        rx = re.compile(pattern)
                    except re.error:
                        continue
                    compiled.append(rx)
                    for key in inst:
                        if rx.search(key):
                            self._walk(sub, inst[key], path + (key,), visited)

            additional = node.get("additionalProperties")
            if isinstance(additional, dict):
                declared = set(props) if isinstance(props, dict) else set()
                for key in inst:
                    if key in declared or any(rx.search(key) for rx in compiled):
                        continue
                    self._walk(additional, inst[key], path + (key,), visited)

        if isinstance(inst, list):
            prefix = node.get("prefixItems")
            start = 0
            if isinstance(prefix, list):
                for i, sub in enumerate(prefix[: len(inst)]):
                    self._walk(sub, inst[i], path + (i,), visited)
                start = len(prefix)

            items = node.get("items")
            if isinstance(items, dict):
                for i in range(start, len(inst)):
                    self._walk(items, inst[i], path + (i,), visited)
            elif isinstance(items, list):
                # 旧版 tuple form（draft 2019-09 及以前）。
                for i, sub in enumerate(items[: len(inst)]):
                    self._walk(sub, inst[i], path + (i,), visited)


@dataclass
class Change:
    """最佳候选相对原实例在单个 JSON Pointer 位置上的最终变更。"""

    pointer: str
    keyword: str
    action: str
    added: bool
    old: Any
    new: Any
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "pointer": self.pointer,
            "keyword": self.keyword,
            "action": self.action,
            "added": self.added,
            "from": self.old,
            "to": self.new,
            "detail": self.detail,
        }


@dataclass
class RepairResult:
    status: str
    reason: str | None
    candidate_instance: Any
    original_instance: Any
    original_valid: bool
    remaining_errors: list[_Err] = field(default_factory=list)
    changes: list[Change] = field(default_factory=list)
    evaluations: int = 0
    stopped_by_budget: bool = False

    @property
    def remaining_error_count(self) -> int:
        return len(self.remaining_errors)

    @property
    def changed_location_count(self) -> int:
        return len(self.changes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "reason": self.reason,
            "candidate_instance": self.candidate_instance,
            "original_instance": self.original_instance,
            "original_valid": self.original_valid,
            "remaining_error_count": self.remaining_error_count,
            "remaining_errors": [
                {
                    "keyword": e.keyword,
                    "pointer": pointer_from_path(e.path),
                    "message": e.message,
                }
                for e in self.remaining_errors
            ],
            "changes": [c.to_dict() for c in self.changes],
            "changed_location_count": self.changed_location_count,
            "evaluations": self.evaluations,
            "stopped_by_budget": self.stopped_by_budget,
        }


class BudgetExhausted(Exception):
    """评估次数预算耗尽（内部控制流异常）。"""


class Repairer:
    """对单个 (schema, instance) 对执行有界自动修复搜索。"""

    def __init__(self, schema: Any, validator, instance: Any, max_evaluations: int):
        self.schema = schema
        self.validator = validator
        self.original = instance
        self.max_evaluations = max_evaluations
        self.evaluations = 0
        self.stopped_by_budget = False
        self.index = _SchemaIndex(schema)

    # ---- 校验 / 预算 -----------------------------------------------------

    def validate(self, instance: Any) -> tuple[tuple[_Err, ...], tuple]:
        """计一次评估并返回 ``(去重错误, 原始错误对象)``；预算耗尽抛异常。"""
        if self.evaluations >= self.max_evaluations:
            self.stopped_by_budget = True
            raise BudgetExhausted()
        self.evaluations += 1
        raw = tuple(self.validator.iter_errors(instance))
        deduped: list[_Err] = []
        seen: set = set()
        for err in raw:
            item = _Err(
                tuple(err.absolute_path),
                err.validator,
                err.message,
            )
            key = (item.path, item.keyword, item.message)
            if key not in seen:
                seen.add(key)
                deduped.append(item)
        return tuple(deduped), raw

    # ---- 主流程 -----------------------------------------------------------

    def repair(
        self,
        initial: tuple[tuple[_Err, ...], tuple] | None = None,
    ) -> RepairResult:
        # 第 1 次评估：原实例可能已经合法（不作修改），也是最佳候选的基线。
        # 允许接口层先单独执行首次校验（便于把不可用 schema 转成 422）。
        if initial is None:
            original_errors, original_raw = self.validate(self.original)
        else:
            original_errors, original_raw = initial
        original_valid = not original_errors

        original_state = _State(self.original, (), original_errors, original_raw)
        if original_valid:
            return self._result(STATUS_VALID, None, original_state, original_valid)

        # 搜索是按 *改动位置数* 分层的 BFS，堆键为
        # (位置数, 剩余错误数, 单调生成序号)。第 k 层的候选恰好触及 k 个
        # JSON Pointer 位置；第 k 层处理完后要么已经找到成功候选（必然是
        # 位置数最少的；同层内按固定序号取首个），要么确定至少需要 k+1 个
        # 位置，进入下一层。
        #
        # best 同时记录 *所有已评估候选*（含尚未弹出堆的）中的最佳，比较键
        # 为 (剩余错误数, 位置数, 固定序号)，保证预算随时截断时仍能返回
        # 已见到的最优候选。
        import heapq

        best = original_state
        best_rank = (len(best.errors), 0, 0)  # (剩余错误数, 位置数, 序号)
        heap: list[tuple[int, int, int, _State]] = [(0, len(best.errors), 0, best)]
        order_counter = 0
        seen_instances = {_canonical(self.original)}

        any_edits = False  # 是否至少构造出过一个单点改动。
        any_candidate = False  # 是否至少评估过一个去重后的新候选。

        while heap:
            locations, _, _, current = heapq.heappop(heap)
            if not current.errors:
                # 按 (位置数, 剩余错误数, 序号) 弹出：首个全通过状态即
                # "位置数最少、固定顺序最前" 的最优解。
                best = current
                return self._result(STATUS_REPAIRED, None, best, original_valid)

            try:
                edits = self._edits_for(current)
            except BudgetExhausted:
                break
            if edits:
                any_edits = True

            budget_dead = False
            for edit in edits:
                candidate_instance = edit.apply(current.instance)
                key = _canonical(candidate_instance)
                if key in seen_instances:
                    continue
                seen_instances.add(key)
                try:
                    candidate_errors, candidate_raw = self.validate(candidate_instance)
                except BudgetExhausted:
                    budget_dead = True
                    break
                except Exception:
                    # 校验器对该候选无法运行（触达坏 $ref 等）：跳过该候选。
                    continue
                any_candidate = True
                order_counter += 1
                candidate_locations = current.locations | {
                    pointer_from_path(edit.target_path)
                }
                candidate_state = _State(
                    candidate_instance,
                    current.edits + (edit,),
                    candidate_errors,
                    candidate_raw,
                    candidate_locations,
                )
                location_count = len(candidate_locations)
                candidate_rank = (
                    len(candidate_errors),
                    location_count,
                    order_counter,
                )
                # 即时更新最佳候选，保证预算截断时仍保留已见到的最优结果。
                if candidate_rank < best_rank:
                    best, best_rank = candidate_state, candidate_rank
                heapq.heappush(
                    heap,
                    (
                        location_count,
                        len(candidate_errors),
                        order_counter,
                        candidate_state,
                    ),
                )
            if budget_dead:
                break

        if self.stopped_by_budget:
            reason = REASON_BUDGET
        elif not any_edits:
            reason = REASON_NO_EDITS
        elif not any_candidate:
            # 所有改动都与已见候选重复（实际中等价于没有新的可行改动）。
            reason = REASON_NO_EDITS
        else:
            reason = REASON_UNREACHABLE
        return self._result(STATUS_UNREPAIRED, reason, best, original_valid)

    # ---- 单点改动构造 -----------------------------------------------------

    def _edits_for(self, state: _State) -> list[_Edit]:
        """按错误列表的固定遍历顺序构造去重后的单点改动候选。

        ``required`` 需要以 *最新候选* 重新做一次实例驱动的 schema 遍历
        （新增默认值后可能出现新路径）；其余关键字直接使用该状态已保存的
        原始校验错误对象中的 ``validator_value``，不额外消耗评估次数。
        """
        edits: list[_Edit] = []
        seen: set = set()
        handled_required: set[tuple] = set()

        def add(edit: _Edit) -> None:
            # 同一位置、同一关键字、同一新值只保留首个（固定顺序）。
            key = (
                pointer_from_path(edit.target_path),
                edit.keyword,
                _canonical(edit.new_value),
            )
            if key not in seen:
                seen.add(key)
                edits.append(edit)

        for raw in state.raw_errors:
            keyword = raw.validator
            path = tuple(raw.absolute_path)

            if keyword == "required":
                # jsonschema 对每个缺失属性各报一个 required 错误；
                # 同一路径统一处理一次，按声明顺序为所有带 default 的缺失
                # 属性各构造一个补默认值改动。
                if path in handled_required:
                    continue
                handled_required.add(path)
                parent, exists = _peek(state.instance, path)
                if not exists or not isinstance(parent, dict):
                    continue
                present = set(parent.keys())
                for name in self.index.required_at(path, state.instance, present):
                    found, default = self.index.default_for(path, name, state.instance)
                    if not found:
                        # 不凭空补必填值：没有 default 就不构造改动。
                        continue
                    add(
                        _Edit(
                            path,
                            name,
                            "required",
                            "add-default",
                            default,
                            f"add missing required property {name!r} from schema default {_brief(default)}",
                        )
                    )
                continue

            value, exists = _peek(state.instance, path)
            if not exists:
                continue
            for edit in self._scalar_edits(keyword, raw.validator_value, value, path):
                add(edit)

        return edits

    def _scalar_edits(
        self, keyword: str, constraint: Any, value: Any, path: tuple
    ) -> list[_Edit]:
        if keyword == "const":
            target = constraint
            if _json_equal(target, value):
                return []
            return [
                _Edit(
                    path,
                    None,
                    "const",
                    "set-const",
                    target,
                    f"set {_brief(value)} to const {_brief(target)}",
                )
            ]

        if keyword == "enum":
            if not isinstance(constraint, list) or not constraint:
                return []
            current_types = _types_of(value)
            same_type = [
                cand
                for cand in constraint
                if (_types_of(cand) & current_types) and not _json_equal(cand, value)
            ]
            other = [
                cand
                for cand in constraint
                if not (_types_of(cand) & current_types) and not _json_equal(cand, value)
            ]
            picked: list = []
            for cand in [*same_type, *other]:
                if all(not _json_equal(cand, p) for p in picked):
                    picked.append(cand)
                if len(picked) >= _MAX_ENUM_CANDIDATES:
                    break
            return [
                _Edit(
                    path,
                    None,
                    "enum",
                    "set-enum",
                    cand,
                    f"set {_brief(value)} to allowed enum value {_brief(cand)}",
                )
                for cand in picked
            ]

        if keyword == "minimum" and _is_num(constraint) and _is_num(value):
            if value >= constraint:
                return []  # 仅 draft 4 布尔 exclusiveMinimum 等场景，无法按 minimum 修复。
            if _is_int(value) and not float(constraint).is_integer():
                new_value: Any = math.ceil(constraint)
            else:
                new_value = constraint
            if _json_equal(new_value, value):
                return []
            return [
                _Edit(
                    path,
                    None,
                    "minimum",
                    "raise-to-minimum",
                    new_value,
                    f"raise {_brief(value)} to {_brief(new_value)} (>= minimum {constraint})",
                )
            ]

        if keyword == "maximum" and _is_num(constraint) and _is_num(value):
            if value <= constraint:
                return []
            if _is_int(value) and not float(constraint).is_integer():
                new_value = math.floor(constraint)
            else:
                new_value = constraint
            if _json_equal(new_value, value):
                return []
            return [
                _Edit(
                    path,
                    None,
                    "maximum",
                    "lower-to-maximum",
                    new_value,
                    f"lower {_brief(value)} to {_brief(new_value)} (<= maximum {constraint})",
                )
            ]

        if keyword == "minLength" and _is_int(constraint) and isinstance(value, str):
            if len(value) >= constraint:
                return []
            pad = constraint - len(value)
            candidates: list[Any] = []
            if value:
                # 重复末字符填充：最可能保住 pattern 等同节点其他约束。
                candidates.append(value + value[-1] * pad)
            filled = value + "x" * pad
            if not any(filled == c for c in candidates):
                candidates.append(filled)
            return [
                _Edit(
                    path,
                    None,
                    "minLength",
                    "pad-to-min-length",
                    cand,
                    f"pad string {len(value)} -> {len(cand)} chars",
                )
                for cand in candidates
            ]

        if keyword == "maxLength" and _is_int(constraint) and isinstance(value, str):
            if len(value) <= constraint:
                return []
            truncated = value[:constraint]
            return [
                _Edit(
                    path,
                    None,
                    "maxLength",
                    "truncate-to-max-length",
                    truncated,
                    f"truncate string {len(value)} -> {len(truncated)} chars",
                )
            ]

        # 其他关键字（type / pattern / multipleOf / exclusiveMinimum / ...）
        # 不构造改动，但仍参与每个候选的完整校验与剩余错误统计。
        return []

    # ---- 结果整理 ----------------------------------------------------------

    def _result(
        self,
        status: str,
        reason: str | None,
        best: _State,
        original_valid: bool,
    ) -> RepairResult:
        return RepairResult(
            status=status,
            reason=reason,
            candidate_instance=best.instance,
            original_instance=self.original,
            original_valid=original_valid,
            remaining_errors=list(best.errors),
            changes=self._changes(best),
            evaluations=self.evaluations,
            stopped_by_budget=self.stopped_by_budget,
        )

    def _changes(self, best: _State) -> list[Change]:
        """把同一位置的多次改动折叠为最终一次（保留触发改动的关键字）。"""
        by_pointer: dict[str, Change] = {}
        order: list[str] = []
        for edit in best.edits:
            pointer = pointer_from_path(edit.target_path)
            old, old_exists = _peek(self.original, edit.target_path)
            new, _ = _peek(best.instance, edit.target_path)
            change = Change(
                pointer=pointer,
                keyword=edit.keyword,
                action=edit.action,
                added=not old_exists,
                old=old if old_exists else None,
                new=new,
                detail=edit.detail,
            )
            if pointer not in by_pointer:
                order.append(pointer)
            by_pointer[pointer] = change
        return [by_pointer[p] for p in order]
