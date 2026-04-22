"""战斗停止条件 — 阈值检查与评估器。

提供 :class:`StopCondition` 数据类与 :class:`StopConditionEvaluator`，
在战斗流程的安全检查点（出征前 / 战斗后）评估是否应停止当前任务。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from autowsgr.types import ConditionFlag


if TYPE_CHECKING:
    from autowsgr.combat.history import CombatHistory
    from autowsgr.context import GameContext


# ═══════════════════════════════════════════════════════════════════════════════
# 数据模型
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True, slots=True)
class StopCondition:
    """战斗停止条件集合。

    所有字段均为可选；未设置的字段不参与评估。
    任意一个条件命中即触发停止。
    """

    ship_count_ge: int | None = None
    """当天获取舰船数 ≥ 此值时停止（上限 500）。"""

    loot_count_ge: int | None = None
    """当天获取战利品数 ≥ 此值时停止（上限 50）。"""

    target_ship_dropped: list[str] = field(default_factory=list)
    """掉落列表中任意一艘舰船时停止。"""


# ═══════════════════════════════════════════════════════════════════════════════
# 评估器
# ═══════════════════════════════════════════════════════════════════════════════


class StopConditionEvaluator:
    """停止条件评估器。

    在战斗前后的安全检查点调用，判断当前上下文是否满足停止条件。
    """

    def __init__(self, condition: StopCondition | None) -> None:
        self._condition = condition

    def evaluate(
        self,
        ctx: GameContext,
        history: CombatHistory | None = None,
    ) -> ConditionFlag | None:
        """评估停止条件。

        Parameters
        ----------
        ctx:
            游戏上下文（含当天掉落计数）。
        history:
            本次战斗历史（用于检查目标船掉落）；
            若为 ``None`` 则跳过掉落相关检查（适用于出征前预检）。

        Returns
        -------
        ConditionFlag | None
            命中的停止标志；未命中返回 ``None``。
        """
        if self._condition is None:
            return None

        cond = self._condition

        # ── 舰船数阈值 ──
        if cond.ship_count_ge is not None:
            if ctx.dropped_ship_count >= cond.ship_count_ge:
                return ConditionFlag.SHIP_FULL

        # ── 战利品阈值 ──
        if cond.loot_count_ge is not None:
            if ctx.dropped_loot_count >= cond.loot_count_ge:
                return ConditionFlag.LOOT_MAX

        # ── 目标船掉落 ──
        if cond.target_ship_dropped and history is not None:
            for fight in history.get_fight_results_list():
                if fight.dropped_ship and fight.dropped_ship in cond.target_ship_dropped:
                    return ConditionFlag.TARGET_SHIP_DROPPED

        return None

    def evaluate_preflight(self, ctx: GameContext) -> ConditionFlag | None:
        """出征前预检 — 仅检查计数阈值（不涉及掉落）。"""
        return self.evaluate(ctx, history=None)
