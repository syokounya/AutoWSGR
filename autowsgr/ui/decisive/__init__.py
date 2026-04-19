"""决战 UI 子包。

包含决战相关的 UI 控制器:

- ``DecisiveBattlePage`` — 决战总览页 (章节导航/购买/进入地图)
- ``DecisiveMapController`` — 决战地图页 (overlay/出征/修理)
- ``DecisiveBattlePreparationPage`` — 决战出征准备页 (决战专用换船)

以及内部模块:

- ``overlay.py`` — 像素签名/坐标常量/overlay 检测函数
- ``battle_page.py`` — 总览页控制器
- ``map_controller.py`` — 地图页控制器
- ``fleet_ocr.py`` — 舰队 OCR 识别 (map_controller 内部委托)
- ``preparation.py`` — 决战专用出征准备页控制器

模块结构::

    decisive/
    ├── __init__.py           ← 本文件 (统一导出)
    ├── battle_page.py        ← DecisiveBattlePage (总览页)
    ├── overlay.py            ← 签名/坐标常量/检测函数/DecisiveOverlay
    ├── map_controller.py     ← DecisiveMapController (地图页 UI 操作)
    ├── fleet_ocr.py          ← 舰队 OCR 识别函数
    └── preparation.py        ← DecisiveBattlePreparationPage (决战换船)
"""

from autowsgr.types import PageName
from autowsgr.ui.decisive.battle_page import DecisiveBattlePage
from autowsgr.ui.decisive.map_controller import DecisiveMapController
from autowsgr.ui.decisive.preparation import DecisiveBattlePreparationPage


__all__ = [
    'DecisiveBattlePage',
    'DecisiveBattlePreparationPage',
    'DecisiveMapController',
]


# ── 页面注册 ────────────────────────────────────────────────────────────
# 决战地图页 (含 overlay) 注册到页面中心，使 get_current_page / goto_page
# 能够识别该状态，避免导航失败时无法标注。

from autowsgr.ui.page import register_page
from autowsgr.ui.decisive.overlay import (
    detect_decisive_overlay,
    get_overlay_signature,
    is_decisive_map_page,
)
from autowsgr.vision.annotation import annotations_from_pixel_signature


def _is_decisive_map(screen):
    """识别决战地图页 (含任何 overlay)。"""
    return is_decisive_map_page(screen) or detect_decisive_overlay(screen) is not None


def _get_decisive_map_annotations(screen):
    """为决战地图页生成标注。

    优先返回命中 overlay 的签名标注；若无 overlay 则返回地图页签名标注。
    """
    overlay = detect_decisive_overlay(screen)
    if overlay is not None:
        sig = get_overlay_signature(overlay)
        return annotations_from_pixel_signature(screen, sig)
    from autowsgr.ui.decisive.overlay import SIG_MAP_PAGE
    return annotations_from_pixel_signature(screen, SIG_MAP_PAGE)


register_page(
    PageName.DECISIVE_MAP,
    _is_decisive_map,
    get_annotations=_get_decisive_map_annotations,
)
