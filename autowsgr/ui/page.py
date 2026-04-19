"""UI 页面注册中心。

提供页面注册与识别功能:

- **register_page** - 注册页面识别函数
- **get_current_page** - 遍历注册表识别当前截图
- **get_registered_pages** - 列出所有已注册页面

导航 / 等待工具函数已迁移至 :mod:`autowsgr.ui.utils`，
此处保留兼容性再导出。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from autowsgr.infra.logger import get_logger
from autowsgr.ui.utils import (
    DEFAULT_NAV_CONFIG,
    NavConfig,
    NavigationError,
    click_and_wait_for_page,
    click_and_wait_leave_page,
    confirm_operation,
    wait_for_page,
    wait_leave_page,
)


if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy as np


_log = get_logger('ui')

# ---------------------------------------------------------------------------
# 页面注册中心
# ---------------------------------------------------------------------------

_PAGE_REGISTRY: dict[str, Callable[[np.ndarray], bool]] = {}
_PAGE_ANNOTATIONS: dict[str, Callable[[np.ndarray], list[object]] | None] = {}


def register_page(
    name: str,
    checker: Callable[[np.ndarray], bool],
    *,
    get_annotations: Callable[[np.ndarray], list[object]] | None = None,
) -> None:
    """注册页面识别函数。

    Parameters
    ----------
    name:
        页面名称。
    checker:
        页面识别函数，接收截图返回是否匹配。
    get_annotations:
        可选的标注生成函数，用于 NavError 截图调试。
    """
    # Python 3.13+ 中 StrEnum 的 str()/format() 返回 'ClassName.MEMBER' 而非值，
    # 显式提取 .value 确保 key 始终为纯 str，避免日志和比较中出现意外格式。
    key: str = name.value if hasattr(name, 'value') else name
    if key in _PAGE_REGISTRY:
        _log.warning("[UI] 页面 '{}' 已注册，将覆盖", key)
    _PAGE_REGISTRY[key] = checker
    _PAGE_ANNOTATIONS[key] = get_annotations
    # _log.debug("[UI] 注册页面: {}", key)


def get_current_page(screen: np.ndarray) -> str | None:
    """识别截图对应的页面名称，无匹配返回 ``None``。"""
    failed_checkers: list[str] = []
    for name, checker in _PAGE_REGISTRY.items():
        try:
            if checker(screen):
                _log.debug('[UI] 当前页面: {}', name)
                return name
        except Exception:
            _log.opt(exception=True).warning("[UI] 页面 '{}' 识别器异常", name)
            failed_checkers.append(name)
    if failed_checkers:
        _log.warning(
            '[UI] 无匹配页面，且以下识别器抛异常: {} (共 {} 个注册页面)',
            failed_checkers,
            len(_PAGE_REGISTRY),
        )
    else:
        _log.debug('[UI] 当前页面: 无匹配 (共 {} 个注册页面)', len(_PAGE_REGISTRY))
    return None


def collect_all_page_annotations(screen: np.ndarray) -> list[object]:
    """收集所有已注册页面的标注（NavError fallback）。

    遍历所有注册了 ``get_annotations`` 的页面，合并其标注结果。
    主要用于 ``ops/navigate.py`` 中页面识别完全失败时的 debug 截图。

    Parameters
    ----------
    screen:
        截图 (HxWx3, RGB)。

    Returns
    -------
    list[Annotation]
        合并后的标注列表，可能为空。
    """
    anns: list[object] = []
    for name, getter in _PAGE_ANNOTATIONS.items():
        if getter is None:
            continue
        try:
            page_anns = getter(screen)
            if page_anns:
                anns.extend(page_anns)
        except Exception:
            _log.opt(exception=True).trace("[UI] 页面 '{}' 标注生成失败", name)
    return anns


def get_registered_pages() -> list[str]:
    """返回所有已注册的页面名称列表。"""
    return list(_PAGE_REGISTRY.keys())


# ---------------------------------------------------------------------------
# 兼容性再导出 - 所有从 page 导入的旧路径继续工作
# ---------------------------------------------------------------------------
__all__ = [
    'DEFAULT_NAV_CONFIG',
    'NavConfig',
    'NavigationError',
    'click_and_wait_for_page',
    'click_and_wait_leave_page',
    'confirm_operation',
    'get_current_page',
    'get_registered_pages',
    'register_page',
    'wait_for_page',
    'wait_leave_page',
]
