"""截图标注工具 — 为 debug / NavError 截图自动绘制 ROI、探测点和匹配结果。

提供:

- :class:`ProbePoint` — 像素探测点（坐标、期望/实际颜色、匹配状态）
- :class:`ROIRect` — 矩形 ROI 框
- :class:`TemplateBox` — 模板匹配结果框
- :class:`TextLabel` — 文本标签
- :func:`draw_annotations` — 统一绘制入口

设计原则
----------
1. **纯函数**: ``draw_annotations`` 不修改输入图像，返回新数组。
2. **零依赖** (除 OpenCV/numpy 外): 不引用 UI 层或设备层。
3. **向后兼容**: 所有参数使用相对坐标 [0.0, 1.0]，与现有视觉层一致。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import cv2
import numpy as np


if TYPE_CHECKING:
    from collections.abc import Sequence


# ═══════════════════════════════════════════════════════════════════════════════
# 标注数据类型
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True, slots=True)
class ProbePoint:
    """单个像素探测点标注。

    Parameters
    ----------
    x, y:
        相对坐标 [0.0, 1.0]。
    expected_color:
        期望的 RGB 颜色，用于绘制外圈。
    actual_color:
        实际截图中的 RGB 颜色，用于填充内圆。
    matched:
        是否匹配通过。决定外圈颜色（通过=绿，失败=红）。
    label:
        可选文本标签（如 ``"距离=42.3"``）。
    radius:
        圆点半径（像素），默认 6。
    """

    x: float
    y: float
    expected_color: tuple[int, int, int] = (255, 255, 255)
    actual_color: tuple[int, int, int] | None = None
    matched: bool = True
    label: str = ''
    radius: int = 6


@dataclass(frozen=True, slots=True)
class ROIRect:
    """矩形 ROI 标注框。

    Parameters
    ----------
    x1, y1, x2, y2:
        相对坐标 [0.0, 1.0]。
    color:
        框线 RGB 颜色，默认蓝色 (0, 128, 255)。
    label:
        可选标签文本。
    thickness:
        框线粗细（像素），默认 2。
    """

    x1: float
    y1: float
    x2: float
    y2: float
    color: tuple[int, int, int] = (0, 128, 255)
    label: str = ''
    thickness: int = 2


@dataclass(frozen=True, slots=True)
class TemplateBox:
    """模板匹配结果框。

    Parameters
    ----------
    x1, y1, x2, y2:
        相对坐标 [0.0, 1.0]。
    confidence:
        匹配置信度 (0.0-1.0)，会显示在标签中。
    color:
        框线 RGB 颜色，默认橙色 (255, 128, 0)。
    label:
        模板名称标签。
    thickness:
        框线粗细（像素），默认 2。
    """

    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float = 1.0
    color: tuple[int, int, int] = (255, 128, 0)
    label: str = ''
    thickness: int = 2


@dataclass(frozen=True, slots=True)
class TextLabel:
    """纯文本标签。

    Parameters
    ----------
    x, y:
        文本左上角相对坐标。
    text:
        文本内容。
    color:
        RGB 颜色，默认白色。
    bg_color:
        背景 RGB 颜色，为 ``None`` 时不绘制背景。
    font_scale:
        字体大小比例，默认 0.5。
    thickness:
        文字粗细，默认 1。
    """

    x: float
    y: float
    text: str
    color: tuple[int, int, int] = (255, 255, 255)
    bg_color: tuple[int, int, int] | None = None
    font_scale: float = 0.5
    thickness: int = 1


Annotation = ProbePoint | ROIRect | TemplateBox | TextLabel


# ═══════════════════════════════════════════════════════════════════════════════
# 绘制引擎
# ═══════════════════════════════════════════════════════════════════════════════


def draw_annotations(
    screen: np.ndarray,
    annotations: Sequence[Annotation],
    *,
    alpha: float = 1.0,
) -> np.ndarray:
    """在截图上绘制一组标注，返回新的 RGB 数组。

    Parameters
    ----------
    screen:
        原始截图 (HxWx3, RGB uint8)。
    annotations:
        标注对象列表。
    alpha:
        背景半透明遮罩强度 (0.0-1.0)。为 0 时不添加遮罩。

    Returns
    -------
    np.ndarray
        绘制后的 RGB 数组（原图副本）。
    """
    vis = screen.copy()
    h, w = vis.shape[:2]

    if alpha > 0:
        overlay = vis.copy()
        cv2.addWeighted(overlay, 1 - alpha, vis, alpha, 0, vis)

    for ann in annotations:
        _draw_one(vis, ann, w, h)

    return vis


def _draw_one(
    canvas: np.ndarray,
    ann: Annotation,
    w: int,
    h: int,
) -> None:
    """单条标注绘制分发。"""
    match ann:
        case ProbePoint():
            _draw_probe_point(canvas, ann, w, h)
        case ROIRect():
            _draw_roi_rect(canvas, ann, w, h)
        case TemplateBox():
            _draw_template_box(canvas, ann, w, h)
        case TextLabel():
            _draw_text_label(canvas, ann, w, h)


def _draw_probe_point(
    canvas: np.ndarray,
    ann: ProbePoint,
    w: int,
    h: int,
) -> None:
    px, py = int(ann.x * w), int(ann.y * h)
    r = ann.radius

    # 外圈颜色：匹配通过=绿色，失败=红色
    outline = (0, 255, 0) if ann.matched else (255, 0, 0)
    # 内圆颜色：实际颜色（若未提供则使用期望颜色）
    fill = ann.actual_color if ann.actual_color is not None else ann.expected_color

    # 外圈（粗）
    cv2.circle(canvas, (px, py), r + 2, outline, 2)
    # 内圆（填充）
    cv2.circle(canvas, (px, py), r, fill, -1)
    # 中心白点（提高可见性）
    cv2.circle(canvas, (px, py), 2, (255, 255, 255), -1)

    if ann.label:
        text_y = py - r - 6
        _put_text_with_bg(canvas, ann.label, (px, text_y), outline)


def _draw_roi_rect(
    canvas: np.ndarray,
    ann: ROIRect,
    w: int,
    h: int,
) -> None:
    p1 = (int(ann.x1 * w), int(ann.y1 * h))
    p2 = (int(ann.x2 * w), int(ann.y2 * h))
    cv2.rectangle(canvas, p1, p2, ann.color, ann.thickness)
    if ann.label:
        _put_text_with_bg(canvas, ann.label, (p1[0], p1[1] - 5), ann.color)


def _draw_template_box(
    canvas: np.ndarray,
    ann: TemplateBox,
    w: int,
    h: int,
) -> None:
    p1 = (int(ann.x1 * w), int(ann.y1 * h))
    p2 = (int(ann.x2 * w), int(ann.y2 * h))
    cv2.rectangle(canvas, p1, p2, ann.color, ann.thickness)

    label = ann.label
    if label:
        label = f'{label} ({ann.confidence:.2f})'
    else:
        label = f'{ann.confidence:.2f}'
    _put_text_with_bg(canvas, label, (p1[0], p1[1] - 5), ann.color)


def _draw_text_label(
    canvas: np.ndarray,
    ann: TextLabel,
    w: int,
    h: int,
) -> None:
    px, py = int(ann.x * w), int(ann.y * h)
    if ann.bg_color is not None:
        _put_text_with_bg(canvas, ann.text, (px, py), ann.color, ann.bg_color, ann.font_scale, ann.thickness)
    else:
        cv2.putText(
            canvas,
            ann.text,
            (px, py),
            cv2.FONT_HERSHEY_SIMPLEX,
            ann.font_scale,
            ann.color,
            ann.thickness,
            cv2.LINE_AA,
        )


def _put_text_with_bg(
    canvas: np.ndarray,
    text: str,
    pos: tuple[int, int],
    color: tuple[int, int, int],
    bg_color: tuple[int, int, int] | None = None,
    font_scale: float = 0.5,
    thickness: int = 1,
) -> None:
    """绘制带半透明背景的文本。"""
    if bg_color is None:
        bg_color = (0, 0, 0)

    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    x, y = pos
    # 确保文本在画布内（不超出顶部）
    y = max(y, th + 4)

    # 背景矩形
    bg_x1, bg_y1 = x, y - th - 4
    bg_x2, bg_y2 = x + tw + 6, y + baseline

    # 半透明背景
    overlay = canvas.copy()
    cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), bg_color, -1)
    cv2.addWeighted(overlay, 0.7, canvas, 0.3, 0, canvas)

    cv2.putText(
        canvas,
        text,
        (x + 3, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 便捷构造（从现有视觉层结果转换）
# ═══════════════════════════════════════════════════════════════════════════════


def from_pixel_detail(detail: object) -> ProbePoint:
    """从 :class:`autowsgr.vision.pixel.PixelDetail` 构造探测点标注。

    Parameters
    ----------
    detail:
        PixelDetail 实例（具有 ``rule``, ``actual``, ``distance``, ``matched`` 属性）。

    Returns
    -------
    ProbePoint
    """
    rule = detail.rule
    actual = detail.actual
    label = f'{rule.color.distance(actual):.1f}' if hasattr(rule.color, 'distance') else ''
    return ProbePoint(
        x=rule.x,
        y=rule.y,
        expected_color=rule.color.as_rgb_tuple(),
        actual_color=actual.as_rgb_tuple(),
        matched=detail.matched,
        label=label,
    )


def from_image_match_detail(detail: object) -> TemplateBox:
    """从 :class:`autowsgr.vision.image_template.ImageMatchDetail` 构造模板框标注。

    Parameters
    ----------
    detail:
        ImageMatchDetail 实例（具有 ``template_name``, ``confidence``,
        ``top_left``, ``bottom_right`` 属性）。

    Returns
    -------
    TemplateBox
    """
    return TemplateBox(
        x1=detail.top_left[0],
        y1=detail.top_left[1],
        x2=detail.bottom_right[0],
        y2=detail.bottom_right[1],
        confidence=detail.confidence,
        label=detail.template_name,
    )
