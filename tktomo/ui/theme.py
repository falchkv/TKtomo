"""The "Midnight & Champagne" look of the tracking UIs.

Design tokens, the Qt stylesheet built from them, the bundled fonts, and
the few things a stylesheet cannot express: the bezel painted around the
projection view, the segmented tab control, section headers with
letter-spaced small caps, the legend swatches, and the colours pyqtgraph
widgets are handed. The spec is the design handoff
``design_handoff_tktomo_track_model_v2`` (README + HTML reference); the
token names below follow its "Design tokens" section.

Everything here is optional dressing: a window that never calls
:func:`install` still works, it just looks like stock Qt. Fonts are
registered with the process-wide font database (SIL OFL, bundled under
``assets/fonts``); the stylesheet is scoped to the window it is
installed on unless :func:`apply` put it on the whole application.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QFontDatabase,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPixmap,
    QRadialGradient,
)
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QStackedWidget,
    QTabBar,
    QVBoxLayout,
    QWidget,
)

ASSETS = Path(__file__).with_name("assets")
FONT_DIR = ASSETS / "fonts"

# ---------------------------------------------------------------- tokens
# surfaces
WINDOW = "#0b1020"
CARD = "#10172a"
WELL = "#0b1122"
INPUT = "#0d1427"
BUTTON = "#161f36"
ACTIVE_ROW = "#18213b"
BAR = "#141c33"
HOVER = "#1c2644"
CANVAS = "#04060d"
# lines
LINE_CARD = "#1e2742"
LINE_SUBTLE = "#222c47"
LINE_INPUT = "#26304d"
LINE_BUTTON = "#2a3453"
LINE_DISABLED = "#1a2340"
# text
TEXT_BRIGHT = "#f1f2f6"
TEXT = "#e6e8ef"
TEXT_STRONG = "#dfe3ec"
TEXT_TABLE = "#cfd4e2"
TEXT_LABEL = "#a3abc2"
TEXT_MUTED = "#8a93ae"
TEXT_HINT = "#6f7896"
TEXT_DISABLED = "#434c68"
# champagne, the primary accent
GOLD = "#c8a978"
GOLD_LIGHT = "#e6cf9f"
GOLD_LIGHTER = "#ecd8ad"
GOLD_LIGHTEST = "#f3e3bf"
GOLD_DARK = "#b8955d"
GOLD_DARKER = "#9c7f4f"
GOLD_DARKEST = "#8a6d40"
GOLD_DATA = "#d9b77a"
# anodised blue: data and rings
BLUE = "#4C82E0"
BLUE_INFO = "#9fb4d8"
# status
RED = "#e0605a"

#: feature colours in id order. The first four are the handoff's; the
#: rest keep ten ids apart on the same dark canvas without reusing the
#: gold and red that mean "one label" and "none" in the plots.
FEATURE_PALETTE = [
    "#3fd6c4", "#4C82E0", "#d9b77a", "#c78ad6",
    "#f0a26a", "#8fd18b", "#9fb4d8", "#f08c9c", "#e6cf9f", "#6ad0f0",
]

UI_FONT = "Jost"
MONO_FONT = "IBM Plex Mono"


def rgb(hex_color: str) -> tuple[int, int, int]:
    """'#rrggbb' -> (r, g, b), the form pyqtgraph pens and brushes take."""
    h = hex_color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def rgba(hex_color: str, alpha: int) -> tuple[int, int, int, int]:
    return (*rgb(hex_color), int(alpha))


def qcolor(hex_color: str, alpha: int = 255) -> QColor:
    return QColor(*rgba(hex_color, alpha))


# ----------------------------------------------------------------- fonts

_fonts_loaded = False


def load_fonts() -> None:
    """Register the bundled Jost and IBM Plex Mono files, once per process.

    Both are SIL Open Font License; the licence texts sit next to the
    files. Registration is harmless when a font is already installed
    system-wide: the family name is the same either way.
    """
    global _fonts_loaded
    if _fonts_loaded:
        return
    _fonts_loaded = True
    for path in sorted(FONT_DIR.glob("*.ttf")):
        QFontDatabase.addApplicationFont(str(path))


def ui_font(px: float, weight: int = 400, spacing: float = 0.0) -> QFont:
    """Jost at a pixel size, with the letter-spacing the spec asks for.

    Letter-spacing has no stylesheet property, which is why the labels
    that need it (section titles, the wordmark) take a QFont instead.
    """
    font = QFont(UI_FONT)
    font.setPixelSize(int(round(px)))
    font.setWeight(QFont.Weight(weight))
    if spacing:
        font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, spacing)
    return font


def mono_font(px: float, weight: int = 400) -> QFont:
    font = QFont(MONO_FONT)
    font.setPixelSize(int(round(px)))
    font.setWeight(QFont.Weight(weight))
    return font


# ------------------------------------------------------------ stylesheet


def _url(name: str) -> str:
    return "url(\"" + (ASSETS / name).as_posix() + "\")"


def stylesheet() -> str:
    """The QSS, with the icon paths resolved for this install."""
    gold_gradient = ("qlineargradient(x1:0, y1:0, x2:0, y2:1, "
                     f"stop:0 {GOLD_LIGHTER}, stop:1 {GOLD_DARK})")
    gold_gradient_hover = ("qlineargradient(x1:0, y1:0, x2:0, y2:1, "
                           f"stop:0 {GOLD_LIGHTEST}, stop:1 {GOLD})")
    tab_gradient = ("qlineargradient(x1:0, y1:0, x2:0, y2:1, "
                    "stop:0 #222d4d, stop:1 #18213a)")
    # the slider thumb: gold ball, a 3 px gap ring in the card colour, a
    # 1 px anodised ring. One radial gradient with hard stops draws all
    # three, because a handle takes one background and one border.
    thumb = ("qradialgradient(cx:0.5, cy:0.5, radius:0.5, fx:0.42, fy:0.36, "
             f"stop:0 #f6e8c8, stop:0.36 {GOLD}, stop:0.66 {GOLD_DARKEST}, "
             f"stop:0.68 {CARD}, stop:0.90 {CARD}, stop:0.92 {BLUE}, "
             f"stop:1 {BLUE})")
    return f"""
QMainWindow {{
    background: qradialgradient(cx:0.3, cy:0, fx:0.3, fy:0, radius:1.1,
        stop:0 #16203c, stop:0.6 {WINDOW}, stop:1 {WINDOW});
}}
QWidget {{
    font-family: "{UI_FONT}";
    font-size: 13px;
    color: {TEXT};
}}
QDialog, QMessageBox, QProgressDialog, QInputDialog, QFileDialog {{
    background: {CARD};
}}
QToolTip {{
    background: {BAR};
    color: {TEXT};
    border: 1px solid {LINE_BUTTON};
    padding: 6px 8px;
    font-size: 12px;
}}

/* title bar: the menu bar wears it, so the File menu and its shortcuts
   stay a real QMenuBar */
QMenuBar {{
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 {BAR}, stop:1 {CARD});
    border-bottom: 1px solid {LINE_CARD};
    padding: 8px 14px;
    min-height: 30px;
    spacing: 6px;
}}
QMenuBar::item {{
    background: transparent;
    color: {TEXT_TABLE};
    font-size: 12.5px;
    padding: 6px 16px;
    border: 1px solid {LINE_INPUT};
    border-radius: 15px;
}}
QMenuBar::item:selected, QMenuBar::item:pressed {{
    border-color: {GOLD};
    background: {BAR};
}}
QMenu {{
    background: {BAR};
    border: 1px solid {LINE_BUTTON};
    border-radius: 16px;
    padding: 8px;
}}
QMenu::item {{
    padding: 8px 12px;
    border-radius: 10px;
    color: {TEXT};
    background: transparent;
}}
QMenu::item:selected {{ background: {HOVER}; }}
QMenu::item:disabled {{ color: {TEXT_DISABLED}; }}
QMenu::separator {{
    height: 1px;
    background: {LINE_SUBTLE};
    margin: 6px 6px;
}}
QMenu::right-arrow {{ image: {_url("chevron-right.svg")}; width: 6px; }}

/* layout chrome */
QSplitter::handle {{ background: transparent; }}
QSplitter::handle:horizontal {{ width: 14px; }}
QSplitter::handle:vertical {{ height: 14px; }}
QScrollArea {{ background: transparent; border: none; }}
QScrollArea > QWidget > QWidget {{ background: transparent; }}
QStackedWidget {{ background: transparent; }}
QScrollBar:vertical {{
    background: transparent; width: 10px; margin: 0;
}}
QScrollBar:horizontal {{
    background: transparent; height: 10px; margin: 0;
}}
QScrollBar::handle {{
    background: #243052; border-radius: 5px; border: 2px solid {CARD};
}}
QScrollBar::handle:vertical {{ min-height: 24px; }}
QScrollBar::handle:horizontal {{ min-width: 24px; }}
QScrollBar::add-line, QScrollBar::sub-line,
QScrollBar::add-page, QScrollBar::sub-page {{
    background: transparent; border: none; height: 0; width: 0;
}}

/* cards and wells */
QFrame#card {{
    background: {CARD};
    border: 1px solid {LINE_CARD};
    border-radius: 22px;
}}
QFrame#segment {{
    background: {WELL};
    border: 1px solid {LINE_CARD};
    border-radius: 18px;
}}
QFrame#chip {{
    background: {INPUT};
    border: 1px solid {LINE_SUBTLE};
    border-radius: 14px;
}}
QFrame#pill {{
    background: {INPUT};
    border: 1px solid {LINE_SUBTLE};
    border-radius: 13px;
}}
QFrame#divider {{
    background: {LINE_CARD};
    border: none;
    min-height: 1px;
    max-height: 1px;
}}
QFrame#vdivider {{
    background: {LINE_INPUT};
    border: none;
    min-width: 1px;
    max-width: 1px;
    min-height: 18px;
    max-height: 18px;
}}
QWidget#controlsPanel {{ background: transparent; }}

/* text */
QLabel {{ background: transparent; }}
QLabel[role="hint"] {{ color: {TEXT_HINT}; font-size: 11px; }}
QLabel[role="muted"] {{ color: {TEXT_MUTED}; font-size: 11px; }}
QLabel[role="label"] {{ color: {TEXT_LABEL}; font-size: 12px; }}
QLabel[role="table"] {{ color: {TEXT_TABLE}; font-size: 12px; }}
QLabel[role="strong"] {{ color: {TEXT_STRONG}; font-size: 12.5px; }}
QLabel[role="gold"] {{ color: {GOLD_LIGHT}; font-size: 12px; }}
QLabel[role="info"] {{ color: {BLUE_INFO}; font-size: 11.5px; }}
QLabel[role="warn"] {{ color: {RED}; font-size: 12px; }}
QLabel[role="title"] {{ color: {GOLD}; }}
QLabel[role="kicker"] {{ color: {TEXT_MUTED}; }}
QLabel[role="wordmark"] {{ color: {TEXT_BRIGHT}; }}
QWidget[mono="true"] {{ font-family: "{MONO_FONT}"; }}

/* buttons: secondary by default, kind="primary" and kind="ghost" */
QPushButton {{
    background: {BUTTON};
    border: 1px solid {LINE_BUTTON};
    border-radius: 15px;
    color: {TEXT_STRONG};
    font-size: 12.5px;
    padding: 6px 14px;
}}
QPushButton:hover {{ border-color: {GOLD}; }}
QPushButton:pressed {{ background: {HOVER}; }}
QPushButton:checked {{
    background: {tab_gradient};
    border-color: {GOLD};
    color: {GOLD_LIGHTER};
}}
QPushButton:disabled {{
    background: transparent;
    border-color: {LINE_DISABLED};
    color: {TEXT_DISABLED};
}}
QPushButton[kind="ghost"] {{
    background: transparent;
    color: {TEXT_LABEL};
    font-size: 12px;
}}
QPushButton[kind="ghost"]:pressed {{ background: {HOVER}; }}
QPushButton[kind="primary"] {{
    background: {gold_gradient};
    border: 1px solid {GOLD_DARK};
    color: {CARD};
    font-weight: 600;
}}
QPushButton[kind="primary"]:hover {{ background: {gold_gradient_hover}; }}
QPushButton[kind="primary"]:pressed {{ background: {GOLD_DARK}; }}
QPushButton[kind="primary"]:disabled {{
    background: {LINE_BUTTON}; border-color: {LINE_BUTTON};
    color: {TEXT_DISABLED};
}}
QPushButton#fitButton {{
    font-size: 14px;
    padding: 10px 16px;
    border-radius: 20px;
}}

/* check boxes, also the pin column of the feature table */
QCheckBox {{ spacing: 7px; color: {TEXT_LABEL}; font-size: 12px; }}
QCheckBox:disabled {{ color: {TEXT_DISABLED}; }}
QCheckBox::indicator, QTableView::indicator {{
    width: 15px; height: 15px;
    border-radius: 4px;
    border: 1px solid {LINE_BUTTON};
    background: {INPUT};
}}
QCheckBox::indicator:hover, QTableView::indicator:hover {{
    border-color: {GOLD};
}}
QCheckBox::indicator:checked, QTableView::indicator:checked {{
    background: {GOLD};
    border-color: {GOLD};
    image: {_url("check.svg")};
}}
QCheckBox::indicator:disabled {{
    border-color: {LINE_DISABLED}; background: transparent;
}}

/* number inputs */
QAbstractSpinBox, QLineEdit {{
    background: {INPUT};
    border: 1px solid {LINE_INPUT};
    border-radius: 8px;
    color: {TEXT};
    font-family: "{MONO_FONT}";
    font-size: 12px;
    padding: 3px 6px;
    selection-background-color: {LINE_BUTTON};
    selection-color: {TEXT_BRIGHT};
}}
QAbstractSpinBox {{ padding-right: 16px; }}
QAbstractSpinBox:focus, QLineEdit:focus {{ border-color: {GOLD}; }}
QAbstractSpinBox:disabled, QLineEdit:disabled {{
    color: {TEXT_DISABLED}; border-color: {LINE_DISABLED};
}}
QAbstractSpinBox::up-button, QAbstractSpinBox::down-button {{
    subcontrol-origin: border;
    width: 14px;
    border: none;
    background: transparent;
}}
QAbstractSpinBox::up-button {{ subcontrol-position: top right; margin-top: 3px; }}
QAbstractSpinBox::down-button {{
    subcontrol-position: bottom right; margin-bottom: 3px;
}}
QAbstractSpinBox::up-arrow {{
    image: {_url("caret-up.svg")}; width: 8px; height: 5px;
}}
QAbstractSpinBox::down-arrow {{
    image: {_url("caret-down.svg")}; width: 8px; height: 5px;
}}
QAbstractSpinBox::up-arrow:disabled, QAbstractSpinBox::up-arrow:off,
QAbstractSpinBox::down-arrow:disabled, QAbstractSpinBox::down-arrow:off {{
    image: none;
}}
QSpinBox#chipSpin {{
    background: {BUTTON};
    border: none;
    border-radius: 11px;
    color: {GOLD_LIGHT};
    padding: 2px 4px;
    min-width: 30px;
}}

/* selects */
QComboBox {{
    background: {INPUT};
    border: 1px solid {LINE_INPUT};
    border-radius: 13px;
    color: {TEXT_TABLE};
    font-size: 12px;
    padding: 4px 24px 4px 12px;
}}
QComboBox:hover, QComboBox:on {{ border-color: {GOLD}; }}
QComboBox:disabled {{ color: {TEXT_DISABLED}; border-color: {LINE_DISABLED}; }}
QComboBox::drop-down {{
    subcontrol-origin: padding;
    subcontrol-position: center right;
    width: 22px;
    border: none;
}}
QComboBox::down-arrow {{
    image: {_url("chevron-down.svg")};
    width: 10px; height: 6px;
    margin-right: 6px;
}}
QComboBox QAbstractItemView {{
    background: {BAR};
    border: 1px solid {LINE_BUTTON};
    color: {TEXT};
    selection-background-color: {HOVER};
    selection-color: {TEXT_BRIGHT};
    outline: none;
    padding: 4px;
}}
QTableView QComboBox {{
    border-radius: 8px;
    padding: 1px 18px 1px 6px;
    font-size: 11px;
}}

/* the view slider */
QSlider {{ min-height: 26px; background: transparent; }}
QSlider::groove:horizontal {{
    height: 3px;
    background: {LINE_BUTTON};
    border-radius: 1px;
    margin: 0 6px;
}}
QSlider::sub-page:horizontal {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 {GOLD_DARKER}, stop:1 {GOLD_LIGHT});
    border-radius: 1px;
}}
QSlider::handle:horizontal {{
    width: 25px; height: 25px;
    margin: -11px -6px;
    border-radius: 12px;
    background: {thumb};
}}

/* the feature table */
QTableView {{
    background: {WELL};
    alternate-background-color: {WELL};
    border: 1px solid {LINE_CARD};
    border-radius: 14px;
    gridline-color: transparent;
    font-family: "{MONO_FONT}";
    font-size: 11px;
    color: {TEXT_TABLE};
    selection-background-color: {ACTIVE_ROW};
    selection-color: {TEXT_BRIGHT};
    outline: none;
}}
QTableView::item {{ padding: 3px 3px; border: none; }}
QTableView::item:selected {{ background: {ACTIVE_ROW}; color: {TEXT_BRIGHT}; }}
QHeaderView {{ background: transparent; border: none; }}
QHeaderView::section {{
    background: transparent;
    color: {TEXT_HINT};
    font-family: "{UI_FONT}";
    font-size: 10px;
    font-weight: 500;
    padding: 6px 3px;
    border: none;
    border-bottom: 1px solid {LINE_CARD};
}}
QTableCornerButton::section {{ background: transparent; border: none; }}

/* text wells: the auto-track report and the fit warnings */
QPlainTextEdit, QTextEdit {{
    background: {WELL};
    border: none;
    border-left: 2px solid {GOLD};
    border-radius: 12px;
    padding: 6px 8px;
    font-family: "{MONO_FONT}";
    font-size: 11px;
    color: {GOLD_LIGHT};
    selection-background-color: {LINE_BUTTON};
}}
QPlainTextEdit#autoReport {{ color: {BLUE_INFO}; }}

/* segmented tabs */
QTabBar {{ background: transparent; border: none; }}
QTabBar::tab {{
    background: transparent;
    color: {TEXT_MUTED};
    padding: 5px 18px;
    border: 1px solid transparent;
    border-radius: 14px;
    margin: 0 1px;
    font-size: 12.5px;
}}
QTabBar::tab:selected {{
    background: {tab_gradient};
    border-color: rgba(200, 169, 120, 115);
    color: {GOLD_LIGHTER};
    font-weight: 500;
}}
QTabBar::tab:hover:!selected {{ color: {TEXT_TABLE}; }}

QProgressBar {{
    background: {WELL}; border: 1px solid {LINE_CARD}; border-radius: 6px;
    text-align: center; color: {TEXT_LABEL};
}}
QProgressBar::chunk {{ background: {GOLD}; border-radius: 5px; }}
QGroupBox {{
    border: 1px solid {LINE_CARD}; border-radius: 12px;
    margin-top: 10px; padding-top: 8px;
}}
QGroupBox::title {{
    subcontrol-origin: margin; left: 12px; padding: 0 4px; color: {GOLD};
}}
"""


def apply(app) -> None:
    """Theme the whole application: fonts, then the stylesheet on `app`."""
    load_fonts()
    app.setStyleSheet(stylesheet())


def install(window: QWidget) -> None:
    """Theme one window (and its popups) without touching the application.

    Used by the window constructors, so a window built by a test or
    embedded in something else still looks right. Skipped when
    :func:`apply` already themed the application, which avoids styling
    everything twice.
    """
    load_fonts()
    from PySide6.QtWidgets import QApplication  # noqa: PLC0415
    instance = QApplication.instance()
    if instance is not None and instance.styleSheet() == stylesheet():
        return
    window.setStyleSheet(stylesheet())


# ---------------------------------------------------------------- helpers


def repolish(widget: QWidget) -> None:
    """Re-run the stylesheet after a dynamic property changed."""
    style = widget.style()
    style.unpolish(widget)
    style.polish(widget)


def set_role(widget: QWidget, role: str) -> QWidget:
    widget.setProperty("role", role)
    repolish(widget)
    return widget


def set_mono(widget: QWidget, mono: bool = True) -> QWidget:
    widget.setProperty("mono", mono)
    repolish(widget)
    return widget


def set_kind(button, kind: str):
    """'primary', 'ghost' or 'secondary' (the default look)."""
    button.setProperty("kind", kind)
    repolish(button)
    return button


def label(text: str = "", *, role: str | None = None, mono: bool = False,
          wrap: bool = False) -> QLabel:
    widget = QLabel(text)
    if role:
        widget.setProperty("role", role)
    if mono:
        widget.setProperty("mono", True)
    if wrap:
        widget.setWordWrap(True)
    return widget


def card(margins: tuple[int, int, int, int] = (18, 16, 18, 16),
         spacing: int = 10) -> tuple[QFrame, QVBoxLayout]:
    """A panel: card surface, hairline ring, the spec's 16 by 18 padding."""
    frame = QFrame()
    frame.setObjectName("card")
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(*margins)
    layout.setSpacing(spacing)
    return frame, layout


def title_label(text: str) -> QLabel:
    """A section title: letter-spaced champagne small caps."""
    head = QLabel(text.upper())
    head.setProperty("role", "title")
    head.setFont(ui_font(11, 500, 2.2))
    return head


def section_header(title: str, hint: str = "") -> QWidget:
    """A section title with a sentence-case hint after it."""
    holder = QWidget()
    row = QHBoxLayout(holder)
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(10)
    row.addWidget(title_label(title))
    if hint:
        row.addWidget(label(hint, role="hint"))
    row.addStretch(1)
    return holder


def kicker(text: str, role: str = "title") -> QLabel:
    """10 px uppercase with 2 px tracking, the VIEW / BIN / ROW labels."""
    widget = QLabel(text.upper())
    widget.setProperty("role", role)
    widget.setFont(ui_font(10, 500, 2.0))
    return widget


def divider() -> QFrame:
    line = QFrame()
    line.setObjectName("divider")
    line.setFrameShape(QFrame.Shape.NoFrame)
    return line


def vdivider() -> QFrame:
    line = QFrame()
    line.setObjectName("vdivider")
    line.setFrameShape(QFrame.Shape.NoFrame)
    return line


def chip(text: str, spinbox) -> QFrame:
    """A pill holding a small label and a buttonless spinbox: the degree
    controls of the model card."""
    from PySide6.QtWidgets import QAbstractSpinBox  # noqa: PLC0415

    frame = QFrame()
    frame.setObjectName("chip")
    row = QHBoxLayout(frame)
    row.setContentsMargins(12, 3, 4, 3)
    row.setSpacing(6)
    row.addWidget(label(text, role="muted"))
    spinbox.setObjectName("chipSpin")
    spinbox.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
    spinbox.setAlignment(Qt.AlignmentFlag.AlignCenter)
    spinbox.setFixedWidth(38)
    row.addWidget(spinbox)
    return frame


class Swatch(QWidget):
    """A legend mark: a coloured dot (optionally with a glow and a ring
    in the same colour) or a short vertical tick."""

    def __init__(self, color: str, *, shape: str = "dot", size: int = 7,
                 glow: bool = False, ring: bool = False, parent=None) -> None:
        super().__init__(parent)
        self._color = qcolor(color)
        self._shape = shape
        self._size = size
        self._glow = glow
        self._ring = ring
        extent = size + (8 if glow else 0) + (8 if ring else 0)
        if shape == "tick":
            self.setFixedSize(2, size)
        else:
            self.setFixedSize(extent, extent)

    def set_color(self, color) -> None:
        """A '#rrggbb' string or an (r, g, b) tuple."""
        self._color = qcolor(color) if isinstance(color, str) else QColor(*color)
        self.update()

    def paintEvent(self, event):  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        if self._shape == "tick":
            painter.fillRect(self.rect(), self._color)
            return
        centre = self.rect().center()
        cx, cy = centre.x() + 0.5, centre.y() + 0.5
        r = self._size / 2.0
        if self._glow:
            glow = QRadialGradient(cx, cy, r + 4)
            faint = QColor(self._color)
            faint.setAlpha(120)
            glow.setColorAt(0.0, faint)
            faint = QColor(self._color)
            faint.setAlpha(0)
            glow.setColorAt(1.0, faint)
            painter.setBrush(glow)
            painter.drawEllipse(QRectF(cx - r - 4, cy - r - 4,
                                       2 * r + 8, 2 * r + 8))
        if self._ring:
            painter.setBrush(self._color)
            painter.drawEllipse(QRectF(cx - r - 4, cy - r - 4,
                                       2 * r + 8, 2 * r + 8))
            painter.setBrush(qcolor(CARD))
            painter.drawEllipse(QRectF(cx - r - 3, cy - r - 3,
                                       2 * r + 6, 2 * r + 6))
        painter.setBrush(self._color)
        painter.drawEllipse(QRectF(cx - r, cy - r, 2 * r, 2 * r))


def _rounded(rect: QRectF, radius: float) -> QPainterPath:
    path = QPainterPath()
    path.addRoundedRect(rect, radius, radius)
    return path


def _ring(outer: QRectF, r_outer: float, inner: QRectF,
          r_inner: float) -> QPainterPath:
    path = _rounded(outer, r_outer)
    path.addRoundedRect(inner, r_inner, r_inner)
    path.setFillRule(Qt.FillRule.OddEvenFill)
    return path


def _knurl_brush() -> QBrush:
    """The bezel texture: 1.5 px ridges every 3.5 px, as a tiled pixmap."""
    tile = QPixmap(7, 4)
    tile.fill(qcolor("#111930"))
    painter = QPainter(tile)
    painter.fillRect(0, 0, 2, 4, qcolor("#1b2544"))
    painter.fillRect(4, 0, 1, 4, qcolor("#1b2544"))
    painter.end()
    return QBrush(tile)


class _BezelOverlay(QWidget):
    """Paints the frame ON TOP of the framed child, see `Bezel`."""

    def __init__(self, bezel: Bezel) -> None:
        super().__init__(bezel)
        self._bezel = bezel
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground)

    def paintEvent(self, event):  # noqa: N802
        self._bezel.paint_frame(QPainter(self), QRectF(self.rect()))


class Bezel(QWidget):
    """A frame painted around one child widget.

    ``style="knurl"``: the projection bezel of the handoff, a knurled
    outer ring with a champagne metal ring and an anodised blue ring
    inside it, over a soft shadow. ``"ring"``: the two metal rings alone
    (the recon slice). ``"well"``: an inset well with a hairline ring,
    for plots.

    The frame is painted by an overlay sibling raised above the child and
    transparent to the mouse, so the child's square corners simply
    disappear under the rounded rings. Masking the child instead would
    give aliased corners, and a pyqtgraph view cannot be clipped any
    other way. Nothing here touches the child's own painting; every
    repaint of the image costs one extra pass over four thin paths.
    """

    #: (shadow margins l, t, r, b), frame thickness, outer radius
    _GEOMETRY = {
        "knurl": ((10, 4, 10, 26), 13, 30),
        "ring": ((6, 2, 6, 14), 4, 22),
        "well": ((0, 0, 0, 0), 1, 14),
    }

    def __init__(self, child: QWidget, style: str = "knurl",
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        if style not in self._GEOMETRY:
            raise ValueError(f"unknown bezel style {style!r}")
        self._style = style
        self._child = child
        margins, thickness, radius = self._GEOMETRY[style]
        self._margins = margins
        self._thickness = thickness
        self._radius = radius
        layout = QVBoxLayout(self)
        # the child sits one pixel under the innermost ring so no seam
        # shows between its edge and the ring's inner curve
        tuck = 1 if style != "well" else 0
        layout.setContentsMargins(
            margins[0] + thickness - tuck, margins[1] + thickness - tuck,
            margins[2] + thickness - tuck, margins[3] + thickness - tuck)
        layout.addWidget(child)
        self._overlay = _BezelOverlay(self)
        self._overlay.raise_()
        self._knurl = _knurl_brush() if style == "knurl" else None

    @property
    def child(self) -> QWidget:
        return self._child

    def resizeEvent(self, event):  # noqa: N802
        super().resizeEvent(event)
        self._overlay.setGeometry(self.rect())
        self._overlay.raise_()

    def frame_rect(self) -> QRectF:
        left, top, right, bottom = self._margins
        rect = QRectF(self.rect())
        return rect.adjusted(left, top, -right, -bottom)

    def paint_frame(self, painter: QPainter, area: QRectF) -> None:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        box = self.frame_rect()
        if self._style == "well":
            hole = box.adjusted(1, 1, -1, -1)
            outside = QPainterPath()
            outside.addRect(area)
            outside.addRoundedRect(hole, 13, 13)
            outside.setFillRule(Qt.FillRule.OddEvenFill)
            painter.fillPath(outside, qcolor(CARD))
            painter.fillPath(_ring(box, 14, hole, 13), qcolor(LINE_CARD))
            return

        if self._style == "knurl":
            metal = box.adjusted(9, 9, -9, -9)
        else:
            metal = box
        hole = metal.adjusted(4, 4, -4, -4)
        # Nothing may land on the image: the shadow below is a stack of
        # translucent layers and even a few of them over the hole would
        # dim the pixels, so everything paints through a clip that
        # excludes the hole.
        outside = QPainterPath()
        outside.addRect(area)
        outside.addRoundedRect(hole, 18, 18)
        outside.setFillRule(Qt.FillRule.OddEvenFill)
        painter.setClipPath(outside)

        # soft drop shadow, 0 24px 50px rgba(0,0,0,.5): a stack of
        # translucent rounded rects, cheap enough to redraw every frame
        for i in range(14, 0, -1):
            alpha = int(46 * (1.0 - i / 15.0) ** 1.6)
            grow = i * 1.3
            rect = box.adjusted(-grow, -grow * 0.4, grow, grow * 1.1)
            rect.translate(0, 6 + i * 0.6)
            painter.fillPath(_rounded(rect, self._radius + grow),
                             QColor(0, 0, 0, alpha))

        if self._style == "knurl":
            knurl = _rounded(box, 30)
            knurl.addRoundedRect(hole, 18, 18)
            knurl.setFillRule(Qt.FillRule.OddEvenFill)
            painter.fillPath(knurl, self._knurl)
            painter.fillPath(_ring(box, 30, box.adjusted(1, 1, -1, -1), 29),
                             QColor(200, 169, 120, 64))
        r_metal, r_blue = 22, 20
        blue = metal.adjusted(2, 2, -2, -2)
        gradient = QLinearGradient(metal.topLeft(), metal.bottomRight())
        gradient.setColorAt(0.0, qcolor(GOLD_LIGHTEST))
        gradient.setColorAt(0.35, qcolor(GOLD_DARKER))
        gradient.setColorAt(0.6, qcolor(GOLD_LIGHT))
        gradient.setColorAt(1.0, qcolor(GOLD_DARKEST))
        painter.fillPath(_ring(metal, r_metal, blue, r_blue), gradient)
        painter.fillPath(_ring(blue, r_blue, hole, r_blue - 2), qcolor(BLUE))


class SegmentedTabs(QWidget):
    """Pill tabs in a track, each page with its own toolbar on the right.

    The toolbar row is one line: the segmented control on the left, the
    current page's tools on the right, so the viewer's own controls
    (colormap, auto levels, high-pass) sit where the handoff puts them
    without leaving the viewer widget that owns them.
    """

    currentChanged = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.bar = QTabBar()
        self.bar.setDrawBase(False)
        self.bar.setExpanding(False)
        self.bar.setUsesScrollButtons(False)
        self.bar.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        segment = QFrame()
        segment.setObjectName("segment")
        seg_layout = QHBoxLayout(segment)
        seg_layout.setContentsMargins(3, 3, 3, 3)
        seg_layout.addWidget(self.bar)
        self.tools = QStackedWidget()
        self.tools.setSizePolicy(self.tools.sizePolicy().horizontalPolicy(),
                                 self.tools.sizePolicy().verticalPolicy())
        self.stack = QStackedWidget()
        top = QHBoxLayout()
        top.setContentsMargins(16, 12, 16, 4)
        top.setSpacing(16)
        top.addWidget(segment)
        top.addStretch(1)
        top.addWidget(self.tools)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addLayout(top)
        layout.addWidget(self.stack, 1)
        self.bar.currentChanged.connect(self._changed)

    def addTab(self, page: QWidget, title: str,  # noqa: N802
               tools: QWidget | None = None) -> int:
        self.stack.addWidget(page)
        self.tools.addWidget(tools if tools is not None else QWidget())
        return self.bar.addTab(title)

    def currentIndex(self) -> int:  # noqa: N802
        return self.bar.currentIndex()

    def setCurrentIndex(self, index: int) -> None:  # noqa: N802
        self.bar.setCurrentIndex(index)

    def count(self) -> int:
        return self.bar.count()

    def _changed(self, index: int) -> None:
        self.stack.setCurrentIndex(index)
        self.tools.setCurrentIndex(index)
        self.currentChanged.emit(index)


# ------------------------------------------------------------- pyqtgraph


def style_plot(plot) -> None:
    """A pyqtgraph PlotWidget on the well surface with muted axes."""
    import pyqtgraph as pg  # noqa: PLC0415

    plot.setBackground(WELL)
    item = plot.getPlotItem()
    for name in ("left", "bottom", "right", "top"):
        axis = item.getAxis(name)
        axis.setPen(pg.mkPen(rgb(LINE_BUTTON)))
        axis.setTextPen(pg.mkPen(rgb(TEXT_HINT)))
        axis.setStyle(tickFont=mono_font(10))
    item.getViewBox().setBorder(None)
    plot.showGrid(x=True, y=True, alpha=0.12)


def style_image_view(image_view) -> None:
    """A pyqtgraph ImageView: canvas-black image area, well-dark histogram."""
    import pyqtgraph as pg  # noqa: PLC0415

    image_view.ui.graphicsView.setBackground(CANVAS)
    histogram = image_view.ui.histogram
    histogram.setBackground(WELL)
    histogram.item.axis.setPen(pg.mkPen(rgb(LINE_BUTTON)))
    histogram.item.axis.setTextPen(pg.mkPen(rgb(TEXT_HINT)))
    histogram.item.axis.setStyle(tickFont=mono_font(9))
    region = histogram.item.region
    region.setBrush(pg.mkBrush(rgba(GOLD, 40)))
    region.setHoverBrush(pg.mkBrush(rgba(GOLD, 70)))
    for line in region.lines:
        line.setPen(pg.mkPen(rgba(GOLD, 200)))
        line.setHoverPen(pg.mkPen(rgb(GOLD_LIGHTEST), width=2))
