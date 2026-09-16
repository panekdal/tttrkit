import numpy as np
from qtpy.QtCore import QDir, Qt, Signal
from qtpy.QtGui import QBrush, QColor, QPainter, QPen
from qtpy.QtWidgets import (
    QApplication,
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from tttrkit.ptuio.reader import TTTRReader
from tttrkit.ptuio.reconstructor import ScanConfig, SegmentReconstructor
from tttrkit.ptuio.utils import (
    _read_probe_chunk,
    estimate_bidirectional_prealign,
    estimate_bidirectional_shift,
)


def _nice_ticks(v_min, v_max, target=6):
    """Tick values on a 1/2/2.5/5 x 10^k ladder, clipped to [v_min, v_max]."""
    span = float(v_max) - float(v_min)
    if not np.isfinite(span) or span <= 0:
        return np.array([float(v_min)])

    raw_step = span / max(target, 1)
    magnitude = 10.0 ** np.floor(np.log10(raw_step))
    step = 10.0 * magnitude
    for multiple in (1.0, 2.0, 2.5, 5.0):
        if multiple * magnitude >= raw_step:
            step = multiple * magnitude
            break

    ticks = np.arange(np.ceil(v_min / step) * step, v_max + step * 0.5, step)
    tolerance = step * 1e-6
    return ticks[(ticks >= v_min - tolerance) & (ticks <= v_max + tolerance)]


class _PlotCanvas(QWidget):
    """Shared dark-canvas painting helpers for the alignment plots."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(400)

    def _plot_rect(self):
        return self.rect().adjusted(65, 20, -25, -45)

    @staticmethod
    def _map_x(values, plot_rect, x_min, x_max):
        span = x_max - x_min if x_max != x_min else 1.0
        scaled = (np.asarray(values, dtype=float) - x_min) * plot_rect.width() / span
        return plot_rect.left() + scaled

    @staticmethod
    def _map_y(values, plot_rect, y_min, y_max):
        span = y_max - y_min if y_max != y_min else 1.0
        scaled = (np.asarray(values, dtype=float) - y_min) * plot_rect.height() / span
        return plot_rect.bottom() - scaled

    def _draw_axes(self, painter, plot_rect, x_min, x_max, y_min, y_max, x_label, y_label):
        painter.setPen(QPen(QColor("white"), 1))
        painter.drawLine(plot_rect.bottomLeft(), plot_rect.bottomRight())
        painter.drawLine(plot_rect.bottomLeft(), plot_rect.topLeft())

        for value in _nice_ticks(x_min, x_max):
            x = int(self._map_x(value, plot_rect, x_min, x_max))
            painter.drawLine(x, plot_rect.bottom(), x, plot_rect.bottom() + 5)
            painter.drawText(x - 20, plot_rect.bottom() + 20, f"{value:.4g}")

        for value in _nice_ticks(y_min, y_max):
            y = int(self._map_y(value, plot_rect, y_min, y_max))
            painter.drawLine(plot_rect.left() - 5, y, plot_rect.left(), y)
            painter.drawText(8, y + 4, f"{value:.4g}")

        painter.drawText(plot_rect.center().x() - 25, self.height() - 8, x_label)
        painter.save()
        painter.translate(15, plot_rect.center().y())
        painter.rotate(-90)
        painter.drawText(0, -5, y_label)
        painter.restore()


class AlignmentPlotWidget(_PlotCanvas):
    """Forward/backward scan profiles with an interactive cursor and selection."""

    selector_changed = Signal(float)
    selection_changed = Signal(float, float)
    selection_cleared = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._x = np.array([], dtype=float)
        self._traces = []
        self._x_label = "time [ms]"
        self.x_unit = "ms"
        self._stop_marker_x = np.array([], dtype=float)
        self._has_line_start_marker = False
        self._show_line_start = True
        self._show_line_stop = True
        self._selector = None
        self._selection = None
        self._dragging = False
        self._drag_origin_px = None
        self._drag_start = None
        self.setMouseTracking(True)

    def clear(self):
        self._x = np.array([], dtype=float)
        self._traces = []
        self._stop_marker_x = np.array([], dtype=float)
        self._has_line_start_marker = False
        self._selector = None
        self._selection = None
        self._dragging = False
        self.update()

    def set_result(self, result):
        """Show the coarse pre-alignment profiles against a time axis in ms."""
        try:
            time_axis = np.asarray(result["time_axis"].values, dtype=float).ravel()
            forward = np.asarray(result["forward"].values, dtype=float).ravel()
            backward = np.asarray(result["backward"].values, dtype=float).ravel()
            backward_aligned = np.asarray(result["backward_aligned"].values, dtype=float).ravel()
            durations_s = np.asarray(result["durations_s"].values, dtype=float).ravel()
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "Expected an xarray Dataset with time_axis, forward, backward, "
                "backward_aligned and durations_s."
            ) from exc

        if not len(time_axis) or len(time_axis) != len(forward) or len(time_axis) != len(backward):
            raise ValueError("time_axis, forward, and backward must be non-empty and equally sized.")

        self._x = time_axis * 1e3
        self._x_label = "time [ms]"
        self.x_unit = "ms"
        self._traces = [
            (forward, QColor("white"), Qt.PenStyle.SolidLine, "forward"),
            (backward, QColor("red"), Qt.PenStyle.DashLine, "backward"),
            (backward_aligned, QColor("lime"), Qt.PenStyle.SolidLine, "backward aligned"),
        ]
        self._stop_marker_x = durations_s * 1e3
        self._has_line_start_marker = True
        self._selector = None
        self._selection = None
        self.update()

    def set_segment(self, forward, backward):
        """Show a reconstruction at the current delays, against pixel index."""
        forward = np.asarray(forward, dtype=float).ravel()
        backward = np.asarray(backward, dtype=float).ravel()
        if not len(forward) or len(forward) != len(backward):
            raise ValueError("forward and backward must be non-empty and equally sized.")

        self._x = np.arange(len(forward), dtype=float)
        self._x_label = "pixel"
        self.x_unit = "px"
        self._traces = [
            (forward, QColor("white"), Qt.PenStyle.SolidLine, "forward"),
            (backward, QColor("lime"), Qt.PenStyle.DashLine, "backward"),
        ]
        self._stop_marker_x = np.array([], dtype=float)
        self._has_line_start_marker = False
        self._selector = None
        self._selection = None
        self.update()

    def set_marker_visibility(self, line_start, line_stop):
        self._show_line_start = bool(line_start)
        self._show_line_stop = bool(line_stop)
        self.update()

    def _value_from_x(self, x_pos):
        if not len(self._x):
            return None

        plot_rect = self._plot_rect()
        x_min = float(self._x[0])
        x_max = float(self._x[-1])
        if x_max <= x_min:
            return x_min
        if x_pos <= plot_rect.left():
            return x_min
        if x_pos >= plot_rect.right():
            return x_max
        return x_min + (x_pos - plot_rect.left()) / plot_rect.width() * (x_max - x_min)

    def mouseMoveEvent(self, event):
        value = self._value_from_x(event.position().toPoint().x())
        if value is None:
            return

        self._selector = value
        self.selector_changed.emit(float(value))
        if self._dragging and self._drag_start is not None:
            self._selection = (min(self._drag_start, value), max(self._drag_start, value))
        self.update()

    def mousePressEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton:
            return

        pos = event.position().toPoint()
        start = self._value_from_x(pos.x())
        if start is None:
            return

        self._dragging = True
        self._drag_origin_px = pos.x()
        self._drag_start = start
        self._selector = start
        self.selector_changed.emit(float(start))
        self.update()

    def mouseReleaseEvent(self, event):
        if not self._dragging:
            return

        pos = event.position().toPoint()
        end = self._value_from_x(pos.x())
        start = self._drag_start
        # A press that barely moved is a click, which resets the selection.
        dragged = abs(pos.x() - self._drag_origin_px) >= 3

        if dragged and start is not None and end is not None:
            self._selection = (min(start, end), max(start, end))
            self.selection_changed.emit(float(self._selection[0]), float(self._selection[1]))
        else:
            self._selection = None
            self.selection_cleared.emit()

        self._dragging = False
        self._drag_origin_px = None
        self._drag_start = None
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("black"))

        plot_rect = self._plot_rect()
        painter.setPen(QPen(QColor("white"), 1))
        painter.drawRect(plot_rect)

        if not len(self._x) or plot_rect.width() <= 0 or plot_rect.height() <= 0:
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No alignment result")
            return

        traces = [trace for trace in self._traces if len(trace[0]) == len(self._x)]

        finite_x = []
        finite_y = []
        for values, _, _, _ in traces:
            mask = np.isfinite(self._x) & np.isfinite(values)
            finite_x.append(self._x[mask])
            finite_y.append(values[mask])

        finite_x = np.concatenate(finite_x) if finite_x else np.array([])
        finite_y = np.concatenate(finite_y) if finite_y else np.array([])
        if not len(finite_x) or not len(finite_y):
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No finite alignment data")
            return

        x_min, x_max = float(np.min(finite_x)), float(np.max(finite_x))
        y_min, y_max = float(np.min(finite_y)), float(np.max(finite_y))
        if x_min == x_max:
            x_min -= 0.5
            x_max += 0.5
        if y_min == y_max:
            y_min -= 1.0
            y_max += 1.0

        if self._selection is not None:
            left = int(self._map_x(self._selection[0], plot_rect, x_min, x_max))
            right = int(self._map_x(self._selection[1], plot_rect, x_min, x_max))
            painter.fillRect(
                left,
                plot_rect.top(),
                max(right - left, 1),
                plot_rect.height(),
                QColor(255, 255, 255, 40),
            )

        if self._show_line_stop and len(self._stop_marker_x):
            painter.setPen(QPen(QColor(0, 112, 240, 64), 1))
            for marker in self._stop_marker_x:
                if not np.isfinite(marker) or marker < x_min or marker > x_max:
                    continue
                x = int(self._map_x(marker, plot_rect, x_min, x_max))
                painter.drawLine(x, plot_rect.top(), x, plot_rect.bottom())

        if self._has_line_start_marker and self._show_line_start and x_min <= 0.0 <= x_max:
            painter.setPen(QPen(QColor(0, 200, 255, 160), 1))
            x = int(self._map_x(0.0, plot_rect, x_min, x_max))
            painter.drawLine(x, plot_rect.top(), x, plot_rect.bottom())

        self._draw_axes(
            painter, plot_rect, x_min, x_max, y_min, y_max, self._x_label, "photon count"
        )

        x_pixels = self._map_x(self._x, plot_rect, x_min, x_max)
        for values, color, style, _ in traces:
            painter.setPen(QPen(color, 1, style))
            y_pixels = self._map_y(values, plot_rect, y_min, y_max)
            previous = None
            for x_value, value, x, y in zip(self._x, values, x_pixels, y_pixels):
                if not np.isfinite(x_value) or not np.isfinite(value):
                    previous = None
                    continue
                point = (int(x), int(y))
                if previous is not None:
                    painter.drawLine(*previous, *point)
                previous = point

        for index, (_, color, _, label) in enumerate(traces):
            painter.setPen(QPen(color))
            painter.drawText(plot_rect.left() + 10, plot_rect.top() + 18 + index * 18, label)

        if self._selector is not None and x_min <= self._selector <= x_max:
            painter.setPen(QPen(QColor("white"), 1, Qt.PenStyle.DashLine))
            x = int(self._map_x(self._selector, plot_rect, x_min, x_max))
            painter.drawLine(x, plot_rect.top(), x, plot_rect.bottom())


class CorrelationPlotWidget(_PlotCanvas):
    """Score-versus-shift curve produced by the fine optimization."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._test_shift_us = np.array([], dtype=float)
        self._scores = np.array([], dtype=float)
        self._fit_shift_us = np.array([], dtype=float)
        self._fit = np.array([], dtype=float)
        self._best_shift_us = None

    def clear(self):
        self._test_shift_us = np.array([], dtype=float)
        self._scores = np.array([], dtype=float)
        self._fit_shift_us = np.array([], dtype=float)
        self._fit = np.array([], dtype=float)
        self._best_shift_us = None
        self.update()

    def set_result(self, result):
        try:
            test_shift = np.asarray(result["test_shift"].values, dtype=float).ravel()
            scores = np.asarray(result["scores"].values, dtype=float).ravel()
            fit_shift = np.asarray(result["fit_shift"].values, dtype=float).ravel()
            fit = np.asarray(result["fit"].values, dtype=float).ravel()
            best_shift = float(result["best_shift"].item())
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "Expected an xarray Dataset with test_shift, scores, fit_shift, fit "
                "and best_shift."
            ) from exc

        self._test_shift_us = test_shift * 1e6
        self._scores = scores
        self._fit_shift_us = fit_shift * 1e6
        self._fit = fit
        self._best_shift_us = best_shift * 1e6
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("black"))

        plot_rect = self._plot_rect()
        painter.setPen(QPen(QColor("white"), 1))
        painter.drawRect(plot_rect)

        if not len(self._test_shift_us) or plot_rect.width() <= 0 or plot_rect.height() <= 0:
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No optimization result")
            return

        score_length = min(len(self._test_shift_us), len(self._scores))
        score_x = self._test_shift_us[:score_length]
        score_y = self._scores[:score_length]
        score_mask = np.isfinite(score_x) & np.isfinite(score_y)

        # A failed Gaussian fit returns all-NaN values sized to `steps`, not 100.
        fit_length = min(len(self._fit_shift_us), len(self._fit))
        fit_x = self._fit_shift_us[:fit_length]
        fit_y = self._fit[:fit_length]
        fit_mask = np.isfinite(fit_x) & np.isfinite(fit_y)

        if not score_mask.any():
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No finite score data")
            return

        x_values = np.concatenate((score_x[score_mask], fit_x[fit_mask]))
        y_values = np.concatenate((score_y[score_mask], fit_y[fit_mask]))
        x_min, x_max = float(np.min(x_values)), float(np.max(x_values))
        y_min, y_max = float(np.min(y_values)), float(np.max(y_values))
        if x_min == x_max:
            x_min -= 0.5
            x_max += 0.5
        if y_min == y_max:
            y_min -= 1.0
            y_max += 1.0

        self._draw_axes(painter, plot_rect, x_min, x_max, y_min, y_max, "shift [μs]", "score")

        if fit_mask.any():
            painter.setPen(QPen(QColor("lime"), 2))
            fit_px = self._map_x(fit_x, plot_rect, x_min, x_max)
            fit_py = self._map_y(fit_y, plot_rect, y_min, y_max)
            previous = None
            for keep, x, y in zip(fit_mask, fit_px, fit_py):
                if not keep:
                    previous = None
                    continue
                point = (int(x), int(y))
                if previous is not None:
                    painter.drawLine(*previous, *point)
                previous = point

        painter.setPen(QPen(QColor("white"), 1))
        painter.setBrush(QBrush(QColor("white")))
        score_px = self._map_x(score_x, plot_rect, x_min, x_max)
        score_py = self._map_y(score_y, plot_rect, y_min, y_max)
        for keep, x, y in zip(score_mask, score_px, score_py):
            if not keep:
                continue
            painter.drawEllipse(int(x) - 3, int(y) - 3, 6, 6)
        painter.setBrush(Qt.BrushStyle.NoBrush)

        if self._best_shift_us is not None and x_min <= self._best_shift_us <= x_max:
            painter.setPen(QPen(QColor("white"), 1, Qt.PenStyle.DashLine))
            x = int(self._map_x(self._best_shift_us, plot_rect, x_min, x_max))
            painter.drawLine(x, plot_rect.top(), x, plot_rect.bottom())
            painter.setPen(QPen(QColor("white")))
            painter.drawText(x + 5, plot_rect.top() + 14, f"{self._best_shift_us:.3f} μs")


class MarkerAnalysisWidget(QWidget):
    """Estimate and display the forward/backward marker pre-alignment."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.resize(960, 860)
        self.setMinimumSize(640, 480)
        self._syncing = False

        self.file_path_edit = QLineEdit()
        self.file_path_edit.setPlaceholderText("Select a .ptu file")
        self.file_path_edit.setClearButtonEnabled(True)

        self.browse_button = QPushButton("Browse…")
        self.browse_button.setFixedWidth(90)
        self.browse_button.clicked.connect(self.select_ptu_file)

        file_layout = QHBoxLayout()
        file_layout.setContentsMargins(0, 0, 0, 0)
        file_layout.addWidget(self.file_path_edit)
        file_layout.addWidget(self.browse_button)

        self.start_marker_dly_input = self._create_delay_input()
        self.stop_marker_dly_input = self._create_delay_input()

        self.shift_input = QDoubleSpinBox()
        self.shift_input.setRange(-100_000.0, 100_000.0)
        self.shift_input.setDecimals(3)
        self.shift_input.setSingleStep(0.1)
        self.shift_input.setFixedWidth(150)

        self.fix_shift_checkbox = QCheckBox("Fix shift")

        shift_layout = QHBoxLayout()
        shift_layout.setContentsMargins(0, 0, 0, 0)
        shift_layout.addWidget(self.shift_input)
        shift_layout.addWidget(self.fix_shift_checkbox)
        shift_layout.addStretch()

        self.pixels_input = QSpinBox()
        self.pixels_input.setRange(16, 4_096)
        self.pixels_input.setSingleStep(16)
        self.pixels_input.setFixedWidth(150)
        self.pixels_input.setValue(512)

        self.chunk_size_input = QSpinBox()
        self.chunk_size_input.setRange(100, 20_000)
        self.chunk_size_input.setSingleStep(100)
        self.chunk_size_input.setFixedWidth(150)
        self.chunk_size_input.setValue(500)

        self.skip_n_chunks_input = QSpinBox()
        self.skip_n_chunks_input.setRange(0, 100)
        self.skip_n_chunks_input.setSingleStep(1)
        self.skip_n_chunks_input.setFixedWidth(150)
        self.skip_n_chunks_input.setValue(5)

        self.max_shift_input = QDoubleSpinBox()
        self.max_shift_input.setRange(0.1, 1_000.0)
        self.max_shift_input.setDecimals(3)
        self.max_shift_input.setSingleStep(0.5)
        self.max_shift_input.setFixedWidth(150)
        self.max_shift_input.setValue(3.0)

        self.steps_input = QSpinBox()
        self.steps_input.setRange(3, 101)
        self.steps_input.setSingleStep(2)
        self.steps_input.setFixedWidth(150)
        self.steps_input.setValue(11)

        self.label = QLabel("Ready")
        self.estimate_button = QPushButton("Estimate align")
        self.estimate_button.setFixedSize(120, 30)
        self.estimate_button.clicked.connect(self.estimate_align)

        self.show_align_button = QPushButton("Show align")
        self.show_align_button.setFixedSize(120, 30)
        self.show_align_button.clicked.connect(self.show_align)

        self.optimize_button = QPushButton("Optimize")
        self.optimize_button.setFixedSize(120, 30)
        self.optimize_button.clicked.connect(self.optimize)

        self.marker_checkbox_layout = QHBoxLayout()
        self.marker_checkbox_layout.setContentsMargins(0, 0, 0, 0)
        self.marker_checkbox_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.marker_checkboxes = {}
        for name, label in [
            ("line_start", "Line start markers"),
            ("line_stop", "Stop markers"),
        ]:
            checkbox = QCheckBox(label)
            checkbox.setChecked(True)
            checkbox.toggled.connect(self._update_marker_visibility)
            self.marker_checkboxes[name] = checkbox
            self.marker_checkbox_layout.addWidget(checkbox)

        self.plot = AlignmentPlotWidget()
        self.correlation_plot = CorrelationPlotWidget()
        self.plot_tabs = QTabWidget()
        self.plot_tabs.addTab(self.plot, "Alignment")
        self.plot_tabs.addTab(self.correlation_plot, "Correlation")

        self.selector_position_input = self._create_readout(-1_000_000_000.0)
        self.selector_position_label = QLabel("Selector position [ms]:")
        self.selection_width_value = QLabel("--")
        self.selection_width_value.setMinimumWidth(110)

        layout = QVBoxLayout(self)
        form_layout = QFormLayout()
        form_layout.addRow("PTU file:", file_layout)
        form_layout.addRow("Line start marker delay [μs]:", self.start_marker_dly_input)
        form_layout.addRow("Line stop marker delay [μs]:", self.stop_marker_dly_input)
        form_layout.addRow("Shift [μs]:", shift_layout)
        form_layout.addRow("Pixels per line:", self.pixels_input)
        form_layout.addRow("Chunk size [x1000]:", self.chunk_size_input)
        form_layout.addRow("Chunks to skip:", self.skip_n_chunks_input)
        form_layout.addRow("Max shift [μs]:", self.max_shift_input)
        form_layout.addRow("Optimization steps:", self.steps_input)
        layout.addLayout(form_layout)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.estimate_button)
        button_layout.addWidget(self.show_align_button)
        button_layout.addWidget(self.optimize_button)
        button_layout.addStretch()
        layout.addLayout(button_layout)
        layout.addLayout(self.marker_checkbox_layout)
        layout.addWidget(self.plot_tabs)

        selector_layout = QHBoxLayout()
        selector_layout.addWidget(self.selector_position_label)
        selector_layout.addWidget(self.selector_position_input)
        selector_layout.addWidget(QLabel("Selection Δ:"))
        selector_layout.addWidget(self.selection_width_value)
        selector_layout.addStretch()
        layout.addLayout(selector_layout)
        layout.addWidget(self.label)

        self.plot.selector_changed.connect(self._update_selector_position)
        self.plot.selection_changed.connect(self._update_selection_width)
        self.plot.selection_cleared.connect(self._clear_selection_width)

        self.start_marker_dly_input.valueChanged.connect(self._resync_stop_delay)
        self.stop_marker_dly_input.valueChanged.connect(self._resync_stop_delay)
        self.shift_input.valueChanged.connect(self._resync_stop_delay)
        self.fix_shift_checkbox.toggled.connect(self._on_fix_shift_toggled)
        self._on_fix_shift_toggled(self.fix_shift_checkbox.isChecked())

    @staticmethod
    def _create_delay_input():
        numeric_input = QDoubleSpinBox()
        numeric_input.setRange(-500_000.0, 500_000.0)
        numeric_input.setDecimals(3)
        numeric_input.setSingleStep(1.0)
        numeric_input.setFixedWidth(150)
        return numeric_input

    @staticmethod
    def _create_readout(minimum):
        numeric_input = QDoubleSpinBox()
        numeric_input.setRange(minimum, 1_000_000_000.0)
        numeric_input.setDecimals(4)
        numeric_input.setSingleStep(0.01)
        numeric_input.setReadOnly(True)
        numeric_input.setFixedWidth(150)
        numeric_input.setSpecialValueText("--")
        return numeric_input

    def _sync_stop_from_shift(self):
        self._syncing = True
        self.stop_marker_dly_input.setValue(
            self.shift_input.value() - self.start_marker_dly_input.value()
        )
        self._syncing = False

    def _on_fix_shift_toggled(self, checked):
        self.stop_marker_dly_input.setReadOnly(checked)
        self.shift_input.setReadOnly(not checked)
        if checked:
            self._sync_stop_from_shift()

    def _resync_stop_delay(self):
        if self._syncing or not self.fix_shift_checkbox.isChecked():
            return
        self._sync_stop_from_shift()

    def _update_selector_position(self, value):
        self.selector_position_input.setValue(float(value))

    def _update_selection_width(self, start, stop):
        width = abs(float(stop) - float(start))
        self.selection_width_value.setText(f"{width:.4g} {self.plot.x_unit}")

    def _clear_selection_width(self):
        self.selection_width_value.setText("--")

    def _reset_readouts(self):
        self.selector_position_input.setValue(self.selector_position_input.minimum())
        self.selection_width_value.setText("--")

    def _update_readout_units(self):
        self.selector_position_label.setText(f"Selector position [{self.plot.x_unit}]:")

    def _update_marker_visibility(self):
        self.plot.set_marker_visibility(
            line_start=self.marker_checkboxes["line_start"].isChecked(),
            line_stop=self.marker_checkboxes["line_stop"].isChecked(),
        )

    def select_ptu_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select PTU file",
            self.file_path_edit.text() or QDir.homePath(),
            "PTU files (*.ptu);;All files (*)",
        )
        if file_path:
            self.file_path_edit.setText(file_path)

    def _build_config(self):
        return ScanConfig(
            bidirectional=True,
            pixels=self.pixels_input.value(),
            line_start_marker_delay=self.start_marker_dly_input.value() * 1e-6,
            line_stop_marker_delay=self.stop_marker_dly_input.value() * 1e-6,
        )

    def _begin_run(self, message):
        for button in (self.estimate_button, self.show_align_button, self.optimize_button):
            button.setEnabled(False)
        self.label.setText(message)
        QApplication.processEvents()

    def _end_run(self):
        for button in (self.estimate_button, self.show_align_button, self.optimize_button):
            button.setEnabled(True)

    def _open_reader(self, file_path):
        reader = TTTRReader(file_path)
        tags = reader.header.tags
        return (
            reader,
            float(tags.get("TTResult_SyncRate", 40e6)),
            int(tags.get("TTResultFormat_WrapAround", 1024)),
        )

    def estimate_align(self):
        file_path = self.file_path_edit.text().strip()
        if not file_path:
            self.label.setText("Select a .ptu file before estimating alignment.")
            return

        self._begin_run("Estimating alignment…")
        self.plot.clear()
        self.correlation_plot.clear()
        self._reset_readouts()

        try:
            reader, sync_rate, wrap = self._open_reader(file_path)
            result = estimate_bidirectional_prealign(
                reader=reader,
                cfg=self._build_config(),
                laser_sync_rate=sync_rate,
                wrap=wrap,
                chunk_length=self.chunk_size_input.value() * 1_000,
                skip_chunks=self.skip_n_chunks_input.value(),
                verbose=False,
            )
            self.plot.set_result(result)
            self._update_marker_visibility()
            self._update_readout_units()

            pixel_shift = int(result["pixel_shift"].item())
            time_shift = float(result["time_shift"].item())
            applied_shift_us = -time_shift * 1e6
            self.shift_input.setValue(applied_shift_us)
            self.fix_shift_checkbox.setChecked(True)
            self._sync_stop_from_shift()
            self.plot_tabs.setCurrentWidget(self.plot)
            self.label.setText(
                f"Alignment estimated: {pixel_shift} pixels; "
                f"shift set to {applied_shift_us:.3f} μs."
            )
        except Exception as exc:
            self.label.setText(f"Alignment failed: {exc}")
        finally:
            self._end_run()

    def show_align(self):
        file_path = self.file_path_edit.text().strip()
        if not file_path:
            self.label.setText("Select a .ptu file before showing the alignment.")
            return

        self._begin_run("Reconstructing at the current delays…")
        self.plot.clear()
        self.correlation_plot.clear()
        self._reset_readouts()

        try:
            reader, sync_rate, wrap = self._open_reader(file_path)
            config = self._build_config()
            corrected_chunk, parity = _read_probe_chunk(
                reader,
                config,
                wrap,
                self.chunk_size_input.value() * 1_000,
                self.skip_n_chunks_input.value(),
                False,
            )
            segments = SegmentReconstructor(config, laser_sync_rate=sync_rate)
            photon_count = segments.reconstruct(corrected_chunk).photon_count.values[parity:]
            if len(photon_count) < 2:
                raise ValueError(
                    "Not enough complete line pairs in the probe chunk. "
                    "Try increasing the chunk size."
                )

            self.plot.set_segment(
                photon_count[0::2].sum(axis=0).astype(float),
                photon_count[1::2].sum(axis=0).astype(float),
            )
            self._update_marker_visibility()
            self._update_readout_units()
            self.plot_tabs.setCurrentWidget(self.plot)
            self.label.setText(
                f"Reconstructed {len(photon_count)} lines at start "
                f"{self.start_marker_dly_input.value():.3f} μs / stop "
                f"{self.stop_marker_dly_input.value():.3f} μs."
            )
        except Exception as exc:
            self.label.setText(f"Show align failed: {exc}")
        finally:
            self._end_run()

    def optimize(self):
        file_path = self.file_path_edit.text().strip()
        if not file_path:
            self.label.setText("Select a .ptu file before optimizing the shift.")
            return

        self._begin_run("Optimizing shift…")
        self.correlation_plot.clear()

        try:
            reader, sync_rate, wrap = self._open_reader(file_path)
            result = estimate_bidirectional_shift(
                reader=reader,
                config=self._build_config(),
                laser_sync_rate=sync_rate,
                wrap=wrap,
                max_shift=self.max_shift_input.value() * 1e-6,
                steps=self.steps_input.value(),
                chunk_length=self.chunk_size_input.value() * 1_000,
                skip_chunks=self.skip_n_chunks_input.value(),
                verbose=False,
            )
            self.correlation_plot.set_result(result)

            best_shift_us = float(result["best_shift"].item()) * 1e6
            if self.fix_shift_checkbox.isChecked():
                self.shift_input.setValue(self.shift_input.value() + best_shift_us)
            else:
                self.stop_marker_dly_input.setValue(
                    self.stop_marker_dly_input.value() + best_shift_us
                )
            self.plot_tabs.setCurrentWidget(self.correlation_plot)
            self.label.setText(
                f"Optimization found {best_shift_us:.3f} μs; stop delay is now "
                f"{self.stop_marker_dly_input.value():.3f} μs."
            )
        except Exception as exc:
            self.label.setText(f"Optimization failed: {exc}")
        finally:
            self._end_run()
