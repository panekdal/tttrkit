import numpy as np
from qtpy.QtCore import QDir, Qt
from qtpy.QtGui import QColor, QPainter, QPen
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
    QVBoxLayout,
    QWidget,
)

from tttrkit.ptuio.reader import TTTRReader
from tttrkit.ptuio.reconstructor import ScanConfig
from tttrkit.ptuio.utils import estimate_bidirectional_prealign


class AlignmentPlotWidget(QWidget):
    """Draw forward and backward pre-alignment profiles against time."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._time_axis = np.array([], dtype=float)
        self._forward = np.array([], dtype=float)
        self._backward = np.array([], dtype=float)
        self.setMinimumHeight(400)

    def clear(self):
        self._time_axis = np.array([], dtype=float)
        self._forward = np.array([], dtype=float)
        self._backward = np.array([], dtype=float)
        self.update()

    def set_result(self, result):
        try:
            time_axis = np.asarray(result["time_axis"].values, dtype=float).ravel()
            forward = np.asarray(result["forward"].values, dtype=float).ravel()
            backward = np.asarray(result["backward"].values, dtype=float).ravel()
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "Expected an xarray Dataset with time_axis, forward, and backward."
            ) from exc

        if not len(time_axis) or len(time_axis) != len(forward) or len(time_axis) != len(backward):
            raise ValueError("time_axis, forward, and backward must be non-empty and equally sized.")

        self._time_axis = time_axis
        self._forward = forward
        self._backward = backward
        self.update()

    def _plot_rect(self):
        return self.rect().adjusted(65, 20, -25, -45)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("black"))

        plot_rect = self._plot_rect()
        painter.setPen(QPen(QColor("white"), 1))
        painter.drawRect(plot_rect)

        if not len(self._time_axis) or plot_rect.width() <= 0 or plot_rect.height() <= 0:
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No alignment result")
            return

        finite_forward = np.isfinite(self._time_axis) & np.isfinite(self._forward)
        finite_backward = np.isfinite(self._time_axis) & np.isfinite(self._backward)
        finite_values = np.concatenate(
            (self._forward[finite_forward], self._backward[finite_backward])
        )
        finite_times = self._time_axis[finite_forward | finite_backward]
        if not len(finite_values) or not len(finite_times):
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No finite alignment data")
            return

        x_min, x_max = float(np.min(finite_times)), float(np.max(finite_times))
        y_min, y_max = float(np.min(finite_values)), float(np.max(finite_values))
        if x_min == x_max:
            x_min -= 0.5
            x_max += 0.5
        if y_min == y_max:
            y_min -= 1.0
            y_max += 1.0

        x_range = x_max - x_min
        y_range = y_max - y_min

        painter.setPen(QPen(QColor("white"), 1))
        painter.drawLine(plot_rect.bottomLeft(), plot_rect.bottomRight())
        painter.drawLine(plot_rect.bottomLeft(), plot_rect.topLeft())

        for value in np.linspace(x_min, x_max, 5):
            x = plot_rect.left() + (value - x_min) * plot_rect.width() / x_range
            painter.drawLine(int(x), plot_rect.bottom(), int(x), plot_rect.bottom() + 5)
            painter.drawText(int(x) - 20, plot_rect.bottom() + 20, f"{value:.3g}")

        for value in np.linspace(y_min, y_max, 5):
            y = plot_rect.bottom() - (value - y_min) * plot_rect.height() / y_range
            painter.drawLine(plot_rect.left() - 5, int(y), plot_rect.left(), int(y))
            painter.drawText(10, int(y) + 4, f"{value:.3g}")

        for values, color in [
            (self._forward, QColor("#00B0F0")),
            (self._backward, QColor("#FF8C00")),
        ]:
            painter.setPen(QPen(color, 2))
            previous = None
            for time, value in zip(self._time_axis, values):
                if not np.isfinite(time) or not np.isfinite(value):
                    previous = None
                    continue
                x = plot_rect.left() + (time - x_min) * plot_rect.width() / x_range
                y = plot_rect.bottom() - (value - y_min) * plot_rect.height() / y_range
                point = (int(x), int(y))
                if previous is not None:
                    painter.drawLine(*previous, *point)
                previous = point

        painter.setPen(QPen(QColor("#00B0F0")))
        painter.drawText(plot_rect.left() + 10, plot_rect.top() + 18, "Forward")
        painter.setPen(QPen(QColor("#FF8C00")))
        painter.drawText(plot_rect.left() + 10, plot_rect.top() + 36, "Backward")

        painter.setPen(QPen(QColor("white"), 1))
        painter.drawText(plot_rect.center().x() - 25, self.height() - 8, "Time (s)")
        painter.save()
        painter.translate(15, plot_rect.center().y())
        painter.rotate(-90)
        painter.drawText(0, -5, "Photon count")
        painter.restore()


class MarkerAnalysisWidget(QWidget):
    """Estimate and display the forward/backward marker pre-alignment."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.resize(960, 720)
        self.setMinimumSize(640, 480)

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

        self.start_marker_dly_input = QDoubleSpinBox()
        self.start_marker_dly_input.setRange(-0.5, 0.5)
        self.start_marker_dly_input.setDecimals(5)
        self.start_marker_dly_input.setSingleStep(0.001)
        self.start_marker_dly_input.setFixedWidth(150)

        self.stop_marker_dly_input = QDoubleSpinBox()
        self.stop_marker_dly_input.setRange(-0.5, 0.5)
        self.stop_marker_dly_input.setDecimals(5)
        self.stop_marker_dly_input.setSingleStep(0.001)
        self.stop_marker_dly_input.setFixedWidth(150)

        self.chunk_size_input = QSpinBox()
        self.chunk_size_input.setRange(100, 20_000)
        self.chunk_size_input.setSingleStep(100)
        self.chunk_size_input.setFixedWidth(150)
        self.chunk_size_input.setValue(100)

        self.skip_n_chunks_input = QSpinBox()
        self.skip_n_chunks_input.setRange(0, 100)
        self.skip_n_chunks_input.setSingleStep(1)
        self.skip_n_chunks_input.setFixedWidth(150)

        self.label = QLabel("Ready")
        self.apply_button = QPushButton("Apply")
        self.apply_button.setFixedSize(90, 30)
        self.apply_button.clicked.connect(self.apply_inputs)

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
            self.marker_checkboxes[name] = checkbox
            self.marker_checkbox_layout.addWidget(checkbox)

        self.plot = AlignmentPlotWidget()

        self.selector_position_input = QDoubleSpinBox()
        self.selector_position_input.setRange(-1_000_000_000.0, 1_000_000_000.0)
        self.selector_position_input.setDecimals(3)
        self.selector_position_input.setSingleStep(0.01)
        self.selector_position_input.setReadOnly(True)
        self.selector_position_input.setFixedWidth(150)
        self.selector_position_input.setSpecialValueText("--")

        layout = QVBoxLayout(self)
        form_layout = QFormLayout()
        form_layout.addRow("PTU file:", file_layout)
        form_layout.addRow("Line start marker delay phase:", self.start_marker_dly_input)
        form_layout.addRow("Line stop marker delay phase:", self.stop_marker_dly_input)
        form_layout.addRow("Chunk size [x1000]:", self.chunk_size_input)
        form_layout.addRow("Chunks to skip:", self.skip_n_chunks_input)
        layout.addLayout(form_layout)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.apply_button)
        button_layout.addStretch()
        layout.addLayout(button_layout)
        layout.addLayout(self.marker_checkbox_layout)
        layout.addWidget(self.plot)

        selector_layout = QHBoxLayout()
        selector_layout.addWidget(QLabel("Selector position [s]:"))
        selector_layout.addWidget(self.selector_position_input)
        selector_layout.addStretch()
        layout.addLayout(selector_layout)
        layout.addWidget(self.label)

    def select_ptu_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select PTU file",
            self.file_path_edit.text() or QDir.homePath(),
            "PTU files (*.ptu);;All files (*)",
        )
        if file_path:
            self.file_path_edit.setText(file_path)

    def apply_inputs(self):
        file_path = self.file_path_edit.text().strip()
        if not file_path:
            self.label.setText("Select a .ptu file before estimating alignment.")
            return

        self.apply_button.setEnabled(False)
        self.label.setText("Estimating alignment…")
        self.plot.clear()
        QApplication.processEvents()

        try:
            reader = TTTRReader(file_path)
            tags = reader.header.tags
            result = estimate_bidirectional_prealign(
                reader=reader,
                cfg=ScanConfig(
                    bidirectional=True,
                    line_start_marker_delay=self.start_marker_dly_input.value(),
                    line_stop_marker_delay=self.stop_marker_dly_input.value(),
                ),
                laser_sync_rate=float(tags.get("TTResult_SyncRate", 40e6)),
                wrap=int(tags.get("TTResultFormat_WrapAround", 1024)),
                chunk_length=self.chunk_size_input.value() * 1_000,
                skip_chunks=self.skip_n_chunks_input.value(),
                verbose=False,
            )
            self.plot.set_result(result)

            pixel_shift = int(result["pixel_shift"].item())
            time_shift = float(result["time_shift"].item())
            self.label.setText(
                f"Alignment estimated: {pixel_shift} pixels ({time_shift * 1e6:.3f} μs)."
            )
        except Exception as exc:
            self.label.setText(f"Alignment failed: {exc}")
        finally:
            self.apply_button.setEnabled(True)
