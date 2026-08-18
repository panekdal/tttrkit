import numpy as np
from qtpy.QtCore import QDir, Qt, Signal
from qtpy.QtGui import QColor, QPainter, QPen
from qtpy.QtWidgets import (
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
from tttrkit.ptuio.reconstructor import TraceReconstructor
from tttrkit.ptuio.decoder import T3OverflowCorrector

class LinePlotWidget(QWidget):
    """Small dependency-free line plot suitable for embedding in a Qt form."""

    selector_changed = Signal(float)
    selection_range_changed = Signal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._series = []
        self._marker_series = {}
        self._selector_time = None
        self._dragging = False
        self._drag_start_time = None
        self._drag_end_time = None
        self.setMouseTracking(True)
        self.setMinimumHeight(400)
        self._x_min = None
        self._x_max = None

    @staticmethod
    def _channel_colors():
        return [
            QColor("#0072B2"),
            QColor("#D55E00"),
            QColor("#009E73"),
            QColor("#CC79A7"),
            QColor("#E69F00"),
        ]

    def set_values(self, values, start_time: float = 0, stop_time: float = 10, visible_channels=None, sum_selected=False, marker_visibility=None,):
        self._series = []
        self._marker_series = {}
        self._selector_time = None
        self._x_min = start_time
        self._x_max = stop_time

        if values is None:
            self.update()
            return

        if hasattr(values, "data_vars") and "photon_count" in values.data_vars:
            photon_count = values["photon_count"]
            time_axis = np.asarray(photon_count.coords["time"].values)

            self._x_min = float(start_time)
            self._x_max = float(stop_time)

            counts = np.asarray(photon_count.values)
            if counts.ndim == 1:
                counts = counts.reshape(-1, 1)
            channel_ids = np.asarray(photon_count.coords["channel"].values)
            if visible_channels is None:
                visible_channels = set(channel_ids.tolist())
            else:
                visible_channels = {int(channel) for channel in visible_channels}

            channel_count = counts.shape[1] if counts.ndim > 1 else 1
            colors = self._channel_colors()
            for channel_index in range(channel_count):
                channel_id = int(channel_ids[channel_index]) if channel_count > 0 and len(channel_ids) > channel_index else channel_index
                if channel_id not in visible_channels:
                    continue
                channel_values = counts[:, channel_index]
                self._series.append((time_axis, channel_values, colors[channel_index % len(colors)], f"Channel {channel_id}"))

            if sum_selected and self._series:
                summed = np.sum(np.asarray([series[1] for series in self._series], dtype=float), axis=0)
                self._series.append((self._series[0][0], summed, QColor("white"), "Sum"))

            for name, label in {
                "line_start": "line_start_times",
                "line_stop": "line_stop_times",
                "frame_start": "frame_start_times",
            }.items():
                if label in values.data_vars:
                    marker_values = np.asarray(values[label].values, dtype=float)
                    if marker_visibility is None or marker_visibility.get(name, True):
                        self._marker_series[name] = (marker_values, {
                            "line_start": QColor("lime"),
                            "line_stop": QColor("red"),
                            "frame_start": QColor("cyan"),
                        }[name])
            self.update()
            return

        if isinstance(values, (list, tuple)) and values and isinstance(values[0], (list, tuple, np.ndarray)):
            flattened = [float(item) for sublist in values for item in sublist]
            self._series = [(
                np.arange(len(flattened)),
                np.asarray(flattened, dtype=float),
                self._channel_colors()[0],
                "Series",
            )]
            self.update()
            return

        flattened = [float(value) for value in values]
        self._series = [(
            np.arange(len(flattened)),
            np.asarray(flattened, dtype=float),
            self._channel_colors()[0],
            "Series",
        )]
        self.update()

    def _plot_rect(self):
        left, top, right, bottom = 60, 20, 25, 40
        return self.rect().adjusted(left, top, -right, -bottom)

    def _time_from_x(self, x_pos):
        if not self._series:
            return None

        plot_rect = self._plot_rect()
        x_min = self._x_min
        x_max = self._x_max
        if x_max <= x_min:
            return x_min

        if x_pos <= plot_rect.left():
            return x_min
        if x_pos >= plot_rect.right():
            return x_max

        relative = (x_pos - plot_rect.left()) / plot_rect.width()
        return x_min + relative * (x_max - x_min)

    def _update_selector_from_event(self, event):
        pos = event.position().toPoint()
        if not self._plot_rect().contains(pos):
            return
        x_pos = pos.x()
        self._selector_time = self._time_from_x(x_pos)
        if self._selector_time is not None:
            self.selector_changed.emit(float(self._selector_time))
        if self._dragging:
            self._drag_end_time = float(self._selector_time) if self._selector_time is not None else None
        self.update()

    def mouseMoveEvent(self, event):
        self._update_selector_from_event(event)

    def mousePressEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton:
            return
        pos = event.position().toPoint()
        if not self._plot_rect().contains(pos):
            return
        self._dragging = True
        self._drag_start_time = self._time_from_x(pos.x())
        self._drag_end_time = self._drag_start_time
        self._selector_time = self._drag_start_time
        if self._selector_time is not None:
            self.selector_changed.emit(float(self._selector_time))
        self.update()

    def mouseReleaseEvent(self, event):
        if not self._dragging:
            return

        pos = event.position().toPoint()
        if self._plot_rect().contains(pos):
            self._drag_end_time = self._time_from_x(pos.x())

        if self._drag_start_time is not None and self._drag_end_time is not None:
            start_time = min(float(self._drag_start_time), float(self._drag_end_time))
            stop_time = max(float(self._drag_start_time), float(self._drag_end_time))
            self.selection_range_changed.emit(start_time, stop_time)

        self._dragging = False
        self._drag_start_time = None
        self._drag_end_time = None
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("black"))

        plot_rect = self._plot_rect()
        painter.setPen(QPen(QColor("white"), 1))
        painter.drawRect(plot_rect)

        if not self._series or plot_rect.width() <= 0 or plot_rect.height() <= 0:
            painter.setPen(QPen(QColor("white")))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No channel selected")
            return

        all_values = []
        x_min = self._x_min if self._x_min is not None else float("inf")
        x_max = self._x_max if self._x_max is not None else float("-inf")
        for x_axis, y_values, _, _ in self._series:
            values = np.asarray(y_values, dtype=float)
            all_values.extend(values.tolist())
            if len(x_axis) > 0:
                x_min = self._x_min if self._x_min is not None else x_min
                x_max = self._x_max if self._x_max is not None else x_max

        if x_min == float("inf") or x_max == float("-inf"):
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Click Apply to plot output")
            return

        minimum = min(all_values)
        maximum = max(all_values)
        if minimum == maximum:
            minimum -= 1.0
            maximum += 1.0
        value_range = maximum - minimum
        if value_range == 0:
            value_range = 1.0

        x_tick_values = np.linspace(x_min, x_max, 5)
        y_tick_values = np.linspace(minimum, maximum, 5)

        painter.setPen(QPen(QColor("white"), 1))
        painter.drawLine(plot_rect.bottomLeft(), plot_rect.bottomRight())
        painter.drawLine(plot_rect.bottomLeft(), plot_rect.topLeft())

        for tick_value in x_tick_values:
            x = plot_rect.left() + (tick_value - x_min) * plot_rect.width() / (x_max - x_min if x_max != x_min else 1.0)
            painter.drawLine(int(x), plot_rect.bottom(), int(x), plot_rect.bottom() + 5)
            painter.drawText(int(x) - 12, plot_rect.bottom() + 18, f"{tick_value:.3g}")

        for tick_value in y_tick_values:
            y = plot_rect.bottom() - (tick_value - minimum) * plot_rect.height() / value_range
            painter.drawLine(plot_rect.left() - 5, int(y), plot_rect.left(), int(y))
            painter.drawText(15, int(y) + 4, f"{tick_value:.3g}")

        for index, (x_axis, y_values, color, label) in enumerate(self._series):
            points = []
            values = np.asarray(y_values, dtype=float)
            samples = np.asarray(x_axis, dtype=float)
            if len(samples) != len(values):
                continue
            for sample_value, signal_value in zip(samples, values):
                x = plot_rect.left() + (sample_value - x_min) * plot_rect.width() / (x_max - x_min if x_max != x_min else 1.0)
                y = plot_rect.bottom() - (signal_value - minimum) * plot_rect.height() / value_range
                points.append((int(x), int(y)))

            painter.setPen(QPen(color, 2))
            for start, end in zip(points, points[1:]):
                painter.drawLine(*start, *end)

            painter.setPen(QPen(color))
            label_y = plot_rect.top() + 12 + index * 16
            painter.drawText(plot_rect.left() + 10, label_y, label)

        for marker_name, (marker_times, color) in self._marker_series.items():
            if len(marker_times) == 0:
                continue
            painter.setPen(QPen(color, 1, Qt.PenStyle.DashLine))
            value_zero = 0.0
            y_zero = plot_rect.bottom() - (value_zero - minimum) * plot_rect.height() / value_range
            y_top = plot_rect.top()
            for marker_time in marker_times:
                if marker_time < x_min or marker_time > x_max:
                    continue
                x = plot_rect.left() + (marker_time - x_min) * plot_rect.width() / (x_max - x_min if x_max != x_min else 1.0)
                painter.drawLine(int(x), int(y_top), int(x), int(y_zero))

        if self._selector_time is not None and x_min <= self._selector_time <= x_max:
            painter.setPen(QPen(QColor("white"), 1, Qt.PenStyle.DashLine))
            selector_x = plot_rect.left() + (self._selector_time - x_min) * plot_rect.width() / (x_max - x_min if x_max != x_min else 1.0)
            painter.drawLine(int(selector_x), int(plot_rect.top()), int(selector_x), int(plot_rect.bottom()))

        painter.setPen(QPen(QColor("white"), 1))
        painter.drawText(plot_rect.center().x() - 30, self.height() - 7, "Time")

        painter.save()
        painter.translate(15, plot_rect.center().y())
        painter.rotate(-90)
        painter.drawText(0, -5, "Photon count")
        painter.restore()


class TrackAnalysisWidget(QWidget):
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

        self.start_time_input = self._create_time_input()
        self.stop_time_input = self._create_time_input()
        self.start_time_input.setValue(0.0)
        self.stop_time_input.setValue(10.0)

        self.max_detector_input = self._create_detector_input()
        self.sensor_bin_width_input = self._create_bin_width_input()

        self.label = QLabel("Ready")
        self.apply_button = QPushButton("Apply")
        
        self.apply_button.setFixedSize(90, 30)
        
        self.plot = LinePlotWidget()
        self.selector_position_input = self._create_selector_position_input()
        self.sum_checkbox = QCheckBox("Sum selected channels")
        self.sum_checkbox.setChecked(False)
        self.sum_checkbox.toggled.connect(self._refresh_plot_from_checkboxes)
        self.channel_checkbox_layout = QHBoxLayout()
        self.channel_checkbox_layout.setContentsMargins(0, 0, 0, 0)
        self.channel_checkbox_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.channel_checkboxes = {}
        self._last_result = None
        self.channel_control_row = QHBoxLayout()
        self.channel_control_row.setContentsMargins(0, 0, 0, 0)
        self.channel_control_row.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.channel_control_row.addWidget(self.sum_checkbox)
        self.channel_control_row.addLayout(self.channel_checkbox_layout)

        self.marker_checkboxes = {}
        self.marker_checkbox_layout = QHBoxLayout()
        self.marker_checkbox_layout.setContentsMargins(0, 0, 0, 0)
        self.marker_checkbox_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        for name, label in [
            ("line_start", "Line start markers"),
            ("line_stop", "Stop markers"),
            ("frame_start", "Frame start markers"),
        ]:
            checkbox = QCheckBox(label)
            checkbox.setChecked(True)
            checkbox.toggled.connect(self._refresh_plot_from_checkboxes)
            self.marker_checkboxes[name] = checkbox
            self.marker_checkbox_layout.addWidget(checkbox)

        layout = QVBoxLayout(self)
        form_layout = QFormLayout()
        form_layout.addRow("PTU file:", file_layout)
        form_layout.addRow("Start time (s):", self.start_time_input)
        form_layout.addRow("Stop time (s):", self.stop_time_input)
        form_layout.addRow("Max detector:", self.max_detector_input)
        form_layout.addRow("Bin width (ms):", self.sensor_bin_width_input)
        layout.addLayout(form_layout)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.apply_button)
        # button_layout.addWidget(self.button)
        button_layout.addStretch()
        layout.addLayout(button_layout)
        layout.addLayout(self.channel_control_row)
        layout.addLayout(self.marker_checkbox_layout)
        layout.addWidget(self.plot)

        selector_layout = QHBoxLayout()
        selector_layout.addWidget(QLabel("Selector position [s]:"))
        selector_layout.addWidget(self.selector_position_input)
        selector_layout.addStretch()
        layout.addLayout(selector_layout)
        layout.addWidget(self.label)

        self.apply_button.clicked.connect(self.apply_inputs)
        # self.button.clicked.connect(self.run)
        self.plot.selector_changed.connect(self._update_selector_position)
        self.plot.selection_range_changed.connect(self._update_time_range_from_drag)

    @staticmethod
    def _create_time_input():
        numeric_input = QDoubleSpinBox()
        numeric_input.setRange(0.0, 999.99)
        numeric_input.setDecimals(3)
        numeric_input.setSingleStep(0.01)
        numeric_input.setFixedWidth(150)
        return numeric_input

    @staticmethod
    def _create_detector_input():
        numeric_input = QSpinBox()
        numeric_input.setRange(1, 10_000_000)
        numeric_input.setSingleStep(1)
        numeric_input.setFixedWidth(150)
        numeric_input.setValue(1)
        return numeric_input

    @staticmethod
    def _create_bin_width_input():
        numeric_input = QDoubleSpinBox()
        numeric_input.setRange(0.001, 10_000.0)
        numeric_input.setDecimals(3)
        numeric_input.setSingleStep(0.01)
        numeric_input.setFixedWidth(150)
        numeric_input.setValue(1.0)
        return numeric_input

    @staticmethod
    def _create_selector_position_input():
        numeric_input = QDoubleSpinBox()
        numeric_input.setRange(-1_000_000_000.0, 1_000_000_000.0)
        numeric_input.setDecimals(3)
        numeric_input.setSingleStep(0.01)
        numeric_input.setReadOnly(True)
        numeric_input.setFixedWidth(150)
        numeric_input.setSpecialValueText("--")
        return numeric_input

    def _update_selector_position(self, value):
        self.selector_position_input.setValue(float(value))

    def _update_time_range_from_drag(self, start_time, stop_time):
        self.start_time_input.setValue(float(min(start_time, stop_time)))
        self.stop_time_input.setValue(float(max(start_time, stop_time)))

    def _set_active_channels(self, result):
        if result is None or "photon_count" not in result.data_vars:
            for checkbox in list(self.channel_checkboxes.values()):
                checkbox.deleteLater()
            self.channel_checkboxes.clear()
            return

        channel_ids = list(result["photon_count"].coords["channel"].values)
        for channel_id in channel_ids:
            if channel_id not in self.channel_checkboxes:
                checkbox = QCheckBox(f"Channel {int(channel_id)}")
                checkbox.setChecked(True)
                checkbox.toggled.connect(self._refresh_plot_from_checkboxes)
                self.channel_checkboxes[int(channel_id)] = checkbox
                self.channel_checkbox_layout.addWidget(checkbox)

        for channel_id in list(self.channel_checkboxes.keys()):
            if channel_id not in channel_ids:
                checkbox = self.channel_checkboxes.pop(channel_id)
                self.channel_checkbox_layout.removeWidget(checkbox)
                checkbox.deleteLater()

    def _selected_channels(self):
        selected = []
        for channel_id, checkbox in self.channel_checkboxes.items():
            if checkbox.isChecked():
                selected.append(channel_id)
        return selected

    def _refresh_plot_from_checkboxes(self):
        if self._last_result is not None:
            self.plot.set_values(
                self._last_result,
                start_time=self.start_time_input.value(),
                stop_time=self.stop_time_input.value(),
                visible_channels=self._selected_channels(),
                sum_selected=self.sum_checkbox.isChecked(),
                marker_visibility={
                    name: checkbox.isChecked()
                    for name, checkbox in self.marker_checkboxes.items()
                },
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

    @staticmethod
    def calculate_output(
        path,
        start_time,
        stop_time,
        max_detector,
        bin_width,
    ):
        reader = TTTRReader(path)
        for key, value in reader.header.tags.items():
            print(f"{key}: {value}")

        wrap = reader.header.tags.get("TTResultFormat_WrapAround", 1024)
        sync_rate = reader.header.tags.get("TTResult_SyncRate", 40e6)

        corrector = T3OverflowCorrector(wraparound=wrap)
        reconstructor = TraceReconstructor(
            start_time=start_time,
            stop_time=stop_time,
            max_detector=max_detector,
            bin_width=bin_width / 1_000,
            sync_rate=sync_rate,
            outputs=["photon_count", "markers"],
        )
        for chunk in reader.iter_chunks():
            reconstructor.update(corrector.correct(chunk))
        result = reconstructor.finalize()
        return result

    def apply_inputs(self):
        file_path = self.file_path_edit.text().strip()
        if not file_path:
            self.label.setText("Select a .ptu file before applying analysis settings.")
            return

        output = self.calculate_output(
            path=file_path,
            start_time=self.start_time_input.value(),
            stop_time=self.stop_time_input.value(),
            max_detector=self.max_detector_input.value(),
            bin_width=self.sensor_bin_width_input.value(),
        )
        self._last_result = output
        self._set_active_channels(output)
        self.plot.set_values(
            output,
            start_time=self.start_time_input.value(),
            stop_time=self.stop_time_input.value(),
            visible_channels=self._selected_channels(),
            sum_selected=self.sum_checkbox.isChecked(),
            marker_visibility={
                name: checkbox.isChecked()
                for name, checkbox in self.marker_checkboxes.items()
            },
        )
        self.label.setText("Plot updated.")

    def run(self):
        file_path = self.file_path_edit.text().strip()
        if not file_path:
            self.label.setText("Select a .ptu file before running.")
            return

        self.label.setText(
            f"Running with {file_path} "
            f"(start={self.start_time_input.value():.2f}s, stop={self.stop_time_input.value():.2f}s, "
            f"max_detector={self.max_detector_input.value()}, bin_width={self.sensor_bin_width_input.value():.3f}ms)"
        )

        try:
            output = self.calculate_output(
                path=file_path,
                start_time=self.start_time_input.value(),
                stop_time=self.stop_time_input.value(),
                max_detector=self.max_detector_input.value(),
                bin_width=self.sensor_bin_width_input.value(),
            )
            self._last_result = output
            self._set_active_channels(output)
            self.plot.set_values(
                output,
                start_time=self.start_time_input.value(),
                stop_time=self.stop_time_input.value(),
                visible_channels=self._selected_channels(),
                sum_selected=self.sum_checkbox.isChecked(),
                marker_visibility={
                    name: checkbox.isChecked()
                    for name, checkbox in self.marker_checkboxes.items()
                },
            )
            self.label.setText("Analysis complete.")
        except Exception as exc:
            self.label.setText(f"Analysis failed: {exc}")
