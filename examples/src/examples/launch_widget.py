from qtpy.QtWidgets import QApplication
from marker_alignment_widget import MarkerAnalysisWidget

app = QApplication.instance() or QApplication([])
widget = MarkerAnalysisWidget()

widget.show()
app.exec()