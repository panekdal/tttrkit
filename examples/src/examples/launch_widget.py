from qtpy.QtWidgets import QApplication
from track_analysis_widget import TrackAnalysisWidget

app = QApplication.instance() or QApplication([])
widget = TrackAnalysisWidget()
widget.show()
app.exec()