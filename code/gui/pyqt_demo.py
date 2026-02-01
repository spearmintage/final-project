import PyQt5.QtWidgets as widgets
import PyQt5.QtCore as core
import PyQt5.QtGui as gui

class Color(widgets.QWidget):
    def __init__(self, color):
        super().__init__()
        self.setAutoFillBackground(True)

        palette = self.palette()
        palette.setColor(gui.QPalette.ColorRole.Window, gui.QColor(color))
        self.setPalette(palette)

class DemoWindow(widgets.QMainWindow):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Birdsong Classification")
        widget = widgets.QWidget()
        self.setFixedSize(core.QSize(800, 600))

        layer1 = widgets.QVBoxLayout()
        layer1.addWidget(widgets.QLabel("helo").setAlignment())

        layer2 = widgets.QVBoxLayout()
        layer2.addWidget(Color("blue"))

        layer3 = widgets.QHBoxLayout()
        layer3.addWidget(Color("green"))
        layer3.addWidget(Color("orange"))

        layer2.addLayout(layer3)
        layer1.addLayout(layer2)
        widget.setLayout(layer1)

        self.setCentralWidget(widget)

app = widgets.QApplication([])
window = DemoWindow()
window.show()
app.exec()