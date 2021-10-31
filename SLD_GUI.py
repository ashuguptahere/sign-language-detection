from PyQt5.QtCore import *  #hacktoberfest
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import sys


class SLD(QDialog):
    def __init__(self):
        QDialog.__init__(self)
        layout = QHBoxLayout()
        self.setWindowTitle("Sign Language Detection")
        label = QLabel("Sign Language Detection")

        win = QWidget()
        l1 = QLabel()
        l1.setPixmap(QPixmap("python.jpg"))

        cbutton = QPushButton("Close")
        browse = QPushButton("Browse")
        # reco = QPushButton("Recognize")
        webCam = QPushButton("Open WebCam")
        self.loc = QLineEdit(self)

        # self.label.move(20, 20)

        self.loc.setPlaceholderText("Your File's Location")

        cbutton.clicked.connect(self.close)
        layout.addWidget(label)
        layout.addWidget(self.loc)
        layout.addWidget(browse)
        # layout.addWidget(reco)
        layout.addWidget(webCam)
        layout.addWidget(cbutton)
        layout.addWidget(l1)

        self.loc.move(80, 20)
        self.loc.resize(200, 32)
        self.setLayout(layout)
        self.setFocus()

        browse.clicked.connect(self.browse_file)

    def browse_file(self):
        # file = QFileDialog.getSaveFileName(self, caption="Save File As", directory=".", filter="All Files (*.*)")
        # file = QFileDialog.getOpenFileName(self, caption="Open File", directory=".", filter="All Files (*.*)")
        # self.save_loc.setText(QDir.toNativeSeparators(save_file))

        name = QFileDialog.getOpenFileName(self, caption='Open File', filter="All Files (*.*)")
        # file = open(name, 'r')

        # save_file = QFileDialog.getSaveFileName(self, caption="Save File As", directory=".", filter="All Files (*.*)")
        # self.loc.setText(QDir.toNativeSeparators(name))
        # self.loc.setText(QDir.toNativeSeparators(name))

    def capImg(self):
        pass

    # def reco(self):
        # pass

    def picshow(self):
        pass


# app = QApplication(sys.argv)
# dialog = SLD()
# dialog.show()
# app.exec_()


if __name__ == "__main__":
    def run_app():
        app = QApplication(sys.argv)
        main_win = SLD()
        main_win.show()
        app.exec_()
    run_app()
