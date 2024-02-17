import sys
import clsmview.gui
from qtpy.QtWidgets import QApplication


def main():
    app = QApplication(sys.argv)
    gui = clsmview.gui.CLSMPixelSelect()
    gui.show()
    app.exec_()


if __name__ == "__main__":
    main()
