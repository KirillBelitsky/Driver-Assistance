from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_DetectVideoWindow(object):

    def setupUi(self, detectVideoWindow):
        detectVideoWindow.setObjectName("detectVideoWindow")
        detectVideoWindow.setFixedSize(1175, 560)
        self.centralwidget = QtWidgets.QWidget(detectVideoWindow)
        self.centralwidget.setObjectName("centralwidget")

        self.__initButtons()
        self.__initLabels()

        detectVideoWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(detectVideoWindow)
        QtCore.QMetaObject.connectSlotsByName(detectVideoWindow)

    def retranslateUi(self, detectVideoWindow):
        _translate = QtCore.QCoreApplication.translate
        detectVideoWindow.setWindowTitle(_translate("detectVideoWindow", "Окно обработки"))
        self.stopButton.setText(_translate("detectVideoWindow", "Остановить"))
        self.statisticLabel.setText(_translate("detectVideoWindow", "Статистика"))
        self.fpsLabel.setText(_translate("detectVideoWindow", "FPS:"))
        self.positionLabel.setText(_translate("detectVideoWindow", "Положение:"))
        self.directionLabel.setText(_translate("detectVideoWindow", "Направление:"))

    def __initButtons(self):
        self.stopButton = QtWidgets.QPushButton(self.centralwidget)
        self.stopButton.setGeometry(QtCore.QRect(1010, 520, 89, 25))
        self.stopButton.setObjectName("stopButton")

    def __initLabels(self):
        self.statisticLabel = QtWidgets.QLabel(self.centralwidget)
        self.statisticLabel.setGeometry(QtCore.QRect(980, 10, 140, 41))
        font = QtGui.QFont()
        font.setPointSize(20)
        self.statisticLabel.setFont(font)
        self.statisticLabel.setObjectName("statisticLabel")

        self.fpsLabel = QtWidgets.QLabel(self.centralwidget)
        self.fpsLabel.setGeometry(QtCore.QRect(950, 60, 40, 40))
        font = QtGui.QFont()
        font.setPointSize(13)
        self.fpsLabel.setFont(font)
        self.fpsLabel.setObjectName("fpsLabel")

        self.gridLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.gridLayoutWidget.setGeometry(QtCore.QRect(0, -1, 930, 550))
        self.gridLayoutWidget.setObjectName("gridLayoutWidget")
        self.gridLayout = QtWidgets.QGridLayout(self.gridLayoutWidget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")

        self.image = QtWidgets.QLabel(self.gridLayoutWidget)
        self.image.setText("")
        self.image.setScaledContents(True)
        self.image.setObjectName("image")
        self.gridLayout.addWidget(self.image, 0, 0, 1, 1)

        self.positionLabel = QtWidgets.QLabel(self.centralwidget)
        self.positionLabel.setGeometry(QtCore.QRect(950, 110, 110, 40))
        font = QtGui.QFont()
        font.setPointSize(13)
        self.positionLabel.setFont(font)
        self.positionLabel.setObjectName("positionLabel")

        self.directionLabel = QtWidgets.QLabel(self.centralwidget)
        self.directionLabel.setGeometry(QtCore.QRect(950, 160, 115, 40))
        font = QtGui.QFont()
        font.setPointSize(13)
        self.directionLabel.setFont(font)
        self.directionLabel.setObjectName("directionLabel")

        self.fpsValue = QtWidgets.QLabel(self.centralwidget)
        self.fpsValue.setGeometry(QtCore.QRect(1020, 60, 170, 40))
        font = QtGui.QFont()
        font.setPointSize(13)
        self.fpsValue.setFont(font)
        self.fpsValue.setText("")
        self.fpsValue.setObjectName("fpsValue")

        self.positionValue = QtWidgets.QLabel(self.centralwidget)
        self.positionValue.setGeometry(QtCore.QRect(1060, 110, 150, 40))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.positionValue.setFont(font)
        self.positionValue.setText("")
        self.positionValue.setObjectName("positionValue")

        self.directionValue = QtWidgets.QLabel(self.centralwidget)
        self.directionValue.setGeometry(QtCore.QRect(1070, 160, 150, 40))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.directionValue.setFont(font)
        self.directionValue.setText("")
        self.directionValue.setObjectName("directionValue")
