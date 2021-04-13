from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_DetectVideoWindow(object):

    def setupUi(self, mainWindow):
        mainWindow.setObjectName("detectVideoWindow")
        mainWindow.setFixedSize(1190, 580)
        self.centralwidget = QtWidgets.QWidget(mainWindow)
        self.centralwidget.setObjectName("centralwidget")

        self.initButtons()
        self.initBars(mainWindow)
        self.initLabels()

        mainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(mainWindow)
        QtCore.QMetaObject.connectSlotsByName(mainWindow)

    def retranslateUi(self, mainWindow):
        _translate = QtCore.QCoreApplication.translate
        mainWindow.setWindowTitle(_translate("detectVideoWindow", "Driver-Assistance"))
        self.stopButton.setText(_translate("detectVideoWindow", "Stop"))
        self.statisticLabel.setText(_translate("detectVideoWindow", "Statistic"))
        self.fpsLabel.setText(_translate("detectVideoWindow", "FPS:"))
        self.positionLabel.setText(_translate("detectVideoWindow", "Position:"))

    def initButtons(self):
        self.stopButton = QtWidgets.QPushButton(self.centralwidget)
        self.stopButton.setGeometry(QtCore.QRect(1010, 520, 89, 25))
        self.stopButton.setObjectName("stopButton")

    def initBars(self, mainWindow):
        self.menubar = QtWidgets.QMenuBar(mainWindow)
        self.menubar.setEnabled(True)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1190, 22))
        self.menubar.setObjectName("menubar")
        mainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(mainWindow)
        self.statusbar.setObjectName("statusbar")
        mainWindow.setStatusBar(self.statusbar)

    def initLabels(self):
        self.statisticLabel = QtWidgets.QLabel(self.centralwidget)
        self.statisticLabel.setGeometry(QtCore.QRect(1010, 10, 121, 41))
        font = QtGui.QFont()
        font.setPointSize(20)
        self.statisticLabel.setFont(font)
        self.statisticLabel.setObjectName("statisticLabel")

        self.fpsLabel = QtWidgets.QLabel(self.centralwidget)
        self.fpsLabel.setGeometry(QtCore.QRect(950, 60, 41, 40))
        font = QtGui.QFont()
        font.setPointSize(15)
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
        self.positionLabel.setGeometry(QtCore.QRect(950, 110, 71, 40))
        font = QtGui.QFont()
        font.setPointSize(13)
        self.positionLabel.setFont(font)
        self.positionLabel.setObjectName("positionLabel")

        self.fpsValue = QtWidgets.QLabel(self.centralwidget)
        self.fpsValue.setGeometry(QtCore.QRect(1020, 60, 170, 40))
        font = QtGui.QFont()
        font.setPointSize(13)
        self.fpsValue.setFont(font)
        self.fpsValue.setText("")
        self.fpsValue.setObjectName("fpsValue")

        self.positionValue = QtWidgets.QLabel(self.centralwidget)
        self.positionValue.setGeometry(QtCore.QRect(1020, 110, 150, 40))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.positionValue.setFont(font)
        self.positionValue.setText("")
        self.positionValue.setObjectName("positionValue")
