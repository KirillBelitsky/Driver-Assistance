from PyQt5 import QtCore, QtWidgets


class Ui_StartWindow(object):

    def __init__(self):
        self.videoPath = None
        self.outputVideoPath = None

    def setupUi(self, mainWindow):
        mainWindow.setObjectName("mainWindow")
        mainWindow.setFixedSize(300, 170)

        self.centralwidget = QtWidgets.QWidget(mainWindow)
        self.centralwidget.setObjectName("centralwidget")

        self.initBars(mainWindow)
        self.initLabels()
        self.initButtons()

        mainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(mainWindow)
        QtCore.QMetaObject.connectSlotsByName(mainWindow)

    def retranslateUi(self, mainWindow):
        _translate = QtCore.QCoreApplication.translate
        mainWindow.setWindowTitle(_translate("mainWindow", "mainWindow"))

        self.label.setText(_translate("mainWindow", "Choose video:"))
        self.chooseVideoButton.setText(_translate("mainWindow", "..."))
        self.startButton.setText(_translate("mainWindow", "Start"))
        self.chooseOutputPathButton.setText(_translate("mainWindow", "..."))
        self.label_2.setText(_translate("mainWindow", "Choose output path:"))

    def initBars(self, mainWindow):
        self.menubar = QtWidgets.QMenuBar(mainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 350, 22))
        self.menubar.setObjectName("menubar")
        mainWindow.setMenuBar(self.menubar)

        self.statusbar = QtWidgets.QStatusBar(mainWindow)
        self.statusbar.setObjectName("statusbar")
        mainWindow.setStatusBar(self.statusbar)

    def initButtons(self):
        self.chooseVideoButton = QtWidgets.QPushButton(self.centralwidget)
        self.chooseVideoButton.setGeometry(QtCore.QRect(190, 20, 90, 25))
        self.chooseVideoButton.setObjectName("chooseVideoButton")

        self.startButton = QtWidgets.QPushButton(self.centralwidget)
        self.startButton.setGeometry(QtCore.QRect(105, 120, 89, 25))
        self.startButton.setObjectName("startButton")

        self.chooseOutputPathButton = QtWidgets.QPushButton(self.centralwidget)
        self.chooseOutputPathButton.setGeometry(QtCore.QRect(190, 60, 90, 25))
        self.chooseOutputPathButton.setObjectName("chooseOutputPathButton")

    def initLabels(self):
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(20, 20, 100, 25))
        self.label.setObjectName("label")

        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(20, 60, 141, 25))
        self.label_2.setObjectName("label_2")