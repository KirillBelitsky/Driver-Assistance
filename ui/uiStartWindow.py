from PyQt5 import QtCore, QtWidgets


class Ui_StartWindow(object):

    def __init__(self):
        pass

    def setupUi(self, startWindow):
        startWindow.setObjectName("mainWindow")
        startWindow.setFixedSize(300, 170)

        self.centralwidget = QtWidgets.QWidget(startWindow)
        self.centralwidget.setObjectName("centralwidget")

        self.__initLabels()
        self.__initButtons()

        startWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(startWindow)
        QtCore.QMetaObject.connectSlotsByName(startWindow)

    def retranslateUi(self, startWindow):
        _translate = QtCore.QCoreApplication.translate
        startWindow.setWindowTitle(_translate("mainWindow", "mainWindow"))

        self.chooseVideoLabel.setText(_translate("mainWindow", "Choose video:"))
        self.chooseVideoButton.setText(_translate("mainWindow", "..."))
        self.startButton.setText(_translate("mainWindow", "Start"))
        self.chooseOutputPathButton.setText(_translate("mainWindow", "..."))
        self.chooseOutputPathLabel.setText(_translate("mainWindow", "Choose output path:"))

    def __initButtons(self):
        self.chooseVideoButton = QtWidgets.QPushButton(self.centralwidget)
        self.chooseVideoButton.setGeometry(QtCore.QRect(190, 20, 90, 25))
        self.chooseVideoButton.setObjectName("chooseVideoButton")

        self.startButton = QtWidgets.QPushButton(self.centralwidget)
        self.startButton.setGeometry(QtCore.QRect(105, 120, 89, 25))
        self.startButton.setObjectName("startButton")

        self.chooseOutputPathButton = QtWidgets.QPushButton(self.centralwidget)
        self.chooseOutputPathButton.setGeometry(QtCore.QRect(190, 60, 90, 25))
        self.chooseOutputPathButton.setObjectName("chooseOutputPathButton")

    def __initLabels(self):
        self.chooseVideoLabel = QtWidgets.QLabel(self.centralwidget)
        self.chooseVideoLabel.setGeometry(QtCore.QRect(20, 20, 100, 25))
        self.chooseVideoLabel.setObjectName("chooseVideoLabel")

        self.chooseOutputPathLabel = QtWidgets.QLabel(self.centralwidget)
        self.chooseOutputPathLabel.setGeometry(QtCore.QRect(20, 60, 141, 25))
        self.chooseOutputPathLabel.setObjectName("chooseOutputPathLabel")