# -*- coding: utf-8 -*-
"""
/***************************************************************************
 ImageProcess
                                 A QGIS plugin
 A plugin for remote sensing images processing
                              -------------------
        begin                : 2019-06-05
        git sha              : $Format:%H$
        copyright            : (C) 2019 by Team Remote Control
        email                : 2016302590109@whu.edu.cn
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/
"""
from PyQt4.QtCore import QSettings, QTranslator, qVersion, QCoreApplication
from PyQt4.QtGui import QAction, QIcon, QFileDialog, QPixmap
# Initialize Qt resources from file resources.py
import resources
# Import the code for the dialog
from ImageProcess_dialog import ImageProcessDialog
import os.path
from K_Means import ImgDataInfo, ColorList, OutPutCenter, KMeans
from PIL import Image
from MyISODATA import Pixel, Cluster, ClusterPair, ISODATA
import cv2
import numpy as np
import matplotlib
from pylab import *
import MyMosaic


class ImageProcess:
    """QGIS Plugin Implementation."""

    def __init__(self, iface):
        """Constructor.

        :param iface: An interface instance that will be passed to this class
            which provides the hook by which you can manipulate the QGIS
            application at run time.
        :type iface: QgisInterface
        """
        # Save reference to the QGIS interface
        self.iface = iface
        # initialize plugin directory
        self.plugin_dir = os.path.dirname(__file__)
        # initialize locale
        locale = QSettings().value('locale/userLocale')[0:2]
        locale_path = os.path.join(self.plugin_dir, 'i18n', 'ImageProcess_{}.qm'.format(locale))

        if os.path.exists(locale_path):
            self.translator = QTranslator()
            self.translator.load(locale_path)

            if qVersion() > '4.3.3':
                QCoreApplication.installTranslator(self.translator)


        # Declare instance attributes
        self.actions = []
        self.menu = self.tr(u'&Image Process')
        # TODO: We are going to let the user set this up in a future iteration
        self.toolbar = self.iface.addToolBar(u'ImageProcess')
        self.toolbar.setObjectName(u'ImageProcess')
        # Create image list
        self.imageList = []
        # Map image name to image path
        self.imgDict = {}

    # noinspection PyMethodMayBeStatic
    def tr(self, message):
        """Get the translation for a string using Qt translation API.

        We implement this ourselves since we do not inherit QObject.

        :param message: String for translation.
        :type message: str, QString

        :returns: Translated version of message.
        :rtype: QString
        """
        # noinspection PyTypeChecker,PyArgumentList,PyCallByClass
        return QCoreApplication.translate('ImageProcess', message)


    def add_action(
        self,
        icon_path,
        text,
        callback,
        enabled_flag=True,
        add_to_menu=True,
        add_to_toolbar=True,
        status_tip=None,
        whats_this=None,
        parent=None):
        """Add a toolbar icon to the toolbar.

        :param icon_path: Path to the icon for this action. Can be a resource
            path (e.g. ':/plugins/foo/bar.png') or a normal file system path.
        :type icon_path: str

        :param text: Text that should be shown in menu items for this action.
        :type text: str

        :param callback: Function to be called when the action is triggered.
        :type callback: function

        :param enabled_flag: A flag indicating if the action should be enabled
            by default. Defaults to True.
        :type enabled_flag: bool

        :param add_to_menu: Flag indicating whether the action should also
            be added to the menu. Defaults to True.
        :type add_to_menu: bool

        :param add_to_toolbar: Flag indicating whether the action should also
            be added to the toolbar. Defaults to True.
        :type add_to_toolbar: bool

        :param status_tip: Optional text to show in a popup when mouse pointer
            hovers over the action.
        :type status_tip: str

        :param parent: Parent widget for the new action. Defaults None.
        :type parent: QWidget

        :param whats_this: Optional text to show in the status bar when the
            mouse pointer hovers over the action.

        :returns: The action that was created. Note that the action is also
            added to self.actions list.
        :rtype: QAction
        """

        # Create the dialog (after translation) and keep reference
        self.dlg = ImageProcessDialog()

        icon = QIcon(icon_path)
        action = QAction(icon, text, parent)
        action.triggered.connect(callback)
        action.setEnabled(enabled_flag)

        if status_tip is not None:
            action.setStatusTip(status_tip)

        if whats_this is not None:
            action.setWhatsThis(whats_this)

        if add_to_toolbar:
            self.toolbar.addAction(action)

        if add_to_menu:
            self.iface.addPluginToMenu(self.menu, action)

        self.actions.append(action)

        return action

    def initGui(self):
        """Create the menu entries and toolbar icons inside the QGIS GUI."""

        icon_path = ':/plugins/ImageProcess/icon.png'
        self.add_action(icon_path, text=self.tr(u'Image Process'), callback=self.run, parent=self.iface.mainWindow())

        self.dlg.btn_LoadImages.clicked.connect(self.LoadImages)
        self.dlg.comboBox.currentIndexChanged.connect(self.LoadChosenImage)
        self.dlg.btn_kmeans.clicked.connect(self.Classify_KMeans)
        self.dlg.btn_remove.clicked.connect(self.RemoveImage)
        self.dlg.btn_isodata.clicked.connect(self.Classify_ISODATA)
        self.dlg.P_NDVI.clicked.connect(self.myPNDVI)
        self.dlg.P_RVI.clicked.connect(self.myPRVI)
        self.dlg.W_NDVI.clicked.connect(self.myWNDVI)
        self.dlg.W_R.clicked.connect(self.myWR)
        self.dlg.btn_cloud.clicked.connect(self.myCloud)
        self.dlg.btn_Merge.clicked.connect(self.fusion)
        self.dlg.btn_Mosaic.clicked.connect(self.mosaic)


    def LoadImages(self):
        filePaths = QFileDialog.getOpenFileNames(self.dlg, self.tr(u'Load Images'), os.path.dirname(__file__), self.tr(u'tiff(*.tif);;bmp(*.bmp);;img(*.img)'))

        self.isLoadImages = True
        for filePath in filePaths:
            self.imageList.append(filePath)
            fileName = filePath.split('.')[0].split('\\')[-1]
            self.imgDict[fileName] = filePath
            self.dlg.comboBox.addItem(fileName)
            self.dlg.comboBox_TM3.addItem(fileName)
            self.dlg.comboBox_TM4.addItem(fileName)
            self.dlg.comboBox_TM3_2.addItem(fileName)
            self.dlg.comboBox_TM4_2.addItem(fileName)
            self.dlg.comboBox_TM3_3.addItem(fileName)
            self.dlg.comboBox_TM4_3.addItem(fileName)
            self.dlg.comboBox_gray.addItem(fileName)
            self.dlg.comboBox_rgb.addItem(fileName)
            self.dlg.comboBox_left.addItem(fileName)
            self.dlg.comboBox_right.addItem(fileName)
        
        curImg = QPixmap(self.imageList[0])
        self.dlg.img_display.setPixmap(curImg)

        curImg = cv2.imread(self.imageList[0], 0)
        plt.clf()
        plt.hist(curImg.ravel(), 256, [0, 256])
        root_dir = "C:\\2016302590109\\"
        curImgName = self.dlg.comboBox.currentText()
        hist_path = root_dir + curImgName + "_hist.tif"
        plt.savefig(hist_path)
        histImg = QPixmap(hist_path).scaled(self.dlg.hist_display.width(), self.dlg.hist_display.height())
        self.dlg.hist_display.clear()
        self.dlg.hist_display.setPixmap(histImg)


    def LoadChosenImage(self):
        if self.isLoadImages:
            self.isLoadImages = False
            return
        fileName = self.dlg.comboBox.currentText()
        curImg = QPixmap(self.imgDict[fileName])
        self.dlg.img_display.clear()
        self.dlg.img_display.setPixmap(curImg)

        curImg = cv2.imread(self.imgDict[fileName], 0)
        plt.clf()
        plt.hist(curImg.ravel(), 256, [0, 256])
        root_dir = "C:\\2016302590109\\"
        curImgName = self.dlg.comboBox.currentText()
        hist_path = root_dir + curImgName + "_hist.tif"
        plt.savefig(hist_path)
        histImg = QPixmap(hist_path).scaled(self.dlg.hist_display.width(), self.dlg.hist_display.height())
        self.dlg.hist_display.clear()
        self.dlg.hist_display.setPixmap(histImg)


    def Classify_KMeans(self):
        my_kmeans = KMeans()
        my_kmeans.imgPaths = self.imageList
        my_kmeans.imgNums = len(self.imageList)
        my_kmeans.centerNums = self.dlg.sBox_cNum.value()
        result_path, result_name = my_kmeans.cluster()
        self.dlg.comboBox.addItem(result_name)
        self.imgDict[result_name] = result_path
        self.dlg.img_display.clear()
        curImg = QPixmap(result_path)
        self.dlg.img_display.setPixmap(curImg)


    def RemoveImage(self):
        file_index = self.dlg.comboBox.currentIndex()
        fileName = self.dlg.comboBox.currentText()
        self.imgDict.pop(fileName)
        self.dlg.comboBox.removeItem(file_index)
        self.dlg.comboBox.setCurrentIndex(0)
        fileName = self.dlg.comboBox.currentText()
        curImg = QPixmap(self.imgDict[fileName])
        self.dlg.img_display.clear()
        self.dlg.img_display.setPixmap(curImg)


    def Classify_ISODATA(self):
        K = int(self.dlg.L.text())
        L = int(self.dlg.L.text())
        I = int(self.dlg.I.text())
        TN = int(self.dlg.TN.text())
        TS = float(self.dlg.TS.text())
        TC = int(self.dlg.TC.text())
        inputFileName = self.dlg.comboBox.currentText()
        inputFilePath = self.imgDict[inputFileName]
        image = Image.open(inputFilePath)
        my_isodata = ISODATA()
        result_path = my_isodata.doISODATARGB(image, K, TN, TS, TC, L, I)
        result_name = result_path.split('.')[0].split('\\')[-1]
        self.dlg.comboBox.addItem(result_name)
        self.imgDict[result_name] = result_path
        self.dlg.img_display.clear()
        curImg = QPixmap(result_path)
        self.dlg.img_display.setPixmap(curImg)

    def myPNDVI(self):
        p_NIR = self.dlg.comboBox_TM4.currentText()
        p_NIR_PATH = self.imgDict[p_NIR]
        p_inputNIR = cv2.imread(p_NIR_PATH, 0)
        p_Red = self.dlg.comboBox_TM3.currentText()
        p_Red_PATH = self.imgDict[p_Red]
        p_inputRed = cv2.imread(p_Red_PATH, 0)

        img = cv2.imread("C:\\2016302590109\\result.tif")
        b = cv2.split(img)[0]
        g = cv2.split(img)[1]
        r = cv2.split(img)[2]

        height = img.shape[0]
        width = img.shape[1]
        P_size = height * width * 2
        self.dlg.P_progress.setMinimum(0)
        self.dlg.P_progress.setMaximum(P_size)

        p_ndvi = np.zeros([height, width], dtype=np.float32)
        P_count = 0

        for i in range(height):
            for j in range(width):
                if float(p_inputNIR[i][j]) + float(p_inputRed[i][j]) == 0:
                    p_ndvi[i][j] = -1.0
                else:
                    p_ndvi[i][j] = float((float(p_inputNIR[i][j]) - float(p_inputRed[i][j])) / (float(p_inputRed[i][j]) + float(p_inputNIR[i][j])))
                P_count += 1
                self.dlg.P_progress.setValue(P_count)

        for i in range(height):
            for j in range(width):
                if p_ndvi[i][j] > float(self.dlg.P_1.text()):
                    b[i][j] = 0
                    g[i][j] = 0
                    r[i][j] = 255
                P_count += 1
                self.dlg.P_progress.setValue(P_count)

        result = cv2.merge([b, g, r])
        result_path = "C:\\2016302590109\\P_NDVI.tif"
        result_name = result_path.split('.')[0].split('\\')[-1]
        cv2.imwrite(result_path, result)
        self.imgDict[result_name] = result_path
        self.dlg.comboBox.addItem(result_name)
        curImg = QPixmap(result_path)
        self.dlg.img_display.clear()
        self.dlg.img_display.setPixmap(curImg)


    def myPRVI(self):
        NIR = self.dlg.comboBox_TM4.currentText()
        NIR_PATH = self.imgDict[NIR]
        inputNIR = cv2.imread(NIR_PATH, 0)
        Red = self.dlg.comboBox_TM3.currentText()
        Red_PATH = self.imgDict[Red]
        inputRed = cv2.imread(Red_PATH, 0)

        img = cv2.imread("C:\\2016302590109\\result.tif")
        b = cv2.split(img)[0]
        g = cv2.split(img)[1]
        r = cv2.split(img)[2]

        height = img.shape[0]
        width = img.shape[1]
        P_size = height * width * 2
        self.dlg.P_progress.setMinimum(0)
        self.dlg.P_progress.setMaximum(P_size)

        ndvi = np.zeros([height, width], dtype=np.float32)
        rvi = np.zeros([height, width], dtype=np.float32)
        P_count1 = 0

        for i in range(height):
            for j in range(width):
                if float(inputNIR[i][j]) + float(inputRed[i][j]) == 0 or float(inputRed[i][j]) == 0:
                    ndvi[i][j] = -1.0
                    rvi[i][j] = -1.0
                else:
                    ndvi[i][j] = float((float(inputNIR[i][j]) - float(inputRed[i][j])) / (float(inputRed[i][j]) + float(inputNIR[i][j])))
                    rvi[i][j] = float(float(inputNIR[i][j]) / float(inputRed[i][j])) 
                P_count1 += 1
                self.dlg.P_progress.setValue(P_count1)

        for i in range(height):
            for j in range(width):
                if ndvi[i][j] > float(self.dlg.P_1.text()) and rvi[i][j] > float(self.dlg.P_2.text()):
                    b[i][j] = 0
                    g[i][j] = 0
                    r[i][j] = 255
                P_count1 += 1
                self.dlg.P_progress.setValue(P_count1)

        result = cv2.merge([b, g, r])
        result_path = "C:\\2016302590109\\P_RVI.tif"
        result_name = result_path.split('.')[0].split('\\')[-1]
        cv2.imwrite(result_path, result)
        self.imgDict[result_name] = result_path
        curImg = QPixmap(result_path)
        self.dlg.comboBox.addItem(result_name)
        self.dlg.img_display.clear()
        self.dlg.img_display.setPixmap(curImg)


    def myWNDVI(self):
        NIR = self.dlg.comboBox_TM4_2.currentText()
        NIR_PATH = self.imgDict[NIR]
        inputNIR = cv2.imread(NIR_PATH, 0)
        Red = self.dlg.comboBox_TM3_2.currentText()
        Red_PATH = self.imgDict[Red]
        inputRed = cv2.imread(Red_PATH, 0)

        img = cv2.imread("C:\\2016302590109\\result.tif")
        b = cv2.split(img)[0]
        g = cv2.split(img)[1]
        r = cv2.split(img)[2]

        height = img.shape[0]
        width = img.shape[1]
        size = height * width * 2
        self.dlg.W_progress.setMinimum(0)
        self.dlg.W_progress.setMaximum(size)
        w_ndvi = np.zeros([height, width], dtype=np.float32)

        W_count = 0
        for i in range(height):
            for j in range(width):
                if float(inputNIR[i][j]) + float(inputRed[i][j]) == 0:
                    w_ndvi[i][j] = -1.0
                else:
                    w_ndvi[i][j] = float((float(inputNIR[i][j]) - float(inputRed[i][j])) / (float(inputRed[i][j]) + float(inputNIR[i][j])))
                W_count += 1
                self.dlg.W_progress.setValue(W_count)

        for i in range(height):
            for j in range(width):
                if w_ndvi[i][j] < float(self.dlg.W_1.text()):
                    b[i][j] = 0
                    g[i][j] = 0
                    r[i][j] = 255
                W_count += 1
                self.dlg.W_progress.setValue(W_count)

        result = cv2.merge([b, g, r])
        result_path = "C:\\2016302590109\\W_NDVI.tif"
        result_name = result_path.split('.')[0].split('\\')[-1]
        cv2.imwrite(result_path, result)
        self.imgDict[result_name] = result_path
        self.dlg.comboBox.addItem(result_name)
        curImg = QPixmap(result_path)
        self.dlg.img_display.clear()
        self.dlg.img_display.setPixmap(curImg)


    def myWR(self):
        NIR = self.dlg.comboBox_TM4_2.currentText()
        NIR_PATH = self.imgDict[NIR]
        inputNIR = cv2.imread(NIR_PATH, 0)
        Red = self.dlg.comboBox_TM3_2.currentText()
        Red_PATH = self.imgDict[Red]
        inputRed = cv2.imread(Red_PATH, 0)

        img = cv2.imread("C:\\2016302590109\\result.tif")
        b = cv2.split(img)[0]
        g = cv2.split(img)[1]
        r = cv2.split(img)[2]

        height = img.shape[0]
        width = img.shape[1]
        size = height * width * 2
        self.dlg.W_progress.setMinimum(0)
        self.dlg.W_progress.setMaximum(size)

        wr_ndvi = np.zeros([height, width], dtype=np.float32)
        wr_rvi = np.zeros([height, width], dtype=np.float32)
        W_count_1 = 0

        for i in range(height):
            for j in range(width):
                if float(inputNIR[i][j]) + float(inputRed[i][j]) == 0 or float(inputRed[i][j]) == 0:
                    wr_ndvi[i][j] = -1.0
                    wr_rvi[i][j] = -1.0
                else:
                    wr_ndvi[i][j] = float((float(inputNIR[i][j]) - float(inputRed[i][j])) / (float(inputRed[i][j]) + float(inputNIR[i][j])))
                    wr_rvi[i][j] = float(inputNIR[i][j])
                W_count_1 += 1
                self.dlg.W_progress.setValue(W_count_1)

        for i in range(height):
            for j in range(width):
                if wr_ndvi[i][j] < float(self.dlg.W_1.text()) and wr_rvi[i][j] < float(self.dlg.W_2.text()):#需要引入的参数
                    b[i][j] = 0
                    g[i][j] = 0
                    r[i][j] = 255
                W_count_1 += 1
                self.dlg.W_progress.setValue(W_count_1)

        result = cv2.merge([b, g, r])#result输出img
        result_path = "C:\\2016302590109\\W_IR.tif"
        result_name = result_path.split('.')[0].split('\\')[-1]
        cv2.imwrite(result_path, result)
        self.imgDict[result_name] = result_path
        curImg = QPixmap(result_path)
        self.dlg.comboBox.addItem(result_name)
        self.dlg.img_display.clear()
        self.dlg.img_display.setPixmap(curImg)


    def myCloud(self):
        NIR = self.dlg.comboBox_TM4_3.currentText()
        NIR_PATH = self.imgDict[NIR]
        inputNIR = cv2.imread(NIR_PATH, 0)
        Red = self.dlg.comboBox_TM3_3.currentText()
        Red_PATH = self.imgDict[Red]
        inputRed = cv2.imread(Red_PATH, 0)

        img = cv2.imread("C:\\2016302590109\\cloud_result.tif")
        b = cv2.split(img)[0]
        g = cv2.split(img)[1]
        r = cv2.split(img)[2]

        height = img.shape[0]
        width = img.shape[1]
        size = height * width
        self.dlg.C_progress.setMinimum(0)
        self.dlg.C_progress.setMaximum(size)
        C_count = 0

        for i in range(height):
            for j in range(width):
                if int(inputRed[i][j]) > 245 and int(inputNIR[i][j]) > 245:
                    b[i][j] = 0
                    g[i][j] = 0
                    r[i][j] = 255
                C_count += 1
                self.dlg.C_progress.setValue(C_count)

        result = cv2.merge([b, g, r])
        result_path = "C:\\2016302590109\\Cloud.tif"
        result_name = result_path.split('.')[0].split('\\')[-1]
        cv2.imwrite(result_path, result)
        self.imgDict[result_name] = result_path
        curImg = QPixmap(result_path)
        self.dlg.comboBox.addItem(result_name)
        self.dlg.img_display.clear()
        self.dlg.img_display.setPixmap(curImg)


    def fusion(self):
        GRAY = self.dlg.comboBox_gray.currentText()
        GRAY_PATH = self.imgDict[GRAY]
        inputGRAY = cv2.imread(GRAY_PATH, 1)
        RGB = self.dlg.comboBox_rgb.currentText()
        RGB_PATH = self.imgDict[RGB]
        inputRGB = cv2.imread(RGB_PATH, 1)

        grayimg = inputGRAY * 1
        width = inputGRAY.shape[0]
        height = inputGRAY.shape[1]
        size = height * width * 2
        self.dlg.M_progress.setMinimum(0)
        self.dlg.M_progress.setMaximum(size)

        M_count = 0
        HSVimg = cv2.cvtColor(inputRGB, cv2.COLOR_BGR2HSV)
        for i in range(width):
            for j in range(height):
                grayimg[i, j] = 0.299 * inputGRAY[i, j, 0] + 0.587 * inputGRAY[i, j, 1] + 0.114 * inputGRAY[i, j, 2]
                M_count += 1
                self.dlg.M_progress.setValue(M_count)

        H, S, V = cv2.split(HSVimg)
        rows, cols = V.shape
        for i in range(rows):
            for j in range(cols):
                V[i, j] = grayimg[i][j][0]
                M_count += 1
                self.dlg.M_progress.setValue(M_count)

        HSgray = cv2.merge([H, S, V])
        HSgray = np.uint8(HSgray)

        RGBimg = cv2.cvtColor(HSgray, cv2.COLOR_HSV2BGR)

        result_path = "C:\\2016302590109\\Merge.tif"
        result_name = result_path.split('.')[0].split('\\')[-1]
        cv2.imwrite(result_path, RGBimg)
        self.imgDict[result_name] = result_path
        curImg = QPixmap(result_path)
        self.dlg.comboBox.addItem(result_name)
        self.dlg.img_display.clear()
        self.dlg.img_display.setPixmap(curImg)
        cv2.imshow("result", RGBimg)
        cv2.waitKey(0)


    def mosaic(self):
        left = self.dlg.comboBox_left.currentText()
        LEFT_PATH = self.imgDict[left]
        inputLEFT = cv2.imread(LEFT_PATH)
        right = self.dlg.comboBox_right.currentText()
        RIGHT_PATH = self.imgDict[right]
        inputRIGHT = cv2.imread(RIGHT_PATH)

        result_path = MyMosaic.Func(inputLEFT, inputRIGHT)
        mosaic_img = cv2.imread(result_path)
        cv2.imshow("result", mosaic_img)
        cv2.waitKey(0)


    def unload(self):
        """Removes the plugin menu item and icon from QGIS GUI."""
        for action in self.actions:
            self.iface.removePluginMenu(self.tr(u'&Image Process'), action)
            self.iface.removeToolBarIcon(action)
        # remove the toolbar
        del self.toolbar


    def run(self):
        """Run method that performs all the real work"""
        # show the dialog
        self.dlg.show()
        # Run the dialog event loop
        result = self.dlg.exec_()
        # See if OK was pressed
        if result:
            # Do something useful here - delete the line containing pass and
            # substitute with your code.
            pass
