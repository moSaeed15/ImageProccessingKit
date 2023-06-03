from fileinput import filename
from logging import error
import string
from typing import Counter
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.uic import  loadUiType
from PyQt5 import QtCore,QtWidgets,QtGui
from cv2 import normalize
import matplotlib.pyplot as plt
from PIL import Image
from pandas import array
from pydicom import dcmread
import os
from os import path 
import sys
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import math
import numpy as np
from sympy import factor
from collections import Counter
import random
import matplotlib.pyplot as plt
import cv2
import statistics
from skimage.data import shepp_logan_phantom
from skimage.transform import radon, rescale
from skimage.transform import iradon

FORM_CLASS,_= loadUiType(path.join(path.dirname(__file__),"untitled.ui"))

def partition(array, low, high):

        # Choose the rightmost element
        pivot = array[high]
    
        # pointer for greater element
        i = low - 1
    
        # traverse through all elements
        # compare each element with pivot
        for j in range(low, high):
            
            if array[j] <= pivot:
    
                # If element smaller than pivot is found
                # swap it with the greater element pointed by i
                i = i + 1
    
                # Swapping element at i with element at j
                (array[i], array[j]) = (array[j], array[i])
    
        # Swap the pivot element with the greater element specified by i
        (array[i + 1], array[high]) = (array[high], array[i + 1])
    
        # Return the position from where partition is done
        return i + 1

def quicksort(array, low, high):
        if low < high:
 
            # Find pivot element such that
            # element smaller than pivot are on the left
            # element greater than pivot are on the right
            pi =partition(array, low, high)
    
            # Recursive call on the left of pivot
            quicksort(array, low, pi - 1)
    
            # Recursive call on the right of pivot
            quicksort(array, pi + 1, high)


class MplCanvas(FigureCanvas):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)

class MainApp(QMainWindow , FORM_CLASS):
    def __init__(self, parent=None):
        super(MainApp,self).__init__(parent)
        QMainWindow.__init__(self)
        self.setupUi(self)
        self.create_phantom()
        self.BrowseButton.triggered.connect(self.Browse)
        self.lineEdit.returnPressed.connect(self.tabTwo)
        self.lineEdit_2.returnPressed.connect(self.tabFour)
        self.lineEdit_2.returnPressed.connect(self.tabSeven)
        self.lineEdit_5.returnPressed.connect(self.tabFour)
        self.pushButton.clicked.connect(self.tabFive)
        self.guassianButton.clicked.connect(self.add_gaussian_noise)
        self.uniformButton.clicked.connect(self.add_uniform_noise)

        # Creting the figure and adding it to the GUI
        self.figure = plt.figure(figsize=(15,5))
        self.Canvas = FigureCanvas(self.figure)
        self.gridLayout.addWidget(self.Canvas,0, 0, 1, 1)
        self.zoomFactor=0 
        self.label_36.hide()
        self.label_35.hide()
        self.fileName = self.ext=''
        # self.tabNine()
        # self.tabEight()
        self.tabTen()


    def Browse(self):
      self.clearData()

      self.fileName ,self.ext=QFileDialog.getOpenFileName(self, 'Open image file', QtCore.QDir.rootPath(),'JPG files(*.jpg);;JPEG files(*.jpeg);;DICOM files(*.dcm);;PNG files(*.png);;BMP files(*.bmp)')      

      if(self.ext=='DICOM files(*.dcm)'):
        try:    
            # Read and parse a DICOM dataset 
            self.ds = dcmread(self.fileName)
            self.dicomArray=self.ds.pixel_array
  
           #Display data as an image
            plt.imshow( self.dicomArray, cmap=plt.cm.bone)
            self.Canvas.draw()
            self.showDicomMetaData()
        except:
            self.handleError('DICOM file missing meta data')
      else:
            try:
                # reading the image into an array
                image = plt.imread(self.fileName)
                plt.imshow(image)
                self.Canvas.draw()
                # self.tabTwo()

                self.showMetaData()
            except:
                self.handleError('The file is Corrupt ')
              
            # self.thirdTab() 
            self.tabSix() 

    def showDicomMetaData(self):
        bitDepth=self.getDicomBitDepth(self.ds.pixel_array);
        size=self.ds.Columns*self.ds.Rows*bitDepth
        self.textEdit.setPlainText(f'{self.ds.Columns}')
        self.textEdit_2.setPlainText(f'{self.ds.Rows}')
        self.textEdit_3.setPlainText(f'{size}')
        self.textEdit_4.setPlainText(f'{bitDepth}')
        self.textEdit_5.setPlainText(f'{self.ds.PhotometricInterpretation}')
        self.textEdit_6.setPlainText(f'{self.ds.Modality}')
        self.textEdit_7.setPlainText(f'{self.ds.PatientName}')
        self.textEdit_8.setPlainText(f'{self.ds.StudyDescription}')
    
    def showMetaData(self):
        # returns an object of the image that can be read
        image=Image.open(self.fileName)
        bitDepth=self.getBitDepth(image);
        size=image.width*image.height*bitDepth
        self.textEdit.setPlainText(f'{image.width}')
        self.textEdit_2.setPlainText(f'{image.height}')
        self.textEdit_3.setPlainText(f'{size}')
        self.textEdit_4.setPlainText(f'{self.getBitDepth(image)}')
    
    # To clear the text boxex and the figure
    def clearData(self):
        self.textEdit.setPlainText('')
        self.textEdit_2.setPlainText('')
        self.textEdit_3.setPlainText('')
        self.textEdit_4.setPlainText('')
        self.textEdit_5.setPlainText('')
        self.textEdit_6.setPlainText('')
        self.textEdit_7.setPlainText('')
        self.textEdit_8.setPlainText('')
        self.label_36.hide()
        self.label_35.hide()
        self.figure = plt.figure(figsize=(15,5))
        self.Canvas = FigureCanvas(self.figure)
        self.gridLayout.addWidget(self.Canvas,0, 0, 1, 1)

    def getDicomBitDepth(self,arr):
        if self.ds.PhotometricInterpretation=='RGB':
            factor=3
        else:
            factor=1
        bitDepth=math.ceil(math.log2(arr.max()-arr.min()+1))*factor
        return bitDepth
    
    def getBitDepth(self,img):
        FactorDic = {'1':1, 'L':1, 'P':1, 'RGB':3, 'RGBA':4, 'CMYK':4, 'YCbCr':3}
        bitDepth = math.ceil(math.log2(np.array(img).max()-np.array(img).min()+1))*FactorDic[img.mode]
        return bitDepth

    def tabTwo(self):
        
        if self.fileName=='' :
            self.handleError('No file opened. Select a file first')
            return
        
        try:
            self.zoomFactor=float(self.lineEdit.text())
            if self.zoomFactor<=0 or type(self.zoomFactor)==string:
                self.handleError('Zoom factor needs to be a number greater than 0')
                return
        except:
            self.handleError('Zoom factor needs to be a number.Please enter a number')
            return


        if(self.ext!='DICOM files(*.dcm)'):
            self.label_36.hide()
            self.label_35.hide()
            orignalImage = Image.open(self.fileName)

            # coverting the image to grayscale
            if orignalImage.mode=='RGB':
                GrayscaleImage= orignalImage.convert('L')
                self.normalArray=np.asarray(GrayscaleImage)

            nearestNeighbourImage=self.nearestNeigbourInterpolation(self.normalArray,GrayscaleImage.size)
                   
            data = Image.fromarray(nearestNeighbourImage)
            NNImage = data.toqpixmap()
            qpixmap=QPixmap(NNImage)
            self.nearestLabel.setPixmap(qpixmap) 

            bilinearImage=self.bilinearInterpolation(self.normalArray)
            data = Image.fromarray(bilinearImage)
            bilinearImage = data.toqpixmap()
            bilinearpixmap=QPixmap(bilinearImage)
            self.bilinearLabel.setPixmap(bilinearpixmap) 
        else:
            # dicomImage = self.ds.convert('L')
            nearestNeighbourImage=self.nearestNeigbourInterpolation(self.dicomArray,(self.ds.Columns,self.ds.Rows))
            data = Image.fromarray(nearestNeighbourImage)
            NNImage = data.toqpixmap()
            qpixmap=QPixmap(NNImage)
            self.nearestLabel.setPixmap(qpixmap)

            bilinearImage=self.bilinearInterpolation(self.dicomArray)
            data = Image.fromarray(bilinearImage)
            bilinearImage = data.toqpixmap()
            bilinearpixmap=QPixmap(bilinearImage)
            self.bilinearLabel.setPixmap(bilinearpixmap)  
    

    def thirdTab(self):

        EqualisedImage,normalizedImageArray=self.HistogramNormalisatonAndEqualisation()
        data = Image.fromarray(self.imageArray)
        orignalImage = data.toqpixmap()
        qpixmap=QPixmap(orignalImage)
        self.orignalImageLabel.setPixmap(qpixmap)

        data2 = Image.fromarray(EqualisedImage)
        equalisedImage = data2.toqpixmap()
        qpixmap=QPixmap(equalisedImage)
        self.equalizedImageLabel.setPixmap(qpixmap)
        

        # sc = MplCanvas(self, width=5, height=4, dpi=100)
        # sc.axes.clear()

        # self.gridLayout2.addWidget(sc,0, 0, 1, 1)

        # sc.axes.bar(range(256),normalizedImageArray)

        equalizedNormalisedArray = np.bincount(EqualisedImage.flatten(), minlength=256)

        # Normalise
        numPixels = np.sum(equalizedNormalisedArray)
        equalizedNormalisedArray = equalizedNormalisedArray/numPixels


    def HistogramNormalisatonAndEqualisation(self): 

        orignalImage = Image.open(self.fileName)
        self.imageArray=0
        if orignalImage.mode=='RGB':
                GrayscaleImage= orignalImage.convert('L')
                self.imageArray=np.asarray(GrayscaleImage)
        else:
            self.imageArray=np.asarray(orignalImage)

        #flatten the image array and calculate the number of histogram bins
        histogramArray = np.bincount(self.imageArray.flatten(), minlength=256)

        # Normalise
        numPixels = np.sum(histogramArray)
        histogramArray = histogramArray/numPixels

        # Normalised Cumalitve Array
        cHistogramArray=np.cumsum(histogramArray)

        # normalizedArray= self.imageArray-min(self.imageArray)/range of self.imageArray

        transformMap = np.floor(255 * cHistogramArray).astype(np.uint8)

        flattendImageList= list(self.imageArray.flatten())

        # transform the pixel values to the equalized pixel values

        equalizedImageList = [transformMap[i] for i in flattendImageList]

        equalizedImageArray = np.reshape(np.asarray(equalizedImageList), self.imageArray.shape)

        return equalizedImageArray,histogramArray

    def nearestNeigbourInterpolation(self,arr,size):
        
        newX=int(size[0]*self.zoomFactor)
        newY=int(size[1]*self.zoomFactor)

        # Calculation row and column Ratio
        rowRatio,columnRatio=np.array(newX)/np.array(size),np.array(newY)/np.array(size)
        
        # Filling the rows and columns array
        rowPostion = (np.ceil(range(1,1+int(size[0]*rowRatio[0]))/rowRatio[0]) - 1).astype(int)

        columnPostion = (np.ceil(range(1,1+int(size[1]*columnRatio[1]))/columnRatio[1]) - 1).astype(int)

        # Filling the New image array with
        newImage=arr[:,rowPostion][columnPostion,:]
     
        return newImage.astype('uint8')


    def bilinearInterpolation(self,arr):
        height,width=arr.shape[:2]
        print(height,width)
        newHeight=int(height*self.zoomFactor)
        newWidth=int(width*self.zoomFactor)
        print(newHeight,newWidth)

        newImage=np.zeros([newHeight,newWidth])

        widthScale=width/newWidth
        heightScale=height/newWidth

        for i in range(newHeight):
            for j in range(newWidth):
                #mapping new coordinates from the original image

                x = i * heightScale
                y = j * widthScale

                # Calculate the coordinates of 4 Surronding pixels
                floorX = math.floor(x)
                ceilX = min( height - 1, math.ceil(x))
                floorY = math.floor(y)
                ceilY = min(width - 1, math.ceil(y))
                
                if (ceilX == floorX) and (ceilY == floorY):
                    LinearInterpolationResult=arr[int(x),int(y)]   
                elif (ceilX == floorX):
                    linearInterpolation1 = arr[int(x), int(floorY)]
                    linearInterpolation2 = arr[int(x), int(ceilY)]
                    LinearInterpolationResult = linearInterpolation1 * (ceilY - y) + linearInterpolation2 * (y - floorY)

                elif (ceilY == floorY):
                    linearInterpolation1 = arr[int(floorX), int(y)]
                    linearInterpolation2 = arr[int(ceilX), int(y)]
                    LinearInterpolationResult = (linearInterpolation1 * (ceilX - x)) + (linearInterpolation2* (x - floorX))
                else:    
                    # Get pixel value form orignal array
                    value1=arr[floorX,floorY]
                    value2=arr[ceilX,floorY]
                    value3=arr[floorX,ceilY]
                    value4=arr[ceilX,ceilY]

                    #calc the result of linear interpolation 
                    linearInterpolation1=value1*(ceilX-x)+value2*(x-floorX)
                    linearInterpolation2=value3*(ceilX-x)+value4*(x-floorX)
                    LinearInterpolationResult=linearInterpolation1*(ceilY-y)+linearInterpolation2*(y-floorY)

                newImage[i,j]=LinearInterpolationResult
        return newImage.astype('uint8')


    def tabFour(self):
        self.label_36.hide()
        self.label_35.hide()
        multiFactor=int(self.lineEdit_5.text())
        kernelSize=int(self.lineEdit_2.text())
        
        if multiFactor<=0:
            self.handleError('Insert a Factor number greater than 0') 
            return
        if kernelSize<=1:
            self.handleError('Insert a Kernel size greater than 1') 
            return

        # intializing the box array
        boxArray=np.ones((kernelSize,kernelSize)) 
        

        normalizedKernal=boxArray*(1/np.size(boxArray))
        try:
            orignalImage = Image.open(self.fileName)
        except:
            self.handleError('Browse for a file before applying the filter') 
            return

        GrayscaleImage= orignalImage.convert('L')
        imageArray=np.asarray(GrayscaleImage)

        # Calculating padding size
        padding=np.floor(kernelSize/2)

        convolvedArray=self.convolveImage(imageArray,normalizedKernal,int(padding))

        # Subtracting the blurred image from the orignal and multiplying it by user define factor
        subtractionArray=(imageArray-convolvedArray)*multiFactor

        enhancedImage=subtractionArray+imageArray
        
        enhancedImage=self.scaleImage(enhancedImage).astype('uint8')
        self.spatialOutput=self.scaleImage(convolvedArray)
        data = Image.fromarray(enhancedImage)
        imageDisplay = data.toqpixmap()
        qpixmap=QPixmap(imageDisplay)
        self.orignalImageLabel_2.setPixmap(qpixmap) 

        data = Image.fromarray(imageArray)
        imageDisplay = data.toqpixmap()
        qpixmap=QPixmap(imageDisplay)
        self.orignalImageLabel_6.setPixmap(qpixmap) 

    def convolveImage(self,image,kernel,padding):
        # define the number of strides taken each step of the convultion
        strides=1

        # Cross correlation
        flippedKernel=np.flipud(np.fliplr(kernel))
        
        # Calculating the shapes of the kernel and the image and the padding
        kernalXShape=flippedKernel.shape[0]
        kernalYShape=flippedKernel.shape[1]
        imageXShape=image.shape[0]
        imageYShape=image.shape[1]

        # Shape of the output array
        if kernalXShape%2!=0:
            outputX=int(((imageXShape-kernalXShape+2*padding)/strides)+1) 
            outputY=int(((imageYShape-kernalYShape+2*padding)/strides)+1) 
        else:
            outputX=int(((imageXShape-kernalXShape+2*padding)/strides)) 
            outputY=int(((imageYShape-kernalYShape+2*padding)/strides)) 

        output=np.zeros((outputX,outputY))

        # Apply padding on each side

        imagePadded=np.zeros((imageXShape+padding*2,imageYShape+padding*2))

        imagePadded[int(padding):int(-1 * padding), int(padding):int(-1 * padding)] = image

        for y in range(image.shape[1]):
            #exit convultion
            if y>image.shape[1]-kernalYShape:
                break
            if y % strides == 0:
                for x in range(image.shape[0]):
                    if x > image.shape[0]-kernalXShape:
                        break
                    try:
                        if x% strides==0:
                            output[x,y]=(kernel*imagePadded[x: x + kernalXShape, y: y + kernalYShape]).sum()
                    except:
                        break

        return output

    def scaleImage(self,image):
        for i in range (image.shape[1]):
            for j in range(image.shape[0]):
                if image[j,i]>255:
                    image[j,i]=255
                elif image[j,i]<0:
                    image[j,i]=0

        xImgShape, yImgShape = image.shape
        if (yImgShape % 2 == 0):
            image = np.vstack((image, image[xImgShape-1 , :]))
        if (xImgShape % 2 == 0):
            image = np.c_[image, image[:, yImgShape-1 ]]  
        return image


    def tabFive(self):

        try:
            orignalImage = Image.open(self.fileName)
        except:
            self.handleError('Browse for a file before applying the filter') 
            return
            
         
        GrayscaleImage= orignalImage.convert('L')
        imageArray=np.asarray(GrayscaleImage)

        noisyImage=self.addSaltAndPepperNoise(imageArray)

        data = Image.fromarray(noisyImage)
        imageDisplay = data.toqpixmap()
        qpixmap=QPixmap(imageDisplay)
        self.orignalImageLabel_7.setPixmap(qpixmap) 
        
        denoisedImage=self.medianFilter(noisyImage)

        data = Image.fromarray(denoisedImage)
        imageDisplay = data.toqpixmap()
        qpixmap=QPixmap(imageDisplay)
        self.orignalImageLabel_8.setPixmap(qpixmap) 


    def addSaltAndPepperNoise(self,image):
        row,col=image.shape

        # Pick a random number between 300 and 10000
        pixelNumber=random.randint(300, 10000)

        for i in range(pixelNumber):
    
            # Pick a random y coordinate
            yCoord=random.randint(0, row - 1)
         
            # Pick a random x coordinate
            xCoord=random.randint(0, col - 1)
         
            # Color that pixel to white
            image[yCoord][xCoord] = 255

        # For black random pixel
        for i in range(pixelNumber):
    
            # Pick a random y coordinate
            yCoord=random.randint(0, row - 1)
         
            # Pick a random x coordinate
            xCoord=random.randint(0, col - 1)
         
            # Color that pixel to black
            image[yCoord][xCoord] = 0
        return image

    def medianFilter(self,data):
        # how large is the filter per each iteration
        filter_size=3

        temp = []
        indexer = filter_size // 2

        # Final data array
        data_final = []
        data_final = np.zeros((len(data),len(data[0])))

        for i in range(len(data)):

            for j in range(len(data[0])):

                for z in range(filter_size):
                    if i + z - indexer < 0 or i + z - indexer > len(data) - 1:
                        for c in range(filter_size):
                            temp.append(0)
                    else:
                        if j + z - indexer < 0 or j + indexer > len(data[0]) - 1:
                            temp.append(0)
                        else:
                            for k in range(filter_size):
                                temp.append(data[i + z - indexer][j + k - indexer])
                # Sorting using quicksort
                quicksort(temp,0,len(temp)-1)

                data_final[i][j] = temp[len(temp) // 2]
                temp = []
        return data_final.astype('uint8')

    def tabSix(self):

        try:
            orignalImage = Image.open(self.fileName)
        except:
            self.handleError('Browse for a file before applying the filter') 
            return

        GrayscaleImage= orignalImage.convert('L')
        imageArray=np.asarray(GrayscaleImage)

        data = Image.fromarray(imageArray)
        imageDisplay = data.toqpixmap()
        qpixmap=QPixmap(imageDisplay)
        self.label_11.setPixmap(qpixmap) 


        imageFourier = np.fft.fftshift(np.fft.fft2(imageArray))

        magnitude=np.sqrt((imageFourier.real ** 2)+(imageFourier.imag ** 2))
        
        phase=np.arctan2(imageFourier.imag,imageFourier.real)
        
        self.drawCanvas(magnitude,self.gridLayout_6)

        self.drawCanvas(phase,self.gridLayout_5)
        


        # Fourier log
        fourierLog=np.log((abs(imageFourier)))

        magnitudeLog=np.log(magnitude+1)

        phaseLog=np.log(phase+2*math.pi)

        self.drawCanvas(magnitudeLog,self.gridLayout_7)

        self.drawCanvas(phaseLog,self.gridLayout_4)

    def tabSeven(self):
        kernelSize=int(self.lineEdit_2.text())

        if kernelSize<=0 or kernelSize%2==0:
            self.handleError('Please write an odd number greater than 0')
            return


        boxArray=np.ones((kernelSize,kernelSize)) 
        
        normalizedKernal=boxArray*(1/np.size(boxArray))

        try:
            orignalImage = Image.open(self.fileName)
        except:
            self.handleError('Browse for a file before applying the filter') 
            return
            
        kernel = np.ones((kernelSize, kernelSize))

        kernel = kernel  / (kernelSize*kernelSize)
                
        GrayscaleImage= orignalImage.convert('L')
        imageArray=np.asarray(GrayscaleImage)

        xImgShape, yImgShape = imageArray.shape

        if (yImgShape % 2 == 0):
            imageArray = np.vstack((imageArray, imageArray[xImgShape-1 , :]))
        if (xImgShape % 2 == 0):
            imageArray = np.c_[imageArray, imageArray[:, yImgShape-1 ]]

        xOutput, yOutput = imageArray.shape
        xKernShape = kernel.shape[0]
        yKernShape = kernel.shape[1]

        kernelPadded = np.zeros((xOutput, yOutput))

        kernH = (yOutput - yKernShape + 1) // 2
        kernW = (xOutput - xKernShape + 1) // 2

        for i in range(kernW, kernW + xKernShape):
            for j in range(kernH, kernH + yKernShape):
                kernelPadded[i, j] = (1 / (xKernShape ** 2))

        fftImage = np.fft.fft2(imageArray)

        fftKernel = np.fft.fft2(kernelPadded)

        c= fftKernel* fftImage

        c=np.fft.ifft2(c)

        # Result of 2
             
        bluredImage=np.abs(np.fft.fftshift(c))
            
        self.drawCanvas(bluredImage,self.gridLayout_8)

        final=bluredImage-self.spatialOutput 

        for i in range(0, final.shape[0]):
            for j in range(0, final.shape[1]):
                if final[i, j] > 255:
                    final[i, j] = 255
                elif final[i, j] < 0:
                    final[i, j] = 0

        self.drawCanvas(final,self.gridLayout_9)
        print(final)
        
    def create_phantom(self):

        # 50, 120, 250
   
        x = np.linspace(-10, 10, 256)
        y = np.linspace(-10, 10, 256)
        x, y = np.meshgrid(x, y)
        x_0 = 0
        y_0 = 0
        circle = np.sqrt((x-x_0)**2+(y-y_0)**2)

        r = 5
        for x in range(256):
            for y in range(256):
                if circle[x,y] < r:
                    circle[x,y] =100
                elif circle[x,y] >= r:
                    circle[x,y] = 0

        squares = np.full((256, 256), 50)
        for i in range(35, 221):
            for j in range(35, 221):
                squares[i,j] = 150

        self.phantom = squares + circle 

        phantomImage= Image.fromarray(np.uint8(self.phantom))
        imageDisplay = phantomImage.toqpixmap()
        qpixmap=QPixmap(imageDisplay)
        self.phantomImageLabel.setPixmap(qpixmap) 

    def add_gaussian_noise(self):

        # Display the image

        # np.random.normal(mean, standard deviation, number of elements)
        gaussian_noise = np.random.normal(0,5,(256,256))
        noisy_phantom = self.phantom + gaussian_noise
        noisy_phantom[noisy_phantom<0] = 0 
        noisy_phantom[noisy_phantom>255] = 255

        
        guassianImage = Image.fromarray(np.uint8(noisy_phantom))

        imageDisplay = guassianImage.toqpixmap()
        qpixmap=QPixmap(imageDisplay)
        self.noisyLabel.setPixmap(qpixmap) 

        self.ROI(noisy_phantom)


    def add_uniform_noise(self):
    # uniform noise a = -10 & b = 10

        a = -10
        b = 10
        mean = (a + b)/ 2
        std_dev = np.sqrt(((b - a) ** 2)/ 12)
        # np.random.normal(mean, standard deviation, number of elements)
        uniform_noise = np.random.uniform(mean, std_dev,(256,256))
        noisy_phantom = self.phantom + uniform_noise
        noisy_phantom[noisy_phantom<0] = 0 
        noisy_phantom[noisy_phantom>255] = 255

        uniformImage = Image.fromarray(np.uint8(noisy_phantom))

        imageDisplay = uniformImage.toqpixmap()
        qpixmap=QPixmap(imageDisplay)
        self.noisyLabel.setPixmap(qpixmap)

        self.ROI(noisy_phantom)

    def ROI(self,noisyImageArray):

        noisyImage= noisyImageArray.astype(np.uint8)

        roi=cv2.selectROI("ROI",noisyImage,False,False)

        # crop the selected region from original image

        croppedROI=noisyImage[int(roi[1]):int(roi[1]+roi[3]),int(roi[0]):int(roi[0]+roi[2])]
        
        histogramArray = np.bincount(croppedROI.flatten(), minlength=256)


        # Normalise
        numPixels = np.sum(histogramArray)
        histogramArray = histogramArray/numPixels
        
        self.drawCanvas(histogramArray,self.gridLayout_2,True)

        
        mean=0
        sumSquareDifference = 0

        # print(statistics.mean(croppedROI))
        # print(statistics.stdev(croppedROI))

        for i in range(histogramArray.shape[0]):
            mean += histogramArray[i]*i

        for i in range(histogramArray.shape[0]):
            sumSquareDifference += ((i-mean)**2)*histogramArray[i]
        
        standardDeviation=np.sqrt(sumSquareDifference)
        
        print(mean)
        print(standardDeviation)


    def drawCanvas(self,image,layout,type=False):

        self.figure=plt.figure(figsize=(15,5))
        self.Canvas=FigureCanvas(self.figure)
        layout.addWidget(self.Canvas,0,0,1,1)
        if type==False:
            plt.imshow(image,cmap='gray')
        else:
            plt.bar(range(256),image)
        self.Canvas.draw()
   
    def handleError(self,errMsg):
        self.label_36.show()
        self.label_35.show()
        self.label_36.setText(errMsg)

    def DrawImage(self,image,label):
        Imagedraw = Image.fromarray(image.astype('uint8'))
        imageDisplay = Imagedraw.toqpixmap()
        qpixmap=QPixmap(imageDisplay)
        label.setPixmap(qpixmap)

    def tabNine(self):
        image = shepp_logan_phantom()
        image = rescale(image, scale=0.64, mode='reflect', channel_axis=None) 
        self.drawCanvas(image,self.gridLayout_3)
        theta = np.arange(0, 180, 1)
        sinogram = radon(image, theta=theta)
        # dx, dy = 0.5 * 180.0 / max(image.shape), 0.5 / sinogram.shape[0]
        sinogram=np.rot90(sinogram)
        self.drawCanvas(sinogram,self.gridLayout_10)

        theta20 = np.arange(0, 161, 20)

        laminogram=self.laminogramConstrction(image,theta20)

        self.drawCanvas(laminogram,self.gridLayout_11)
        
        laminogram=self.laminogramConstrction(image,theta)

        self.drawCanvas(laminogram,self.gridLayout_12)

        laminogram=self.laminogramConstrction(image,theta,'ramp')

        self.drawCanvas(laminogram,self.gridLayout_13)

        laminogram=self.laminogramConstrction(image,theta,'hamming')

        self.drawCanvas(laminogram,self.gridLayout_14)
        
    def laminogramConstrction(self,image,theta,filter=None):
        sinogram = radon(image, theta=theta)
        Laminogram = iradon(sinogram, theta=theta, filter_name= filter) # , filter_name='ramp'

        return Laminogram
    
    def tabTen(self):
        pass
        # load image
        # image= cv2.imread('./images/binary_image.png',0)

        # imageArray=np.asarray(image)
        # orig_shape = imageArray.shape
    
        # # structure element
        # structure=np.array([[0, 1,1,1,0],
        #           [1,1,1,1,1],
        #           [1,1,1,1,1],
        #           [1,1,1,1,1],
        #           [0,1,1,1,0]])
        # # apply erosion(fit)  


        # erodedImage=self.erosion(imageArray,structure)

        # self.DrawImage(erodedImage,self.label_17)
        # # apply dilation(hit)
        # dilatedImage=self.dilation(imageArray,structure)
        # self.DrawImage(dilatedImage,self.label_18)
        

        # # apply opening 
        # openedImage=self.dilation(erodedImage,structure)
        # self.DrawImage(openedImage,self.label_21)
        # # apply closing
        # closingImage=self.erosion(dilatedImage,structure)
        # self.DrawImage(closingImage,self.label_24)

        # # Denoising the image
        # # open the image
        # # A different structure array because the previous one is too aggresive in eroding the image
        # structure=np.array([
        #           [0,1,0],
        #           [1,1,1],
        #           [0,1,0],
        #           ]) 

        # erodedNoiseImage=self.erosion(imageArray,structure)
        # openedNoiseImage=self.dilation(erodedNoiseImage,structure)
        # dilateOpenedImage=self.dilation(openedNoiseImage,structure)
        # denoisedImage=self.erosion(dilateOpenedImage,structure)
        # self.DrawImage(openedNoiseImage,self.label_30)


    def erosion(self,image,structure):
        m,n= image.shape
        erodedImage= np.zeros((m,n), dtype=np.uint8)
        steps=(structure.shape[0]-1)//2
        countOfOnes=np.count_nonzero(structure == 1)
        for i in range(steps, m-steps):
            for j in range(steps,n-steps):
                temp= image[i-steps:i+steps+1, j-steps:j+steps+1]
                product= temp*structure
                value=self.countOnes(product,countOfOnes) 
                erodedImage[i,j]= value 
        return erodedImage

    def dilation(self,image,structure):
        m,n= image.shape
        dilatedImage= np.zeros((m,n), dtype=np.uint8)
        steps=(structure.shape[0]-1)//2
 
        for i in range(steps, m-steps):
            for j in range(steps,n-steps):
                temp= image[i-steps:i+steps+1, j-steps:j+steps+1]
                product= temp*structure
                value=self.countOnes(product,_,True) 
                dilatedImage[i,j]= value 
        return dilatedImage

    def countOnes(self,array,countOfOnes=21,dilation=False):
        count=0
        if(dilation==False):
            for i in range(0, len(array)):
                for j in range(0,len(array)):
                    if array[i][j]==255:
                        count+=1
            if count==countOfOnes:
                return 255       
            return 0
        else:
            for i in range(0, len(array)):
                for j in range(0,len(array)):
                    if array[i][j]==255:
                        count+=1
            if count>0:
                return 255       
            return 0
def main ():
    app = QApplication(sys.argv)
    window = MainApp ()
    window.show()
    app.exec_()
if __name__ == '__main__':
    main()
