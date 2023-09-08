# ------------------------- Imports
import matplotlib.pyplot as plt
import matplotlib.colors as clr
import numpy as np
import cv2
import scipy.fftpack as fft
import math

# ------------------------- Variables

ConversionTableYCbCr = np.array([[0.299, 0.587, 0.114],
                                 [-0.168736, -0.331264, 0.5],
                                 [0.5, -0.418688, -0.081312]])

QuantizationMatrixY = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                                [12, 12, 14, 19, 26, 58, 60, 55],
                                [14, 13, 16, 24, 40, 57, 69, 56],
                                [14, 17, 22, 29, 51, 87, 80, 62],
                                [18, 22, 37, 56, 68, 109, 103, 77],
                                [24, 35, 55, 64, 81, 104, 113, 92],
                                [49, 64, 78, 87, 103, 121, 120, 101],
                                [72, 92, 95, 98, 112, 100, 103, 99]])

QuantizationMatrixC = np.array([[17, 18, 24, 47, 99, 99, 99, 99],
                                [18, 21, 26, 66, 99, 99, 99, 99],
                                [24, 26, 56, 99, 99, 99, 99, 99],
                                [47, 66, 99, 99, 99, 99, 99, 99],
                                [99, 99, 99, 99, 99, 99, 99, 99],
                                [99, 99, 99, 99, 99, 99, 99, 99],
                                [99, 99, 99, 99, 99, 99, 99, 99],
                                [99, 99, 99, 99, 99, 99, 99, 99]])


# ------------------------- Functions

def showImage(image, title, show):
    if show:
        plt.figure()
        plt.imshow(image)
        plt.axis('off')
        plt.title(title)
        plt.show()


# Ex 3.2
def createColorMap(name, R, G, B):
    return clr.LinearSegmentedColormap.from_list(name, [(0, 0, 0), (R, G, B)], N=256)


def getColormap(cor):
    if cor == "red":
        return clr.LinearSegmentedColormap.from_list("myRed", [(0, 0, 0), (1, 0, 0)], N=256)
    if cor == "green":
        return clr.LinearSegmentedColormap.from_list("myGreen", [(0, 0, 0), (0, 1, 0)], N=256)
    if cor == "blue":
        return clr.LinearSegmentedColormap.from_list("myBlue", [(0, 0, 0), (0, 0, 1)], N=256)
    if cor == "gray":
        return clr.LinearSegmentedColormap.from_list("myGray", [(0, 0, 0), (1, 1, 1)], N=256)


# Ex 3.3
def imageColormap(image, title, cMap, show):
    if show:
        plt.figure()
        plt.imshow(image, cMap)
        plt.axis('off')
        plt.title(title)
        plt.show()


# Ex 3.4
def getComponents(image):
    R = image[:, :, 0]
    G = image[:, :, 1]
    B = image[:, :, 2]
    return R, G, B


def getImageFromComponents(R, G, B):
    return np.dstack((R, G, B))


# Ex 3.5
def imageAndChannels(image, show):
    showImage(image, "Original Image", show)

    R, G, B = getComponents(image)

    imageColormap(R, "R Channel", getColormap("red"), show)
    imageColormap(G, "G Channel", getColormap("green"), show)
    imageColormap(B, "B Channel", getColormap("blue"), show)


# Ex 4.1
def padding(image, n):
    l = image.shape[0]
    c = image.shape[1]

    paddedImg = image

    if (l % n) != 0:
        v = n - (l % n)
        paddedImg = np.vstack([paddedImg, np.repeat([paddedImg[-1, :, :]], v, axis=0)])

    if (c % n) != 0:
        v = n - (c % n)
        paddedImg = np.hstack([paddedImg, np.repeat(paddedImg[:, -1:, :], v, axis=1)])

    return paddedImg


def unpadding(nl, nc, paddedImg):
    return paddedImg[:nl, :nc, :]


# Ex 5.1
def convertRGBtoYCbCr(img, Tc):
    R, G, B = getComponents(img)

    YCbCr = np.empty_like(img).astype("float")

    # Y
    YCbCr[:, :, 0] = Tc[0, 0] * R + Tc[0, 1] * G + Tc[0, 2] * B
    # Cb
    YCbCr[:, :, 1] = Tc[1, 0] * R + Tc[1, 1] * G + Tc[1, 2] * B + 128
    # Cr
    YCbCr[:, :, 2] = Tc[2, 0] * R + Tc[2, 1] * G + Tc[2, 2] * B + 128

    '''YCbCr = np.round(YCbCr)
    YCbCr[YCbCr > 255] = 255
    YCbCr[YCbCr < 0] = 0'''

    return YCbCr


def convertYCbCrtoRGB(Y, Cb, Cr, Tc):
    TcInvertida = np.linalg.inv(Tc)

    '''Y = Y.astype(float)
    Cb = Cb.astype(float)
    Cr = Cr.astype(float)'''

    l, c = Y.shape
    RGB = np.zeros((l, c, 3))

    # Y
    RGB[:, :, 0] = TcInvertida[0, 0] * Y + TcInvertida[0, 1] * (Cb - 128) + TcInvertida[0, 2] * (Cr - 128)
    # Cb
    RGB[:, :, 1] = TcInvertida[1, 0] * Y + TcInvertida[1, 1] * (Cb - 128) + TcInvertida[1, 2] * (Cr - 128)
    # Cr
    RGB[:, :, 2] = TcInvertida[2, 0] * Y + TcInvertida[2, 1] * (Cb - 128) + TcInvertida[2, 2] * (Cr - 128)

    RGB = np.round(RGB)
    RGB[RGB > 255] = 255
    RGB[RGB < 0] = 0
    RGB = RGB.astype(np.uint8)

    return RGB


# Ex 5.3
def imageAndChannelsYCbCr(image, show):
    showImage(image, "YCbCr Image", False)

    Y, Cb, Cr = getComponents(image)

    cMap = getColormap("gray")
    showImage(Y, "Y Channel", False)
    imageColormap(Y, "Y Channel",  cMap, show)
    showImage(Cb, "Cb Channel", False)
    imageColormap(Cb, "Cb Channel", cMap, show)
    showImage(Cr, "Cr Channel", False)
    imageColormap(Cr, "Cr Channel", cMap, show)


# Ex 6.1
def createDownSampling(YCbCr, method, interpol):
    sx = 0
    sy = 0

    if method == "4:2:0":
        sx = 0.5
        sy = 0.5

    elif method == "4:2:2":
        sx = 0.5
        sy = 1

    stx = int(1 // sx)
    sty = int(1 // sy)

    if interpol == 0:
        Cb_d = YCbCr[::sty, ::stx, 1]
        Cr_d = YCbCr[::sty, ::stx, 2]

    else:
        Cb_d = cv2.resize(YCbCr[:, :, 1], None, fx=sx, fy=sy, interpolation=cv2.INTER_LINEAR)
        Cr_d = cv2.resize(YCbCr[:, :, 2], None, fx=sx, fy=sy, interpolation=cv2.INTER_LINEAR)

    return YCbCr[:, :, 0], Cb_d, Cr_d


def createUpSampling(Y_d, Cb_d, Cr_d, method, interpol):
    sx = 0
    sy = 0

    if method == "4:2:0":
        sx = 0.5
        sy = 0.5

    elif method == "4:2:2":
        sx = 0.5
        sy = 1

    stx = int(1 // sx)
    sty = int(1 // sy)

    if interpol == 0:
        Cb = np.repeat(Cb_d, stx, axis=1)
        Cr = np.repeat(Cr_d, stx, axis=1)
        Cb = np.repeat(Cb, sty, axis=0)
        Cr = np.repeat(Cr, sty, axis=0)

    else:
        Cb = cv2.resize(Cb_d, None, fx=stx, fy=sty, interpolation=cv2.INTER_LINEAR)
        Cr = cv2.resize(Cr_d, None, fx=stx, fy=sty, interpolation=cv2.INTER_LINEAR)

    return Y_d, Cb, Cr


# Ex 6.2
def visualizeDs(Y_d, Cb_d, Cr_d, show):
    if show:
        print(Y_d.shape)
        print(Cb_d.shape)
        print(Cr_d.shape)

    cMap = getColormap("gray")
    imageColormap(Y_d, "Downsampling - Y Channel", cMap, show)
    imageColormap(Cb_d, "Downsampling - Cb Channel", cMap, show)
    imageColormap(Cr_d, "Downsampling - Cr Channel", cMap, show)


# Ex 7.1.1
def calculateDCT(channel):
    return fft.dct(fft.dct(channel, norm="ortho").T, norm="ortho").T

def calculateDCTLog(dctChannel):
    return np.log(np.abs(dctChannel) + 0.0001)

def calculateInvDCT(dctChannel):
    return fft.idct(fft.idct(dctChannel, norm="ortho").T, norm="ortho").T


def getDCT(Y_d, Cb_d, Cr_d):
    return calculateDCT(Y_d), calculateDCT(Cb_d), calculateDCT(Cr_d)

def getInvDCT(Y_d, Cb_d, Cr_d):
    return calculateInvDCT(Y_d), calculateInvDCT(Cb_d), calculateInvDCT(Cr_d)


# Ex 7.1.2
def visualizeDctDsChannels(Y_d, Cb_d, Cr_d, show):
    cMap = getColormap("gray")
    imageColormap(calculateDCTLog(calculateDCT(Y_d)), "DCT - Y Channel", cMap, show)
    imageColormap(calculateDCTLog(calculateDCT(Cb_d)), "DCT - Cb Channel", cMap, show)
    imageColormap(calculateDCTLog(calculateDCT(Cr_d)), "DCT - Cr Channel", cMap, show)


# Ex 7.2.1
def dctInBlocks(channel, size, case, quantization, dpcm):
    l, c = channel.shape
    nl = int(l / size)
    nc = int(c / size)
    dc0 = 0

    channel = np.vsplit(channel, nl)
    for i in range(nl):
        channel[i] = np.hsplit(channel[i], nc)
        for j in range(nc):
            if case == "DCT":
                channel[i][j] = calculateDCT(channel[i][j])

                if 0 <= quantization[0] <= 100:
                    channel[i][j] = quantizeChannelBlock(channel[i][j], quantization[1])

                    if dpcm == 1:
                        channel[i][j][0][0], dc0 = makeDPCM(channel[i][j][0][0], i, j, dc0, case)

            elif case == "InvDCT":
                if 0 < quantization[0] <= 100:
                    if dpcm == 1:
                        channel[i][j][0][0], dc0 = makeDPCM(channel[i][j][0][0], i, j , dc0, case)

                    channel[i][j] = dequantizeChannelBlock(channel[i][j], quantization[1])

                channel[i][j] = calculateInvDCT(channel[i][j])

    channel = np.array(channel)
    channel = np.hstack(channel)
    channel = np.hstack(channel)
    return channel


def getDctInBlocks(Y_d, Cb_d, Cr_d, size, case, quantization, dpcm):
    if 0 < quantization <= 100:
        QM_Y, QM_C = createQuantizationMatrixes(quantization)

        Y = dctInBlocks(Y_d, size, case, (quantization, QM_Y), dpcm)
        Cb = dctInBlocks(Cb_d, size, case, (quantization, QM_C), dpcm)
        Cr = dctInBlocks(Cr_d, size, case, (quantization, QM_C), dpcm)
    else:
        Y = dctInBlocks(Y_d, size, case, (-1,), dpcm)
        Cb = dctInBlocks(Cb_d, size, case, (-1,), dpcm)
        Cr = dctInBlocks(Cr_d, size, case, (-1,), dpcm)

    return Y, Cb, Cr


# Ex 7.2.2
def visualizeDctInBlocksChannels(Y_DCT8, Cb_DCT8, Cr_DCT8, show):
    cMap = getColormap("gray")
    imageColormap(calculateDCTLog(Y_DCT8), "DCT In Blocks - Y Channel", cMap, show)
    imageColormap(calculateDCTLog(Cb_DCT8), "DCT In Blocks - Cb Channel", cMap, show)
    imageColormap(calculateDCTLog(Cr_DCT8), "DCT In Blocks - Cr Channel", cMap, show)


# Ex 8.1
def createQuantizationMatrixes(quality):
    if quality >= 50:
        sf = (100 - quality) / 50
    else:
        sf = 50 / quality

    if sf != 0:
        QM_Y = np.round(QuantizationMatrixY * sf)
        QM_C = np.round(QuantizationMatrixC * sf)
    else:
        QM_Y = np.ones(QuantizationMatrixY.shape)
        QM_C = np.ones(QuantizationMatrixC.shape)

    QM_Y[QM_Y > 255] = 255
    QM_C[QM_C > 255] = 255

    QM_Y[QM_Y < 1] = 1
    QM_C[QM_C < 1] = 1

    return QM_Y, QM_C


def quantizeChannelBlock(block, QM):
    quantizedBlock = np.round(block / QM)
    return quantizedBlock


def dequantizeChannelBlock(block, QM):
    dequantizedBlock = block * QM
    return dequantizedBlock


# Ex 9.1
def makeDPCM(channel, i, j, dc0, case):
    if i == 0 and j == 0:
        return channel, channel
    else:
        dc = channel
        if case == "DCT":
            channel = dc - dc0
            return channel, dc
        elif case == "InvDCT":
            channel = dc + dc0
            return channel, channel


# Ex 10.2
def getMSE(original, reconstructed, nl, nc):
    original = original.astype('float')
    reconstructed = reconstructed.astype('float')
    aux = np.square(original-reconstructed)
    return (1/(nl*nc)) * (np.sum(aux))

def getRMSE(original, reconstructed, nl, nc):
    return math.sqrt(getMSE(original, reconstructed, nl, nc))

def getSNR(original, reconstructed, nl, nc):
    original = original.astype('float')
    aux = (1/(nl*nc)) * (np.sum(np.square(original)))
    return 10 * math.log10(aux/ getMSE(original, reconstructed, nl, nc))

def getPSNR(original, reconstructed, nl, nc):
    original = original.astype('float')
    aux = np.square(np.amax(original))
    return 10 * math.log10(aux / getMSE(original, reconstructed, nl, nc))


def getDistortionMetrics(original, reconstructed, nl, nc, show):
    mse = getMSE(original, reconstructed, nl, nc)
    rmse = getRMSE(original, reconstructed, nl, nc)
    snr = getSNR(original, reconstructed, nl, nc)
    psnr = getPSNR(original, reconstructed, nl, nc)

    if show:
        print("MSE: %f" %mse)
        print("RMSE: %f" %rmse)
        print("SNR: %f" %snr)
        print("PSNR: %f" %psnr)


def showYChannelDiference(original, reconstructed, show):
    Y_O = getComponents(convertRGBtoYCbCr(original, ConversionTableYCbCr))[0]
    Y_R = getComponents(convertRGBtoYCbCr(reconstructed, ConversionTableYCbCr))[0]
    cMap = getColormap("gray")
    imageColormap(np.abs(Y_O-Y_R), "Original and Decoded Images Difference", cMap, show)


# ------------------------- Encoder and Decoder

def encoder(image, quality):
    # Ex 3.1
    img = plt.imread(image)
    nl = img.shape[0]
    nc = img.shape[1]

    # Ex 3.2
    cMap = getColormap("blue")

    # Ex 3.3
    imageColormap(img[:,:,2], "Image with Colormap", cMap, False)

    # Ex 3.4
    R, G, B = getComponents(img)

    img2 = getImageFromComponents(R, G, B)
    showImage(img2, "Image with Merged Channels", False)

    # Ex 3.5
    imageAndChannels(img, False)

    # Ex 4.1
    padingDimension = 16
    paddedImage = padding(img, padingDimension)
    showImage(paddedImage, "Image with Padding", False)

    unpaddedImage = unpadding(nl, nc, paddedImage)
    showImage(unpaddedImage, "Image with Unpadding", False)

    # Ex 5.1
    YCbCr = convertRGBtoYCbCr(paddedImage, ConversionTableYCbCr)
    showImage(YCbCr.astype("uint8"), "Image converted to YCbCr", False)

    Y, Cb, Cr = getComponents(YCbCr)
    RGB = convertYCbCrtoRGB(Y, Cb, Cr, ConversionTableYCbCr)
    showImage(RGB, "Image converted to RGB", False)

    # Ex 5.3
    imageAndChannelsYCbCr(YCbCr, False)

    # Ex 6.1
    samplingType = "4:2:0"
    Y_d, Cb_d, Cr_d = createDownSampling(YCbCr, samplingType, 1)

    usImg = createUpSampling(Y_d, Cb_d, Cr_d, samplingType, 1)
    showImage(getImageFromComponents(usImg[0], usImg[1], usImg[2]).astype("uint8"), "Image with Upsampling", False)

    # Ex 6.2
    visualizeDs(Y_d, Cb_d, Cr_d, False)

    # Ex 7.1.1
    dctY, dctCb, dctCr = getDCT(Y_d, Cb_d, Cr_d)

    dctLogCb = calculateDCTLog(dctCb)
    showImage(dctLogCb, "Cb Channel with DCTLog", False)
    invDct = calculateInvDCT(dctCb)
    showImage(invDct, "Cb Channel with InvDCT", False)

    # Ex 7.1.2
    visualizeDctDsChannels(Y_d, Cb_d, Cr_d, False)

    # Ex 7.2.1, Ex 8.1 and Ex 9.1
    blockSize = 8                     # to run without using blocks it is necessary to use the function calculateDCT(channel)
    quantizationQuality = quality     # to run without quantization it is necessary to use an invalid quantization quality (e.g. -1)
    DPCM = 1                          # to run without DPCM it is necessary to use the value 0 (false)
    Y_DCT8, Cb_DCT8, Cr_DCT8 = getDctInBlocks(Y_d, Cb_d, Cr_d, blockSize, "DCT", quantizationQuality, DPCM)

    # Ex 7.2.2, Ex 8.2 and Ex 9.2
    visualizeDctInBlocksChannels(Y_DCT8, Cb_DCT8, Cr_DCT8, False)

    return img, (Y_DCT8, Cb_DCT8, Cr_DCT8), nl, nc, samplingType, blockSize, quantizationQuality, DPCM


def decoder(imageEncoded, nl, nc, samplingType, blockSize, quantizationQuality, DPCM):
    Y_d, Cb_d, Cr_d = getDctInBlocks(imageEncoded[0], imageEncoded[1], imageEncoded[2], blockSize, "InvDCT", quantizationQuality, DPCM)

    Y, Cb, Cr = createUpSampling(Y_d, Cb_d, Cr_d, samplingType, 1)

    RGB = convertYCbCrtoRGB(Y, Cb, Cr, ConversionTableYCbCr)

    unpaddedImg = unpadding(nl, nc, RGB)

    showImage(unpaddedImg, "Decoded Image", True)

    return unpaddedImg


########################################################################################################################

def main():
    plt.close('all')

    imgOriginal, imgEncoded, nl, nc, samplingType, blockSize, quantizationQuality, DPCM = encoder('images/barn_mountains.bmp', 75)

    imgReconstructed = decoder(imgEncoded, nl, nc, samplingType, blockSize, quantizationQuality, DPCM)

    getDistortionMetrics(imgOriginal, imgReconstructed, nl, nc, True)

    showYChannelDiference(imgOriginal, imgReconstructed, False)


if __name__ == '__main__':
    main()