import numpy as np
import platform
import tempfile
import os
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
import scipy as sp
# requires skimage package
import skimage as sk
from skimage import io as skio
import skimage.feature as skf
from scipy import ndimage as ndi
import skimage.morphology as morpho

# def viewimage(im,normalise=True,MINI=0.0, MAXI=255.0):
#     """This function displays the image in GREYSCALE in GIMP.
#         By default, the image is normalized between 0 and 255 before being saved.
#         Otherwise, the mnimum will be MINI and the MAXI will be set to 255
#     """
#     imt=np.float32(im.copy())
#     if platform.system()=='Darwin': #Mac
#         prephrase='open -a GIMP '
#         endphrase=' '
#     else: #llinux
#         prephrase='gimp '
#         endphrase= ' &'
#
#     if normalise:
#         m=imt.min()
#         imt=imt-m
#         M=imt.max()
#         if M>0:
#             imt=imt/M
#
#     else:
#         imt=(imt-MINI)/(MAXI-MINI)
#         imt[imt<0]=0
#         imt[imt>1]=1
#
#     nomfichier=tempfile.mktemp('TPIMA.png')
#     commande=prephrase +nomfichier+endphrase
#     skio.imsave(nomfichier,imt)
#     os.system(commande)
#
# def viewimage_color(im,normalise=True,MINI=0.0, MAXI=255.0):
#     """This function displays the image in COLORS in GIMP.
#         By default, the image is normalized between 0 and 255 before being saved.
#         Otherwise, the mnimum will be MINI and the MAXI will be set to 255
#     """
#     imt=np.float32(im.copy())
#     if platform.system()=='Darwin': #Mac
#         prephrase='open -a GIMP '
#         endphrase= ' '
#     else: #Linux
#         prephrase='gimp '
#         endphrase=' &'
#
#     if normalise:
#         m=imt.min()
#         imt=imt-m
#         M=imt.max()
#         if M>0:
#             imt=imt/M
#     else:
#         imt=(imt-MINI)/(MAXI-MINI)
#         imt[imt<0]=0
#         imt[imt>1]=1
#
#     nomfichier=tempfile.mktemp('TPIMA.pgm')
#     commande=prephrase +nomfichier+endphrase
#     skio.imsave(nomfichier,imt)
#     os.system(commande)
#
#
#
# def noise(im,br):
#     """ This function adds a white noise of standard deviation br to the image"""
#     imt=np.float32(im.copy())
#     sh=imt.shape
#     bruit=br*np.random.randn(*sh)
#     imt=imt+bruit
#     return imt
#
# def quantize(im,n=2):
#     """
#     Returns a quantized image over n(=2 by default) levels
#     """
#     imt=np.float32(im.copy())
#     if np.floor(n)!= n or n<2:
#         raise Exception("The n value is not correct in quantize")
#     else:
#         m=imt.min()
#         M=imt.max()
#         imt=np.floor(n*((imt-m)/(M-m)))*(M-m)/n+m
#         imt[imt==M]=M-(M-m)/n #maximum value case
#         return imt
#
#
# def seuil(im,s):
#     """ returns a white pixel where im>=s and a black one everywhere else.
#     """
#     imt=np.float32(im.copy())
#     mask=imt<s
#     imt[mask]=0
#     imt[~mask]=255
#     return imt
#
# def gradx(im):
#     "returns the horizontal gradient"
#     imt=np.float32(im)
#     gx=0*imt
#     gx[:,:-1]=imt[:,1:]-imt[:,:-1]
#     return gx
#
# def grady(im):
#     "returns the vertical gradient"
#     imt=np.float32(im)
#     gy=0*imt
#     gy[:-1,:]=imt[1:,:]-imt[:-1,:]
#     return gy
#
# def view_spectre(im,option=1,hamming=False):
#     """ displays the spectre of an image
#      if option =1 visualize the intensity linearly
#      if option =2 visualize the log of the intensity
#      if hamming=True (defaut False) a Hamming winow is applied before doing the fourier transform
#      """
#     imt=np.float32(im.copy())
#     (ty,tx)=im.shape
#     pi=np.pi
#     if hamming:
#         XX=np.ones((ty,1))@(np.arange(0,tx).reshape((1,tx)))
#         YY=(np.arange(0,ty).reshape((ty,1)))@np.ones((1,tx))
#         imt=(1-np.cos(2*pi*XX/(tx-1)))*(1-np.cos(2*pi*YY/(ty-1)))*imt
#     aft=np.fft.fftshift(abs(np.fft.fft2(imt)))
#
#     if option==1:
#         viewimage(aft)
#     else:
#         viewimage(np.log(0.1+aft))
#
#
# def filterlow(im):
#     """apply a perfectlow pass filter to the image for grayscale image of even size"""
#     (ty,tx)=im.shape
#     imt=np.float32(im.copy())
#     pi=np.pi
#     XX=np.concatenate((np.arange(0,tx/2+1),np.arange(-tx/2+1,0)))
#     XX=np.ones((ty,1))@(XX.reshape((1,tx)))
#
#     YY=np.concatenate((np.arange(0,ty/2+1),np.arange(-ty/2+1,0)))
#     YY=(YY.reshape((ty,1)))@np.ones((1,tx))
#     mask=(abs(XX)<tx/4) & (abs(YY)<ty/4)
#     imtf=np.fft.fft2(imt)
#     imtf[~mask]=0
#     return np.real(np.fft.ifft2(imtf))
#
# def filtergauss(im):
#     """apply a gaussian low pass filtering to a grayscale image. cuts approximately to f0/4"""
#
#     (ty,tx)=im.shape
#     imt=np.float32(im.copy())
#     pi=np.pi
#     XX=np.concatenate((np.arange(0,tx/2+1),np.arange(-tx/2+1,0)))
#     XX=np.ones((ty,1))@(XX.reshape((1,tx)))
#
#     YY=np.concatenate((np.arange(0,ty/2+1),np.arange(-ty/2+1,0)))
#     YY=(YY.reshape((ty,1)))@np.ones((1,tx))
#
#     sig=(tx*ty)**0.5/2/(pi**0.5)
#     mask=np.exp(-(XX**2+YY**2)/2/sig**2)
#     imtf=np.fft.fft2(imt)*mask
#     return np.real(np.fft.ifft2(imtf))
#



##### MATHEMATICAL MORPHOLOGY


def strel(shape, size):
    """returns the chosen structuring element
     'diamond'  closed ball for the  L1 of radius size
     'disk'     closed ball for the  L2 of radius size
     'square'   square  of size size
    """

    if shape == 'diamond':
        return morpho.selem.diamond(size)
    if shape == 'disk':
        return morpho.selem.disk(size)
    if shape == 'square':
        return morpho.selem.square(size)

    raise RuntimeError('Erreur dans fonction strel: forme incomprise')


def main_one():
    # basics
    ## load image
    img = skio.imread('images/b.tif', as_gray=True)

    ## show image
    # plt.imshow(img, cmap='gray')

    ## show histogram
    # plt.hist(img.reshape((-1,)), bins=255)

    # FFT
    fft = np.fft.fft2(img)

    ### check if the inverse will give the original image
    # plt.imshow(np.abs(np.fft.ifft2(fft)))

    ### center at (0, 0)
    centered = np.fft.fftshift(fft)

    ### fft is complex -> gotta take the absolute
    realfft = np.abs(centered)

    ### take log (because high number shadows small ones)
    logfft = np.log(realfft)
    # plt.imshow(logfft)

    ### take angle of fft (aka phase) (you can't see anything..)
    fftphase = np.angle(realfft)
    # plt.imshow(fftphase)

    # subsampling
    width = np.shape(img)[0]
    height = np.shape(img)[1]

    ## take only every x-th pixel from the original (basically change resolution)
    factor = 10
    resampled = img[0:width:factor, 0:height:factor] # from 0 to width, step length = factor
    # plt.imshow(resampled)

    # linear filtering
    ## low-pass-filter
    ### diy mean convolution
    mean_filter = (.04)*np.ones((factor*4,factor*4), dtype=np.float32) # use float32 to get higher precision

    #### take a look at the convolution in function space
    # plt.imshow(np.log(np.abs(np.fft.fftshift(np.fft.fft2(mean_filter)))))

    #### convolve the image
    mean = sp.signal.convolve2d(img, mean_filter, mode='valid')
    # plt.imshow(mean)

    ### scikit gaussian filter
    gaussian = sk.filters.gaussian(img, sigma=20) # sigma == radius
    # plt.imshow(gaussian)

    plt.show()


def main_two():
    img = skio.imread('images/spot.tif')
    # plt.imshow(img)

    ## show histogram
    plt.hist(img.reshape((-1,)), bins=255)

    # ...

    plt.show()


def main_three():
    # img = skio.imread('images/cell.tif')
    img = skio.imread('images/bat.tif', as_gray=True)
    structural_elements = []#strel('square', 1), strel('square', 5)]#, strel('diamond', 1), strel('diamond', 5), strel('disk', 1), strel('disk', 5)]

    for se in structural_elements:
        plt.title('{}'.format(se))
        plt.subplot(221)
        plt.title('erosion')
        plt.imshow(morpho.erosion(img, se), cmap='gray')
        plt.subplot(222)
        plt.title('dilation')
        plt.imshow(morpho.dilation(img, se), cmap='gray')
        plt.subplot(223)
        plt.title('opening')
        plt.imshow(morpho.opening(img, se), cmap='gray')
        plt.subplot(224)
        plt.title('closing')
        plt.imshow(morpho.closing(img, se), cmap='gray')
        # plt.show()

    plt.title('Top hat')
    for radius in [2, 3, 4, 5]:
        se = strel('square', radius)
        tophat = np.abs(morpho.opening(img, se) - img)
        plt.subplot(2, 2, radius-1)
        plt.title('Square {}'.format(radius))
        plt.imshow(tophat)

    plt.show()

if __name__ == '__main__':
    # main_one()
    # main_two()
    main_three()