import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import cv2 as cv
import spectral.io.envi as envi
import torch
import torchvision
import sys
from spectral import *
import wget

sys.path.append("..")
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

wget.download("https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth")

def imread(data, HSI_or_RGB = "RGB"):

  if HSI_or_RGB == "RGB":

    img = cv.cvtColor(cv.imread(data),cv.COLOR_BGR2RGB)
    return img

  elif HSI_or_RGB == "HSI":

    HSI = envi.open(data)[:,:,:]
    wavelength = envi.read_envi_header(data)['wavelength']
    wavelength = [int(float(i)) for i in wavelength]

    RBand = HSI[:,:,70]
    GBand = HSI[:,:,53]
    BBand = HSI[:,:,19]

    RGB = np.stack((RBand,GBand,BBand),axis = 2)
    RGB = np.array(RGB*255,dtype = np.uint8)

    return HSI,RGB, wavelength

  else:

    return print("INDIQUE QUE TIPO DE IMAGEN QUIERE LEER (RGB O HIS)")


def unfolding(img):

    features = tuple(img.shape)
    img = img.reshape(-1,features[2])

    return img, features

def SAM_Mask_Generator(data):

  sam_checkpoint = "sam_vit_h_4b8939.pth"
  model_type = "vit_h"
  device = "cuda"


  sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
  sam.to(device=device)


  mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=32,
    pred_iou_thresh=0.98,
    stability_score_thresh=0.92,
    crop_n_layers=1,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=10,
  )

  masks = mask_generator.generate(data)

  return masks


def Generate_Mask(Data,Mask,number_mask):

  anns = Mask
  image = np.copy(Data)

  sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)

  img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
  img[:,:,3] = 0

  for ann in sorted_anns:
      m = ann['segmentation']
      color_mask = np.concatenate([np.random.random(3), [1]])
      img[m] = color_mask


  Fondo = sorted_anns[number_mask]['segmentation']
  Fondo_idx = np.where(Fondo.reshape(-1) == 0)
  image2d = image.reshape(-1,3)
  image2d[Fondo_idx] = np.array([0,0,0])

  idx_white = np.where(img.reshape(-1,4) == [1,1,1,0])[0]
  image2d[idx_white] = np.array([0,0,0])

  img_ = image2d.reshape(image.shape)

  plt.figure(figsize = (10,10))
  plt.imshow(img_)
  #plt.imshow(img,alpha = 0.5)
  plt.axis("off")
  plt.show()

  return img_


def NormalRGB(img,white_limit):


    features = img.shape
    data = np.reshape(img,(-1,3))

    std = data.std(1)
    hist, bin_edges = np.histogram(std,bins = 50)
    bin_centers = (bin_edges[:-1] + bin_edges[1:])/2
    sorted_idx = np.argsort(hist)
    max_idx = sorted_idx[-1]

    mean_hist = bin_centers[max_idx]
    sigma = np.std(std)

    sigma_idx = np.where(std <= mean_hist + (0.5*sigma))
    sigma_filt = np.std(std[sigma_idx])
    stdlim = mean_hist + 3*sigma_filt

    idx_background = np.where(std <= stdlim)[0]
    background_reference = data[idx_background]
    background_reference_mean = background_reference.mean(1)
    white_idx = np.where(background_reference_mean >= white_limit)[0]
    white_reference =  background_reference[white_idx]
    mean = white_reference.mean(0)

    construction_data = np.copy(data)
    ref_white = np.copy(data)

    construction_data = construction_data/mean
    np.place(construction_data, construction_data > 1,1)

    ref_white[idx_background[white_idx]] = np.array([0,0,0])
    construction_data = np.reshape(construction_data,features)
    ref_white = np.reshape(ref_white,features)

    return construction_data, ref_white, mean


def RGB2Lab(img_data):

    e =  216/24389

    non_zero_rgb = img_data[:,:]**(2.2)

    X = non_zero_rgb[:, 0] * 0.4124 + non_zero_rgb[:, 1] * 0.3576 + non_zero_rgb[:, 2] * 0.1805
    Y = non_zero_rgb[:, 0] * 0.2126 + non_zero_rgb[:, 1] * 0.7152 + non_zero_rgb[:, 2] * 0.0722
    Z = non_zero_rgb[:, 0] * 0.0193 + non_zero_rgb[:, 1] * 0.1192 + non_zero_rgb[:, 2] * 0.9505

    fx = (X/0.94811)
    fy = (Y/1)
    fz = (Z/1.07304)

    if fx.any() > e:
        fx = fx**(1/3)
    else:
        fx = ((903.3*fx)+16)/116

    if fy.any() > e:
        fy = fy**(1/3)
    else:
        fy = ((903.3*fy)+16)/116

    if fz.any() > e:
        fz = fz**(1/3)
    else:
        fz = ((903.3*fz)+16)/116

    a = (fx - fy) * 500
    b = (fy - fz) * 200
    L = (fy * 116) - 16

    data_CLab = np.column_stack((L, a, b))

    return data_CLab


def White_reference(datacube,stdlim):
    illumination_reference = np.genfromtxt('/content/drive/MyDrive/Proyecto_Investigación/HSI/D65.csv',delimiter=',')
    TriEst = np.genfromtxt('/content/drive/MyDrive/Proyecto_Investigación/HSI/XYZOK.csv', delimiter= ',')
    std = datacube.std(1)
    White_idx = np.where(std <= stdlim)[0]
    White = datacube[White_idx]
    White_mean = White.mean(1)
    Teflon_idx = np.where(White_mean >= 0.90)[0]
    Teflon = datacube[White_idx[Teflon_idx]].mean(0)
    k = 1/np.sum(illumination_reference[:130]*TriEst[:,3])
    Xo = k*np.sum(illumination_reference[:130]*Teflon[:130]*TriEst[:,2])
    Yo = k*np.sum(illumination_reference[:130]*Teflon[:130]*TriEst[:,3])
    Zo = k*np.sum(illumination_reference[:130]*Teflon[:130]*TriEst[:,1])
    white_reference = (Xo,Yo,Zo)

    datacube[White_idx[Teflon_idx]] = np.zeros(204)

    plt.imshow(datacube.reshape(512,512,-1)[:,:,50])
    return white_reference,Teflon


def Spectra2Lab(datacube,white_reference = (1,1,1)):

    TriEst = np.genfromtxt('/content/drive/MyDrive/Proyecto_Investigación/HSI/XYZOK.csv', delimiter= ',')
    illumination_reference = np.genfromtxt('/content/drive/MyDrive/Proyecto_Investigación/HSI/D65.csv',delimiter=',')
    k = 1/np.sum(illumination_reference[:130]*TriEst[:,3])
    X = k*np.sum(illumination_reference[:130]*datacube[:,:130]*TriEst[:,2],axis=1)
    Y = k*np.sum(illumination_reference[:130]*datacube[:,:130]*TriEst[:,3],axis=1)
    Z = k*np.sum(illumination_reference[:130]*datacube[:,:130]*TriEst[:,1],axis=1)

    L = 116*((Y/white_reference[1])**(1/3)) - 16
    a = 500*(((X/white_reference[0])**(1/3))-((Y/white_reference[1])**(1/3)))
    b = 200*(((Y/white_reference[1])**(1/3))-((Z/white_reference[2])**(1/3)))

    Lab = np.column_stack((L,a,b))

    return Lab


def image_segmentation(img,k):

    img = np.array(img, dtype=np.float64)
    img,features = unfolding(img)

    kmeans= KMeans(n_clusters = k, init = "k-means++",random_state = 0, n_init = "auto")
    kmeans.fit(img)

    segmented_img = kmeans.labels_

    info = []
    Lab = []

    for i in range(0,k):

        idx  = np.where(segmented_img == i)
        mask = img[idx]
        info.append(mask.shape[0])

        if i > 0:
            Lab_mean = mask.mean(0)
            Lab.append(Lab_mean)


    info = np.array(info[1:])
    pix_sample = np.sum(info)
    info = (info/pix_sample)*100

    Lab = np.array(Lab)

    segmented_img = np.reshape(segmented_img,features[:-1])
    return segmented_img,info,Lab