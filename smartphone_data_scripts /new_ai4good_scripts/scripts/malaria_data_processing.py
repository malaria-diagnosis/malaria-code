import pandas as pd
import os
from matplotlib import pyplot as plt
import cv2
import numpy as np
import math

# from zipfile import ZipFile
#%%

df_parasite_loc = pd.read_csv("data/patientid_cellmapping_parasitized.csv")
df_uninfected = pd.read_csv("data/patientid_cellmapping_uninfected.csv")

#%%

# with ZipFile('data/cell_images.zip', 'r') as zipObj:
#    zipObj.extractall('data/cell_images')
def get_patients_and_imgs(dir_):
    patients = list(os.listdir(dir_))
    patients.remove(".DS_Store")
    pat_img_dic = {}
    for pat in patients:
        imgs = list(os.listdir(os.path.join(dir_, pat)))
        imgs.remove("data.txt")
        pat_img_dic[pat] = imgs
    return patients, pat_img_dic
patients, pat_img_dic = get_patients_and_imgs("data/thick_smear_imgs")
#%%
# find imgs with no info
def get_no_info_imgs(patients, pat_img_dic):
    no_info_imgs = {}
    for pat in patients:
        no_info_imgs[pat] = []
        for img in pat_img_dic[pat]:
            try:
                pd.read_csv(
                    "data/image_locs/GT_updated/" + pat + "/" + img.split(".")[0] + ".txt"
                )
            except:
                no_info_imgs[pat].append(img)
    return no_info_imgs
no_info_imgs = get_no_info_imgs(patients, pat_img_dic)
#%%
# delete imgs with no info
def del_imgs_no_info(no_info_imgs):
    for pat in no_info_imgs:
        if no_info_imgs[pat]:
            for img in no_info_imgs[pat]:
                os.remove("data/thick_smear_imgs/" + pat + "/" + img)
    return None

#%% patient 3 img 0 is nice
def get_img_dir(pat_no, img_no):
    pat_img_dir = os.path.join(patients[pat_no],
                               pat_img_dic[patients[pat_no]][img_no])
    return pat_img_dir
pat_img_dir = get_img_dir(4, 0)
img = cv2.imread(
    "data/thick_smear_imgs/" + pat_img_dir
)
#%%

df = pd.read_csv('data/image_locs/GT_updated/'+pat_img_dir.replace('jpg', 'txt'))
no_contenders = df.columns[0]
img_height = int(df.columns[1])
img_width = int(df.columns[2])
df.reset_index(inplace=True)
df.columns = ['id', 'classification', 'comments', 'circle_or_point', 'circ_point_num',
              'x_centre', 'y_centre', 'x_circle_point', 'y_circle_point']
x_diff = df['x_centre'] - df['x_circle_point']
y_diff = df['y_centre'] - df['y_circle_point']
radius = (x_diff**2 + y_diff**2)**0.5
df['radius'] = radius.apply(lambda x: math.ceil(x) if not math.isnan(x) else 0) + 1
df['radius'] = df['radius'].astype(int)
for point in ['x_centre', 'y_centre', 'x_circle_point', 'y_circle_point']:
    df[point] = df[point].apply(lambda x: round(x) if not math.isnan(x) else 0)
    df[point] = df[point].astype(int)
    
#%%
mask_parasite = np.zeros((img_height, img_width))
mask_nucleus = np.zeros((img_height, img_width))
mask_all = np.zeros((img_height, img_width))

df_p_c = df[np.logical_and(df['classification'] == 'Parasite',
                           df['circle_or_point'] == 'Circle')]
df_p_p = df[np.logical_and(df['classification'] == 'Parasite',
                           df['circle_or_point'] == 'Point')]
df_wbc_c = df[np.logical_and(df['classification'] == 'White_Blood_Cell',
                           df['circle_or_point'] == 'Circle')]
df_wbc_p = df[np.logical_and(df['classification'] == 'White_Blood_Cell',
                           df['circle_or_point'] == 'Point')]

def create_circle_masks(df):
    mask_outline = np.zeros((img_height, img_width))
    mask_fill = np.zeros((img_height, img_width))
    for ii in range(len(df)):
        row = df.iloc[ii,:]
        for y in range(row['y_centre'] - row['radius'],row['y_centre'] + row['radius'] +1):
            x_plus = int(round(row['x_centre'] + (-(y-row['y_centre'])**2 + row['radius']**2)**0.5))
            x_minus = int(round(row['x_centre'] - (-(y-row['y_centre'])**2 + row['radius']**2)**0.5))
            mask_outline[x_plus+1, -(y+1)] = 255
            mask_outline[x_minus+1, -(y+1)] = 255
            for x in range(x_minus, x_plus+1):
                mask_fill[x+1, -(y+1)] = 255
    return mask_fill, mask_outline

def create_point_mask(df):
    mask = np.zeros((img_height, img_width))
    for ii in range(len(df)):
        row = df.iloc[ii,:]
        for x_co in range(row['x_centre'] - 1, row['x_centre'] + 2):
            for y_co in range(row['y_centre'] - 1, row['y_centre'] + 2):
                mask[x_co, -y_co] = 255
    return mask

para_mask, para_mask_outline = create_circle_masks(df_p_c)
para_mask += create_point_mask(df_p_p)

wbc_mask, wbc_mask_outline = create_circle_masks(df_wbc_c)
wbc_mask = create_point_mask(df_wbc_p)

mask_all = np.maximum(para_mask, wbc_mask)

img_mask = img.copy()
img_mask[np.where(para_mask_outline == 255)] = 255

cv2.imwrite('data/thick_smear_masks/test.jpg', img_mask)
cv2.imwrite('data/thick_smear_masks/test_mask.jpg', mask_all)

#%%

def crop_img(img, mask, colour_n):
  y_points = np.any(img > colour_n, axis=0)
  x_l, x_r = np.where(y_points)[0][[0, -1]]

  x_points = np.any(img > colour_n, axis=1)
  y_u, y_d = np.where(x_points)[0][[0, -1]]

  width_len = x_r - x_l
  height_len = y_d - y_u

  if width_len >= height_len:
    diff = width_len - height_len
    y_d = int(diff/2) + y_d
    y_u = y_u - math.ceil(diff/2)
    if y_u < 0:
      diff_y = 0 - y_u
      y_u += diff_y
      y_d += diff_y
    if y_d >= img.shape[1]:
      diff_y = img.shape[1] - y_d
      y_u -= diff_y
      y_d -= diff_y
  else:
    diff = height_len - width_len
    x_r = int(diff/2) + x_r
    x_l = x_l - math.ceil(diff/2)
    if x_l < 0:
      diff_x = 0 - x_l
      x_l += diff_x
      x_r += diff_x
    if x_r >= img.shape[0]:
      diff_x = img.shape[0] - x_r
      x_r -= diff_x
      x_l -= diff_x

  img = img[y_u:y_d, x_l:x_r, :]
  mask = mask[y_u:y_d, x_l:x_r]
  return img, mask

test, mask = crop_img(img, para_mask, 20)
test.shape
plt.imshow(test)
plt.imshow(mask)
pd.Series(img[1, :, 0]).value_counts()

#%%

for pat in patients[:2]:
    for img in pat_img_dic[pat]:
        pat_img_dir = os.path.join(pat, img)
        img = cv2.imread("data/thick_smear_imgs/" + pat_img_dir)
        plt.imshow(img)
        plt.show()
        mask = 

