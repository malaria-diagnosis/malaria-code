import pandas as pd
import os
from matplotlib import pyplot as plt
import cv2
import numpy as np
import math
from tqdm import tqdm
# from zipfile import ZipFile

DATA_DIR = '/Users/Alex/Google Drive/AI4good/datasets/thick_smears_150/'
LABELS_DIR = DATA_DIR+'GT_updated/'
DATA_PREPARED_DIR = '/Users/Alex/Google Drive/AI4good/datasets/thick_smears_150_prepared/'
if not os.path.exists(DATA_PREPARED_DIR):
    os.mkdir(DATA_PREPARED_DIR)

# with ZipFile('data/cell_images.zip', 'r') as zipObj:
#    zipObj.extractall('data/cell_images')
def get_patients_and_imgs(dir_):
    '''
    Parameters
    ----------
    dir_ : str
        directory containing the patient folders

    Returns
    -------
    patients : list
        list of patients
    pat_img_dic : dict
        dictionary where key is the patient and value is a list of imgages
        from the patient
    '''
    patients = [p for p in list(os.listdir(dir_)) if p !='.DS_Store' and not p.endswith('.docx')]
    pat_img_dic = {}
    for pat in patients:
        imgs = [i for i in list(os.listdir(os.path.join(dir_, pat))) if i !='data.txt']
        pat_img_dic[pat] = imgs
    return patients, pat_img_dic
#%%
# find imgs with no info
def get_no_info_imgs(patients, pat_img_dic):
    '''
    Parameters
    ----------
    patients : list
        list of patients
    pat_img_dic : dict
        dictionary where key is the patient and value is a list of imgages
        from the patient
    Returns
    -------
    no_info_imgs : dict
        keys are patients values are list of images containing no textfile of 
        information
    '''
    no_info_imgs = {}
    for pat in patients:
        no_info_imgs[pat] = []
        for img in pat_img_dic[pat]:
            try:
                pd.read_csv(
                    LABELS_DIR + pat + '/' + img.split('.')[0] + '.txt'
                )
            except:
                no_info_imgs[pat].append(img)
    return no_info_imgs

#%%
# delete imgs with no info
def del_imgs_no_info(no_info_imgs):
    '''
    Deletes images with no textfile of information

    Parameters
    ----------
    no_info_imgs : dict
        keys are patients values are list of images containing no textfile of 
        information

    Returns
    -------
    None.

    '''
    for pat in no_info_imgs:
        if no_info_imgs[pat]:
            for img in no_info_imgs[pat]:
                os.remove(DATA_DIR + pat + '/' + img)
    return None

#%% patient 3 img 0 is nice
def get_img_dir(pat_no, img_no):
    '''
    

    Parameters
    ----------
    pat_no : TYPE
        DESCRIPTION.
    img_no : TYPE
        DESCRIPTION.

    Returns
    -------
    pat_img_dir : TYPE
        DESCRIPTION.

    '''
    pat_img_dir = os.path.join(patients[pat_no],
                               pat_img_dic[patients[pat_no]][img_no])
    return pat_img_dir

#%%
def get_img_info(pat_img_dir):
    '''
    Creates a df used to create parasite masks

    Parameters
    ----------
    pat_img_dir : string
        directory of image

    Returns
    -------
    df : DataFrame
        dataframe of parasite and WBC locations
    img_width : int
        image width
    img_height : int
        image height
    '''
    df = pd.read_csv(LABELS_DIR+pat_img_dir.replace('jpg', 'txt'))
    img_height = int(df.columns[1])
    img_width = int(df.columns[2])
    df.reset_index(inplace=True)
    #TODO! read the df better so we get WBC masks when there are no circle masks
    if df.shape[1] <= 1 or df.shape[1] != 9: #if the textfile contains no info return None
        df = None
        return df, img_width, img_height
    df.columns = ['id', 'classification', 'comments', 'circle_or_point', 'circ_point_num',
                  'y_centre', 'x_centre', 'y_point_circle', 'x_point_circle']
    y_diff = df['y_centre'] - df['y_point_circle']
    x_diff = df['x_centre'] - df['x_point_circle']
    radius = (x_diff**2 + y_diff**2)**0.5
    df['radius'] = radius.apply(lambda x: math.ceil(x) if not math.isnan(x) else 0) + 1
    df['radius'] = df['radius'].astype(int)
    for point in ['y_centre', 'x_centre', 'y_point_circle', 'x_point_circle']:
        df[point] = df[point].apply(lambda x: round(x) if not math.isnan(x) else 0)
        df[point] = df[point].astype(int)
    return df, img_width, img_height
    
#%%

def seperate_img_info_df(df):
    '''
    

    Parameters
    ----------
    df : DataFrame
        contains all mask information

    Returns
    -------
    df_p_c : DataFrame
        contains parasite circles
    df_p_p : DataFrame
        contains parasite points
    df_wbc_c : DataFrame
        contains WBC circles
    df_wbc_p : DataFrame
        contains WBC points
    '''
    df_p_c = df[np.logical_and(df['classification'] == 'Parasite',
                               df['circle_or_point'] == 'Circle')]
    df_p_p = df[np.logical_and(df['classification'] == 'Parasite',
                               df['circle_or_point'] == 'Point')]
    df_wbc_c = df[np.logical_and(df['classification'] == 'White_Blood_Cell',
                               df['circle_or_point'] == 'Circle')]
    df_wbc_p = df[np.logical_and(df['classification'] == 'White_Blood_Cell',
                               df['circle_or_point'] == 'Point')]
    return df_p_c, df_p_p, df_wbc_c, df_wbc_p

#%%
def create_circle_masks(df, img_height, img_width):
    '''
    Creates masks

    Parameters
    ----------
    df : DataFrame
        contains all mask information
    img_height : int
        image height
    img_width : int
        image width

    Returns
    -------
    mask_fill : numpy.ndarray
        mask of parasites or WBC
    mask_outline : numpy.ndarray
        outline of parasites or WBC

    '''
    mask_outline = np.zeros((img_height, img_width))
    mask_fill = np.zeros((img_height, img_width))
    for ii in range(len(df)):
        row = df.iloc[ii,:]
        for x in range(row['x_centre'] - row['radius'],row['x_centre'] + row['radius'] +1):
            y_plus = int(round(row['y_centre'] + (-(x-row['x_centre'])**2 + row['radius']**2)**0.5))
            y_minus = int(round(row['y_centre'] - (-(x-row['x_centre'])**2 + row['radius']**2)**0.5))
            mask_outline[y_plus+1, -(x+1)] = 255
            mask_outline[y_minus+1, -(x+1)] = 255
            for y in range(y_minus, y_plus+1):
                mask_fill[y+1, -(x+1)] = 255
    return mask_fill, mask_outline

#%%
def create_point_mask(df, img_height, img_width):
    '''
    Creates masks

    Parameters
    ----------
    df : DataFrame
        contains all mask information
    img_height : int
        image height
    img_width : int
        image width

    Returns
    -------
    mask : numpy.ndarray
        mask points of parasites or WBC

    '''
    mask = np.zeros((img_height, img_width))
    for ii in range(len(df)):
        row = df.iloc[ii,:]
        for y_co in range(row['y_centre'] - 1, row['y_centre'] + 2):
            for x_co in range(row['x_centre'] - 1, row['x_centre'] + 2):
                mask[y_co, -x_co] = 255 #x_co is negative as data reads from right to left
    return mask

#%%
def create_total_mask(df_p_c, df_p_p, img_height, img_width):
    '''
    combines two masks

    Parameters
    ----------
    df_p_c : DataFrame
        df of circle mask info
    df_p_p : DataFrame
        df of point mask info
    img_height : int
        image height
    img_width : int
        image width

    Returns
    -------
    para_mask : numpy.ndarray
        mask of points and circles

    '''
    para_mask, para_mask_outline = create_circle_masks(df_p_c, img_height, img_width)
    para_mask = np.maximum(para_mask, create_point_mask(df_p_p, img_height, img_width))
    return para_mask

#%%

def crop_img(img, mask, colour_n):
    '''
    

    Parameters
    ----------
    img : numpy.ndarray
        image of thick blood smear
    mask : numpy.ndarray
        mask of thick blood smear
    colour_n : int
        point at which any colour pixel can be until image is cropped

    Returns
    -------
    img : numpy.ndarray
        cropped image
    mask : numpy.ndarray
        cropped mask

    '''
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
#%%
    
def get_para_mask(pat_img_dir):
    '''
    

    Parameters
    ----------
    pat_img_dir : str
        directory of image

    Returns
    -------
    para_mask : TYPE
        numpy.ndarray

    '''
    df, img_width, img_height = get_img_info(pat_img_dir)
    if df is not None: #checks if df exists
        df_p_c, df_p_p, df_wbc_c, df_wbc_p = seperate_img_info_df(df)
        para_mask = create_total_mask(df_p_c, df_p_p, img_height, img_width)
    else:
        para_mask = np.zeros((img_height, img_width))
    return para_mask

#%%

def create_and_save_small_imgs(img, para_mask, small_img_size,  dir_):
    '''
    Creates 25 small_img_size x small_img_size images for each picture
    This is useful for u-net model

    Parameters
    ----------
    img : numpy.ndarray
        image of thick blood smear
    para_mask : numpy.ndarray
        parasite mask of thick blood smear
    small_img_size : int
        size of the small image
    dir_ : string
        string of the directory to save images in

    Returns
    -------
    None.

    '''
    for yy in range(5):
        y_start = int(yy*(img.shape[0]/5 - 5*(small_img_size - img.shape[0]/5)/4))
        for xx in range(5):
            x_start = int(xx*(img.shape[1]/5 - 5*(small_img_size - img.shape[1]/5)/4))
        
            small_img = img[y_start:y_start + small_img_size,
                            x_start:x_start + small_img_size,
                            :]
            small_mask = para_mask[y_start:y_start + small_img_size,
                                   x_start:x_start + small_img_size]
            plt.imshow(small_mask)
            plt.show()
            assert(np.logical_and(small_img.shape[:2] == small_mask.shape,
                                  small_mask.shape == (610, 610)))
            cv2.imwrite(dir_+'mask_x'+str(xx)+'_y' +str(yy)+'.jpg', small_mask)
            cv2.imwrite(dir_+'img_x'+str(xx)+'_y' +str(yy)+'.jpg', small_img)
    return None

#%% create small images
patients, pat_img_dic = get_patients_and_imgs(DATA_DIR)

thick_smear_dir = DATA_DIR
for pat in tqdm(patients):
    pat_small_dir = DATA_PREPARED_DIR+str(pat)
    if pat not in os.listdir(DATA_PREPARED_DIR):
        os.mkdir(pat_small_dir)
    for img_name in pat_img_dic[pat]:
        pat_img_path = os.path.join(pat, img_name)
        img = cv2.imread(thick_smear_dir + pat_img_path)
        para_mask = get_para_mask(pat_img_path)
        img, para_mask = crop_img(img, para_mask, 100)
        small_img_dir = os.path.join(pat_small_dir, img_name[:-4]) #-4 to remove .jpg
        if img_name[:-4] not in os.listdir(pat_small_dir + '/'):
            os.mkdir(small_img_dir)
        if len(os.listdir(pat_small_dir + '/')) > 50:
            create_and_save_small_imgs(img, para_mask, 610, small_img_dir+'/')





