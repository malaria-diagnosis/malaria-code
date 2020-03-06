import pandas as pd
import os
from matplotlib import pyplot as plt
import cv2
import numpy as np
import math
from tqdm import tqdm
import time
import shutil

#%%
def get_patients_and_imgs(dir_):
    """
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
    """
    patients = list(os.listdir(dir_))
    if ".DS_Store" in patients:
        patients.remove(".DS_Store")
    pat_img_dic = {}
    for pat in patients:
        imgs = list(os.listdir(os.path.join(dir_, pat)))
        if "data.txt" in imgs:
            imgs.remove("data.txt")
        pat_img_dic[pat] = imgs
    return patients, pat_img_dic


#%%
def get_no_info_imgs(patients, pat_img_dic, image_loc_path=image_loc_path):
    """
    Finds images with no textfile of information
    
    Parameters
    ----------
    patients : list
        list of patients
    pat_img_dic : dict
        dictionary where key is the patient and value is a list of imgages
        from the patient
    image_loc_path : str
        image location path
    Returns
    -------
    no_info_imgs : dict
        keys are patients values are list of images containing no textfile of 
        information
    """
    image_loc_path = os.path.join(image_loc_path, "GT_updated")
    no_info_imgs = {}
    for pat in patients:
        path = os.path.join(image_loc_path, pat)
        no_info_imgs[pat] = []
        for img in pat_img_dic[pat]:
            filename = img.split(".")[0] + ".txt"
            path_and_filename = os.path.join(path, filename)
            try:
                pd.read_csv(path_and_filename)
            except:
                no_info_imgs[pat].append(img)
    return no_info_imgs


#%%
# delete imgs with no info
def del_imgs_no_info(no_info_imgs, thick_img_path):
    """
    Deletes images with no textfile of information

    Parameters
    ----------
    no_info_imgs : dict
        keys are patients values are list of images containing no textfile of 
        information

    Returns
    -------
    None.

    """
    for pat in no_info_imgs:
        pat_path = os.path.join(thick_img_path, pat)
        if no_info_imgs[pat]:
            for img in no_info_imgs[pat]:
                img_path = os.path.join(pat_path, img)
                if os.path.exists(img_path):
                    os.remove(img_path)
    return None


#%%
def get_img_dir(pat_no, img_no):
    """
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

    """
    pat_img_dir = os.path.join(patients[pat_no], pat_img_dic[patients[pat_no]][img_no])
    return pat_img_dir


#%%
def get_img_info(pat_img_dir):
    """
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
    """
    patients_info_path = os.path.join(image_loc_path, "GT_updated")
    pat_img_info_path = os.path.join(
        patients_info_path, pat_img_dir.replace("jpg", "txt")
    )
    df = pd.read_csv(pat_img_info_path)
    img_height = int(df.columns[1])
    img_width = int(df.columns[2])
    df.reset_index(inplace=True)
    # TODO! read the df better so we get WBC masks when there are no circle masks
    if (df.shape[1] <= 1 or df.shape[1] != 9):
        # if the textfile contains no info return None
        df = None
        return df, img_width, img_height
    df.columns = [
        "id",
        "classification",
        "comments",
        "circle_or_point",
        "circ_point_num",
        "y_centre",
        "x_centre",
        "y_point_circle",
        "x_point_circle",
    ] #reads in y first as the read me is the wrong way around
    y_diff = df["y_centre"] - df["y_point_circle"]
    x_diff = df["x_centre"] - df["x_point_circle"]
    radius = (x_diff ** 2 + y_diff ** 2) ** 0.5
    df["radius"] = radius.apply(lambda x: math.ceil(x) if not math.isnan(x) else 0) + 1
    df["radius"] = df["radius"].astype(int)
    for point in ["y_centre", "x_centre", "y_point_circle", "x_point_circle"]:
        df[point] = df[point].apply(lambda x: round(x) if not math.isnan(x) else 0)
        df[point] = df[point].astype(int)
    return df, img_width, img_height


#%%
def seperate_img_info_df(df):
    """
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
    """
    df_p_c = df[
        np.logical_and(
            df["classification"] == "Parasite", df["circle_or_point"] == "Circle"
        )
    ]
    df_p_p = df[
        np.logical_and(
            df["classification"] == "Parasite", df["circle_or_point"] == "Point"
        )
    ]
    df_wbc_c = df[
        np.logical_and(
            df["classification"] == "White_Blood_Cell",
            df["circle_or_point"] == "Circle",
        )
    ]
    df_wbc_p = df[
        np.logical_and(
            df["classification"] == "White_Blood_Cell", df["circle_or_point"] == "Point"
        )
    ]
    return df_p_c, df_p_p, df_wbc_c, df_wbc_p


#%%
def create_circle_masks(df, img_height, img_width):
    """
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

    """
    mask_outline = np.zeros((img_height, img_width))
    mask_fill = np.zeros((img_height, img_width))
    for ii in range(len(df)):
        row = df.iloc[ii, :]
        for x in range(
            row["x_centre"] - row["radius"], row["x_centre"] + row["radius"] + 1
        ):
            y_plus = int(
                round(
                    row["y_centre"]
                    + (-((x - row["x_centre"]) ** 2) + row["radius"] ** 2) ** 0.5
                )
            )
            y_minus = int(
                round(
                    row["y_centre"]
                    - (-((x - row["x_centre"]) ** 2) + row["radius"] ** 2) ** 0.5
                )
            )
            mask_outline[y_plus + 1, -(x + 1)] = 255
            mask_outline[y_minus + 1, -(x + 1)] = 255
            for y in range(y_minus, y_plus + 1):
                mask_fill[y + 1, -(x + 1)] = 255
                # x_co is negative as data reads from right to left
    return mask_fill, mask_outline


#%%
def create_point_mask(df, img_height, img_width):
    """
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

    """
    mask = np.zeros((img_height, img_width))
    for ii in range(len(df)):
        row = df.iloc[ii, :]
        for y_co in range(row["y_centre"] - 1, row["y_centre"] + 2):
            for x_co in range(row["x_centre"] - 1, row["x_centre"] + 2):
                mask[y_co, -x_co] = 255
                # x_co is negative as data reads from right to left
    return mask


#%%
def create_total_mask(df_p_c, df_p_p, img_height, img_width):
    """
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

    """
    para_mask, para_mask_outline = create_circle_masks(df_p_c, img_height, img_width)
    para_mask = np.maximum(para_mask, create_point_mask(df_p_p, img_height, img_width))
    return para_mask


#%%
def crop_img(img, mask, colour_n):
    """
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

    """
    y_points = np.any(img > colour_n, axis=0)
    x_l, x_r = np.where(y_points)[0][[0, -1]]

    x_points = np.any(img > colour_n, axis=1)
    y_u, y_d = np.where(x_points)[0][[0, -1]]

    width_len = x_r - x_l
    height_len = y_d - y_u

    if width_len >= height_len:
        diff = width_len - height_len
        y_d = int(diff / 2) + y_d
        y_u = y_u - math.ceil(diff / 2)
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
        x_r = int(diff / 2) + x_r
        x_l = x_l - math.ceil(diff / 2)
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
    """
    Parameters
    ----------
    pat_img_dir : str
        directory of image

    Returns
    -------
    para_mask : TYPE
        numpy.ndarray

    """
    df, img_width, img_height = get_img_info(pat_img_dir)
    if df is not None:  # checks if df exists
        df_p_c, df_p_p, df_wbc_c, df_wbc_p = seperate_img_info_df(df)
        para_mask = create_total_mask(df_p_c, df_p_p, img_height, img_width)
    else:
        para_mask = np.zeros((img_height, img_width))
    return para_mask


#%%
def create_and_save_small_imgs(img, para_mask, small_img_size, dir_):
    """
    Creates 25 images of shape (small_img_size x small_img_size) for each picture
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

    """
    for yy in range(5):
        y_start = int(
            yy * (img.shape[0] / 5 - 5 * (small_img_size - img.shape[0] / 5) / 4)
        )
        # y_start and x_start are the starting points of each small image
        # this works for different sized images
        for xx in range(5):
            x_start = int(
                xx * (img.shape[1] / 5 - 5 * (small_img_size - img.shape[1] / 5) / 4)
            )

            small_img = img[
                y_start : y_start + small_img_size,
                x_start : x_start + small_img_size,
                :,
            ]
            small_mask = para_mask[
                y_start : y_start + small_img_size, x_start : x_start + small_img_size
            ]
            assert np.logical_and(
                small_img.shape[:2] == small_mask.shape,
                small_mask.shape == (small_img_size, small_img_size),
            )
            mask_filename = "mask_x" + str(xx) + "_y" + str(yy) + ".jpg"
            img_filename = "img_x" + str(xx) + "_y" + str(yy) + ".jpg"
            cv2.imwrite(os.path.join(dir_, mask_filename), small_mask)
            cv2.imwrite(os.path.join(dir_, img_filename), small_img)
    return None


#%%
def define_folder_names():
    """
    Returns
    -------
    thick_img_path : str
        thick smear images path.
    image_loc_path : str
        image information path.
    small_thick_smear_path : str
        small thick smear images path

    """
    thick_img_path = os.path.join("data", "thick_smear_imgs")
    image_loc_path = os.path.join("data", "image_locs")
    small_thick_smear_path = os.path.join("data", "small_thick_smear_imgs")
    return thick_img_path, image_loc_path, small_thick_smear_path


#%%
def setup_folder_structure():
    """
    Moves folders around and sets up the folder structure.
    Only works if the folders have the same setup as:
    ftp://lhcftp.nlm.nih.gov/Open-Access-Datasets/Malaria/Thick_Smears_150
    and python is in this directory
    
    Returns
    -------
    None.

    """
    patient_names = pd.Series(os.listdir())
    patient_names = patient_names[patient_names.str[:2] == "TF"]
    thick_img_path, image_loc_path, small_thick_smear_path = define_folder_names()
    folders_to_make = ["data", thick_img_path, image_loc_path, small_thick_smear_path]
    for folder in folders_to_make:
        if folder not in os.listdir():
            try:
                os.mkdir(folder)
            except OSError as err:
                print(err)
    for pat in patient_names:
        if pat in os.listdir():
            shutil.move(pat, thick_img_path)
    if "GT_updated" in os.listdir():
        shutil.move("GT_updated", image_loc_path)
    return None


#%% create small images

if __name__ == "__main__":
    thick_img_path, image_loc_path, small_thick_smear_path = define_folder_names()

    setup_folder_structure()
    patients, pat_img_dic = get_patients_and_imgs(thick_img_path)
    no_info_imgs = get_no_info_imgs(patients, pat_img_dic)
    del_imgs_no_info(no_info_imgs, thick_img_path)

    for pat in tqdm(patients):
        pat_small_dir = os.path.join(small_thick_smear_path, str(pat))
        if pat not in os.listdir(small_thick_smear_path):
            os.mkdir(pat_small_dir)
        for img_name in pat_img_dic[pat]:
            pat_img_path = os.path.join(pat, img_name)
            img = cv2.imread(os.path.join(thick_img_path, pat_img_path))
            para_mask = get_para_mask(pat_img_path)
            img, para_mask = crop_img(img, para_mask, 100)
            small_img_dir = os.path.join(
                pat_small_dir, img_name[:-4]
            )  # -4 to remove .jpg
            if img_name[:-4] not in os.listdir(pat_small_dir):
                os.mkdir(small_img_dir)
            if len(os.listdir(pat_small_dir)) < 50:
                create_and_save_small_imgs(img, para_mask, 610, small_img_dir)
                time.sleep(2) #2 second wait seems to help me, but feel free to delete
