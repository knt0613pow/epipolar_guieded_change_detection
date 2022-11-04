import os
import sys
import cv2
import numpy as np

def xyz2lonlat(xyz):
    atan2 = np.arctan2
    asin = np.arcsin

    norm = np.linalg.norm(xyz, axis=-1, keepdims=True)
    xyz_norm = xyz / norm
    x = xyz_norm[..., 0:1]
    y = xyz_norm[..., 1:2]
    z = xyz_norm[..., 2:]

    lon = atan2(x, z)
    lat = asin(y)
    lst = [lon, lat]

    out = np.concatenate(lst, axis=-1)
    return out

def lonlat2XY(lonlat, shape):
    X = (lonlat[..., 0:1] / (2 * np.pi) + 0.5) * (shape[1] - 1)
    Y = (lonlat[..., 1:] / (np.pi) + 0.5) * (shape[0] - 1)
    lst = [X, Y]
    out = np.concatenate(lst, axis=-1)

    return out 
    
def Panorama2lonlat(label, shape):
    """
    shape : (H, W, 3)
    label : (N,2) # of (x, y) coordinate or panorama image
    x is right direction coordinate
    y is down direction coordinate
    return : (2, N)
    """
    lon = (label[:,0]/(shape[1]) -0.5)*(2*np.pi)
    lat = (label[:,1]/(shape[0])-0.5)*(np.pi)
    lonlat = [lon, lat]
    return np.array(lonlat)

    
def lonlat2xyz(lonlat):
    """
    lonlat : (2,N) 
    xyz : len(xyz) == 3
    return : (N,3)
    """
    lon, lat = lonlat
    x = np.cos(lat) * np.sin(lon)
    y = np.sin(lat)
    z = np.cos(lat) * np.cos(lon)

    return np.concatenate((x,y,z), axis = -1)
    
    
def valid_label(lonlat, THETA):


    lon, lat = lonlat
    theta = THETA / np.pi
    left = (THETA-90)/np.pi
    right = (THETA+90)/np.pi
    if THETA <= -90:
        idx = np.where( (2*np.pi+left)<lon or lon <right)
    elif THETA < 90:
        idx = np.where(left < lon and lon<right)
    else:
        idx = np.where(left < lon or (-2*np.pi +right) > lon)
    return lonlat[:, idx]

class Equirectangular:
    def __init__(self, img_name, label):
        self._img = cv2.imread(img_name, cv2.IMREAD_COLOR)
        self._label = label
        [self._height, self._width, _] = self._img.shape
        #cp = self._img.copy()  
        #w = self._width
        #self._img[:, :w/8, :] = cp[:, 7*w/8:, :]
        #self._img[:, w/8:, :] = cp[:, :7*w/8, :]
    

    def GetPerspective(self, FOV, THETA, PHI, height, width):
        #
        # THETA is left/right angle, PHI is up/down angle, both in degree
        #

        f = 0.5 * width * 1 / np.tan(0.5 * FOV / 180.0 * np.pi)
        cx = (width - 1) / 2.0
        cy = (height - 1) / 2.0
        K = np.array([
                [f, 0, cx],
                [0, f, cy],
                [0, 0,  1],
            ], np.float32)
        K_inv = np.linalg.inv(K)
        x = np.arange(width)
        y = np.arange(height)
        x, y = np.meshgrid(x, y)  
        z = np.ones_like(x)
        xyz = np.concatenate([x[..., None], y[..., None], z[..., None]], axis=-1)
        xyz = xyz @ K_inv.T

        y_axis = np.array([0.0, 1.0, 0.0], np.float32)
        x_axis = np.array([1.0, 0.0, 0.0], np.float32)
        R1, _ = cv2.Rodrigues(y_axis * np.radians(THETA))
        R2, _ = cv2.Rodrigues(np.dot(R1, x_axis) * np.radians(PHI))
        R = R2 @ R1
        xyz = xyz @ R.T
        lonlat = xyz2lonlat(xyz) 
        XY = lonlat2XY(lonlat, shape=self._img.shape).astype(np.float32)
        persp = cv2.remap(self._img, XY[..., 0], XY[..., 1], cv2.INTER_CUBIC, borderMode=cv2.BORDER_WRAP)

        return persp
    def GetPerspective_label(self, FOV, THETA, PHI, height, width):
        lonlat = Panorama2lonlat(self._label, shape = self._img.shape)
        lonlat = valid_label(lonlat, THETA)
        xyz = lonlat2xyz(lonlat)
        f = 0.5 * width * 1 / np.tan(0.5 * FOV / 180.0 * np.pi)
        
        y_axis = np.array([0.0, 1.0, 0.0], np.float32)
        x_axis = np.array([1.0, 0.0, 0.0], np.float32)
        R1, _ = cv2.Rodrigues(y_axis * np.radians(THETA))
        R2, _ = cv2.Rodrigues(np.dot(R1, x_axis) * np.radians(PHI))
        R = R2 @ R1
        R_inv = np.linalg.inv(R)
        camera_xyz = xyz@R_inv.T
        camera_xyz =camera_xyz / camera_xyz[:,2]

        
        
        cx = (width - 1) / 2.0
        cy = (height - 1) / 2.0
        K = np.array([
                [f, 0, cx],
                [0, f, cy],
                [0, 0, 1],
            ], np.float32)        
        camera_xyz = camera_xyz @ K.T
        
        return camera_xyz
        
        
        
        
        
        
import json
        
        
if __name__ == '__main__':
    img_path = 'test.jpg'
    label = 'test.json'
    Er = Equirectangular(img_path, label)
    per = Er.GetPerspective(90, -60, 0, 720, 1080)
    with open('test.json', 'r') as f:
      json_data = json.load(f)
    breakpoint()
    Er._label = np.array([928.125, 562.5]).reshape(1,-1)
    
    
    label_idx = Er.GetPerspective_label(90, -60, 0, 720, 1080)
    breakpoint()
    per = cv2.circle(per,(int(label_idx[0,0]), int(label_idx[0,1])),8,(0,0,255),3)
    cv2.imwrite('test_result.jpg', per)
    #should be ~345,30
