import os
import cv2
import re

import yaml

import numpy as np


def safe_make(path, exist_ok = True): 
    '''
    safely makes a directory and returns the path
    '''
    os.makedirs(path, exist_ok = exist_ok)
    return path


def format_sample_input(x):

    if isinstance(x, int): 
        return x
    elif isinstance(x, list): 
        return tuple(x) 
    else: 
        return None
    

def scale_coords(coords, orig_res, new_res):
        '''
        given 2d coordinates, scales the x and y pixels 
        independently according to H / H' and W / W' where 
        orig_res = (H, W) and new_res = (H', W')
        '''

        # must be 2d
        assert coords.shape[-1] == 2

        scale = np.array([orig_res[1] / new_res[1],
                          orig_res[0] / new_res[0]])

        coords_scaled = coords / scale

        return coords_scaled


def extract_name(fname, pattern):
    '''
    uses regex to extract name from data path
    '''
    pattern_compiled = re.compile(pattern)

    base_name = os.path.basename(fname)
    m = pattern_compiled.search(base_name)

    if m is not None:
        name = m[0]
    else: 
        name = ''

    return name


def extract_num(fname, pattern):
    '''
    uses regex to extract num from data path
    '''

    base_name = os.path.basename(fname)
    m = re.findall(pattern, base_name)

    if m is not None:
        name = m[0]
        if name.isdigit():
            name = int(name)
    else: 
        name = ''

    return name


def get_dirs(path): 

    dirs = os.listdir(path)

    dirs = [d for d in dirs if os.path.isdir(os.path.join(path, d))
            and not d.startswith('.')]
    
    # sort numerically if all folders are numerical, 
    # otherwise sort alphabetically
    if all(d.isdigit() for d in dirs):
        dirs = sorted(dirs, key = int)
    else: 
        dirs = sorted(dirs) 
    
    return dirs


def load_yaml(path): 
    '''
    safely loads data from a yaml file
    '''
    with open(path, 'r') as f: 
        data = yaml.safe_load(f)

    return data


def disassemble_extrinsics(extrinsics): 
    ''' 
    returns the rotation and translation vectors 
    given the extrinsics matrix
    '''
    extrinsics = np.array(extrinsics)
    rotation_matrix = extrinsics[:3, :3]
    rvec, _ = cv2.Rodrigues(rotation_matrix)
    tvec = extrinsics[:3, 3]

    return rvec, tvec