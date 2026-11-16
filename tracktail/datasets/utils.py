import os
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
    import cv2
    extrinsics = np.array(extrinsics)
    rvec, _ = cv2.Rodrigues(extrinsics[:3, :3])
    tvec = extrinsics[:3, 3]

    return rvec, tvec


def sort_camera_names(cam_names):
    '''sort camera names numerically if they are all digits, else alphabetically'''
    cam_names = list(cam_names)
    if all(str(c).isdigit() for c in cam_names):
        return sorted(cam_names, key = int)
    return sorted(cam_names)


def load_camera_metadata(camera_metadata, cam_names = None, fisheye = False):
    '''
    build an aniposelib CameraGroup from a metadata.yaml (path or loaded dict).

    returns (cgroup, offset_dict, cam_type, moving_ext). cameras are sorted numerically
    or alphabetically unless cam_names gives the cameras (and their order) to build.
    with `moving_cams: true` the extrinsics are per frame (T,4,4): the Camera holds the
    frame-0 pose and the full stack is returned in moving_ext {cam_name: (T,4,4)};
    per-frame intrinsics/distortions (time-invariant) collapse to frame 0. otherwise
    moving_ext is None. fisheye=True builds FisheyeCameras (only viz3d uses this; the
    model reads cam_type from the returned value).
    '''
    from aniposelib.cameras import Camera, CameraGroup, FisheyeCamera

    if isinstance(camera_metadata, (str, os.PathLike)):
        source = camera_metadata
        cam_metadata = load_yaml(camera_metadata)
    else:
        source = 'camera metadata'
        cam_metadata = camera_metadata

    offset_dict = cam_metadata.get('offset_dict', None)
    if offset_dict:
        # aniposelib cameras store str names; key the per-camera dicts the same way
        offset_dict = {str(k): v for k, v in offset_dict.items()}
    cam_type = cam_metadata.get('cam_type', 'pinhole')
    moving = bool(cam_metadata.get('moving_cams', False))
    moving_ext = {} if moving else None

    intrinsics_dict = cam_metadata['intrinsic_matrices']
    extrinsics_dict = cam_metadata['extrinsic_matrices']
    distortions_dict = cam_metadata['distortion_matrices']
    widths_dict = cam_metadata.get('camera_widths')
    heights_dict = cam_metadata.get('camera_heights')

    if cam_names is None:
        cam_names = sort_camera_names(intrinsics_dict.keys())

    cam_cls = FisheyeCamera if fisheye else Camera
    cams = []
    for cam_name in cam_names:
        ext = np.asarray(extrinsics_dict[cam_name], dtype = np.float64)
        mat = np.asarray(intrinsics_dict[cam_name], dtype = np.float64)
        dist = np.asarray(distortions_dict[cam_name], dtype = np.float64)
        if moving:
            if ext.ndim != 3 or ext.shape[-2:] != (4, 4):
                # raise rather than degrade: a static extrinsic for a rig the metadata says
                # is moving would make the pose lag the subject with no diagnostic.
                raise ValueError(
                    f'{source}: moving_cams=true but extrinsics for {cam_name} have shape '
                    f'{tuple(ext.shape)} (expected (T,4,4)). Refusing to fall back to a '
                    f'static extrinsic.')
            moving_ext[str(cam_name)] = ext
            ext = ext[0]
            mat = mat[0] if mat.ndim == 3 else mat
            dist = dist[0] if dist.ndim == 2 else dist

        size = None
        if widths_dict is not None and heights_dict is not None:
            size = (widths_dict[cam_name], heights_dict[cam_name])

        cam = cam_cls.from_extrinsics_mat(mat, ext, dist = dist, name = str(cam_name))
        cam.set_size(size)  # kept as given (from_extrinsics_mat would round to int)
        cams.append(cam)

    return CameraGroup(cams), offset_dict, cam_type, moving_ext