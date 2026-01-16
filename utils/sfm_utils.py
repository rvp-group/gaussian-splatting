import numpy as np
import scipy
from scene.colmap_loader import qvec2rotmat, read_extrinsics_binary
import os
import PIL
import torchvision.transforms.functional as tf
import re
from icecream import ic

def split_train_test(image_files, llffhold=80, n_views=None, verbose=True):
    """
    Splits images into train/test sets.

    If n_views is specified, uniformly samples n_views from all images for training,
    then applies llffhold to the remaining images for testing.

    If n_views is not specified, uses LLFF holdout logic (every llffhold-th image for test).
    """
    total_images = len(image_files)

    if n_views is not None and n_views > 0:
        if n_views >= total_images:
            if verbose:
                print(
                    f" ! Requested {n_views} views, but only {total_images} are available. Using all for training."
                )
            train_idx = list(range(total_images))
            test_idx = []
        else:
            # Uniformly sample n_views for training
            selector = np.linspace(0, total_images - 1, num=n_views, dtype=int)
            train_idx = selector.tolist()

            # Get remaining indices
            remaining_idx = [i for i in range(total_images) if i not in train_idx]

            # Apply llffhold to remaining images
            if llffhold > 0 and llffhold < len(remaining_idx):
                # Every llffhold-th image from remaining becomes test
                test_idx = [remaining_idx[i] for i in range(0, len(remaining_idx), llffhold)]
            else:
                # If llffhold >= remaining images, use all remaining
                test_idx = remaining_idx
    else:
        if llffhold > 0:
            test_idx = np.arange(0, total_images, llffhold).tolist()
        else:
            test_idx = []
        train_idx = [i for i in range(total_images) if i not in test_idx]

    train_img_files = [image_files[i] for i in train_idx]
    test_img_files = [image_files[i] for i in test_idx]

    if verbose:
        print(">> Splitting Train-Test Set:")
        print(f" - Total images found: {total_images}")
        print(f" - Train indices ({len(train_idx)}): {train_idx}")
        print(f" - Test indices ({len(test_idx)}): {test_idx}")

    return train_img_files, test_img_files

def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []
    for fname in os.listdir(renders_dir):
        render = PIL.Image.open(renders_dir / fname)
        gt = PIL.Image.open(gt_dir / fname)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)
    
    # Sort based on numerical values in filenames
    def extract_number(filename):
        match = re.search(r'\d+', filename)
        return int(match.group()) if match else float('inf')
    
    # Create sorting indices based on image_names
    indices = sorted(range(len(image_names)), key=lambda k: extract_number(image_names[k]))
    
    # Reorder all lists using the indices
    renders = [renders[i] for i in indices]
    gts = [gts[i] for i in indices]
    image_names = [image_names[i] for i in indices]
    
    return renders, gts, image_names

def align_pose(pose1, pose2):
    mtx1 = np.array(pose1, dtype=np.double, copy=True)
    mtx2 = np.array(pose2, dtype=np.double, copy=True)

    if mtx1.ndim != 2 or mtx2.ndim != 2:
        raise ValueError("Input matrices must be two-dimensional")
    if mtx1.shape != mtx2.shape:
        raise ValueError("Input matrices must be of same shape")
    if mtx1.size == 0:
        raise ValueError("Input matrices must be >0 rows and >0 cols")

    # translate all the data to the origin
    mtx1 -= np.mean(mtx1, 0)
    mtx2 -= np.mean(mtx2, 0)

    norm1 = np.linalg.norm(mtx1)
    norm2 = np.linalg.norm(mtx2)

    if norm1 == 0 or norm2 == 0:
        raise ValueError("Input matrices must contain >1 unique points")

    # change scaling of data (in rows) such that trace(mtx*mtx') = 1
    mtx1 /= norm1
    mtx2 /= norm2

    # transform mtx2 to minimize disparity
    R, s = scipy.linalg.orthogonal_procrustes(mtx1, mtx2)
    mtx2 = mtx2 * s

    return mtx1, mtx2, R

def read_colmap_gt_pose(gt_pose_path, llffhold=8):
    colmap_cam_extrinsics = read_extrinsics_binary(gt_pose_path + '/sparse_5/0/images.bin')
    colmap_cam_extrinsics = {k: v for k, v in sorted(colmap_cam_extrinsics.items(), key=lambda item: item[1].name)}
    all_pose=[]
    for idx, key in enumerate(colmap_cam_extrinsics):
        extr = colmap_cam_extrinsics[key]
        # print(idx, extr.name)
        R = np.transpose(qvec2rotmat(extr.qvec))
        # R = np.array(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)
        pose = np.eye(4,4)
        pose[:3, :3] = R
        pose[:3, 3] = T
        all_pose.append(pose)
    colmap_pose = np.array(all_pose)
    return colmap_pose

