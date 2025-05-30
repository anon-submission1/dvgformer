# modified from https://github.com/lisiyao21/Bailando/blob/main/utils/metrics_new.py

import os
import io
import json
import re
import h5py
import shutil
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import re
from PIL import Image
from scipy import linalg
from src.cimtr.cimtr import CImTr, CImTrConfig
from src.preparation.youtube_download import get_video_stat
from src.data.state_action_conversion import get_states_actions, reverse_states_actions, state_avg, state_std, action_avg, action_std
from src.data.drone_path_seq_dataset import speed_multipliers
from src.blender.blender_camera_env import R_blender_cam_dir, R_blender_from_colmap, R_colmap_from_blender
from src.utils.quaternion_operations import euler2quat, euler2mat, mat2quat, interpolate_tvecs, interpolate_qvecs, add_angular_velocity_to_quaternion, convert_to_local_frame, convert_to_global_frame
from src.utils.padding import padding
from scipy.spatial.distance import pdist


fps = 15


def extract_feats(pred_dir, gt_dir, gt_h5fname='dataset_full.h5', split_name='val', feature_type='kinetic',
                  cimtr_path='logs/CImTr-trans-ALds5-lr0.001b512ep200-losst2t1contrast0.1-augFSTC-flashed-cobbler', quiet=True):
    """"
    Extract kinetic features from camera trajectories and save them.
    Parameters:
        pred_dir (str): Directory containing the predicted camera trajectories.
        gt_dir (str): Directory containing the ground truth camera trajectories.
        gt_h5fname (str): Name of the ground truth h5 file.
        split_name (str): Name of the split (e.g., 'train', 'val').
        feature_type (str): Type of feature to extract. Either 'kinetic' or 'cimtr'.
        cimtr_path (str): Path to CImTr model checkpoint.
        quiet (bool): If True, suppress output messages.
    """
    from transformers import logging as hf_logging
    hf_logging.set_verbosity_error()

    pred_traj_dir = f'{pred_dir}/camera_trajs'
    if os.path.exists(pred_traj_dir):
        shutil.rmtree(pred_traj_dir)
    pred_init_frame_dir = f'{pred_dir}/init_frames'
    if os.path.exists(pred_init_frame_dir):
        shutil.rmtree(pred_init_frame_dir)
    pred_feature_dir = f'{pred_dir}/{feature_type}_features'
    if os.path.exists(pred_feature_dir):
        shutil.rmtree(pred_feature_dir)
    extract_pred_trajs_images(pred_dir)
    calc_and_save_feats(pred_dir, feature_type=feature_type,
                        cimtr_path=cimtr_path,
                        quiet=quiet)

    gt_traj_dir = f'{gt_dir}/{gt_h5fname.replace(".h5", "")}/camera_trajs'
    gt_init_frame_dir = f'{gt_dir}/{gt_h5fname.replace(".h5", "")}/init_frames'
    gt_feature_dir = f'{gt_dir}/{gt_h5fname.replace(".h5", "")}/{feature_type}_features'
    if not os.path.exists(gt_traj_dir) or not os.path.exists(gt_init_frame_dir):
        if os.path.exists(gt_traj_dir):
            shutil.rmtree(gt_traj_dir)
        if os.path.exists(gt_init_frame_dir):
            shutil.rmtree(gt_init_frame_dir)
        extract_gt_trajs_images(gt_dir, gt_h5fname, split_name=split_name)
    if not os.path.exists(gt_feature_dir):
        calc_and_save_feats(f'{gt_dir}/{gt_h5fname.replace(".h5", "")}', feature_type=feature_type,
                            cimtr_path=cimtr_path,
                            quiet=quiet)


def calc_and_save_feats(root, feature_type='kinetic', cimtr_path=None,
                        t_resolution=5, batch_size=16, quiet=True):
    """
    Calculate and save kinetic features for camera trajectories.
    Parameters:
        root (str): Root directory for camera trajectories, init images, and saving the features.
        feature_type (str): Type of feature to extract. Either 'kinetic' or 'cimtr'.
        cimtr_path (str): Path to CImTr model checkpoint.
        t_resolution (int): Temporal resolution of the camera trajectory. Only used for 'kinetic'.
        batch_size (int): Batch size for processing. Only used for 'cimtr'.
        quiet (bool): If True, suppress output messages.
    """

    if feature_type == 'cimtr':
        assert cimtr_path is not None, "CImTr model path is required for feature_type 'cimtr'"
        model = CImTr.from_pretrained(cimtr_path)
        model.eval()
        model.to('cuda')
        transform = T.Compose([
            T.Resize(model.config.image_resolution),
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
    else:
        assert feature_type == 'kinetic', "feature_type must be either 'kinetic' or 'cimtr'"
        model = None
        transform = None

    camera_traj_dir = f'{root}/camera_trajs'
    init_frame_dir = f'{root}/init_frames'
    camera_feature_dir = f'{root}/{feature_type}_features'

    fnames = []
    for fname in sorted(os.listdir(camera_traj_dir)):
        if fname.endswith('states.txt'):
            fnames.append(fname)
            assert os.path.exists(
                f'{camera_traj_dir}/{fname.replace("states.txt", "actions.txt")}')
            if init_frame_dir is not None:
                assert os.path.exists(
                    f'{init_frame_dir}/{fname.replace("states.txt", "init_frame.jpg")}')

    all_fnames = []
    all_states = []
    all_actions = []
    all_images = []
    all_lengths = []

    feats = []
    for i, fname in enumerate(tqdm(fnames)):
        # load the camera trajectory
        states = np.loadtxt(
            f'{camera_traj_dir}/{fname}', dtype=np.float32)
        actions = np.loadtxt(
            f'{camera_traj_dir}/{fname.replace("states.txt", "actions.txt")}', dtype=np.float32)

        # normalize the states and actions
        states = (states - state_avg) / state_std
        actions = (actions - action_avg) / action_std

        # skip if the camera trajectory is too short
        if len(states) < t_resolution:
            continue

        if feature_type == 'kinetic':
            feat_dict = extract_camera_kinetic_features(
                actions, t_resolution, )
            all_feat_dicts = {fname: feat_dict}
            feats.append(feat_dict['linear'])
        elif feature_type == 'cimtr':
            # get the initial frame
            image = Image.open(
                f'{init_frame_dir}/{fname.replace("states.txt", "init_frame.jpg")}').convert('RGB')
            image = transform(image)

            if len(all_states) <= batch_size:
                all_fnames.append(fname)
                all_states.append(
                    torch.tensor(states, dtype=torch.float32))
                all_actions.append(
                    torch.tensor(actions, dtype=torch.float32))
                all_images.append(image)
                all_lengths.append(len(states))

                all_feat_dicts = {}
            if len(all_states) == batch_size:
                all_feat_dicts = extract_camera_cimtr_features(
                    all_fnames, all_states, all_actions, all_images, all_lengths, model)
                all_fnames = []
                all_states = []
                all_actions = []
                all_images = []
                all_lengths = []
                feats.extend(
                    [feat_dict['img'] for feat_dict in all_feat_dicts.values()])

        for fname, feat_dict in all_feat_dicts.items():
            for key, value in feat_dict.items():
                os.makedirs(f'{camera_feature_dir}/{key}', exist_ok=True)
                np.save(f'{camera_feature_dir}/{key}/feat_{fname.replace("_states.txt", f".npy")}',
                        value)
        pass
    if len(all_states) > 0:
        all_feat_dicts = extract_camera_cimtr_features(
            all_fnames, all_states, all_actions, all_images, all_lengths, model)
        feats.extend(
            [feat_dict['img'] for feat_dict in all_feat_dicts.values()])
    for fname, feat_dict in all_feat_dicts.items():
        for key, value in feat_dict.items():
            os.makedirs(f'{camera_feature_dir}/{key}', exist_ok=True)
            np.save(f'{camera_feature_dir}/{key}/feat_{fname.replace("_states.txt", f".npy")}',
                    value)
    # set print options to float
    if not quiet:
        print(f'avg feat: \t{np.array(feats).mean(axis=0)}')


def traj_to_state_action(camera_traj, coord_sys="blender"):
    """
    Convert camera trajectory to state and action vectors.
    Parameters:
        camera_traj (ndarray): Array of shape (N, 6) or (N, 7) with 3D locations and rotations.
        coord_sys (str): Coordinate system of the camera trajectory. Either "blender" or "colmap".
    Returns:
        states (ndarray): Array of shape (N, 6) with state vectors.
        actions (ndarray): Array of shape (N, 6) with action vectors.
    """

    locations, rotations = camera_traj[:, :3], camera_traj[:, 3:]

    # get tvecs and qvecs in colmap convention
    if coord_sys == "blender":

        # # in colmap convention
        # # include the speed multiplier to the tvec and speed
        # raw_tvec_multiplied, raw_qvec, _, _ = convert_to_global_frame(
        #     self.raw_tvec0, self.raw_qvec0,
        #     next_tvecs[i], next_qvecs[i], None, None)
        # # in blender convention
        # loc = R_blender_from_colmap @ raw_tvec_multiplied
        # # R_colmap is for rotating the colmap world plane to the actual camera direction
        # R_colmap = quat2mat(raw_qvec)
        # # R_blender is for rotating the blender world plane to the actual camera direction
        # R_blender = R_blender_from_colmap @ R_colmap
        # # retrieve the global rotation (R_rot) needed for rotating the blender default camera direction
        # # R_rot @ R_bcam = R_blender
        # R_rot = R_blender @ R_blender_cam_dir.T
        # # euler angles conversion
        # rot = mat2euler(R_rot, axes='sxyz')
        assert rotations.shape[1] == 3

        raw_tvecs = (R_colmap_from_blender @ locations.T).T
        raw_qvecs = []
        for i in range(len(rotations)):
            R_rot = euler2mat(*rotations[i], axes='sxyz')
            R_blender = R_rot @ R_blender_cam_dir
            R_colmap = R_colmap_from_blender @ R_blender
            qvec = mat2quat(R_colmap)
            raw_qvecs.append(qvec)
        raw_qvecs = np.array(raw_qvecs)
    else:
        assert rotations.shape[1] == 4
        raw_tvecs = locations
        raw_qvecs = rotations

    # reference frame
    ref_tvec, ref_qvec = raw_tvecs[0], raw_qvecs[0]
    # change the global coord system to the initial frame
    tvecs = np.zeros_like(raw_tvecs)
    qvecs = np.zeros_like(raw_qvecs)
    for i in range(len(raw_tvecs)):
        tvecs[i], qvecs[i], _, _ = convert_to_local_frame(
            ref_tvec, ref_qvec,
            raw_tvecs[i], raw_qvecs[i])

    # calculate the speed and angular velocity from actions
    states, actions = get_states_actions(tvecs, qvecs, motion_option='local')
    return states, actions


def extract_camera_kinetic_features(actions, t_resolution):
    """
    Extracts kinetic features from camera locations and rotations.
    Parameters:
        actions (ndarray): Array of shape (N, 6) with vs and omegas.
        t_resolution (int): Temporal resolution of the camera trajectory.
    Returns:
        kinetic_feature_dict (dict): Dictionary containing kinetic features.
    """

    # downsample the temporal resolution
    t_step = len(actions) // t_resolution
    _actions = actions[:t_step * t_resolution].reshape(
        [t_resolution, t_step, -1]).mean(axis=1)

    v, omega = _actions[:, :3], _actions[:, 3:]
    # normalize v as the original colmap has no scale
    v /= np.mean(np.linalg.norm(v, axis=1))
    # a = np.diff(v, axis=0)

    # calculate the kinetic features
    linear_feature_vector = np.concatenate([
        v[:, 0] ** 2,
        v[:, 1] ** 2,
        v[:, 2] ** 2,
    ])
    angular_feature_vector = np.concatenate([
        omega[:, 0] ** 2,
        omega[:, 1] ** 2,
        omega[:, 2] ** 2,
    ])
    # acceleration_feature_vector = np.linalg.norm(a, axis=1)
    # delta_kinetic = np.abs(np.diff(np.linalg.norm(actions[:, :3], axis=1)))

    kinetic_feature_dict = {
        'linear': linear_feature_vector,
        # 'angular': angular_feature_vector,
        # 'acceleration': acceleration_feature_vector,
        # 'overall': np.concatenate([
        #     linear_feature_vector,
        #     angular_feature_vector,
        #     # acceleration_feature_vector
        # ]),
    }

    return kinetic_feature_dict


def extract_camera_cimtr_features(all_fnames, all_states, all_actions, all_images, all_lengths, model):
    """
    Extracts CImTr features from camera states and actions.
    Parameters:
        all_fnames (list): List of filenames for the camera trajectories.
        all_states (list): List of camera states.
        all_actions (list): List of camera actions.
        all_images (list): List of images corresponding to the camera trajectories.
        all_lengths (list): List of lengths of the camera trajectories.
        model (CImTr): CImTr model for feature extraction.
    Returns:
        cimtr_feature_dict (dict): Dictionary containing CImTr features.
    """

    states = padding(all_states).cuda()
    actions = padding(all_actions).cuda()
    images = torch.stack(all_images, dim=0).cuda()
    lengths = torch.tensor(all_lengths, dtype=torch.int64).cuda()

    # foward pass
    with torch.no_grad():
        output = model(images=images[:, None], states=states[:, :, None],
                       actions=actions[:, :, None], lengths=lengths)

    img_latents = output['img_latents'].cpu().numpy()
    traj_latents = output['traj_latents'].cpu().numpy()
    cimtr_feature_dict = {}
    for i in range(len(all_fnames)):
        cimtr_feature_dict[all_fnames[i]] = {
            'img': img_latents[i],
            'traj': traj_latents[i],
        }

    return cimtr_feature_dict


def extract_gt_trajs_images(root, h5fname, split_name='val'):
    """"
    Extract ground truth camera trajectories and the images from the h5 file and save them.
    Parameters:
        root (str): Root directory for saving the camera trajectories.
        h5fname (str): Name of the h5 file containing the camera trajectories.
        split_name (str): Name of the split (e.g., 'train', 'val').
    """

    # get video_ids in the split
    split_video_ids = []
    with open(f'{root}/{split_name}_video_ids.txt', 'r') as f:
        for line in f.readlines():
            split_video_ids.append(line.strip())

    gt_traj_dir = f'{root}/{h5fname.replace(".h5", "")}/camera_trajs'
    os.makedirs(gt_traj_dir, exist_ok=True)

    gt_init_frame_dir = f'{root}/{h5fname.replace(".h5", "")}/init_frames'
    os.makedirs(gt_init_frame_dir, exist_ok=True)

    # open the hdf5 file
    with h5py.File(f'{root}/{h5fname}', 'r') as f:
        # list all groups
        h5_video_ids = list(f.keys())
        for video_id in tqdm(h5_video_ids):
            # if it is a group
            if isinstance(f[video_id], h5py.Group):
                if video_id not in split_video_ids:
                    continue
                try:
                    data = f[video_id]['data.json'][:]
                    data = io.BytesIO(data.tobytes())
                    metadata = json.load(data)
                    video_stat = get_video_stat(metadata)
                    if not video_stat['is_drone'] or video_stat['has_skip_words']:
                        continue
                    if not video_stat['is_landscape']:
                        continue
                except Exception as e:
                    # print(f'Error reading metadata for {video_id}: {e}')
                    continue
                # list all datasets
                result_fnames = list(f[video_id].keys())
                for result_fname in result_fnames:

                    if not ('-score' in result_fname and result_fname.endswith('.csv')):
                        continue
                    scene = os.path.basename(result_fname).split(
                        '-')[0].replace('scene', '')
                    recons_index = int(os.path.basename(result_fname).split(
                        '-')[1].replace('recons', ''))
                    score = int(re.search(r'-score(\d+)',
                                          result_fname).group(1))
                    valid = '_invalid' not in result_fname
                    if not score or not valid:
                        continue
                    frame_folder = f'scene{scene}-recons{recons_index}-frames/'
                    if frame_folder not in f[video_id]:
                        continue
                    # get the dataset
                    data = f[video_id][result_fname][:]
                    data = io.BytesIO(data.tobytes())
                    # do something with the dataset
                    # if result_fname.endswith('json'):
                    #     # load the config
                    #     data = json.load(dataset)
                    # elif result_fname.endswith('txt'):
                    #     data = np.loadtxt(dataset)
                    # elif result_fname.endswith('csv'):
                    recons_df = pd.read_csv(data, comment='#')

                    is_fpv = int(video_stat['is_fpv'])
                    # camera poses
                    # note: the original tvecs and qvecs in read_images_binary() gives camera extrinsic matrix [R=quat2mat(qvec), t=tvec],
                    # but the camera pose (location and orientation) in the global coord system is [-R.T @ t, R.T]
                    # recons_df include the converted tvecs and qvecs (all in world coord system)
                    recons_array = recons_df.to_numpy()
                    # camera path in global coord system (measurements)
                    # raw_tvecs = recons_array[:, 1:4].astype(float)
                    # raw_qvecs = recons_array[:, 4:8].astype(float)
                    # raw_vs = recons_array[:, 8:11].astype(float)
                    # raw_omegas = recons_array[:, 11:14].astype(float)
                    # camera path the global coord system (estimation)
                    raw_tvecs = recons_array[:, 14:17].astype(float)
                    raw_qvecs = recons_array[:, 17:21].astype(float)
                    raw_vs = recons_array[:, 21:24].astype(float)
                    raw_omegas = recons_array[:, 24:27].astype(float)
                    # add the final speed and angular velocity to extend the sequence
                    final_tvec = raw_tvecs[-1] + raw_vs[-1]
                    final_qvec = add_angular_velocity_to_quaternion(
                        raw_qvecs[-1], raw_omegas[-1], 1)
                    raw_tvecs = np.concatenate(
                        [raw_tvecs, final_tvec[None]], axis=0)
                    raw_qvecs = np.concatenate(
                        [raw_qvecs, final_qvec[None]], axis=0)
                    # modulate the speed based on the drone type
                    # 0: non-fpv max speed of 8 m/s = 0.5 m/frame at 15 fps
                    # 1: fpv max speed of 16 m/s = 1.0 m/frame at 15 fps
                    # raw_tvecs *= speed_multipliers[is_fpv]

                    # colmap data at 15 fps, downsample to target fps
                    camera_traj = np.concatenate(
                        [raw_tvecs, raw_qvecs], axis=1)[::15 // fps]

                    # np.savetxt(f'{gt_traj_dir}/{video_id}_{"fpv" if int(video_stat["is_fpv"]) else "nonfpv"}_{result_fname.replace(".csv", ".txt")}',
                    #            camera_traj,
                    #            fmt='%f',
                    #            header=(f'{video_id}/{result_fname}\n'
                    #                    'locs: x, y, z; rots: qw, qx, qy, qz\n'))

                    states, actions = traj_to_state_action(
                        camera_traj, coord_sys='colmap')
                    np.savetxt(f'{gt_traj_dir}/{video_id}_{result_fname.replace("-score1.csv", "_states.txt")}',
                               states,)
                    np.savetxt(f'{gt_traj_dir}/{video_id}_{result_fname.replace("-score1.csv", "_actions.txt")}',
                               actions,)

                    # save the init frame
                    frame_folder = f'scene{scene}-recons{recons_index}-frames'
                    img_fname = recons_array[0, 0]
                    data = f[video_id][frame_folder][img_fname][:]
                    data = io.BytesIO(data.tobytes())
                    img = Image.open(data).convert('RGB')
                    img.save(
                        f'{gt_init_frame_dir}/{video_id}_{result_fname.replace("-score1.csv", "_init_frame.jpg")}')

                    pass
            pass


def extract_pred_trajs_images(pred_dir):
    pred_traj_dir = f'{pred_dir}/camera_trajs'
    os.makedirs(pred_traj_dir, exist_ok=True)

    pred_init_frame_dir = f'{pred_dir}/init_frames'
    os.makedirs(pred_init_frame_dir, exist_ok=True)

    for fname in sorted(os.listdir(f'{pred_dir}/videos')):
        if fname.endswith('_config.txt'):
            fpath = f'{pred_dir}/videos/{fname}'
            camera_traj = np.loadtxt(fpath)
            # blender results are in 30 fps, downsample to target fps
            camera_traj = camera_traj[::30 // fps]

            states, actions = traj_to_state_action(
                camera_traj, coord_sys='blender')

            # normalize the scale of the trajectory
            v = actions[:, :3]
            avg_v_norm = np.mean(np.linalg.norm(v, axis=1))
            states[:, :3] /= avg_v_norm
            actions[:, :3] /= avg_v_norm

            np.savetxt(f'{pred_traj_dir}/{fname.replace("_config.txt", "_states.txt")}',
                       states,)
            np.savetxt(f'{pred_traj_dir}/{fname.replace("_config.txt", "_actions.txt")}',
                       actions,)
            # save the init frame
            matching_files = [f for f in os.listdir(f'{pred_dir}/videos')
                              if f.startswith(fname.replace('_config.txt', '')) and
                              f.endswith('fps.jpg')]
            if matching_files:
                img_fpath = f'{pred_dir}/videos/{matching_files[0]}'
                img = Image.open(img_fpath).convert('RGB')
                # get the image height and width
                img_width, img_height = img.size
                # crop the left most 16:9 area
                if img_height % 9 == 0:
                    target_width = img_height // 9 * 16
                elif img_height == 168:
                    target_width = 294
                else:
                    raise Exception
                img = img.crop((0, 0, target_width, img_height))
                img.save(
                    f'{pred_init_frame_dir}/{fname.replace("_config.txt", "_init_frame.jpg")}')
            else:
                raise Exception(
                    f'No matching image file found for {fname} in {pred_dir}/videos')


def normalize(feat, feat2):
    mean = feat.mean(axis=0)
    std = feat.std(axis=0)

    return (feat - mean) / (std + 1e-10), (feat2 - mean) / (std + 1e-10)


def quantized_metrics(pred_dir, gt_dir, gt_h5fname='dataset_full.h5'):

    metrics = {}
    # kinetic features
    pred_k_feat_root = f'{pred_dir}/kinetic_features'
    gt_k_feat_root = f'{gt_dir}/{gt_h5fname.replace(".h5", "")}/kinetic_features'
    for kinetic_type in sorted(os.listdir(pred_k_feat_root)):
        assert (os.path.isdir(f'{pred_k_feat_root}/{kinetic_type}') and
                os.path.isdir(f'{gt_k_feat_root}/{kinetic_type}'))

        pred_feats, gt_feats = [], []

        for fname in os.listdir(f'{pred_k_feat_root}/{kinetic_type}'):
            pred_feats.append(
                np.load(f'{pred_k_feat_root}/{kinetic_type}/{fname}'))

        for fname in os.listdir(f'{gt_k_feat_root}/{kinetic_type}'):
            gt_feats.append(
                np.load(f'{gt_k_feat_root}/{kinetic_type}/{fname}'))

        pred_feats = np.array(pred_feats)
        gt_feats = np.array(gt_feats)

        gt_feats, pred_feats = normalize(gt_feats, pred_feats)

        metrics[f'fid_{kinetic_type}'] = calc_fid(pred_feats, gt_feats)

        if kinetic_type == 'linear':
            div_linear_pred = calc_avg_distance(pred_feats)
            gt_div_fname = f'{gt_k_feat_root}/linear_diversity.txt'
            if os.path.exists(gt_div_fname):
                div_linear_gt = np.loadtxt(gt_div_fname).item()
            else:
                div_linear_gt = calc_avg_distance(gt_feats)
                np.savetxt(gt_div_fname, [div_linear_gt])
            metrics['div_linear_pred'] = div_linear_pred
            metrics['div_linear_gt'] = div_linear_gt

    # CImTr features
    pred_cimtr_feat_root = f'{pred_dir}/cimtr_features'
    gt_cimtr_feat_root = f'{gt_dir}/{gt_h5fname.replace(".h5", "")}/cimtr_features'

    pred_img_feats, pred_traj_feats = [], []
    gt_img_feats, gt_traj_feats = [], []
    for fname in os.listdir(f'{pred_cimtr_feat_root}/img'):
        assert os.path.exists(
            f'{pred_cimtr_feat_root}/img/{fname}') and os.path.exists(
            f'{pred_cimtr_feat_root}/traj/{fname}')
        pred_img_feats.append(
            np.load(f'{pred_cimtr_feat_root}/img/{fname}'))
        pred_traj_feats.append(
            np.load(f'{pred_cimtr_feat_root}/traj/{fname}'))
    for fname in os.listdir(f'{gt_cimtr_feat_root}/img'):
        assert os.path.exists(
            f'{gt_cimtr_feat_root}/img/{fname}') and os.path.exists(
            f'{gt_cimtr_feat_root}/traj/{fname}')
        gt_img_feats.append(
            np.load(f'{gt_cimtr_feat_root}/img/{fname}'))
        gt_traj_feats.append(
            np.load(f'{gt_cimtr_feat_root}/traj/{fname}'))
    pred_img_feats = np.array(pred_img_feats)
    pred_traj_feats = np.array(pred_traj_feats)
    gt_img_feats = np.array(gt_img_feats)
    gt_traj_feats = np.array(gt_traj_feats)

    # average similarity across image and trajectory features
    pred_sims = pred_img_feats[:, None] @ pred_traj_feats[:, :, None]
    gt_sims = gt_img_feats[:, None] @ gt_traj_feats[:, :, None]
    # CLIP score calculation
    # https://lightning.ai/docs/torchmetrics/stable/multimodal/clip_score.html?utm_source=chatgpt.com
    pred_sims = np.clip(pred_sims[:, 0, 0], 0, 1) * 100
    gt_sims = np.clip(gt_sims[:, 0, 0], 0, 1) * 100
    metrics['clip_score_pred'] = np.mean(pred_sims).item()
    metrics['clip_score_gt'] = np.mean(gt_sims).item()

    # trajectory feat fid
    metrics['fid_traj'] = calc_fid(pred_traj_feats, gt_traj_feats)

    div_linear_pred = calc_avg_distance(pred_traj_feats)
    gt_div_fname = f'{gt_cimtr_feat_root}/diversity.txt'
    if os.path.exists(gt_div_fname):
        div_linear_gt = np.loadtxt(gt_div_fname).item()
    else:
        div_linear_gt = calc_avg_distance(gt_traj_feats)
        np.savetxt(gt_div_fname, [div_linear_gt])
    metrics['div_cimtr_pred'] = div_linear_pred
    metrics['div_cimtr_gt'] = div_linear_gt

    return metrics


def calc_fid(kps_gen, kps_gt):

    mu_gen = np.mean(kps_gen, axis=0)
    sigma_gen = np.cov(kps_gen, rowvar=False)

    mu_gt = np.mean(kps_gt, axis=0)
    sigma_gt = np.cov(kps_gt, rowvar=False)

    mu1, mu2, sigma1, sigma2 = mu_gen, mu_gt, sigma_gen, sigma_gt

    diff = mu1 - mu2
    eps = 1e-5
    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1)
            + np.trace(sigma2) - 2 * tr_covmean)


def calc_avg_distance(feature_list):
    feature_list = np.stack(feature_list)
    dist = np.mean(pdist(feature_list, metric='euclidean'))
    return dist


if __name__ == '__main__':
    gt_dir = 'youtube_drone_videos'
    gt_h5fname = 'dataset_full.h5'
    np.set_printoptions(precision=4, suppress=True)
    save = True
    quiet = True

    print('--------------new--------------')
    logdirs = sorted(
        [f'logs/{dir}' for dir in os.listdir('logs') if 'CImTr' not in dir])
    # logdirs = [
    #     'None',
    # ]
    results_dict = {}
    for logdir in logdirs:
        if not os.path.exists(f'{logdir}/videos') or 'DEBUG_' in logdir:
            continue
        print(logdir)
        extract_feats(logdir, gt_dir, gt_h5fname,
                      feature_type='kinetic', quiet=quiet)
        extract_feats(logdir, gt_dir, gt_h5fname,
                      feature_type='cimtr', quiet=quiet)
        metrics = quantized_metrics(logdir, gt_dir, gt_h5fname)
        for key, value in metrics.items():
            print(f'{key}: \t{value}')

        crash_fname = None
        duration_fname = None

        for fname in os.listdir(logdir):
            if fname.startswith('crash_'):
                crash_fname = fname
            if fname.startswith('duration_'):
                duration_fname = fname
        if crash_fname is not None and duration_fname is not None:
            crash_rate = np.loadtxt(f'{logdir}/{crash_fname}').item()
            avg_duration = np.loadtxt(f'{logdir}/{duration_fname}').item()
        else:
            # retrieve the average duration and crash rate
            # fname example: 04403_fpv_snowy_mountain_763af71a_return10.00_crashNone_2025-05-06_11-01-13_config.txt
            # fname example: 04301_fpv_river_c02114_return-5.33_crashFront_2025-05-06_11-00-33_config.txt
            durations = []
            crashes = []
            for fname in os.listdir(f'{logdir}/videos'):
                if fname.endswith('_config.txt'):
                    return_value = float(
                        re.search(r'_return([-\d.]+)_', fname).group(1))
                    duration = 10 if return_value == 10 else return_value + 10
                    crash = re.search(r'_crash(\w+)_', fname).group(1)
                    crash = 0 if crash == 'None' else 1
                    durations.append(duration)
                    crashes.append(crash)
            avg_duration = np.mean(durations)
            crash_rate = np.mean(crashes) * 100
        print(f'avg_duration: \t{avg_duration}')
        print(f'crash_rate: \t{crash_rate}')

        results_dict[logdir] = {
            'avg_duration': avg_duration,
            'crash_rate': crash_rate,
        }
        results_dict[logdir].update(metrics)

        if save:
            # remove current metrics files
            for fname in os.listdir(logdir):
                if fname.startswith('fid_') or fname.startswith('div_') or fname.startswith('clip_') or fname.startswith('crash_') or fname.startswith('duration_'):
                    os.remove(f'{logdir}/{fname}')

            # save the metrics as file
            for name, value in metrics.items():
                with open(f'{logdir}/{name}_{value:.3f}', 'w') as f:
                    f.write(f'{value}')

            # save the crash rate as file
            with open(f'{logdir}/crash_{crash_rate:.2f}', 'w') as f:
                f.write(f'{crash_rate}')
            # save the average duration as file
            with open(f'{logdir}/duration_{avg_duration:.2f}', 'w') as f:
                f.write(f'{avg_duration}')
    if save:
        # save everything as csv
        df = pd.DataFrame.from_dict(results_dict, orient='index')
        df.to_csv('results.csv')
