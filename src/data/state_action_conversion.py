import os
import re
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List
import numpy as np
from transforms3d.quaternions import qinverse, qconjugate, qmult, qnorm, quat2mat, mat2quat, quat2axangle, axangle2quat, nearly_equivalent
from transforms3d.euler import euler2quat, quat2euler, euler2mat, mat2euler
import pandas as pd
import torch
from src.utils.quaternion_operations import convert_to_local_frame, convert_to_global_frame, add_angular_velocity_to_quaternion, quaternions_to_angular_velocity
from src.utils.flexible_fs import FlexibleFileSystem
from src.utils.pytorch3d_rotation_conversion import euler_angles_to_matrix, matrix_to_euler_angles, quaternion_to_matrix, matrix_to_quaternion

# non fpv (30854 clips, 3622930 frames)
# states: mean=[ 0.80087  0.10283  7.01349  0.99618  0.0134  -0.00143 -0.00112], std=[23.91302 17.78868 16.91058  0.00793  0.05216  0.04776  0.01807]
# actions: mean=[ 0.01284  0.01375  0.07347  0.00045 -0.00005 -0.00004], std=[0.32955 0.2588  0.22428 0.00159 0.00146 0.00057]
# stops: mean=0.02355
# lengths: 23.48435
# norms: tvec=31.76287, v=0.49480, omega=0.00197

# fpv (68149 clips, 5973175 frames)
# states: mean=[-0.00963 -5.92972 37.03546  0.84379  0.04756 -0.03609 -0.00873], std=[24.11987 18.97639 35.83229  0.28517  0.13522  0.35493  0.15497]
# actions: mean=[ 0.04631 -0.04666  0.64891  0.00307 -0.00205 -0.00064], std=[0.44675 0.35029 0.47971 0.00831 0.01921 0.01118]
# stops: mean=0.03519
# lengths: 17.52975
# norms: tvec=51.89588, v=1.00195, omega=0.01946

# both (99003 clips, 9596105 frames)
# states: mean=[ 0.29259 -3.64301 25.47237  0.9068   0.03375 -0.02301 -0.00576], std=[23.68868 18.63657 32.92044  0.21947  0.1064   0.27115  0.11438]
# actions: mean=[ 0.03357 -0.0242   0.42988  0.00202 -0.00126 -0.00039], std=[0.39975 0.31296 0.48182 0.00619 0.01413 0.00791]
# stops: mean=0.03080
# lengths: 19.38548
# norms: tvec=43.92965, v=0.80739, omega=0.01250


# state: tvec, qvec (all in global reference frame)
state_avg = np.array([0.30, -3.6, 25, 0.91, 0.033, -0.023, -0.0058])
state_std = np.array([24, 19, 33, 0.22, 0.11, 0.27, 0.11])

# action: v, omega (both local, relative to the current frame)
action_avg = np.array([0.034, -0.024, 0.43, 0.0020, -0.0013, -0.00039])
action_std = np.array([0.40, 0.31, 0.48, 0.0062, 0.014, 0.0079])

# state_avg, state_std = np.zeros_like(state_avg), np.ones_like(state_std)
# action_avg, action_std = np.zeros_like(action_avg), np.ones_like(action_std)


def get_states_actions(tvecs, qvecs, motion_option='global'):
    '''
    Get the states and actions from tvecs, qvecs, vs, and omegas.
        tvec(t)
        qvec(t)
        v(t) = tvec(t+1) - tvec(t)
        omega(t) = quaternions_to_angular_velocity(qvec(t), qvec(t+1), 1)

    state = [tvec(t), qvec(t)] for frames in the global coordinate system
    action = [v(t), omega(t)] for frames in the global/local coordinate system

    Args:
        tvecs (array): (N + 1) x 3 array of translation vectors for all frames.
        qvecs (array): (N + 1) x 4 array of rotation quaternions for all frames.
        motion_option (str): 'global' or 'local' motion.
    Returns:
        states (array): N x 7, States (position, rotation).
        actions (tensor): N x 6, Actions (velocity, angular velocity).
    '''
    N = len(tvecs) - 1
    # States: camera pose (location and rotation) in the global coordinate system
    states = np.concatenate([tvecs, qvecs], axis=1)[:N]
    # Actions: camera motion (velocities and angular velocities) in the global/local coordinate system (relative to the *current* frame)
    vs = np.zeros([N, 3])
    omegas = np.zeros([N, 3])
    for i in range(N):
        vs[i] = tvecs[i + 1] - tvecs[i]
        omegas[i] = quaternions_to_angular_velocity(
            qvecs[i], qvecs[i + 1], 1)
    if motion_option == 'global':
        actions = np.concatenate([vs, omegas], axis=1)
    elif motion_option == 'local':
        actions = []
        for i in range(N):
            _, _, v_local, omega_local = convert_to_local_frame(
                tvecs[i], qvecs[i],
                None, None, vs[i], omegas[i])
            action = np.concatenate([v_local, omega_local])
            actions.append(action)
        actions = np.stack(actions)
    else:
        raise ValueError('Invalid motion_option')
    return states, actions


def reverse_states_actions(states, actions, motion_option='global'):
    '''
    Reconstruct tvecs, qvecs, vs, and omegas using the states and actions.
        tvec(t)
        qvec(t)
        v(t) = tvec(t+1) - tvec(t)
        omega(t) = quaternions_to_angular_velocity(qvec(t), qvec(t+1), 1)

    for t  = [0,1,2,3,...],
    state  = [tvec(t), qvec(t)] in the global coordinate system
    action = [v(t), omega(t)] in the global/local coordinate system
    return   next_tvec/next_qvec for t+1=[1,2,3,4,...] and v/omega for t=[0,1,2,3,...] in global coord system

    Args:
        states (array): N x 7 array of states (camera pose in global coord system).
        actions (array): N x 6 array of actions (camera motion in global/local coord system).
        motion_option (str): 'global' or 'local' motion.
    Returns:
        next_tvecs (array): N x 3 array of translation vector.
        next_qvecs (array): N x 4 array of rotation quaternions.
        vs (array): N x 3 array of velocities.
        omegas (array): N x 3 array of angular.
    '''
    N = len(states)
    # global coord system (w.r.t. the initial frame)
    next_tvecs = np.zeros([N, 3])
    next_qvecs = np.zeros([N, 4])
    vs = np.zeros([N, 3])
    omegas = np.zeros([N, 3])

    # for the initial frame, the global position and orientation are specified by state
    last_tvec, last_qvec = states[0, :3], states[0, 3:]

    # Apply actions to reconstruct subsequent frames
    for i in range(N):
        if motion_option == 'global':
            vs[i], omegas[i] = actions[i][:3], actions[i][3:]
        elif motion_option == 'local':
            # convert to the global coordinate system
            _, _, vs[i], omegas[i] = convert_to_global_frame(
                states[i, :3], states[i, 3:],
                None, None, actions[i][:3], actions[i][3:])
        else:
            raise ValueError('Invalid motion_option')
        # location and rotation for the next frame
        next_tvecs[i] = last_tvec + vs[i]
        next_qvecs[i] = add_angular_velocity_to_quaternion(
            last_qvec, omegas[i], 1)
        # update the last location and rotation
        last_tvec, last_qvec = next_tvecs[i], next_qvecs[i]

    return next_tvecs, next_qvecs, vs, omegas

# TODO: the tensor version is not correct...
# def reverse_states_actions_tensor(states, actions, motion_option='global'):
#     '''
#     Differentiable version of reverse_states_actions.
#     Args:
#         states (tensor): [N, 7] tensor of states (camera pose in global coord system).
#         actions (tensor): [N, 6] tensor of  of actions (camera motion in global/local coord system).
#     Returns:
#         next_tvecs (tensor): [N, 3] tensor of translation vector.
#         next_qvecs (tensor): [N, 4] tensor of rotation quaternions.
#         vs (tensor): [N, 3] tensor of velocities.
#         omegas (tensor): [N, 3] tensor of angular velocities.
#     '''
#     N = len(states)
#     # global coord system (w.r.t. the initial frame)
#     next_tvecs = []
#     next_qvecs = []
#     vs = []
#     omegas = []

#     # Apply actions to reconstruct subsequent frames
#     for i in range(N):
#         R1 = quaternion_to_matrix(states[i, 3:])
#         if motion_option == 'global':
#             v, omega = actions[i, :3], actions[i, 3:]
#         elif motion_option == 'local':
#             # convert to the global coordinate system
#             v = R1 @ actions[i, :3]
#             omega = R1 @ actions[i, 3:]
#         else:
#             raise ValueError('Invalid motion_option')
#         vs.append(v)
#         omegas.append(omega)
#         delta_R = euler_angles_to_matrix(omega, 'XYZ')
#         # location and rotation for the next frame
#         next_tvec = states[i, :3] + v
#         next_tvecs.append(next_tvec)
#         next_R = delta_R @ R1
#         next_qvec = matrix_to_quaternion(next_R)
#         next_qvecs.append(next_qvec)
#     next_tvecs = torch.stack(next_tvecs)
#     next_qvecs = torch.stack(next_qvecs)
#     vs = torch.stack(vs)
#     omegas = torch.stack(omegas)

#     return next_tvecs, next_qvecs, vs, omegas


def main():
    import time
    from transforms3d.quaternions import qinverse, qmult, qnorm, quat2mat, mat2quat, quat2axangle, axangle2quat, nearly_equivalent
    from src.utils.quaternion_operations import add_angular_velocity_to_quaternion, quaternions_to_angular_velocity

    root, filter_results_path = 'youtube_drone_videos', 'dataset_mini.h5'
    fps_downsample = 5
    motion_option = 'global'

    result_fpaths = []
    h5_fs = FlexibleFileSystem(
        f'{root}/{filter_results_path}')
    for video_id in sorted(h5_fs.listdir(root)):
        for result_fname in sorted(h5_fs.listdir(f'{root}/{video_id}')):
            if 'score' in result_fname and result_fname.endswith('.csv'):
                score = int(re.search(r'-score(\d+)',
                                      result_fname).group(1))
                valid = '_invalid' not in result_fname
                if score and valid:
                    result_fpath = f'{root}/{video_id}/{result_fname}'
                    result_fpaths.append(result_fpath)

    # data_index = np.random.randint(len(result_fpaths))
    data_index = 21
    print(data_index, result_fpaths[data_index])
    with h5_fs.open(result_fpaths[data_index], 'r') as f:
        recons_df = pd.read_csv(f, comment='#')

    recons_array = recons_df.to_numpy()
    # camera path in global coord system (measurements)
    raw_tvecs = recons_array[:, 1:4].astype(float)
    raw_qvecs = recons_array[:, 4:8].astype(float)
    raw_vs = recons_array[:, 8:11].astype(float)
    raw_omegas = recons_array[:, 11:14].astype(float)
    # add the final speed and angular velocity to extend the sequence
    final_tvec = raw_tvecs[-1] + raw_vs[-1]
    final_qvec = add_angular_velocity_to_quaternion(
        raw_qvecs[-1], raw_omegas[-1], 1)
    raw_tvecs = np.concatenate([raw_tvecs, final_tvec[None]], axis=0)
    raw_qvecs = np.concatenate([raw_qvecs, final_qvec[None]], axis=0)
    # change the global coord system to the initial frame
    tvecs = np.zeros_like(raw_tvecs)
    qvecs = np.zeros_like(raw_qvecs)
    vs = np.zeros_like(raw_vs)
    omegas = np.zeros_like(raw_omegas)
    # change the global coord system to the initial frame
    for i in range(len(raw_tvecs)):
        tvecs[i], qvecs[i], _, _ = convert_to_local_frame(
            raw_tvecs[0], raw_qvecs[0],
            raw_tvecs[i], raw_qvecs[i])
    for i in range(len(raw_vs)):
        _, _, vs[i], omegas[i] = convert_to_local_frame(
            raw_tvecs[0], raw_qvecs[0],
            None, None, raw_vs[i], raw_omegas[i])
    # sequence length
    seq_length = len(recons_array) // fps_downsample

    t0 = time.time()

    # State and Action
    # global coord system (w.r.t. the initial frame)
    states, actions = get_states_actions(
        tvecs, qvecs, motion_option=motion_option)
    _next_tvecs, _next_qvecs, _vs, _omegas = reverse_states_actions(
        states, actions, motion_option=motion_option)
    print(np.abs(_next_tvecs[:-1] - tvecs[1:-1]).max(),
          np.abs(_next_qvecs[:-1] - qvecs[1:-1]).max(),
          np.abs(_vs - vs).max(),
          np.abs(_omegas - omegas).max())
    # __next_tvecs, __next_qvecs, __vs, __omegas = reverse_states_actions_tensor(
    #     torch.tensor(states, dtype=torch.float32),
    #     torch.tensor(actions, dtype=torch.float32),
    #     motion_option=motion_option)
    # print(np.abs(__next_tvecs.numpy()[:-1] - tvecs[1:-1]).max(),
    #       np.abs(__next_qvecs.numpy()[:-1] - qvecs[1:-1]).max(),
    #       np.abs(__vs.numpy() - vs).max(),
    #       np.abs(__omegas.numpy() - omegas).max())
    print(f'time: {time.time() - t0:.4f}s')

    action_downsample = 5
    t0 = time.time()
    states_, actions_ = get_states_actions(
        tvecs[::action_downsample], qvecs[::action_downsample],
        motion_option=motion_option)
    next_tvecs_, next_qvecs_, _, _ = reverse_states_actions(
        states_, actions_, motion_option=motion_option)
    print(np.abs(next_tvecs_ - tvecs[action_downsample:-1:action_downsample]).max(),
          np.abs(next_qvecs_ - qvecs[action_downsample:-1:action_downsample]).max())
    print(f'time: {time.time() - t0:.4f}s')
    pass


if __name__ == '__main__':
    main()
