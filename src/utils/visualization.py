import os
import bpy
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
from transforms3d.quaternions import qinverse, qmult, quat2mat, mat2quat, qnorm, rotate_vector
from transforms3d.euler import euler2quat, quat2euler, euler2mat, mat2euler
from transforms3d.axangles import axangle2mat
from scipy.spatial.transform import Rotation
from src.utils.colmap_official_read_write_model import read_model
from src.blender.blender_camera_env import R_colmap_from_blender, R_blender_cam_dir
from src.utils.quaternion_operations import convert_to_global_frame, convert_to_local_frame, interpolate_eulers, interpolate_tvecs


def colmap_pose_to_extrinsic(qvec, tvec, camera_to_world=False):
    """
    Convert COLMAP qvec and tvec to a 4x4 transformation matrix.

    Parameters:
        qvec (np.ndarray): Quaternion [qw, qx, qy, qz]
        tvec (np.ndarray): Translation vector [tx, ty, tz]
        camera_to_world (bool): If True, returns camera-to-world matrix.
                                If False, returns world-to-camera.

    Returns:
        4x4 numpy.ndarray: Extrinsic transformation matrix.
    """
    R = quat2mat(qvec)
    t = np.array(tvec).reshape(3, 1)

    # World-to-camera
    extrinsic_wc = np.eye(4)
    extrinsic_wc[:3, :3] = R
    extrinsic_wc[:3, 3] = t.squeeze()

    if camera_to_world:
        # Invert to get camera-to-world
        R_inv = R.T
        t_inv = -R_inv @ t
        extrinsic_cw = np.eye(4)
        extrinsic_cw[:3, :3] = R_inv
        extrinsic_cw[:3, 3] = t_inv.squeeze()
        return extrinsic_cw
    else:
        return extrinsic_wc


def draw_camera_frustum(intrinsic, extrinsic, image_size, scale=100, color=[1, 0, 0]):
    """
    Draws a camera frustum in Open3D.

    Parameters:
        intrinsic (np.ndarray): 3x3 intrinsic matrix.
        extrinsic (np.ndarray): 4x4 camera-to-world transformation matrix.
        image_size (tuple): (width, height) of the image.
        scale (float): Scale factor for the frustum size.
        color (list): RGB color of the frustum lines.

    Returns:
        open3d.geometry.LineSet: The frustum as a LineSet.
    """
    w, h = image_size
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]

    # Define 4 corners of the image plane in camera coordinates
    corners = np.array([
        [(0 - cx) / fx, (0 - cy) / fy, 1],
        [(w - cx) / fx, (0 - cy) / fy, 1],
        [(w - cx) / fx, (h - cy) / fy, 1],
        [(0 - cx) / fx, (h - cy) / fy, 1]
    ]) * scale

    # Add camera origin
    points = np.vstack(([0, 0, 0], corners))  # (5, 3)

    # Transform to world coordinates
    R = extrinsic[:3, :3]
    t = extrinsic[:3, 3]
    points_world = (R @ points.T).T + t

    # Define lines: 0 is camera origin
    lines = [
        [0, 1], [0, 2], [0, 3], [0, 4],
        [1, 2], [2, 3], [3, 4], [4, 1]
    ]

    # Create LineSet
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points_world),
        lines=o3d.utility.Vector2iVector(lines)
    )
    line_set.colors = o3d.utility.Vector3dVector([color for _ in lines])
    return line_set


def create_cylinder_line(p1, p2, radius=0.005, color=[0.8, 0.2, 0.8]):
    """
    Create a cylinder between two points p1 and p2.
    """
    # Create a mesh cylinder of the given radius
    mesh_cylinder = o3d.geometry.TriangleMesh.create_cylinder(
        radius=radius, height=np.linalg.norm(p2 - p1))
    mesh_cylinder.paint_uniform_color(color)

    # Compute the transformation for the cylinder
    cyl_transform = np.eye(4)
    cyl_transform[0:3, 3] = (p1 + p2) / 2
    # Align the cylinder with the line p1 -> p2
    v = p2 - p1
    v /= np.linalg.norm(v)
    axis = np.array([0, 0, 1])  # Initial axis of the cylinder
    axis_x_v = np.cross(axis, v)
    angle = np.arctan2(np.linalg.norm(axis_x_v), np.dot(axis, v))

    cyl_transform[0:3, 0:3] = o3d.geometry.get_rotation_matrix_from_axis_angle(
        axis_x_v / np.linalg.norm(axis_x_v) * angle)

    # Apply the transformation
    mesh_cylinder.transform(cyl_transform)

    return mesh_cylinder


def lineset_to_cylinders(lineset, radius=0.01):
    """
    Convert a LineSet to a list of cylinders to simulate thick lines.

    Parameters:
        lineset (o3d.geometry.LineSet): Input LineSet
        radius (float): Cylinder radius

    Returns:
        List of TriangleMesh (cylinders)
    """
    points = np.asarray(lineset.points)
    lines = np.asarray(lineset.lines)
    colors = np.asarray(lineset.colors) if lineset.has_colors() else [
        [0.5, 0.5, 0.5]] * len(lines)

    cylinders = []
    for (i, (start_idx, end_idx)) in enumerate(lines):
        p1, p2 = points[start_idx], points[end_idx]
        color = colors[i % len(colors)]
        cyl = create_cylinder_line(p1, p2, radius, color)
        if cyl is not None:
            cylinders.append(cyl)

    return cylinders


def visualize_camera_coords(tvecs, qvecs, size=0.5):
    # Visualize camera poses
    geometries = []
    for tvec, qvec in zip(tvecs, qvecs):
        R = quat2mat(qvec)
        # Custom camera frustum with extended forward axis and arrow
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=size, origin=[0, 0, 0])
        # Apply rotation
        mesh_frame.rotate(R, center=(0, 0, 0))
        # Apply translation
        mesh_frame.translate(tvec)

        geometries.append(mesh_frame)
    return geometries


def plot_camera_lines(tvecs, color=[0.8, 0.2, 0.8]):
    """
    Plots a camera path as a curve from a sequence of translation vectors.

    Parameters:
    - tvecs: A list or numpy array of translation vectors (Nx3).
    - color: The RGB color of the curve.
    - line_width: The width of the lines. Note that Open3D might not support line width in all environments.
    """
    # Ensure tvecs is a numpy array for easier manipulation
    tvecs = np.asarray(tvecs)

    # Create points and lines for the LineSet
    lines = [[i, i + 1]
             for i in range(len(tvecs) - 1)]  # Connect consecutive points

    # Create a LineSet object and set its properties
    camera_path = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(tvecs),
        lines=o3d.utility.Vector2iVector(lines),
    )
    camera_path.paint_uniform_color(color)  # Set the color of the curve

    return [camera_path]


def export_point_cloud_to_ply(blend_file_path, output_ply_path):
    """
    Opens a .blend file and exports its point cloud data to a .ply file

    Parameters:
    blend_file_path (str): Path to the .blend file
    output_ply_path (str): Path where the .ply file will be saved
    """
    # Clear existing data
    bpy.ops.wm.open_mainfile(filepath=blend_file_path)

    # Get all mesh objects in the scene
    mesh_objects = [
        obj for obj in bpy.context.scene.objects if obj.type == 'MESH']

    if not mesh_objects:
        print("No mesh objects found in the .blend file")
        return False

    # Collect all vertices from all mesh objects
    all_vertices = []

    skip_name_list = ['outofview', 'CameraRigs',
                      'atmosphere', 'inview', 'unapplied']

    for obj in mesh_objects:
        if any(substring in obj.name for substring in skip_name_list):
            continue
        # Get world matrix for the object
        world_matrix = obj.matrix_world

        # Get mesh data
        mesh = obj.data

        # Extract vertices in world space
        for vertex in mesh.vertices:
            # Transform vertex to world space
            world_vertex = world_matrix @ vertex.co
            all_vertices.append(
                (world_vertex.x, world_vertex.y, world_vertex.z))

    if not all_vertices:
        print("No vertices found in the mesh objects")
        return False

    # Convert to numpy array for easier processing
    points = np.array(all_vertices)

    # Write to PLY file
    with open(output_ply_path, 'w') as f:
        # Write header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")

        # Write vertex data
        for point in points:
            f.write(f"{point[0]} {point[1]} {point[2]}\n")

    print(f"Successfully exported {len(points)} points to {output_ply_path}")


def main(fpath):

    # get the blender fpath
    with open(fpath, 'r') as f:
        line = f.readline()
    blender_fpath = line.split(' ')[1].strip()
    # convert .blend file to .ply file with only the point cloud
    point_cloud_fpath = blender_fpath.replace('.blend', '.ply')
    # Define the screenshot path
    screenshot_path = fpath.replace('.txt', '.jpg')

    # image size
    h, w = 225, 400
    # focal length in mm
    focal_length = float(line.split(' ')[-1].split(':')[-1].strip())
    # sensor width in mm
    sensor_width = 36
    # convert focal length to pixel
    focal_length = (focal_length / sensor_width) * w
    # camera intrinsic matrix
    intrinsic = np.array([[focal_length, 0, w / 2],
                          [0, focal_length, h / 2],
                          [0, 0, 1]])

    # blender coordinates in 30 fps
    loc_rot = np.loadtxt(fpath)
    # convert to 15 fps
    locations = loc_rot[::2, :3]
    directions = loc_rot[::2, 3:]
    # tvec and qvec from blender
    tvecs = locations
    qvecs = np.zeros((len(directions), 4))
    for i in range(len(directions)):
        rot = directions[i]
        # R_rot is for rotating the blender defualt camera direction in the blender world plane
        R_rot = euler2mat(*rot, axes='sxyz')
        # apply the global (blender world plane) rotation to the blender defualt camera direction
        # R_blender is for rotating the blender world plane to the actual camera direction
        R_blender = R_rot @ R_blender_cam_dir
        qvec = mat2quat(R_blender)
        qvecs[i] = qvec

    # Load the mesh or point cloud
    if not os.path.exists(point_cloud_fpath):
        # Export the point cloud from the .blend file
        export_point_cloud_to_ply(blender_fpath, point_cloud_fpath)
    # Read the mesh from the .ply file
    mesh = o3d.io.read_triangle_mesh(point_cloud_fpath)
    # Create a point cloud from the vertices of the mesh
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = mesh.vertices

    # Downsample with a voxel size of (for example) 0.02
    voxel_size = 2.0
    downsampled_pcd = point_cloud.voxel_down_sample(voxel_size=voxel_size)
    if 'infinigen' in blender_fpath:
        # filter out points with z < 0
        downsampled_pcd = downsampled_pcd.select_by_index(
            np.where(np.asarray(downsampled_pcd.points)[:, 2] > 0)[0])
    # add color to the downsampled point cloud according to their z value
    z = np.asarray(downsampled_pcd.points)[:, 2]
    z = (z - z.min()) / (z.max() - z.min())
    downsampled_pcd.colors = o3d.utility.Vector3dVector(
        # plt.cm.viridis(z)[:, :3]
        plt.cm.rainbow(z)[:, :3]
    )

    target_color = np.array([0.8, 0.2, 0.8])
    white_color = np.array([1, 1, 1])
    min_color_ratio = 0.4
    # visualize(tvecs, qvecs)
    geometries = []
    # geometries.extend(visualize_points(points3D))
    # geometries.extend(visualize_camera_coords(tvecs, qvecs, 0.5))
    # geometries.extend(plot_camera_lines(
    #     tvecs, color=(target_color * 0.3 + white_color * 0.7)))
    geometries.extend([downsampled_pcd])

    is_fpv = '_fpv_' in fpath
    if is_fpv:
        fps_downsample = 5  # to 3 fps
        camera_scale = 1
    else:
        fps_downsample = 15
        camera_scale = 10

    # camera frustum
    for i in range(0, len(tvecs), fps_downsample):
        tvec = tvecs[i]
        qvec = qvecs[i]
        alpha = min_color_ratio + \
            (i / (len(tvecs) - 1)) * (1 - min_color_ratio)
        color = alpha * target_color + (1 - alpha) * white_color
        extrinsic = colmap_pose_to_extrinsic(qvec, tvec)
        frustum = draw_camera_frustum(
            intrinsic, extrinsic, (w, h), scale=camera_scale, color=color)
        geometries.append(frustum)

    # replace the lines in geometries with cylinders
    cylinders_all = []
    for i in reversed(range(len(geometries))):
        g = geometries[i]
        if isinstance(g, o3d.geometry.LineSet):
            cylinders = lineset_to_cylinders(g, radius=0.2)
            cylinders_all.extend(cylinders)
            geometries.pop(i)
    # add the cylinders to the geometries
    geometries.extend(cylinders_all)

    # Create a custom visualizer function with screenshot capability
    def custom_draw_geometries_with_screenshot_callback(geometries):
        # Create a visualizer object
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window()

        # Set render options
        vis.get_render_option().point_size = 10

        # Add geometries
        for geometry in geometries:
            vis.add_geometry(geometry)

        # Set up the view
        view_ctl = vis.get_view_control()
        camera_params = {
            "front": tvecs[0] - tvecs[len(tvecs) // 2],
            "lookat": tvecs[0],
            "up": np.array([0., 0., 1.]),
            "zoom": 0.001
        }
        view_ctl.set_front(camera_params["front"])
        view_ctl.set_lookat(camera_params["lookat"])
        view_ctl.set_up(camera_params["up"])
        view_ctl.set_zoom(camera_params["zoom"])

        # Define the screenshot callback function
        def take_screenshot(vis):
            print(f"Saving screenshot to {screenshot_path}")
            vis.capture_screen_image(screenshot_path, True)
            print(f"Screenshot saved to {screenshot_path}")
            return True

        # Register the 'S' key for saving screenshot
        vis.register_key_callback(ord('S'), take_screenshot)

        # Print instructions
        print("Press 'S' to save a screenshot.")

        # Run the visualizer
        vis.run()
        vis.destroy_window()

        # Save camera parameters after window closes
        camera_params = view_ctl.convert_to_pinhole_camera_parameters()
        R, t = camera_params.extrinsic[:3, :3], camera_params.extrinsic[:3, 3]
        camera_location = -R.T @ t
        lookat_point = tvecs[0]

        # Save the camera parameters
        np.savetxt("camera_location.txt", camera_location, fmt='%f')
        np.savetxt("lookat_point.txt", lookat_point, fmt='%f')

        return camera_location, lookat_point

    # Use our custom function instead of the original visualization code
    camera_location, lookat_point = custom_draw_geometries_with_screenshot_callback(
        geometries)

    # Save the 3D scene files (this remains the same)
    combined_pcd = o3d.geometry.PointCloud()
    for g in geometries:
        if isinstance(g, o3d.geometry.PointCloud):
            combined_pcd += g
    o3d.io.write_point_cloud("combined_pointcloud.ply", combined_pcd)
    # merges vertices & triangles
    combined_mesh = o3d.geometry.TriangleMesh()
    for g in geometries:
        if isinstance(g, o3d.geometry.TriangleMesh):
            combined_mesh += g
    o3d.io.write_triangle_mesh("combined_mesh.ply", combined_mesh)


if __name__ == '__main__':
    logdir = input("Please enter the log directory: ")
    main(logdir)
