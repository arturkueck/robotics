# robot_visualisation.py

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

def plot_coordinate_frame(ax, T, frame_size=0.1, alpha=1.0):
    origin = T[:3, 3]
    x_axis = origin + T[:3, 0] * frame_size
    y_axis = origin + T[:3, 1] * frame_size
    z_axis = origin + T[:3, 2] * frame_size

    ax.plot([origin[0], x_axis[0]],
            [origin[1], x_axis[1]],
            [origin[2], x_axis[2]],
            c='r', alpha=alpha)
    ax.plot([origin[0], y_axis[0]],
            [origin[1], y_axis[1]],
            [origin[2], z_axis[2]],
            c='g', alpha=alpha)
    ax.plot([origin[0], z_axis[0]],
            [origin[1], z_axis[1]],
            [origin[2], z_axis[2]],
            c='b', alpha=alpha)

class Robot3D:
    def __init__(self, transformations=None):
        if transformations is None:
            transformations = []
        self.transformations = transformations

    def set_transformations(self, transformations):
        self.transformations = transformations

    def get_absolute_poses(self):
        poses = []
        current_pose = np.eye(4)
        for T in self.transformations:
            current_pose = current_pose @ T
            poses.append(current_pose.copy())
        return poses

    def plot(self, frame_size=0.1):
        """
        Old method: show a blocking 3D figure of the robot.
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)

        base = np.eye(4)
        plot_coordinate_frame(ax, base, frame_size=frame_size)

        absolute_poses = self.get_absolute_poses()
        prev_origin = base[:3, 3]

        for pose in absolute_poses:
            plot_coordinate_frame(ax, pose, frame_size=frame_size, alpha=0.8)
            current_origin = pose[:3, 3]
            ax.plot(
                [prev_origin[0], current_origin[0]],
                [prev_origin[1], current_origin[1]],
                [prev_origin[2], current_origin[2]],
                'k--'
            )
            prev_origin = current_origin

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.title("3D Robot Visualization")
        plt.show()

    def save_current_plot(self, file_path, frame_size=0.1, elev=None, azim=None):
        """
        Renders the robot's current configuration to a 3D figure,
        sets the camera view, saves the figure to 'file_path', then closes it.
        
        :param file_path: path to the image file (e.g. 'folder/view_x/step_0.png').
        :param frame_size: scaling factor for drawn coordinate frames.
        :param elev: elevation angle in degrees. If None, use default matplotlib angle.
        :param azim: azimuth angle in degrees. If None, use default matplotlib angle.
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Adjust axes limit if needed:
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)

        if elev is not None or azim is not None:
            ax.view_init(elev=elev, azim=azim)

        base = np.eye(4)
        plot_coordinate_frame(ax, base, frame_size=frame_size)

        absolute_poses = self.get_absolute_poses()
        prev_origin = base[:3, 3]

        for pose in absolute_poses:
            plot_coordinate_frame(ax, pose, frame_size=frame_size, alpha=0.8)
            current_origin = pose[:3, 3]
            ax.plot(
                [prev_origin[0], current_origin[0]],
                [prev_origin[1], current_origin[1]],
                [prev_origin[2], current_origin[2]],
                'k--'
            )
            prev_origin = current_origin

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.title("3D Robot Visualization")

        # Ensure folders exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Save & close
        plt.savefig(file_path, dpi=150)
        plt.close(fig)