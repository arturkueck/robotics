import numpy as np
import random
import os
from robot_visualisation import Robot3D


class KukaRobot(Robot3D):
    def __init__(self, joint_angles=None):
        super().__init__()
        if joint_angles is None:
            joint_angles = [0, 0, 0, 0, 0, 0]
        self.joint_angles = joint_angles

        # Fake DH params: (alpha, a, d, theta_offset)
        self.dh_params = [
            [   0.0,  0.0,  0.34,   0.0 ],
            [-90.0,  0.2,   0.0,    0.0 ],
            [   0.0,  0.2,   0.34,  0.0 ],
            [-90.0,  0.0,   0.34,  0.0 ],
            [ 90.0,  0.0,   0.0,   0.0 ],
            [-90.0,  0.0,   0.18,  0.0 ],
        ]
        self._update_transformations()

    def set_joint_angles(self, joint_angles):
        if len(joint_angles) != 6:
            raise ValueError("KukaRobot requires exactly 6 joint angles.")
        self.joint_angles = joint_angles
        self._update_transformations()

    def _update_transformations(self):
        # Recompute forward kinematics from self.joint_angles
        Ts = []
        for i in range(6):
            alpha_i, a_i, d_i, theta_off = self.dh_params[i]
            theta_rad = np.deg2rad(self.joint_angles[i] + theta_off)
            alpha_rad = np.deg2rad(alpha_i)

            ct, st = np.cos(theta_rad), np.sin(theta_rad)
            ca, sa = np.cos(alpha_rad), np.sin(alpha_rad)

            T_i = np.array([
                [    ct,    -st,     0,       a_i ],
                [ st*ca, ct*ca, -sa, -sa*d_i ],
                [ st*sa, ct*sa,  ca,  ca*d_i ],
                [     0,     0,     0,       1   ]
            ])
            Ts.append(T_i)
        self.transformations = Ts

    def get_end_effector_pos(self):
        """Return the end-effector (tool) position [x,y,z] in the base frame."""
        abs_poses = self.get_absolute_poses()
        if not abs_poses:
            return np.array([0,0,0])
        return abs_poses[-1][:3, 3]

    def inverse_kinematics_random_search(self,
                                         target_pos,
                                         joint_limits=None,
                                         max_tries=10000,
                                         tolerance=0.05):
        """
        Naive random search to find a set of joint angles that brings
        the end-effector close to 'target_pos' (only position).
        
        :param target_pos: [x, y, z] in base frame
        :param joint_limits: list of (min_deg, max_deg) for each of the 6 joints
                             e.g. [(-180, 180), ..., ...]
                             If None, we assume all -180..180 for demonstration.
        :param max_tries: number of random samples
        :param tolerance: acceptable distance in meters

        :return: (best_sol, best_dist)
            best_sol: 6 angles in degrees (may be None if no solutions at all)
            best_dist: distance from that solution to target
        """
        if joint_limits is None:
            # Very broad limits for demonstration
            joint_limits = [(-180, 180)] * 6

        best_sol = None
        best_dist = float('inf')

        original_angles = self.joint_angles[:]

        for _ in range(max_tries):
            # Generate random angles
            rand_joints = [
                random.uniform(lims[0], lims[1]) for lims in joint_limits
            ]
            self.set_joint_angles(rand_joints)
            ee_pos = self.get_end_effector_pos()
            dist = np.linalg.norm(ee_pos - target_pos)
            if dist < best_dist:
                best_dist = dist
                best_sol = rand_joints[:]
                # Early stop if within tolerance
                if dist < tolerance:
                    break

        # Restore original
        self.set_joint_angles(original_angles)
        return best_sol, best_dist

    def plan_trajectory(self, start_joints, end_joints, steps=5):
        """
        Very naive "interpolation" in joint space from start_joints to end_joints.
        Returns a list of joint angles (each is [j1, j2, j3, j4, j5, j6]).
        """
        start_joints = np.array(start_joints)
        end_joints   = np.array(end_joints)
        trajectory   = []
        for i in range(steps + 1):
            alpha = i / steps
            interp = (1-alpha) * start_joints + alpha * end_joints
            trajectory.append(interp.tolist())
        return trajectory


if __name__ == "__main__":
    kuka = KukaRobot()

    # Let's define two sets of joint angles for a simple path
    start_config = [0, 0, 0, 0, 0, 0]
    end_config   = [30, 20, -15, 45, 10, 0]  # in degrees
    steps = 5

    trajectory = kuka.plan_trajectory(start_config, end_config, steps=steps)
    
    # We'll store four views in separate subfolders:
    # 1) default_view/
    # 2) x_view/
    # 3) y_view/
    # 4) z_view/
    #
    # Because we want 4 vantage points for each step.

    for i, config in enumerate(trajectory):
        # Update robot
        kuka.set_joint_angles(config)

        # 1) Default view (use matplotlib's defaults)
        folder = "4_views/default_view"
        filename = f"step_{i}.png"
        path = os.path.join(folder, filename)
        kuka.save_current_plot(path, frame_size=0.1, elev=None, azim=None)

        # 2) X-axis side view (looking along +X direction => elev=0, azim=0)
        folder = "4_views/x_view"
        filename = f"step_{i}.png"
        path = os.path.join(folder, filename)
        kuka.save_current_plot(path, frame_size=0.1, elev=0, azim=0)

        # 3) Y-axis side view (looking along +Y => elev=0, azim=90)
        folder = "4_views/y_view"
        filename = f"step_{i}.png"
        path = os.path.join(folder, filename)
        kuka.save_current_plot(path, frame_size=0.1, elev=0, azim=90)

        # 4) Top-down (looking along -Z => elev=90, azim=-90 or 0)
        folder = "4_views/z_view"
        filename = f"step_{i}.png"
        path = os.path.join(folder, filename)
        kuka.save_current_plot(path, frame_size=0.1, elev=90, azim=-90)

        print(f"Saved step {i} in all 4 views.")