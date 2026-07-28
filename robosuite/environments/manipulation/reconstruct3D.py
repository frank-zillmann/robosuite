from collections import OrderedDict

import numpy as np

from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject, CylinderObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import CustomMaterial
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import UniformRandomSampler
from robosuite.utils.transform_utils import convert_quat

# MuJoCo geom group holding the collision geoms (group 1 holds their visual duplicates)
COLLISION_GEOM_GROUP = 0


class Reconstruct3D(ManipulationEnv):
    """
    Environment for 3D exploration and reconstruction using multiple view geometry.
    The robot must explore a scene with multiple objects and collect camera observations
    for 3D reconstruction.

    Args:
        robots (str or list of str): Specification for specific robot arm(s) to be instantiated within this env
            (e.g: "Sawyer" would generate one arm; ["Panda", "Panda", "Sawyer"] would generate three robot arms)
            Note: Must be a single single-arm robot!

        env_configuration (str): Specifies how to position the robots within the environment (default is "default").
            For most single arm environments, this argument has no impact on the robot setup.

        controller_configs (str or list of dict): If set, contains relevant controller parameters for creating a
            custom controller. Else, uses the default controller for this specific task. Should either be single
            dict if same controller is to be used for all robots or else it should be a list of the same length as
            "robots" param

        gripper_types (str or list of str): type of gripper, used to instantiate
            gripper models from gripper factory. Default is "default", which is the default grippers(s) associated
            with the robot(s) the 'robots' specification. None removes the gripper, and any other (valid) model
            overrides the default gripper. Should either be single str if same gripper type is to be used for all
            robots or else it should be a list of the same length as "robots" param

        base_types (None or str or list of str): type of base, used to instantiate base models from base factory.
            Default is "default", which is the default base associated with the robot(s) the 'robots' specification.
            None results in no base, and any other (valid) model overrides the default base. Should either be
            single str if same base type is to be used for all robots or else it should be a list of the same
            length as "robots" param

        initialization_noise (dict or list of dict): Dict containing the initialization noise parameters.
            The expected keys and corresponding value types are specified below:

            :`'magnitude'`: The scale factor of uni-variate random noise applied to each of a robot's given initial
                joint positions. Setting this value to `None` or 0.0 results in no noise being applied.
                If "gaussian" type of noise is applied then this magnitude scales the standard deviation applied,
                If "uniform" type of noise is applied then this magnitude sets the bounds of the sampling range
            :`'type'`: Type of noise to apply. Can either specify "gaussian" or "uniform"

            Should either be single dict if same noise value is to be used for all robots or else it should be a
            list of the same length as "robots" param

            :Note: Specifying "default" will automatically use the default noise settings.
                Specifying None will automatically create the required dict with "magnitude" set to 0.0.

        table_full_size (3-tuple): x, y, and z dimensions of the table.

        table_friction (3-tuple): the three mujoco friction parameters for
            the table.

        use_camera_obs (bool): if True, every observation includes rendered image(s)

        use_object_obs (bool): if True, include object information in
            the observation.

        placement_initializer (ObjectPositionSampler): if provided, will
            be used to place objects on every reset, else a UniformRandomSampler
            is used by default.

        has_renderer (bool): If true, render the simulation state in
            a viewer instead of headless mode.

        has_offscreen_renderer (bool): True if using off-screen rendering

        render_camera (str): Name of camera to render if `has_renderer` is True. Setting this value to 'None'
            will result in the default angle being applied, which is useful as it can be dragged / panned by
            the user using the mouse.

            :Note: Available "camera" names for 1 Panda arm = ('frontview', 'birdview', 'agentview', 'sideview', 'robot0_robotview', 'robot0_eye_in_hand')

        render_collision_mesh (bool): True if rendering collision meshes in camera. False otherwise.

        render_visual_mesh (bool): True if rendering visual meshes in camera. False otherwise.

        render_gpu_device_id (int): corresponds to the GPU device id to use for offscreen rendering.
            Defaults to -1, in which case the device will be inferred from environment variables
            (GPUS or CUDA_VISIBLE_DEVICES).

        control_freq (float): how many control signals to receive in every second. This sets the amount of
            simulation time that passes between every action input.

        lite_physics (bool): Whether to optimize for mujoco forward and step calls to reduce total simulation overhead.
            Set to False to preserve backward compatibility with datasets collected in robosuite <= 1.4.1.

        horizon (int): Every episode lasts for exactly @horizon timesteps.

        ignore_done (bool): True if never terminating the environment (ignore @horizon).

        hard_reset (bool): If True, re-loads model, sim, and render object upon a reset call, else,
            only calls sim.reset and resets all robosuite-internal variables

        camera_names (str or list of str): name of camera to be rendered. Should either be single str if
            same name is to be used for all cameras' rendering or else it should be a list of cameras to render.

            :Note: At least one camera must be specified if @use_camera_obs is True.

            :Note: To render all robots' cameras of a certain type (e.g.: "robotview" or "eye_in_hand"), use the
                convention "all-{name}" (e.g.: "all-robotview") to automatically render all camera images from each
                robot's camera list).

            :Note: Available "camera" names for 1 Panda arm = ('frontview', 'birdview', 'agentview', 'sideview', 'robot0_robotview', 'robot0_eye_in_hand')

        camera_heights (int or list of int): height of camera frame. Should either be single int if
            same height is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.

        camera_widths (int or list of int): width of camera frame. Should either be single int if
            same width is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.

        camera_depths (bool or list of bool): True if rendering RGB-D, and RGB otherwise. Should either be single
            bool if same depth setting is to be used for all cameras or else it should be a list of the same length as
            "camera names" param.

        camera_segmentations (None or str or list of str or list of list of str): Camera segmentation(s) to use
            for each camera. Valid options are:

                `None`: no segmentation sensor used
                `'instance'`: segmentation at the class-instance level
                `'class'`: segmentation at the class level
                `'element'`: segmentation at the per-geom level

            If not None, multiple types of segmentations can be specified. A [list of str / str or None] specifies
            [multiple / a single] segmentation(s) to use for all cameras. A list of list of str specifies per-camera
            segmentation setting(s) to use.

        sdf_size (int): Resolution of the SDF grid (default 32). The SDF will be computed on a cubic grid
            of shape (sdf_size, sdf_size, sdf_size).

        bbox_padding (float): Padding ratio for the SDF bounding box (default 0.05 = 5% padding on each side).
            This ensures surfaces at boundaries are not cut off.

    Raises:
        AssertionError: [Invalid number of robots specified]
    """

    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types=None,
        base_types="default",
        initialization_noise="default",
        table_full_size=(1.0, 1.0, 0.05),
        table_friction=(1.0, 5e-3, 1e-4),
        use_camera_obs=True,
        use_object_obs=False,
        placement_initializer=None,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="robot0_eye_in_hand",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=4,
        lite_physics=True,
        horizon=32, # 8 seconds at control_freq=4
        ignore_done=True,
        hard_reset=False,
        camera_names="robot0_eye_in_hand",
        camera_heights=128,
        camera_widths=128,
        camera_depths=True,
        camera_segmentations=None,  # {None, instance, class, element}
        renderer="mjviewer",
        renderer_config=None,
        seed=None,
        # SDF-related parameters
        sdf_size=32,
        bbox_padding=0.05,
    ):
        # settings for table top
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = np.array((0, 0, 0.8))

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        # object placement initializer
        self.placement_initializer = placement_initializer

        # static env mesh related variables (initialized to None, computed via compute_gt_mesh)
        self.gt_vertices = None
        self.gt_faces = None

        # GT surface point cloud + incremental coverage state (for point_cloud_coverage)
        self.gt_surface_points = None
        self.gt_covered = None
        self.gt_tree = None
        self.n_recon_consumed = 0

        # Bounding box definition for SDF
        self.bbox_padding = bbox_padding
        self.bbox_center = None
        self.bbox_size = None

        # SDF-related object variables (initialized to None, computed via compute_static_env_sdf)
        self.sdf_grid = None
        self.sdf_size = sdf_size

        # Table surface z-coordinate (top of table plate)
        self.table_surface_z = self.table_offset[2] + self.table_full_size[2] / 2

        self.cached_error = None

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            base_types=base_types,
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            lite_physics=lite_physics,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            camera_segmentations=camera_segmentations,
            renderer=renderer,
            renderer_config=renderer_config,
            seed=seed,
        )

    def reward(
        self,
        action=None,
        reconstruction=None,
        reconstruction_metric=None,
        truncation_distance=None,
        reward_mode=None,
        reward_scale=1.0,
        characteristic_error=1.0,
        action_penalty_scale=0.0,
        coverage_distance=0.01,
        coverage_n_samples=50000,
        output_info_dict=False,
    ):
        """
        Reward function for the 3D reconstruction task.

        Args:
            action (np array): Robot action for action penalty computation.
            reconstruction: The reconstruction data. Type depends on reconstruction_metric:
                - For "chamfer_distance": tuple (vertices, faces)
                - For "voxelwise_tsdf_error": numpy array of shape (sdf_size, sdf_size, sdf_size)
            reconstruction_metric (str): "chamfer_distance" or "voxelwise_tsdf_error".
            truncation_distance (float): TSDF truncation distance in meters.
            output_info_dict (bool): If True, also return the info dict.
            reward_mode (str): "exponential" or "delta".
            reward_scale (float): Scales the reward.
            characteristic_error (float): Normalizes the error in reward computation.
            action_penalty_scale (float): Scales the action penalty term in the reward.

        Returns:
            float: reward value
            If output_info_dict=True, returns tuple (reward, info_dict)
        """
        if reconstruction is None:
            # This happens when reward() is called automatically by the parent step() method
            # which doesn't pass the reconstruction parameter. This is expected behavior.
            # The Gym wrapper will call reward() again manually with the reconstruction.
            return None

        # Compute error based on metric
        info_dict = {}
        if reconstruction_metric == "chamfer_distance":
            vertices, faces = reconstruction

            # Handle empty mesh (no observations yet - expected during early training)
            if len(vertices) == 0:
                error = float("inf")
                print(
                    f"[Reconstruct3D] Empty reconstruction mesh detected in step {self.timestep}. Assigning infinite Chamfer distance error."
                )
            else:
                # Compute Chamfer distance between reconstructed mesh and ground truth mesh
                error = self.compute_modified_chamfer_distance(vertices, faces, n_samples=coverage_n_samples)

            info_dict["chamfer_distance"] = error

        elif reconstruction_metric == "voxelwise_tsdf_error":
            # Unpack (sdf_grid, weights_grid) tuple
            sdf_grid, weights_grid = reconstruction
            error, components = self.compute_voxelwise_tsdf_error(
                sdf_grid,
                truncation_distance=truncation_distance,
            )
            info_dict.update(components)
            info_dict["voxelwise_tsdf_error"] = error

        elif reconstruction_metric == "point_cloud_coverage":
            error, components = self.compute_point_cloud_coverage_error(
                recon_points=reconstruction,
                coverage_distance=coverage_distance,
                n_samples=coverage_n_samples,
            )
            info_dict.update(components)
            info_dict["point_cloud_coverage_error"] = error

        else:
            raise ValueError(f"Unknown reconstruction_metric: '{reconstruction_metric}'. ")

        # Compute reward based on mode
        if reward_mode == "exponential":
            reward = reward_scale * np.exp(-error / characteristic_error)
        elif reward_mode == "delta":
            if self.cached_error is None:
                if self.timestep > 0:
                    print(
                        f"[Reconstruct3D] Warning: No cached error found in step {self.timestep} > 0 (should not happen)."
                    )
                if reconstruction_metric == "chamfer_distance":
                    self.cached_error = error  # No reward at first step
                if reconstruction_metric == "voxelwise_tsdf_error":
                    self.cached_error = 1.0  # 1.0 as default error, corresponds to all missing but no distance error
                if reconstruction_metric == "point_cloud_coverage":
                    self.cached_error = 1.0  # 1.0 = nothing covered yet

            reward = reward_scale * (self.cached_error - error) / characteristic_error
        else:
            raise ValueError(f"Unknown reward_mode: '{reward_mode}'. Use 'exponential' or 'delta'.")

        self.cached_error = error

        # Apply action penalty
        info_dict["pre_action_penalty_reward"] = reward
        action_penalty = self.compute_action_penalty(action, action_penalty_scale=action_penalty_scale)
        reward -= action_penalty
        info_dict["action_penalty"] = action_penalty
        info_dict["reward"] = reward

        if output_info_dict:
            return reward, info_dict
        else:
            return reward

    def compute_action_penalty(self, action, action_penalty_scale):
        """
        Compute the penalty for large actions (torques, velocities, deltas, etc.).

        Args:
            action (np.ndarray): The action taken by the robot.

        Returns:
            float: The computed penalty.
        """
        if action is None or action_penalty_scale == 0.0:
            return 0.0

        # Normalize by action dimension to keep penalty scale consistent across different action spaces
        # penealty = action_penalty_scale * np.linalg.norm(action, ord=2) // np.sqrt(len(action))

        return action_penalty_scale * np.mean(np.abs(action))

    def compute_modified_chamfer_distance(self, recon_vertices, recon_faces, n_samples=10000):
        """
        Compute the Chamfer distance between the reconstructed mesh and the ground truth mesh.

        The Chamfer distance is the mean of the squared distances from points on one surface
        to the closest points on the other surface, averaged over both directions.

        Args:
            recon_vertices (np.ndarray): Vertices of reconstructed mesh, shape (N, 3)
            recon_faces (np.ndarray): Faces of reconstructed mesh, shape (M, 3)
            n_samples (int): Number of points to sample from each mesh surface for distance computation.
                Higher values give more accurate results but are slower. Default: 10000.

        Returns:
            float: Chamfer distance (lower is better, 0 means identical surfaces)

        Raises:
            RuntimeError: If ground truth mesh has not been computed yet
        """
        from scipy.spatial import cKDTree

        # Check if ground truth mesh has been computed
        if self.gt_vertices is None or self.gt_faces is None:
            raise RuntimeError("Ground truth mesh has not been computed yet. Call compute_gt_mesh() first.")

        # Sample points from both mesh surfaces
        gt_points = self._sample_points_from_mesh(self.gt_vertices, self.gt_faces, n_samples)
        recon_points = self._sample_points_from_mesh(recon_vertices, recon_faces, n_samples)

        # Build KD-trees for efficient nearest neighbor queries
        gt_tree = cKDTree(gt_points)
        recon_tree = cKDTree(recon_points)

        # Compute distances from reconstructed points to ground truth
        dist_recon_to_gt, _ = gt_tree.query(recon_points, k=1)

        # Compute distances from ground truth points to reconstructed
        dist_gt_to_recon, _ = recon_tree.query(gt_points, k=1)

        # Chamfer distance is the mean of squared distances in both directions
        norm_exponent = 1  # L1 norm
        chamfer_dist = np.power(
            0.5
            * (np.mean(np.power(dist_recon_to_gt, norm_exponent)) + np.mean(np.power(dist_gt_to_recon, norm_exponent))),
            1 / norm_exponent,
        )

        return chamfer_dist

    def _sample_points_from_mesh(self, vertices, faces, n_samples):
        """
        Sample points uniformly (area-weighted) from a triangle mesh surface via trimesh.

        Args:
            vertices (np.ndarray): Mesh vertices, shape (N, 3)
            faces (np.ndarray): Mesh faces (triangle indices), shape (M, 3)
            n_samples (int): Number of points to sample

        Returns:
            np.ndarray: Sampled points, shape (n_samples, 3)
        """
        import trimesh

        if len(faces) == 0 or len(vertices) == 0:
            return np.zeros((n_samples, 3))

        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        if mesh.area == 0:
            return np.zeros((n_samples, 3))

        points, _ = trimesh.sample.sample_surface(mesh, n_samples)
        return np.asarray(points)

    def compute_point_cloud_coverage_error(self, recon_points, coverage_distance=0.01, n_samples=50000):
        """
        Fraction of ground-truth surface points not yet covered by the reconstruction.

        A GT surface point is "covered" if the reconstructed point cloud has a point
        within ``coverage_distance`` of it. Error is the fraction of uncovered GT points
        (1.0 = nothing covered, 0.0 = fully covered).

        Args:
            recon_points (np.ndarray): Reconstructed point cloud, shape (N, 3). Assumed
                append-only across steps within an episode, if it shrinks an error will be raised.
            coverage_distance (float): Neighborhood radius in meters.
            n_samples (int): Number of GT surface points to sample (cached per episode).

        Returns:
            tuple: (error, info_dict)
        """
        from scipy.spatial import cKDTree


        # Sample and cache the GT surface point cloud once per episode
        if self.gt_surface_points is None:
            self.gt_surface_points = self._sample_points_from_mesh(self.gt_vertices, self.gt_faces, n_samples)
            self.gt_covered = np.zeros(len(self.gt_surface_points), dtype=bool)
            self.gt_tree = cKDTree(self.gt_surface_points)
            self.n_recon_consumed = 0

        gt = self.gt_surface_points
        n_recon_points = len(recon_points)

        # Cloud shrank => not append-only (e.g. a different policy) => must not happen
        if n_recon_points < self.n_recon_consumed:
            raise RuntimeError("Number of reconstruction points decreased within an episode. Must not happen due to caching mechanism.")

        new_points = recon_points[self.n_recon_consumed :] if n_recon_points else []
        self.n_recon_consumed = n_recon_points

        if len(new_points) > 0:
            # GT points within coverage_distance of any newly added reconstruction point
            neighbors = self.gt_tree.query_ball_point(
                new_points, r=coverage_distance, workers=-1, return_sorted=False
            )
            hits = np.concatenate(neighbors) if len(neighbors) > 0 else []
            if len(hits) > 0:
                self.gt_covered[hits.astype(np.intp)] = True

        error = float(np.mean(~self.gt_covered))
        info = {
            "n_covered": int(self.gt_covered.sum()),
            "n_gt": len(gt),
            "n_recon_points": int(n_recon_points),
        }
        return error, info

    def compute_voxelwise_tsdf_error(
        self,
        input_sdf,
        truncation_distance: float,
        truncation_relax_factor=0.5,
        missing_voxel_penalty=1.0,
        unobserved_threshold=90.0,
        exclude_below_table_surface=True,
    ):
        """
        Compute error between a dense TSDF grid and the stored ground truth SDF.

        This method computes error in two parts:
        1. For observed voxels (not truncated): MSE between TSDF and clamped ground truth
        2. For voxels that should have been observed but weren't: fixed penalty

        A voxel "should have been observed" if its ground truth SDF is within
        missing_voxel_truncation_factor * truncation_distance (i.e., it's near the surface).

        Args:
            input_sdf (np.ndarray): Input TSDF grid to compare against ground truth.
                Must have the same shape as self.sdf_grid.
            truncation_distance (float): The truncation distance used by the TSDF (in meters).
            truncation_relax_factor (float): Fraction by which truncation_distance is relaxed 
                when determining which voxels should have been observed.
            missing_voxel_penalty (float): Fixed penalty added for the fraction of missing voxels
                that should have been observed. Default 1.0.
            unobserved_threshold (float): Threshold above which TSDF values are considered
                unobserved (sentinel value). Default 90.0.
            exclude_below_table_surface (bool): If True, exclude voxels below the table top
                surface from error computation. This prevents penalizing the agent for not
                observing the underside of the table. Default True.

        Returns:
            float: Combined error (MSE on observed + penalty for missing observations).
                   Returns inf if no voxels are observed and none should have been.

        Raises:
            RuntimeError: If compute_static_env_sdf has not been called yet
            ValueError: If input_sdf shape does not match ground truth SDF shape
        """
        # Check if SDF has been computed
        if self.sdf_grid is None:
            raise RuntimeError("Ground truth SDF has not been computed yet. Call compute_static_env_sdf() first.")

        # Check shape match
        if input_sdf.shape != self.sdf_grid.shape:
            raise ValueError(
                f"Input SDF shape {input_sdf.shape} does not match ground truth SDF shape {self.sdf_grid.shape}"
            )

        sdf_is = input_sdf
        sdf_should = self.sdf_grid
        error_components = {}

        # Build above-table mask: exclude voxels whose world z-coordinate is below the table surface.
        if exclude_below_table_surface and self.bbox_center is not None:
            z_world = self.bbox_center[2] + np.linspace(-self.bbox_size / 2, self.bbox_size / 2, sdf_is.shape[2])
            above_table_mask = z_world[np.newaxis, np.newaxis, :] >= (self.table_surface_z - truncation_relax_factor * truncation_distance)  # small delte for robustness
            # Broadcast to full 3D shape for element-wise ops
            above_table_mask = np.broadcast_to(above_table_mask, sdf_is.shape).copy()
        else:
            above_table_mask = np.ones(sdf_is.shape, dtype=bool)

        # Identify observed voxels (TSDF values below sentinel threshold), restricted to above-table region
        observed_mask = (np.abs(sdf_is) < unobserved_threshold) & above_table_mask
        n_observed = observed_mask.sum()

        # Part 1: Compute MSE on observed voxels
        if n_observed > 0:
            sdf_is_observed = sdf_is[observed_mask]
            sdf_should_observed = sdf_should[observed_mask]

            # Clamp ground truth to truncation range for fair comparison
            # sdf_should_observed = np.clip(sdf_should_observed, -truncation_distance, truncation_distance)

            # Compute squared error on observed
            diff = np.abs(sdf_is_observed - sdf_should_observed)

            norm_exponent = 1  # L1 norm
            mean_diff_norm_exponent = np.mean(np.power(diff, norm_exponent))
            observed_error = np.power(mean_diff_norm_exponent, 1 / norm_exponent)

            # write statistics about diff to error components
            error_components["observed_diff_max"] = np.max(diff)
            error_components["observed_diff_min"] = np.min(diff)

        else:
            observed_error = 0.0
            print(
                f"[Reconstruct3D] No observed voxels detected in step {self.timestep}. Assigning voxelwise TSDF error to 1.0."
            )

        # Part 2: Add penalty for missing observations
        # Identify voxels that should have been observed (GT SDF within missing_voxel_truncation_factor * truncation_distance)
        # Also restricted to above-table region to avoid penalizing for not observing below the table
        should_observe_mask = (np.abs(sdf_should) < (truncation_relax_factor * truncation_distance)) & above_table_mask
        n_should_observe = should_observe_mask.sum()

        # Count voxels that should have been observed but weren't
        n_missing = (should_observe_mask & ~observed_mask).sum()

        missing_error = (n_missing / n_should_observe) * missing_voxel_penalty if n_should_observe > 0 else 0.0

        error_components["observed_error"] = observed_error
        error_components["missing_error"] = missing_error
        error_components["n_observed"] = n_observed
        error_components["n_should_observe"] = n_should_observe
        error_components["n_missing"] = n_missing

        return observed_error + missing_error, error_components

    def compute_static_env_sdf(self):
        """
        Compute SDF from the static environment mesh and store in object variables.

        This method extracts the mesh from the environment (table + objects), normalizes it,
        computes the SDF using mesh2sdf, and stores the result along with bounding box
        information for later use.

        Note: the GT mesh only contains observable surfaces and is therefore open, so
        mesh2sdf has no reliable inside/outside — treat the resulting signs with care.

        Returns:
            np.ndarray: The computed SDF grid of shape (sdf_size, sdf_size, sdf_size)
            np.ndarray: The center of the SDF bounding box in global coordinates (shape (3,))
            float: The size of the SDF bounding box (length of the longest side)
        """
        import mesh2sdf

        vertices, faces = self.compute_gt_mesh()

        # Normalize vertices to [-1, 1] for mesh2sdf
        vertices_normalized = (vertices - self.bbox_center) / (self.bbox_size / 2)

        # Convert to SDF using mesh2sdf (returns values in normalized [-1, 1] space)
        sdf_normalized = mesh2sdf.compute(vertices_normalized, faces, size=self.sdf_size, fix=False, return_mesh=False)

        # Convert SDF values back to world units (meters)
        self.sdf_grid = sdf_normalized * (self.bbox_size / 2)

        return self.sdf_grid, self.bbox_center, self.bbox_size

    def compute_gt_mesh(self, min_normal_z=-0.1):
        """
        Ground-truth mesh of the static environment (table + objects), and its bounding box.

        Uses only the table's top face and each object's faces except
        the (near-)downward one it rests on — since only those should be reconstructed.

        Args:
            min_normal_z (float): An object face is kept if its world normal's z exceeds this.

        Returns:
            tuple: (vertices, faces)
        """
        body_names = {obj.root_body for obj in self.primitives_on_table} | {"table"}

        all_vertices, all_faces, offset = [], [], 0
        for geom_id in range(self.sim.model.ngeom):
            body_name = self.sim.model.body(self.sim.model.geom_bodyid[geom_id]).name

            if body_name not in body_names:
                continue
            # Every body carries a collision geom (group 0) and a visual duplicate (group 1);
            # taking only collision geoms avoids double surfaces and drops the table legs,
            # which exist as visual geoms only.
            if self.sim.model.geom_group[geom_id] != COLLISION_GEOM_GROUP:
                continue

            vertices, faces, normals = self._geom_mesh(geom_id, self.sim.model.geom_type[geom_id])

            # To world coordinates (normals only rotate)
            geom_mat = self.sim.data.geom_xmat[geom_id].reshape(3, 3)
            vertices = vertices @ geom_mat.T + self.sim.data.geom_xpos[geom_id]
            normal_z = (normals @ geom_mat.T)[:, 2]

            # The table is only ever seen from above; objects hide the face they stand on.
            faces = faces[normal_z > (0.9 if body_name == "table" else min_normal_z)]
            if len(faces) == 0:
                continue

            all_vertices.append(vertices)
            all_faces.append(faces + offset)
            offset += len(vertices)

        vertices = np.vstack(all_vertices) if all_vertices else np.zeros((0, 3))
        faces = np.vstack(all_faces) if all_faces else np.zeros((0, 3), dtype=np.int32)

        # Drop vertices left unreferenced by the removed faces, so the bounding box below
        # covers the observable region only
        if len(faces) > 0:
            used, faces = np.unique(faces, return_inverse=True)
            vertices, faces = vertices[used], faces.reshape(-1, 3)

        self.gt_vertices, self.gt_faces = vertices, faces

        # Bounding box, padded to avoid cutting off surfaces at the boundaries
        bbox_min, bbox_max = vertices.min(axis=0), vertices.max(axis=0)
        self.bbox_center = (bbox_min + bbox_max) / 2
        self.bbox_size = (bbox_max - bbox_min).max() * (1 + 2 * self.bbox_padding)

        return self.gt_vertices, self.gt_faces

    def _geom_mesh(self, geom_id, geom_type):
        """
        Triangle mesh of a single geom in its local frame.

        Returns:
            tuple: (vertices, faces, face_normals) with outward normals — `trimesh.creation`
                primitives are wound outward, imported meshes are re-oriented per closed body.
        """
        import mujoco
        import trimesh

        model = self.sim.model
        size = model.geom_size[geom_id]

        if geom_type == mujoco.mjtGeom.mjGEOM_BOX:
            mesh = trimesh.creation.box(extents=2.0 * np.asarray(size[:3]))
        elif geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
            mesh = trimesh.creation.cylinder(radius=size[0], height=2.0 * size[1], sections=16)
        elif geom_type == mujoco.mjtGeom.mjGEOM_MESH:
            mesh_id = model.geom_dataid[geom_id]
            last = mesh_id >= model.nmesh - 1
            v0, v1 = model.mesh_vertadr[mesh_id], len(model.mesh_vert) if last else model.mesh_vertadr[mesh_id + 1]
            f0, f1 = model.mesh_faceadr[mesh_id], len(model.mesh_face) if last else model.mesh_faceadr[mesh_id + 1]
            mesh = trimesh.Trimesh(model.mesh_vert[v0:v1], model.mesh_face[f0:f1], process=False)
            mesh.fix_normals()
        else:
            raise NotImplementedError(f"Unsupported geom type {geom_type} for mesh extraction.")

        return np.asarray(mesh.vertices), np.asarray(mesh.faces, dtype=np.int32), np.asarray(mesh.face_normals)

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()

        # Adjust base pose accordingly
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        # load model for table top workspace
        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        # initialize objects of interest
        tex_attrib = {
            "type": "cube",
        }
        mat_attrib = {
            "texrepeat": "1 1",
            "specular": "0.4",
            "shininess": "0.1",
        }
        redwood = CustomMaterial(
            texture="WoodRed",
            tex_name="redwood",
            mat_name="redwood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        greenwood = CustomMaterial(
            texture="WoodGreen",
            tex_name="greenwood",
            mat_name="greenwood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )

        self.primitives_on_table = []

        self.primitives_on_table.append(
            BoxObject(
                name="Box_red",
                size_min=[0.05, 0.05, 0.05],
                size_max=[0.2, 0.2, 0.2],
                rgba=[1, 0, 0, 1],
                material=redwood,
            )
        )

        self.primitives_on_table.append(
            BoxObject(
                name="Box_green",
                size_min=[0.05, 0.05, 0.05],
                size_max=[0.2, 0.2, 0.2],
                rgba=[0, 1, 0, 1],
                material=greenwood,
            )
        )

        self.primitives_on_table.append(
            CylinderObject(
                name="Cylinder_red",
                size_min=[0.05, 0.05],
                size_max=[0.2, 0.2],
                rgba=[0, 0, 1, 1],
                material=redwood,
            )
        )

        self.primitives_on_table.append(
            CylinderObject(
                name="Cylinder_green",
                size_min=[0.05, 0.05],
                size_max=[0.2, 0.2],
                rgba=[0, 0, 1, 1],
                material=greenwood,
            )
        )

        # Create placement initializer
        if self.placement_initializer is not None:
            self.placement_initializer.reset()
            self.placement_initializer.add_objects(self.primitives_on_table)
        else:
            self.placement_initializer = UniformRandomSampler(
                name="ObjectSampler",
                mujoco_objects=self.primitives_on_table,
                x_range=[self.table_full_size[0] * -0.4, self.table_full_size[0] * 0.4],
                y_range=[self.table_full_size[1] * -0.4, self.table_full_size[1] * 0.4],
                rotation=None,  # uniform random rotation
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.00,  # place on table surface
                rng=self.rng,
            )

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self.primitives_on_table,
        )

    def _setup_references(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._setup_references()

        self.primitives_on_table_ids = [self.sim.model.body_name2id(obj.root_body) for obj in self.primitives_on_table]

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()
        self.cached_error = None
        self.gt_surface_points = None
        self.gt_covered = None
        self.gt_tree = None
        self.n_recon_consumed = 0

        # Reset all object positions using initializer sampler if we're not directly loading from an xml
        if not self.deterministic_reset:

            # Sample from the placement initializer for all objects
            object_placements = self.placement_initializer.sample()

            # Loop through all objects and reset their positions
            for obj_pos, obj_quat, obj in object_placements.values():
                self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)]))

    def _setup_observables(self):
        """
        Sets up observables to be used for this environment. Creates object-based observables if enabled

        Returns:
            OrderedDict: Dictionary mapping observable names to its corresponding Observable object
        """
        observables = super()._setup_observables()

        # No extra observables for now

        return observables

    def _check_success(self):
        """
        Check if reconstruction task is successfully completed.

        Returns:
            bool: False (so far no success condition is defined)
        """
        return False

