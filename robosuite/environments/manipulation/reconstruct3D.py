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
        control_freq=10,
        lite_physics=True,
        horizon=100,
        ignore_done=True,
        hard_reset=True,
        camera_names="robot0_eye_in_hand",
        camera_heights=128,
        camera_widths=128,
        camera_depths=True,
        camera_segmentations=None,  # {None, instance, class, element}
        renderer="mjviewer",
        renderer_config=None,
        seed=None,
    ):
        # settings for table top
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = np.array((0, 0, 0.8))

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        # object placement initializer
        self.placement_initializer = placement_initializer

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

    def reward(self, action):
        from robosuite.utils.log_utils import ROBOSUITE_DEFAULT_LOGGER

        ROBOSUITE_DEFAULT_LOGGER.warning(
            "Reward function for Reconstruct3D environment is not implemented yet. Returning 0."
        )

        return 0

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

    def get_static_env_mesh(self, geom_groups=None):
        """
        Extract static environment mesh (table + objects) from MuJoCo simulation.

        Returns:
            tuple: (vertices, faces) where:
                - vertices: (N, 3) array of vertex positions
                - faces: (M, 3) array of triangle face indices
        """
        import mujoco

        all_vertices = []
        all_faces = []
        vertex_offset = 0

        # Get body names for objects we want to include
        object_body_names = set()
        for obj in self.primitives_on_table:
            object_body_names.add(obj.root_body)

        # Add table body (note: body name is "table", not "table_collision")
        object_body_names.add("table")

        # Normalize geom_groups input
        if geom_groups is not None:
            geom_groups = set(geom_groups)

        # Iterate over all geoms to find table and objects
        for geom_id in range(self.sim.model.ngeom):
            geom_type = self.sim.model.geom_type[geom_id]
            body_id = self.sim.model.geom_bodyid[geom_id]
            body_name = self.sim.model.body(body_id).name

            # Only include table and objects
            if body_name not in object_body_names:
                continue

            # If geom_groups filter provided, enforce it
            if geom_groups is not None:
                geom_group = self.sim.model.geom_group[geom_id]
                if geom_group not in geom_groups:
                    continue

            # Get geom pose
            geom_pos = self.sim.data.geom_xpos[geom_id]
            geom_mat = self.sim.data.geom_xmat[geom_id].reshape(3, 3)

            # Handle meshes
            if geom_type == mujoco.mjtGeom.mjGEOM_MESH:
                mesh_id = self.sim.model.geom_dataid[geom_id]

                # Extract mesh vertices
                vert_start = self.sim.model.mesh_vertadr[mesh_id]
                vert_end = (
                    self.sim.model.mesh_vertadr[mesh_id + 1]
                    if mesh_id < self.sim.model.nmesh - 1
                    else self.sim.model.mesh_vert.shape[0]
                )
                vertices = self.sim.model.mesh_vert[vert_start:vert_end].copy()

                # Extract mesh faces
                face_start = self.sim.model.mesh_faceadr[mesh_id]
                face_end = (
                    self.sim.model.mesh_faceadr[mesh_id + 1]
                    if mesh_id < self.sim.model.nmesh - 1
                    else self.sim.model.mesh_face.shape[0]
                )
                faces = self.sim.model.mesh_face[face_start:face_end].copy()

            else:
                # Generate mesh for primitive shapes
                vertices, faces = self._generate_primitive_mesh(geom_type, geom_id)

            # Transform vertices to world coordinates
            vertices = vertices @ geom_mat.T + geom_pos

            # Add to combined mesh
            all_vertices.append(vertices)
            all_faces.append(faces + vertex_offset)
            vertex_offset += len(vertices)

        # Combine all meshes
        combined_vertices = np.vstack(all_vertices) if all_vertices else np.zeros((0, 3))
        combined_faces = np.vstack(all_faces) if all_faces else np.zeros((0, 3), dtype=np.int32)

        return combined_vertices, combined_faces

    def _generate_primitive_mesh(self, geom_type, geom_id):
        """Generate mesh for primitive shapes (box, cylinder, etc.)."""
        import mujoco

        geom_size = self.sim.model.geom_size[geom_id]

        if geom_type == mujoco.mjtGeom.mjGEOM_BOX:
            # Box: 8 vertices, 12 triangles
            sx, sy, sz = geom_size
            vertices = np.array(
                [
                    [-sx, -sy, -sz],
                    [sx, -sy, -sz],
                    [sx, sy, -sz],
                    [-sx, sy, -sz],
                    [-sx, -sy, sz],
                    [sx, -sy, sz],
                    [sx, sy, sz],
                    [-sx, sy, sz],
                ]
            )
            faces = np.array(
                [
                    [0, 1, 2],
                    [0, 2, 3],
                    [4, 5, 6],
                    [4, 6, 7],  # bottom, top
                    [0, 1, 5],
                    [0, 5, 4],
                    [2, 3, 7],
                    [2, 7, 6],  # front, back
                    [0, 3, 7],
                    [0, 7, 4],
                    [1, 2, 6],
                    [1, 6, 5],  # left, right
                ]
            )

        elif geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
            # Cylinder: approximate with triangular mesh
            radius, half_height = geom_size[0], geom_size[1]
            n_segments = 16
            vertices = []
            faces = []

            # Generate vertices
            for i in range(n_segments):
                angle = 2 * np.pi * i / n_segments
                x, y = radius * np.cos(angle), radius * np.sin(angle)
                vertices.append([x, y, -half_height])
                vertices.append([x, y, half_height])

            # Add center vertices for caps
            vertices.append([0, 0, -half_height])
            vertices.append([0, 0, half_height])

            vertices = np.array(vertices)

            # Generate faces
            for i in range(n_segments):
                i1, i2 = 2 * i, 2 * ((i + 1) % n_segments)
                # Side faces
                faces.append([i1, i2, i1 + 1])
                faces.append([i1 + 1, i2, i2 + 1])
                # Bottom cap
                faces.append([2 * n_segments, i1, i2])
                # Top cap
                faces.append([2 * n_segments + 1, i2 + 1, i1 + 1])

            faces = np.array(faces)

        else:
            # Fallback for unsupported types
            vertices = np.zeros((0, 3))
            faces = np.zeros((0, 3), dtype=np.int32)
            print(f"Warning: Unsupported geom type {geom_type} for mesh extraction.")
            raise NotImplementedError(f"Unsupported geom type {geom_type} for mesh extraction.")

        return vertices, faces
