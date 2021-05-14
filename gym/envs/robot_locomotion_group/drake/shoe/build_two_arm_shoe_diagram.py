from future.utils import iteritems
import math
import numpy as np
import os

from pydrake.common.cpp_param import List
from pydrake.common.eigen_geometry import Quaternion
from pydrake.common.value import Value
from pydrake.math import (
    RigidTransform,
    RollPitchYaw,
    RotationMatrix
)
from pydrake.multibody.math import SpatialForce
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import (
    ExternallyAppliedSpatialForce
)
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import (
    BasicVector,
    DiagramBuilder,
    LeafSystem
)
from pydrake.systems.primitives import (
    ConstantVectorSource
)
from pydrake.manipulation.planner import (
    DifferentialInverseKinematicsParameters)
from pydrake.systems.primitives import FirstOrderLowPassFilter

from gym.envs.robot_locomotion_group.drake.shoe.differential_ik import DifferentialIK

from gym.envs.robot_locomotion_group.drake.shoe.rope_utils import (
    post_finalize_rope_settings,
    initialize_rope_zero
)
from gym.envs.robot_locomotion_group.drake.shoe.transform_utils import (
    transform_to_vector
)
from gym.envs.robot_locomotion_group.drake.shoe.manipulation_diagram import ManipulationDiagram

OPEN_GRIPPER_STATE = [-0.05, 0.05]
CLOSE_GRIPPER_STATE = [-0.008, 0.008]
ARM_UP_POSITION = [0.0118836, 0.53509565, -0.02763152, -2.28621888, 0.00601891, -1.2116513 , -3.12522438]
ARM_DOWN_POSITION = [0, 0.2, 0, -2.5, 0, 0.44, -math.pi]

ARM_STEP_1 = [-0.26331536,  0.63833692, -0.18554668, -2.18210709, -0.35944272, -1.52592672, -2.94267866]
ARM_STEP_2 = [-0.2295779 ,  0.70016039, -0.16652777, -2.02923555, -0.31088144, -1.43394083, -2.93283167]
ARM_STEP_2_OUT = [-0.30372117, 0.72598096, -0.20686482, -1.9706917, -0.39988438, -1.43769007, -2.85823189]

class ArmsController(LeafSystem):
    def __init__(self):
        LeafSystem.__init__(self)
        self.rpyxyz_left_output_port = self.DeclareVectorOutputPort(
            "rpy_xyz_left",
            BasicVector(6),
            self.DoCalcLeftPose
        )
        self.rpyxyz_right_output_port = self.DeclareVectorOutputPort(
            "rpy_xyz_right",
            BasicVector(6),
            self.DoCalcRightPose
        )
        self.gripper_left_output_port = self.DeclareVectorOutputPort(
            "gripper_left",
            BasicVector(1),
            self.DoCalcLeftGripper
        )
        self.gripper_right_output_port = self.DeclareVectorOutputPort(
            "gripper_right",
            BasicVector(1),
            self.DoCalcRightGripper
        )
        self.rpyxyz_left = [0, 0, 0, 0, 0, 0]
        self.rpyxyz_right = [0, 0, 0, 0, 0, 0]
        self.left_width = 0
        self.right_width = 0

    def DoCalcLeftPose(self, context, y_data):
        y_data.SetFromVector(self.rpyxyz_left)

    def DoCalcRightPose(self, context, y_data):
        y_data.SetFromVector(self.rpyxyz_right)

    def DoCalcLeftGripper(self, context, y_data):
        y_data.SetFromVector([self.left_width])

    def DoCalcRightGripper(self, context, y_data):
        y_data.SetFromVector([self.right_width])

    def rotate(self, arm, angle, rpy):
        def func():
            if rpy == "roll":
                index = 0
            elif rpy == "pitch":
                index = 1
            elif rpy == "yaw":
                index = 2
            if arm == "left":
                self.rpyxyz_left[index] += angle
            elif arm == "right":
                self.rpyxyz_right[index] += angle
        return func

    def translate(self, arm, xyz, dis):
        def func():
            if xyz == "x":
                index = 3
            elif xyz == "y":
                index = 4
            elif xyz == "z":
                index = 5
            if arm == "left":
                self.rpyxyz_left[index] += dis
            elif arm == "right":
                self.rpyxyz_right[index] += dis
        return func

    def gripper(self, arm, width):
        def func():
            if arm == "left":
                self.left_width = width
            elif arm == "right":
                self.right_width = width
        return func

    def set_rpyxyz(self, rpyxyz, arm):
        def func():
            if arm == "left":
                self.rpyxyz_left = rpyxyz
            elif arm == "right":
                self.rpyxyz_right  = rpyxyz
        return func

    def set_drpyxyzw(self, drpyxyzw):
        def func():
            self.set_rpyxyz(self.rpyxyz_left + drpyxyzw["left"][:6], "left")()
            self.gripper("left", self.left_width + drpyxyzw["left"][6])()

            self.set_rpyxyz(self.rpyxyz_right + drpyxyzw["right"][:6], "right")()
            self.gripper("right", self.right_width + drpyxyzw["right"][6])()
        return func

def setup_ee_position_input_port(config, builder, station, arm_name, filter_const=None):
    """
    if filter_const is set, we override the one from the config
    """
    robot = station.get_control_mbp(arm_name)
    params = DifferentialInverseKinematicsParameters(robot.num_positions(),
                                                     robot.num_velocities())
    time_step = config['env']['mbp_dt']
    params.set_timestep(time_step)
    # True velocity limits for the IIWA14 (in rad, rounded down to the first
    # decimal)
    iiwa14_velocity_limits = np.array([1.4, 1.4, 1.7, 1.3, 2.2, 2.3, 2.3])
    # Stay within a small fraction of those limits for this teleop demo.
    factor = config['env']['velocity_limit_factor']
    params.set_joint_velocity_limits((-factor*iiwa14_velocity_limits,
                                      factor*iiwa14_velocity_limits))

    differential_ik = builder.AddSystem(DifferentialIK(
        robot, robot.GetFrameByName(f"iiwa_link_7"), params, time_step))

    builder.Connect(differential_ik.GetOutputPort("joint_position_desired"),
                    station.GetInputPort(f"{arm_name}_position"))
    return differential_ik

def initialize_arm_from_values(diagram, simulator, station, arm_name, default_joints, gripper_values):
    simulator_context = simulator.get_mutable_context()
    station_context = diagram.GetMutableSubsystemContext(station, simulator_context)
    station.set_model_state(station_context, arm_name, default_joints, np.zeros(7))
    station.set_model_state(station_context, f"{arm_name}_gripper", gripper_values, [0, 0])

def build_two_rope_two_arm_diagram(config):
    builder = DiagramBuilder()

    station = builder.AddSystem(ManipulationDiagram(config))
    station.add_rope_and_ground(include_ground=False)
    station.add_arms_from_config(config)

    parser = Parser(station.mbp, station.sg)
    shoe_dir = os.path.dirname(os.path.abspath(__file__))

    shoe_file = os.path.join(shoe_dir, "model/shoe_realistic.sdf")
    shoe_model = parser.AddModelFromFile(shoe_file, "shoe")
    if config["env"]["visualization"]:
        station.connect_to_drake_visualizer()
    visualizer = station.connect_to_meshcat()

    station.finalize()
    post_finalize_rope_settings(config, station.mbp, station.sg)
    left_diff_ik = setup_ee_position_input_port(config, builder,
                                                             station, "left")
    right_diff_ik = setup_ee_position_input_port(config, builder,
                                                               station, "right")

    arms_controller = builder.AddSystem(ArmsController())
    builder.Connect(arms_controller.GetOutputPort("rpy_xyz_left"),
                    left_diff_ik.GetInputPort("rpy_xyz_desired"))
    builder.Connect(arms_controller.GetOutputPort("rpy_xyz_right"),
                    right_diff_ik.GetInputPort("rpy_xyz_desired"))

    left_grip_filter = builder.AddSystem(FirstOrderLowPassFilter(
        time_constant=config['env']['filter_time_const'],
        size=1))
    right_grip_filter = builder.AddSystem(FirstOrderLowPassFilter(
        time_constant=config['env']['filter_time_const'],
        size=1))
    builder.Connect(arms_controller.GetOutputPort("gripper_left"),
                    left_grip_filter.get_input_port(0))
    builder.Connect(arms_controller.GetOutputPort("gripper_right"),
                    right_grip_filter.get_input_port(0))
    builder.Connect(left_grip_filter.get_output_port(0),
                    station.GetInputPort("left_gripper_position"))
    builder.Connect(right_grip_filter.get_output_port(0),
                    station.GetInputPort("right_gripper_position"))

    diagram = builder.Build()

    simulator = Simulator(diagram)

    sim_context = simulator.get_mutable_context()
    station_context = diagram.GetMutableSubsystemContext(station, sim_context)

    station.GetInputPort("left_feedforward_torque").FixValue(
        station_context, np.zeros(7))
    station.GetInputPort("right_feedforward_torque").FixValue(
        station_context, np.zeros(7))
    station.GetInputPort("left_gripper_force_limit").FixValue(
                         station_context, 40.)
    station.GetInputPort("right_gripper_force_limit").FixValue(
                         station_context, 40.)

    systems = {"station": station,
               "left_grip_filter": left_grip_filter,
               "right_grip_filter": right_grip_filter,
               "left_diff_ik": left_diff_ik,
               "right_diff_ik": right_diff_ik,
               "arms_controller": arms_controller,
               "visualizer": visualizer}

    default_joints = ARM_UP_POSITION
    default_gripper = CLOSE_GRIPPER_STATE
    left_rpyxyz, right_rpyxyz = reset_simulator(config, simulator, diagram, systems)

    simulator.set_target_realtime_rate(config['env']['target_realtime_rate'])
    return simulator, diagram, systems

def reset_simulator(config, simulator, diagram, systems):
    # Set initial position
    default_joints = ARM_UP_POSITION
    default_gripper = CLOSE_GRIPPER_STATE
    for rope_name, _ in iteritems(config['env']['ropes']):
        initialize_rope_zero(diagram, simulator, systems["station"], rope_name)
    initialize_arm_from_values(diagram, simulator, systems["station"], "left",
                               default_joints, default_gripper)
    initialize_arm_from_values(diagram, simulator, systems["station"], "right",
                               default_joints, default_gripper)

    sim_context = simulator.get_mutable_context()
    station_context = diagram.GetMutableSubsystemContext(
        systems["station"], sim_context)
    left_grip_filter_context = diagram.GetMutableSubsystemContext(
        systems["left_grip_filter"], sim_context)
    right_grip_filter_context = diagram.GetMutableSubsystemContext(
        systems["right_grip_filter"], sim_context)
    left_ik_context = diagram.GetMutableSubsystemContext(
        systems["left_diff_ik"], sim_context)
    right_ik_context = diagram.GetMutableSubsystemContext(
        systems["right_diff_ik"], sim_context)

    sim_context.SetTime(0.)

    left_q0 = systems["station"].GetOutputPort("left_position_measured").Eval(
        station_context)
    right_q0 = systems["station"].GetOutputPort("right_position_measured").Eval(
        station_context)
    systems["left_diff_ik"].parameters.set_nominal_joint_position(left_q0)
    systems["right_diff_ik"].parameters.set_nominal_joint_position(right_q0)
    systems["left_diff_ik"].SetPositions(left_ik_context, left_q0)
    systems["right_diff_ik"].SetPositions(right_ik_context, right_q0)

    # TODO: Remove duplication here
    left_tf = systems["left_diff_ik"].ForwardKinematics(left_q0)
    left_rpyxyz = transform_to_vector(left_tf)

    right_tf = systems["right_diff_ik"].ForwardKinematics(right_q0)
    right_rpyxyz = transform_to_vector(right_tf)

    left_grip_value = systems["station"].get_model_state(station_context, f"left_gripper")["position"]
    left_grip_width = left_grip_value[1] - left_grip_value[0]
    right_grip_value = systems["station"].get_model_state(station_context, f"right_gripper")["position"]
    right_grip_width = right_grip_value[1] - right_grip_value[0]
    systems["left_grip_filter"].set_initial_output_value(left_grip_filter_context, [left_grip_width])
    systems["right_grip_filter"].set_initial_output_value(right_grip_filter_context, [right_grip_width])


    systems["arms_controller"].set_rpyxyz(left_rpyxyz, "left")()
    systems["arms_controller"].set_rpyxyz(right_rpyxyz, "right")()
    systems["arms_controller"].gripper("left", left_grip_width)()
    systems["arms_controller"].gripper("right", right_grip_width)()
    return left_rpyxyz, right_rpyxyz