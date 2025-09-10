from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sensors.ray_caster import RayCasterCfg, patterns
from isaaclab.sensors import ContactSensorCfg
import isaaclab.sim as sim_utils
import math

ROBOT_CFG = ArticulationCfg(
   debug_vis=True,
   spawn=sim_utils.UsdFileCfg(
        usd_path=f"/home/kevin/Desktop/MasterRSCH_Suck_V2.usd"
   ),
   # Add proper init_state to prevent tensor shape issues in acceleration calculation
   init_state=ArticulationCfg.InitialStateCfg(
       pos=(-6.5, 0.0, 0.1),  # Slightly above ground
       rot=(1, 0.0, 0.0, 0),  # No rotation
       joint_pos={
           ".*": 0.0,  # All joints at zero position
       },
       joint_vel={
           ".*": 0.0,  # All joints at zero velocity
       },
   ),
   # Add actuators for all joints with proper limits

   ## Index 0: dof_elevator_1
    # Index 1: dof_elevator_2
    # Index 2: dof_lturret_1
    # Index 3: dof_lturret_2
    # Index 4: dof_rturret_1
    # Index 5: dof_rturret_2
    # Index 6: dof_intake_1
    # Index 7: dof_lwheel_1
    # Index 8: dof_lwheel_2
    # Index 9: dof_rwheel_1
    # Index 10: dof_rwheel_2
    # Index 11: FixedJoint (ignore)

    #Intake Piece Prim: /Root/MasterRSCH/Gripper/Intake_Piece
   actuators= {
       # All joints (except FixedJoint) get a default actuator config
       "wheel_turret": ImplicitActuatorCfg(
           joint_names_expr=["dof_lturret_1", "dof_lturret_2", "dof_rturret_1", "dof_rturret_2"],  # All joints
           effort_limit=40.0,  # Reasonable effort limit
           velocity_limit=2 * math.pi,  # Reasonable velocity limit
           stiffness=105.0,
           damping=30.0,
       ),
       # Wheels (if you want to override for specific wheels)
       "wheels": ImplicitActuatorCfg(
           joint_names_expr=["dof_lwheel_1", "dof_lwheel_2", "dof_rwheel_1", "dof_rwheel_2"],
           effort_limit=50.0,
           stiffness=10.0,
           damping=800.0,
       ),
       # Elevator DOFs
       "elevator_dof": ImplicitActuatorCfg(
           joint_names_expr=["dof_elevator_1", "dof_elevator_2"],
           stiffness=12000.0,
           damping=2500.0,
       ),
       # Intake DOF
       "intake_dof": ImplicitActuatorCfg(
           joint_names_expr=["dof_intake_1"],
           stiffness=120.0,
           damping=25.0
       )

   }
)

