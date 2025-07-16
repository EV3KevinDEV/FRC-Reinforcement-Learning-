// Copyright (c) FIRST and other WPILib contributors.
// Open Source Software; you can modify and/or share it under the terms of
// the WPILib BSD license file in the root directory of this project.

package frc.robot.commands;

import edu.wpi.first.networktables.BooleanPublisher;
import edu.wpi.first.networktables.NetworkTable;
import edu.wpi.first.networktables.NetworkTableInstance;
import edu.wpi.first.wpilibj.smartdashboard.SmartDashboard;
import edu.wpi.first.wpilibj2.command.Command;
import edu.wpi.first.wpilibj2.command.CommandScheduler;
import frc.robot.subsystems.ArmElevator;
import frc.robot.subsystems.Intake;
import frc.robot.subsystems.SwerveSubsystem.CommandSwerveDrivetrain;

/**
 * A command to toggle between manual control and NetworkTable (RL) control.
 * When enabled, it starts the NetworkTableControlCommand.
 * When disabled, it stops the NetworkTable control and returns to manual control.
 */
public class ToggleNetworkTableControl extends Command {
    private final CommandSwerveDrivetrain m_drivetrain;
    private final ArmElevator m_armElevator;
    private final Intake m_intake;
    
    private final NetworkTable m_controlTable;
    private final BooleanPublisher m_enabledPub;
    
    private NetworkTableControlCommand m_networkTableCommand;
    private boolean m_isEnabled = false;
    
    /**
     * Creates a new ToggleNetworkTableControl command.
     */
    public ToggleNetworkTableControl(CommandSwerveDrivetrain drivetrain,
                                   ArmElevator armElevator,
                                   Intake intake) {
        m_drivetrain = drivetrain;
        m_armElevator = armElevator;
        m_intake = intake;
        
        // Initialize NetworkTables for status reporting
        NetworkTableInstance inst = NetworkTableInstance.getDefault();
        m_controlTable = inst.getTable("RL_Control");
        m_enabledPub = m_controlTable.getBooleanTopic("enabled").publish();
        
        // Don't require subsystems since this is just a toggle command
    }
    
    @Override
    public void initialize() {
        // Toggle the enabled state
        m_isEnabled = !m_isEnabled;
        
        if (m_isEnabled) {
            // Enable NetworkTable control
            enableNetworkTableControl();
        } else {
            // Disable NetworkTable control
            disableNetworkTableControl();
        }
        
        // Publish the current state
        m_enabledPub.set(m_isEnabled);
    }
    
    @Override
    public void execute() {
        // Nothing to do during execution
    }
    
    @Override
    public void end(boolean interrupted) {
        // Command ends immediately after toggling
    }
    
    @Override
    public boolean isFinished() {
        return true; // This command finishes immediately after toggling
    }
    
    /**
     * Enables NetworkTable control by starting the NetworkTableControlCommand
     */
    private void enableNetworkTableControl() {
        if (m_networkTableCommand == null) {
            m_networkTableCommand = new NetworkTableControlCommand(m_drivetrain, m_armElevator, m_intake);
        }
        
        // Cancel any existing default commands and start NetworkTable control
        CommandScheduler.getInstance().cancel(m_drivetrain.getDefaultCommand());
        m_networkTableCommand.schedule();
        
        System.out.println("NetworkTable control ENABLED - Robot is now controlled by RL agent");
        
        // Set up initial NetworkTable values with safe defaults
        m_controlTable.getEntry("velocity_x").setDouble(0.0);
        m_controlTable.getEntry("velocity_y").setDouble(0.0);
        m_controlTable.getEntry("angular_velocity").setDouble(0.0);
        m_controlTable.getEntry("elevator_height").setDouble(0.0);
        m_controlTable.getEntry("arm_angle").setDouble(0.0);
        m_controlTable.getEntry("intake_command").setDouble(0.0);
        
        // Update SmartDashboard status
        SmartDashboard.putString("RL_Control_Status", "ENABLED - Robot controlled by NetworkTables");
    }
    
    /**
     * Disables NetworkTable control and returns to manual control
     */
    private void disableNetworkTableControl() {
        if (m_networkTableCommand != null && m_networkTableCommand.isScheduled()) {
            m_networkTableCommand.cancel();
        }
        
        System.out.println("NetworkTable control DISABLED - Robot returned to manual control");
        
        // Update SmartDashboard status
        SmartDashboard.putString("RL_Control_Status", "DISABLED - Manual control active");
        
        // The drivetrain's default command should automatically resume
    }
    
    /**
     * Returns whether NetworkTable control is currently enabled
     */
    public boolean isNetworkTableControlEnabled() {
        return m_isEnabled;
    }
}

/*
 * TESTING INSTRUCTIONS:
 * 
 * To test the NetworkTable control system:
 * 
 * 1. Deploy this code to your robot
 * 2. Open SmartDashboard or Shuffleboard
 * 3. Look for these entries under "NT_Control" tab:
 *    - Test_Mode: Set to true to use SmartDashboard values instead of NetworkTables
 *    - Test_VelX: Forward/backward velocity (-5.0 to 5.0 m/s)
 *    - Test_VelY: Left/right velocity (-5.0 to 5.0 m/s) 
 *    - Test_AngVel: Rotational velocity (-6.0 to 6.0 rad/s)
 *    - Test_ElevHeight: Elevator height (0.0 to 2.5 m)
 *    - Test_ArmAngle: Arm angle (-90.0 to 120.0 degrees)
 *    - Test_Intake: Intake command (1=intake, 0=stop, -1=eject)
 * 
 * 4. Also look for "RL_Control_Status" to see if control is enabled
 * 
 * TESTING PROCEDURE:
 * 1. First, bind the ToggleNetworkTableControl command to a button in RobotContainer
 * 2. Press the button to enable NetworkTable control
 * 3. Set "Test_Mode" to true in SmartDashboard
 * 4. Adjust the test values and observe robot movement
 * 5. Monitor "Current_*" values to see what the robot is receiving
 * 
 * SAFETY NOTES:
 * - All values are clamped to safe ranges
 * - The robot will stop moving when the command is disabled
 * - Test in simulation first before deploying to real robot
 * 
 * NETWORKTABLES STRUCTURE FOR EXTERNAL RL AGENT:
 * 
 * Control Topics (RL agent writes to these):
 * - /RL_Control/velocity_x (double)
 * - /RL_Control/velocity_y (double) 
 * - /RL_Control/angular_velocity (double)
 * - /RL_Control/elevator_height (double)
 * - /RL_Control/arm_angle (double)
 * - /RL_Control/intake_command (double)
 * - /RL_Control/enabled (boolean)
 * 
 * State Topics (RL agent reads from these):
 * - /RL_State/pose_x (double)
 * - /RL_State/pose_y (double)
 * - /RL_State/pose_rotation (double)
 * - /RL_State/velocity_x (double)
 * - /RL_State/velocity_y (double)
 * - /RL_State/angular_velocity (double)
 * - /RL_State/elevator_height (double)
 * - /RL_State/arm_angle (double)
 * - /RL_State/at_setpoint (boolean)
 * - /RL_State/has_game_piece (boolean)
 * - /RL_State/timestamp (double)
 * 
 * EXAMPLE PYTHON CODE FOR RL AGENT:
 * 
 * import ntcore
 * import time
 * 
 * # Connect to robot NetworkTables
 * inst = ntcore.NetworkTableInstance.getDefault()
 * inst.startClient4("RL_Agent")
 * inst.setServerTeam(XXXX)  # Replace with your team number
 * 
 * # Get tables
 * control_table = inst.getTable("RL_Control")
 * state_table = inst.getTable("RL_State")
 * 
 * # Enable control
 * control_table.putBoolean("enabled", True)
 * 
 * # Control loop
 * while True:
 *     # Read current state
 *     pose_x = state_table.getNumber("pose_x", 0.0)
 *     pose_y = state_table.getNumber("pose_y", 0.0)
 *     # ... read other state values
 *     
 *     # Run your RL algorithm here to determine actions
 *     action_vx = your_rl_algorithm.get_action_vx(state)
 *     action_vy = your_rl_algorithm.get_action_vy(state)
 *     # ... other actions
 *     
 *     # Send control commands
 *     control_table.putNumber("velocity_x", action_vx)
 *     control_table.putNumber("velocity_y", action_vy)
 *     # ... send other commands
 *     
 *     time.sleep(0.02)  # 50Hz control loop
 */
