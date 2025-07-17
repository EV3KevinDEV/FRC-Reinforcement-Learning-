// Copyright (c) FIRST and other WPILib contributors.
// Open Source Software; you can modify and/or share it under the terms of
// the WPILib BSD license file in the root directory of this project.

package frc.robot.commands;

import edu.wpi.first.math.kinematics.ChassisSpeeds;
import edu.wpi.first.networktables.DoubleSubscriber;
import edu.wpi.first.networktables.NetworkTable;
import edu.wpi.first.networktables.NetworkTableInstance;
import edu.wpi.first.wpilibj.smartdashboard.SmartDashboard;
import edu.wpi.first.wpilibj2.command.Command;
import frc.robot.subsystems.ArmElevator;
import frc.robot.subsystems.Intake;
import frc.robot.subsystems.SwerveSubsystem.CommandSwerveDrivetrain;
import com.ctre.phoenix6.swerve.SwerveRequest;

/**
 * A command that reads control values from NetworkTables and controls the robot accordingly.
 * This is designed for reinforcement learning where an external agent sends commands
 * through NetworkTables to control the robot's drivetrain, arm, elevator, and intake.
 */
public class NetworkTableControlCommand extends Command {
    // Subsystems
    private final CommandSwerveDrivetrain m_drivetrain;
    private final ArmElevator m_armElevator;
    private final Intake m_intake;
    
    // NetworkTables subscribers for receiving control commands
    private final NetworkTable m_controlTable;
    private final DoubleSubscriber m_velocityXSub;
    private final DoubleSubscriber m_velocityYSub;
    private final DoubleSubscriber m_angularVelocitySub;
    private final DoubleSubscriber m_elevatorHeightSub;
    private final DoubleSubscriber m_armAngleSub;
    private final DoubleSubscriber m_intakeCommandSub;
    
    // Swerve request for controlling drivetrain
    // private final SwerveRequest.RobotCentric m_driveRequest = new SwerveRequest.RobotCentric();
    private final SwerveRequest.RobotCentric m_driveRequest = new SwerveRequest.RobotCentric();
    
    // Control values (defaults to safe values)
    private double m_velocityX = 0.0;      // Forward/backward velocity (m/s)
    private double m_velocityY = 0.0;      // Left/right velocity (m/s)
    private double m_angularVelocity = 0.0; // Rotational velocity (rad/s)
    private double m_elevatorHeight = 0.0;  // Elevator height (m)
    private double m_armAngle = 0.0;        // Arm angle (degrees)
    private double m_intakeCommand = 0.0;   // Intake command (0=off, 1=intake, -1=eject)
    
    // Testing mode flag
    private boolean m_testMode = false;
    
    /**
     * Creates a new NetworkTableControlCommand.
     * 
     * @param drivetrain The drivetrain subsystem
     * @param armElevator The arm/elevator subsystem  
     * @param intake The intake subsystem
     */
    public NetworkTableControlCommand(CommandSwerveDrivetrain drivetrain, 
                                    ArmElevator armElevator, 
                                    Intake intake) {
        m_drivetrain = drivetrain;
        m_armElevator = armElevator;
        m_intake = intake;
        
        // Initialize NetworkTables
        NetworkTableInstance inst = NetworkTableInstance.getDefault();
        m_controlTable = inst.getTable("RL_Control");
        
        // Subscribe to control topics
        m_velocityXSub = m_controlTable.getDoubleTopic("velocity_x").subscribe(0.0);
        m_velocityYSub = m_controlTable.getDoubleTopic("velocity_y").subscribe(0.0);
        m_angularVelocitySub = m_controlTable.getDoubleTopic("angular_velocity").subscribe(0.0);
        m_elevatorHeightSub = m_controlTable.getDoubleTopic("elevator_height").subscribe(0.0);
        m_armAngleSub = m_controlTable.getDoubleTopic("arm_angle").subscribe(0.0);
        m_intakeCommandSub = m_controlTable.getDoubleTopic("intake_command").subscribe(0.0);
        
        // Require all subsystems
        addRequirements(drivetrain, armElevator, intake);
    }
    
    @Override
    public void initialize() {
        // Initialize with safe defaults
        m_velocityX = 0.0;
        m_velocityY = 0.0;
        m_angularVelocity = 0.0;
        m_elevatorHeight = 0.0;
        m_armAngle = 0.0;
        m_intakeCommand = 0.0;
        
        // Initialize SmartDashboard entries
        initializeSmartDashboard();
    }
    
    @Override
    public void execute() {
        // Always ensure SmartDashboard entries exist (in case they get cleared)
        if (!SmartDashboard.containsKey("NT_Control/Test_Mode")) {
            initializeSmartDashboard();
        }
        
        // Check if we're in test mode (using SmartDashboard values)
        // Also check if NetworkTables "enabled" flag is set by Python agent
        boolean networkTablesEnabled = m_controlTable.getEntry("enabled").getBoolean(false);
        boolean testModeRequested = SmartDashboard.getBoolean("NT_Control/Test_Mode", true);
        
        // If Python agent is connected and enabled, override test mode
        m_testMode = testModeRequested && !networkTablesEnabled;
        
        if (m_testMode) {
            // Read values from SmartDashboard for testing
            m_velocityX = SmartDashboard.getNumber("NT_Control/Test_VelX", 0.0);
            m_velocityY = SmartDashboard.getNumber("NT_Control/Test_VelY", 0.0);
            m_angularVelocity = SmartDashboard.getNumber("NT_Control/Test_AngVel", 0.0);
            m_elevatorHeight = SmartDashboard.getNumber("NT_Control/Test_ElevHeight", 0.0);
            m_armAngle = SmartDashboard.getNumber("NT_Control/Test_ArmAngle", 0.0);
            m_intakeCommand = SmartDashboard.getNumber("NT_Control/Test_Intake", 0.0);
        } else {
            // Read latest values from NetworkTables
            m_velocityX = m_velocityXSub.get();
            m_velocityY = m_velocityYSub.get();
            m_angularVelocity = m_angularVelocitySub.get();
            m_elevatorHeight = m_elevatorHeightSub.get();
            m_armAngle = m_armAngleSub.get();
            m_intakeCommand = m_intakeCommandSub.get();
        }
        
        // Apply safety limits
        m_velocityX = clamp(m_velocityX, -5.0, 5.0);        // Max 5 m/s
        m_velocityY = clamp(m_velocityY, -5.0, 5.0);        // Max 5 m/s  
        m_angularVelocity = clamp(m_angularVelocity, -6.0, 6.0); // Max 6 rad/s
        m_elevatorHeight = clamp(m_elevatorHeight, 0.0, 2.5);    // 0-2.5m range
        m_armAngle = clamp(m_armAngle, -90.0, 120.0);            // -90 to 120 degrees
        
        // Update SmartDashboard with current values
        updateSmartDashboard();
        
        // Control drivetrain
        m_drivetrain.setControl(m_driveRequest
            .withVelocityX(m_velocityX)
            .withVelocityY(m_velocityY)
            .withRotationalRate(m_angularVelocity));
        
        // Control arm and elevator
        m_armElevator.setElevatorGoal(m_elevatorHeight);
        m_armElevator.setArmGoal(m_armAngle);
        
        // Control intake
        if (m_intakeCommand > 0.5) {
            // Run intake
            m_intake.setRunningIntake(true);
        } else if (m_intakeCommand < -0.5) {
            // Eject game piece
            if (m_intake.hasGamePiece()) {
                m_intake.ejectGamePiece();
            }
        } else {
            // Stop intake
            m_intake.setRunningIntake(false);
        }
        
        // Publish current state back to NetworkTables for feedback
        publishState();
    }
    
    @Override
    public void end(boolean interrupted) {
        // Stop all motion when command ends
        m_drivetrain.setControl(new SwerveRequest.SwerveDriveBrake());
        m_intake.setRunningIntake(false);
    }
    
    @Override
    public boolean isFinished() {
        // This command runs indefinitely until interrupted
        return false;
    }
    
    /**
     * Clamps a value between min and max
     */
    private double clamp(double value, double min, double max) {
        return Math.max(min, Math.min(max, value));
    }
    
    /**
     * Publishes current robot state back to NetworkTables for the RL agent
     */
    private void publishState() {
        NetworkTable stateTable = NetworkTableInstance.getDefault().getTable("RL_State");
        
        // Robot pose
        var pose = m_drivetrain.getMapleSimSwerveDrivetrain().getMapleSimDrive().getSimulatedDriveTrainPose();
        stateTable.getEntry("pose_x").setDouble(pose.getX());
        stateTable.getEntry("pose_y").setDouble(pose.getY());
        stateTable.getEntry("pose_rotation").setDouble(pose.getRotation().getDegrees());
        
        // Robot velocities
        var speeds = m_drivetrain.getState().Speeds;
        stateTable.getEntry("velocity_x").setDouble(speeds.vxMetersPerSecond);
        stateTable.getEntry("velocity_y").setDouble(speeds.vyMetersPerSecond);
        stateTable.getEntry("angular_velocity").setDouble(speeds.omegaRadiansPerSecond);
        
        // Arm/Elevator state
        stateTable.getEntry("elevator_height").setDouble(m_armElevator.getElevatorHeight());
        stateTable.getEntry("arm_angle").setDouble(m_armElevator.getArmAngle());
        stateTable.getEntry("at_setpoint").setBoolean(m_armElevator.atSetpoint());
        // Publish whether robot currently has a game piece
        stateTable.getEntry("has_game_piece").setBoolean(m_intake.hasGamePiece());
        

        
        // Add timestamp for synchronization
        stateTable.getEntry("timestamp").setDouble(edu.wpi.first.wpilibj.Timer.getFPGATimestamp());
        
        // Note: Enhanced game state information is published directly by Robot.java
        // via Logger.recordOutput() to the same RL_State table, so no need to duplicate here
    }
    
    /**
     * Initializes SmartDashboard entries for testing the NetworkTable control
     */
    private void initializeSmartDashboard() {
        // Test mode toggle - start with test mode enabled for easy testing
        SmartDashboard.putBoolean("NT_Control/Test_Mode", true);
        
        // Test input values
        SmartDashboard.putNumber("NT_Control/Test_VelX", 0.0);
        SmartDashboard.putNumber("NT_Control/Test_VelY", 0.0);
        SmartDashboard.putNumber("NT_Control/Test_AngVel", 0.0);
        SmartDashboard.putNumber("NT_Control/Test_ElevHeight", 0.0);
        SmartDashboard.putNumber("NT_Control/Test_ArmAngle", 0.0);
        SmartDashboard.putNumber("NT_Control/Test_Intake", 0.0);
        
        // Current values display
        SmartDashboard.putNumber("NT_Control/Current_VelX", 0.0);
        SmartDashboard.putNumber("NT_Control/Current_VelY", 0.0);
        SmartDashboard.putNumber("NT_Control/Current_AngVel", 0.0);
        SmartDashboard.putNumber("NT_Control/Current_ElevHeight", 0.0);
        SmartDashboard.putNumber("NT_Control/Current_ArmAngle", 0.0);
        SmartDashboard.putNumber("NT_Control/Current_Intake", 0.0);
        
        SmartDashboard.putString("NT_Control/Status", "Initialized - Ready for Testing");
        
        // Force update SmartDashboard to make sure entries appear
        SmartDashboard.updateValues();
    }
    
    /**
     * Updates SmartDashboard with current control values
     */
    private void updateSmartDashboard() {
        // Update current values
        SmartDashboard.putNumber("NT_Control/Current_VelX", m_velocityX);
        SmartDashboard.putNumber("NT_Control/Current_VelY", m_velocityY);
        SmartDashboard.putNumber("NT_Control/Current_AngVel", m_angularVelocity);
        SmartDashboard.putNumber("NT_Control/Current_ElevHeight", m_elevatorHeight);
        SmartDashboard.putNumber("NT_Control/Current_ArmAngle", m_armAngle);
        SmartDashboard.putNumber("NT_Control/Current_Intake", m_intakeCommand);
        
        // Show NetworkTables enabled status
        boolean networkTablesEnabled = m_controlTable.getEntry("enabled").getBoolean(false);
        SmartDashboard.putBoolean("NT_Control/NetworkTables_Enabled", networkTablesEnabled);
        
        // Update status
        String status;
        if (networkTablesEnabled && !m_testMode) {
            status = "Python Agent Control Active";
        } else if (m_testMode) {
            status = "SmartDashboard Test Mode Active";
        } else {
            status = "Waiting for Control Input";
        }
        SmartDashboard.putString("NT_Control/Status", status);
        
        // Add some top-level entries that should always be visible
        SmartDashboard.putBoolean("RL_Control_Active", true);
        SmartDashboard.putString("RL_Mode", m_testMode ? "SmartDashboard Test" : "NetworkTables");
        SmartDashboard.putString("Control_Source", networkTablesEnabled ? "Python Agent" : "SmartDashboard");
    }
}
