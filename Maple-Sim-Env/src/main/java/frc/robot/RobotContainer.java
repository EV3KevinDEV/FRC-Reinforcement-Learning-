// Copyright (c) FIRST and other WPILib contributors.
// Open Source Software; you can modify and/or share it under the terms of
// the WPILib BSD license file in the root directory of this project.

package frc.robot;

import static edu.wpi.first.units.Units.*;

import org.ironmaple.simulation.SimulatedArena;
import org.ironmaple.simulation.seasonspecific.reefscape2025.ReefscapeCoralAlgaeStack;

import com.ctre.phoenix6.swerve.SwerveModule.DriveRequestType;
import com.ctre.phoenix6.swerve.SwerveRequest;
import edu.wpi.first.math.geometry.Pose2d;
import edu.wpi.first.math.geometry.Rotation2d;
import edu.wpi.first.math.geometry.Translation2d;
import edu.wpi.first.wpilibj.smartdashboard.SmartDashboard;
import edu.wpi.first.wpilibj2.command.Command;
import edu.wpi.first.wpilibj2.command.Commands;
import edu.wpi.first.wpilibj2.command.button.CommandXboxController;
import frc.robot.commands.SetArmElevatorGoals;
import frc.robot.commands.NetworkTableControlCommand;
import frc.robot.subsystems.ArmElevator;
import frc.robot.subsystems.Intake;
import frc.robot.subsystems.SwerveSubsystem.CommandSwerveDrivetrain;
import frc.robot.subsystems.SwerveSubsystem.TunerConstants;

public class RobotContainer {
    private double MaxSpeed = TunerConstants.kSpeedAt12Volts.in(MetersPerSecond); // kSpeedAt12Volts desired top speed
    private double MaxAngularRate =
            RotationsPerSecond.of(0.75).in(RadiansPerSecond); // 3/4 of a rotation per second max angular velocity

    /* Setting up bindings for necessary control of the swerve drive platform */
    private final SwerveRequest.FieldCentric drive = new SwerveRequest.FieldCentric()
            .withDeadband(MaxSpeed * 0.1)
            .withRotationalDeadband(MaxAngularRate * 0.1) // Add a 10% deadband
            .withDriveRequestType(DriveRequestType.OpenLoopVoltage); // Use open-loop control for drive motors
    private final SwerveRequest.SwerveDriveBrake brake = new SwerveRequest.SwerveDriveBrake();
    private final SwerveRequest.PointWheelsAt point = new SwerveRequest.PointWheelsAt();
    private final SwerveRequest.RobotCentric forwardStraight =
            new SwerveRequest.RobotCentric().withDriveRequestType(DriveRequestType.OpenLoopVoltage);

    private final Telemetry logger = new Telemetry(MaxSpeed);

    private final CommandXboxController joystick = new CommandXboxController(0);
    

    public final static CommandSwerveDrivetrain drivetrain = TunerConstants.createDrivetrain();
    public final static Intake m_intake = new Intake(drivetrain.getMapleSimSwerveDrivetrain().getMapleSimDrive());
    public static final ArmElevator m_armElevator = new ArmElevator();

    /* Path follower */

    public RobotContainer() {
        SimulatedArena.getInstance().addGamePiece(new ReefscapeCoralAlgaeStack(new Translation2d(2,2)));

        configureBindings();
        SmartDashboard.putNumber("Elevator Height Goal", 0);
        SmartDashboard.putNumber("Arm Angle Goal", 0);
        drivetrain.resetPose(new Pose2d(3, 3, new Rotation2d()));
    }

    private void configureBindings() {
        // Note that X is defined as forward according to WPILib convention,
        // and Y is defined as to the left according to WPILib convention.
        
        // Set NetworkTable Control as the default command for drivetrain
        // This command runs continuously and can be controlled via NetworkTables or SmartDashboard
        drivetrain.setDefaultCommand(new NetworkTableControlCommand(drivetrain, m_armElevator, m_intake));

        drivetrain.registerTelemetry(logger::telemeterize);

        // Manual override - Press and hold A button to temporarily return to manual control
        joystick.a().whileTrue(
            drivetrain.applyRequest(
                () -> drive.withVelocityX(
                                -joystick.getLeftY() * MaxSpeed) // Drive forward with negative Y (forward)
                        .withVelocityY(-joystick.getLeftX() * MaxSpeed) // Drive left with negative X (left)
                        .withRotationalRate(-joystick.getRightX()
                                * MaxAngularRate) // Drive counterclockwise with negative X (left)
            ));

        // ===== COMMENTED OUT MANUAL CONTROLS =====
        // The robot is now controlled by NetworkTables by default
        // Use SmartDashboard "NT_Control/Test_Mode" = true to test with manual values
        
        /*
        // Intake control
        joystick.a()
                .whileTrue(Commands.run(() -> {
                            // When the 'A' button is pressed, run the intake
                            m_intake.setRunningIntake(true);
                        })).whileFalse(Commands.runOnce(() -> {
                                // When the 'A' button is released, stop the intake
                                m_intake.setRunningIntake(false);
                        }));

        // Home position
        joystick.a().onTrue(new SetArmElevatorGoals(m_armElevator, 0, 0));

        // Arm/Elevator positioning buttons (without ejection)
        // L1 Position - Low level
        joystick.x().onTrue(new SetArmElevatorGoals(m_armElevator, 0.5, 15));
        
        // L2 Position - Mid level  
        joystick.y().onTrue(new SetArmElevatorGoals(m_armElevator, 1.2, 45));
        
        // L3 Position - High level
        joystick.b().onTrue(new SetArmElevatorGoals(m_armElevator, 1.8, 90));
        
        // L4 Position - Maximum height
        joystick.rightBumper().onTrue(new SetArmElevatorGoals(m_armElevator, 2.2, 110));

        // Eject game piece button
        joystick.b()
                .onTrue(Commands.runOnce(() -> {
                    if (m_intake.hasGamePiece()) {
                        m_intake.ejectGamePiece();
                    }
                }));
        */
    }

    public Command getAutonomousCommand() {
        /* Run the path selected from the auto chooser */
        return new SetArmElevatorGoals(m_armElevator, 0.5, 45.0);
    }
}
