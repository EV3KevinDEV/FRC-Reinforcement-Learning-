// Copyright (c) FIRST and other WPILib contributors.
// Open Source Software; you can modify and/or share it under the terms of
// the WPILib BSD license file in the root directory of this project.

package frc.robot.subsystems;

import edu.wpi.first.math.geometry.Pose3d;
import edu.wpi.first.math.geometry.Rotation2d;
import edu.wpi.first.math.geometry.Pose2d;
import edu.wpi.first.math.geometry.Translation2d;
import edu.wpi.first.wpilibj2.command.Command;
import edu.wpi.first.wpilibj2.command.SubsystemBase;
import frc.robot.RobotContainer;
import frc.robot.Constants.Arm;

import static edu.wpi.first.units.Units.Degrees;
import static edu.wpi.first.units.Units.Meters;
import static edu.wpi.first.units.Units.MetersPerSecond;

import org.dyn4j.geometry.Triangle;
import org.dyn4j.geometry.Vector2;
import org.ironmaple.simulation.IntakeSimulation;
import org.ironmaple.simulation.SimulatedArena;
import org.ironmaple.simulation.drivesims.AbstractDriveTrainSimulation;
import org.ironmaple.simulation.gamepieces.*;
import org.ironmaple.simulation.seasonspecific.reefscape2025.ReefscapeCoralOnFly;
import org.littletonrobotics.junction.AutoLogOutput;
import org.littletonrobotics.junction.Logger;

public class Intake extends SubsystemBase {
    IntakeSimulation intakeSimulation;
    AbstractDriveTrainSimulation driveSimulation;

    /** Creates a new ExampleSubsystem. */
    public Intake(AbstractDriveTrainSimulation driveTrainSim) {
        driveSimulation = driveTrainSim; // Store the reference to the drive simulation
        this.intakeSimulation = IntakeSimulation.InTheFrameIntake(
                // Specify the type of game pieces that the intake can collect
                "Coral",
                // Specify the drivetrain to which this intake is attached
                driveTrainSim,
                // The extension length of the intake beyond the robot's frame (when activated)
                Meters.of(1.0),
                // The intake is mounted on the back side of the chassis
                IntakeSimulation.IntakeSide.FRONT,
                // The intake can hold up to 1 note
                1);
    }

    /**
     * Example command factory method.
     *
     * @return a command
     */
    public Command exampleMethodCommand() {
        // Inline construction of command goes here.
        // Subsystem::RunOnce implicitly requires `this` subsystem.
        return runOnce(() -> {
            /* one-time action goes here */
        });
    }

    /**
     * An example method querying a boolean state of the subsystem (for example, a
     * digital sensor).
     *
     * @return value of some boolean subsystem state, such as a digital sensor.
     */
    public boolean exampleCondition() {
        // Query some boolean state, such as a digital sensor.
        return false;
    }

    public void setRunningIntake(boolean running) {
        Logger.recordOutput("IntakeRunning", running); // Log the intake state to the dashboard
        if (running) {
            intakeSimulation
                    .startIntake(); // Extends the intake out from the chassis frame and starts detecting contacts
                                    // with
            // game pieces
        } else {
            intakeSimulation
                    .stopIntake(); // Retracts the intake into the chassis frame, disabling game piece collection
        }
    }
    /**
     * Ejects the game piece from the intake mechanism.
     * The game piece is ejected at a specific angle and speed.
     *
     * @param ArmTip The position of the arm tip in the robot's coordinate system.
     */
    public void ejectGamePiece() {
        if (!hasGamePiece()) {
            Logger.recordOutput("Ejecting Game Piece", false); // Log the failure to eject
            return; // Do not proceed if there is no game piece to eject
        }
        Pose3d ArmTip = RobotContainer.m_armElevator.getArmPose3dFromMech2d(); // Get the arm tip position in the robot's coordinate system
        Logger.recordOutput("Ejecting Game Piece", true); // Log the ejection action
        
        // Transform arm tip position from robot coordinates to field coordinates        
        SimulatedArena.getInstance()
        .addGamePieceProjectile(new ReefscapeCoralOnFly(
            // Obtain robot position from drive simulation
            driveSimulation.getSimulatedDriveTrainPose().getTranslation(),
            // The scoring mechanism is installed at (0.46, 0) (meters) on the robot
            new Translation2d(-ArmTip.getY(), 0),
            // Obtain robot speed from drive simulation
            driveSimulation.getDriveTrainSimulatedChassisSpeedsFieldRelative(),
            // Obtain robot facing from drixve simulation
            driveSimulation.getSimulatedDriveTrainPose().getRotation(),
            // The height at which the coral is ejected
            Meters.of(ArmTip.getZ()),
            // The initial speed of the coral
            MetersPerSecond.of(2),
            // The coral is ejected at a 35-degree slope
            ArmTip.getRotation().getMeasureY()));

            intakeSimulation.obtainGamePieceFromIntake();
    }

    public boolean hasGamePiece() {
        return intakeSimulation.getGamePiecesAmount() != 0; // Returns true if the intake is currently in contact with a
                                                            // game piece
    }

    @Override
    public void periodic() {
        Logger.recordOutput("hasGamePiece", hasGamePiece());
        // is Intake running?
        Logger.recordOutput("Can Intake", intakeSimulation.isSensor());
    }

    @Override
    public void simulationPeriodic() {
        // This method will be called once per scheduler run during simulation
    }
}
