// Copyright (c) FIRST and other WPILib contributors.
// Open Source Software; you can modify and/or share it under the terms of
// the WPILib BSD license file in the root directory of this project.

package frc.robot.subsystems;

import org.littletonrobotics.junction.Logger;

import edu.wpi.first.math.controller.ArmFeedforward;
import edu.wpi.first.math.controller.ElevatorFeedforward;
import edu.wpi.first.math.controller.PIDController;
import edu.wpi.first.math.geometry.Pose3d;
import edu.wpi.first.math.trajectory.TrapezoidProfile;
import edu.wpi.first.math.util.Units;
import edu.wpi.first.wpilibj.Encoder;
import edu.wpi.first.wpilibj.RobotController;
import edu.wpi.first.wpilibj.motorcontrol.PWMSparkMax;
import edu.wpi.first.wpilibj.simulation.BatterySim;
import edu.wpi.first.wpilibj.simulation.ElevatorSim;
import edu.wpi.first.wpilibj.simulation.EncoderSim;
import edu.wpi.first.wpilibj.simulation.RoboRioSim;
import edu.wpi.first.wpilibj.simulation.SingleJointedArmSim;
import edu.wpi.first.wpilibj.smartdashboard.Mechanism2d;
import edu.wpi.first.wpilibj.smartdashboard.MechanismLigament2d;
import edu.wpi.first.wpilibj.smartdashboard.MechanismRoot2d;
import edu.wpi.first.wpilibj.smartdashboard.SmartDashboard;
import edu.wpi.first.wpilibj.util.Color;
import edu.wpi.first.wpilibj.util.Color8Bit;
import edu.wpi.first.wpilibj2.command.SubsystemBase;
import frc.robot.Constants;

public class ArmElevator extends SubsystemBase {
        // --- Real Components ---
        private final PWMSparkMax m_elevatorMotor;
        private final Encoder m_elevatorEncoder;
        private final PIDController m_elevatorPid;
        private final ElevatorFeedforward m_elevatorFeedforward;
        private final TrapezoidProfile m_elevatorTrapezoidProfile;
        private TrapezoidProfile.State m_elevatorGoal;
        private TrapezoidProfile.State m_elevatorSetpoint;

        private final PWMSparkMax m_armMotor;
        private final Encoder m_armEncoder;
        private final PIDController m_armPid;
        private final ArmFeedforward m_armFeedforward;
        private final double offset = Math.PI; // Offset so zero points downward (π/2 radians = 90 degrees)
        private final double armStartingHeight = 0.97155; // Starting height of the arm base in meters

        private double m_armGoal;

        // --- Visualization ---
        private final Mechanism2d m_mech2d;
        private final MechanismRoot2d m_root;
        private final MechanismLigament2d m_elevatorLigament;
        private final MechanismLigament2d m_armLigament;

        // --- Simulation Components ---
        private final EncoderSim m_elevatorEncoderSim;
        private final ElevatorSim m_elevatorSim;

        private final EncoderSim m_armEncoderSim;
        private final SingleJointedArmSim m_armSim;

        /** Creates a new ArmElevator subsystem. */
        public ArmElevator() {
                // --- Real Component Initialization ---
                m_elevatorMotor = new PWMSparkMax(Constants.Elevator.kMotorPort);
                m_elevatorEncoder = new Encoder(
                                Constants.Elevator.kEncoderPorts[0],
                                Constants.Elevator.kEncoderPorts[1],
                                Constants.Elevator.kEncoderReversed);
                m_elevatorEncoder.setDistancePerPulse(Constants.Elevator.kDistancePerPulse);
                m_elevatorPid = new PIDController(Constants.Elevator.kP, Constants.Elevator.kI, Constants.Elevator.kD);
                m_elevatorFeedforward = new ElevatorFeedforward(Constants.Elevator.kS, Constants.Elevator.kG,
                                Constants.Elevator.kV);
                m_elevatorTrapezoidProfile = new TrapezoidProfile(
                                new TrapezoidProfile.Constraints(Constants.Elevator.kMaxVelocity,
                                                Constants.Elevator.kMaxAcceleration));
                m_elevatorGoal = new TrapezoidProfile.State(0, 0);
                m_elevatorSetpoint = new TrapezoidProfile.State(0, 0);

                m_armMotor = new PWMSparkMax(Constants.Arm.kMotorPort);
                m_armEncoder = new Encoder(
                                Constants.Arm.kEncoderPorts[0], Constants.Arm.kEncoderPorts[1],
                                Constants.Arm.kEncoderReversed);
                m_armEncoder.setDistancePerPulse(Constants.Arm.kDistancePerPulse);
                m_armPid = new PIDController(Constants.Arm.kP, Constants.Arm.kI, Constants.Arm.kD);
                m_armFeedforward = new ArmFeedforward(Constants.Arm.kS, Constants.Arm.kG, Constants.Arm.kV);
                m_armGoal = 0;

                // --- Visualization Initialization ---
                m_mech2d = new Mechanism2d(3, 3);
                m_root = m_mech2d.getRoot("Root", 1.5, 0.2);
                m_elevatorLigament = m_root.append(
                                new MechanismLigament2d("Elevator", getElevatorHeight(), 90, 6,
                                                new Color8Bit(Color.kBlue)));
                m_armLigament = m_elevatorLigament.append(new MechanismLigament2d(
                                "Arm", Constants.Arm.kArmLength, getArmAngle(), 6, new Color8Bit(Color.kYellow)));
                SmartDashboard.putData("Mech2d", m_mech2d);

                // --- Simulation Initialization ---
                m_elevatorEncoderSim = new EncoderSim(m_elevatorEncoder);
                m_elevatorSim = new ElevatorSim(
                                Constants.Elevator.kElevatorGearbox,
                                Constants.Elevator.kElevatorGearing,
                                Constants.Elevator.kCarriageMass,
                                Constants.Elevator.kElevatorDrumRadius,
                                Constants.Elevator.kMinElevatorHeightMeters,
                                Constants.Elevator.kMaxElevatorHeightMeters,
                                true, // Simulate gravity
                                0.0); // Start at height 0

                m_armEncoderSim = new EncoderSim(m_armEncoder);
                m_armSim = new SingleJointedArmSim(
                                Constants.Arm.kArmGearbox,
                                Constants.Arm.kArmGearing,
                                Constants.Arm.kArmMomentOfInertia,
                                Constants.Arm.kArmLength,
                                Constants.Arm.kMinAngleRads,
                                Constants.Arm.kMaxAngleRads,
                                true, // Simulate gravity
                                0.0); // Start at angle 0
        }

        public Pose3d getArmPose3dVisualizer() {
                double elevatorHeight = getElevatorHeight();
                double armAngleRadians = Units.degreesToRadians(getArmAngle());
                double armLength = Constants.Arm.kArmLength;

                double z = elevatorHeight;

                return new Pose3d(0, 0.0, z, new edu.wpi.first.math.geometry.Rotation3d(0, -armAngleRadians, 0));
        }

        public Pose3d getElevaPose3dVisualizer() {
                double elevatorHeight = getElevatorHeight();

                double x = 0;
                double z = elevatorHeight + armStartingHeight;
                double y = 0;

                return new Pose3d(x, y, z, new edu.wpi.first.math.geometry.Rotation3d(0, 0, 0));
        }

        public Pose3d getArmPose3dFromMech2d() {
                // Get the arm tip position directly from the Mechanism2d ligament
                // MechanismLigament2d does not provide getX(), so we calculate armTipX manually
                double armAngleRadians = Units.degreesToRadians(m_armLigament.getAngle());
                double armTipX = Constants.Arm.kArmLength * Math.cos(armAngleRadians);
                double armTipY = Constants.Arm.kArmLength * Math.sin(armAngleRadians);
                double armTipZ = getElevatorHeight();
                
                
                // The pose represents the arm's end effector position and orientation
                return new Pose3d(
                    armTipX, 
                    armTipY, 
                    armTipZ, 
                    new edu.wpi.first.math.geometry.Rotation3d(0, armAngleRadians, 0)
                );
        }

        @Override
        public void periodic() {
                // --- Control Logic ---
                m_elevatorSetpoint = m_elevatorTrapezoidProfile.calculate(Constants.kDt, m_elevatorSetpoint,
                                m_elevatorGoal);
                double elevatorPidOutput = m_elevatorPid.calculate(m_elevatorEncoder.getDistance(),
                                m_elevatorSetpoint.position);
                double elevatorFeedforwardOutput = m_elevatorFeedforward.calculate(m_elevatorSetpoint.velocity);
                m_elevatorMotor.setVoltage(elevatorPidOutput + elevatorFeedforwardOutput);

                double armPidOutput = m_armPid.calculate(m_armEncoder.getDistance(), m_armGoal);
                double armFeedforwardOutput = m_armFeedforward.calculate(Units.degreesToRadians(m_armGoal), 0);
                m_armMotor.setVoltage(armPidOutput + armFeedforwardOutput);

                // --- Visualization Update ---
                m_elevatorLigament.setLength(getElevatorHeight());
                m_armLigament.setAngle(getArmAngle() + offset * (180/Math.PI) ); // Add 90 degrees to make zero point downward

                //Logging 3d Pose
                Logger.recordOutput("ArmPose3d", getArmPose3dVisualizer());
                Logger.recordOutput("ElevatorPose3d", getElevaPose3dVisualizer());
                Logger.recordOutput("ArmTipPose3d", getArmPose3dFromMech2d());
        }

        @Override
        public void simulationPeriodic() {
                // --- Simulation Logic ---
                // 1. Set simulator inputs from motor outputs
                m_elevatorSim.setInputVoltage(m_elevatorMotor.get() * RobotController.getBatteryVoltage());
                m_armSim.setInputVoltage(m_armMotor.get() * RobotController.getBatteryVoltage());

                // 2. Update the simulation state
                m_elevatorSim.update(Constants.kDt);
                m_armSim.update(Constants.kDt);

                // 3. Update sensor outputs with simulated data
                m_elevatorEncoderSim.setDistance(m_elevatorSim.getPositionMeters());
                m_elevatorEncoderSim.setRate(m_elevatorSim.getVelocityMetersPerSecond());

                m_armEncoderSim.setDistance(Units.radiansToDegrees(m_armSim.getAngleRads()));
                m_armEncoderSim.setRate(Units.radiansToDegrees(m_armSim.getVelocityRadPerSec()));

                // 4. Update battery simulation
                double totalCurrent = m_elevatorSim.getCurrentDrawAmps() + m_armSim.getCurrentDrawAmps();
                RoboRioSim.setVInVoltage(BatterySim.calculateDefaultBatteryLoadedVoltage(totalCurrent));
        }

        // --- Methods ---
        public void setElevatorGoal(double height) {
                m_elevatorGoal = new TrapezoidProfile.State(height, 0);
        }

        public double getElevatorHeight() {
                return m_elevatorEncoder.getDistance();
        }

        public void setArmGoal(double angle) {
                m_armGoal = angle;
        }

        public double getArmAngle() {
                return m_armEncoder.getDistance();
        }

        public boolean atSetpoint() {
                boolean atElevatorGoal = Math
                                .abs(getElevatorHeight() - m_elevatorGoal.position) < Constants.Elevator.kTolerance;
                boolean atArmGoal = Math.abs(getArmAngle() - m_armGoal) < Constants.Arm.kTolerance;
                return atElevatorGoal && atArmGoal;
        }
}
