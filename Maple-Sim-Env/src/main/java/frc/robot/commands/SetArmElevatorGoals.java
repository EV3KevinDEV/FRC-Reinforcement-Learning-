// Copyright (c) FIRST and other WPILib contributors.
// Open Source Software; you can modify and/or share it under the terms of
// the WPILib BSD license file in the root directory of this project.

package frc.robot.commands;

import edu.wpi.first.wpilibj2.command.Command;
import frc.robot.subsystems.ArmElevator;

/** A command that moves the arm and elevator to a specific setpoint and waits until they arrive. */
public class SetArmElevatorGoals extends Command {
    @SuppressWarnings({"PMD.UnusedPrivateField", "PMD.SingularField"})
    private final ArmElevator m_armElevator;

    private final double m_elevatorHeight;
    private final double m_armAngle;

    /**
     * Creates a new SetArmElevatorGoals command.
     *
     * @param subsystem The ArmElevator subsystem used by this command.
     * @param elevatorHeight The desired height for the elevator in meters.
     * @param armAngle The desired angle for the arm in degrees.
     */
    public SetArmElevatorGoals(ArmElevator subsystem, double elevatorHeight, double armAngle) {
        m_armElevator = subsystem;
        m_elevatorHeight = elevatorHeight;
        m_armAngle = armAngle;
        // Use addRequirements() here to declare subsystem dependencies.
        addRequirements(subsystem);
    }

    // Called when the command is initially scheduled.
    @Override
    public void initialize() {}

    // Called every time the scheduler runs while the command is scheduled.
    @Override
    public void execute() {
        m_armElevator.setElevatorGoal(m_elevatorHeight);
        m_armElevator.setArmGoal(m_armAngle);
        // The subsystem's periodic() method handles the movement, so nothing is needed here.
    }

    // Called once the command ends or is interrupted.
    @Override
    public void end(boolean interrupted) {}

    // Returns true when the command should end.
    @Override
    public boolean isFinished() {
        return m_armElevator.atSetpoint();
    }
}
