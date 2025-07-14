package frc.robot;

import edu.wpi.first.math.system.plant.DCMotor;
import edu.wpi.first.math.util.Units;

public final class Constants {
    public static final double kDt = 0.02; // Timestep for periodic loops

    public static final class Elevator {
        public static final int kMotorPort = 0;
        public static final int[] kEncoderPorts = {0, 1};
        public static final boolean kEncoderReversed = false;
        
        // This should be calculated based on your gear reduction and encoder resolution
        public static final double kDistancePerPulse = (1.0 / 2048.0) * (Units.inchesToMeters(2) * Math.PI) / 10.0; // Example value: (1/2048) * (2in * pi) / 10:1 gearing

        // Control Gains
        public static final double kP = 5.0;
        public static final double kI = 0.0;
        public static final double kD = 0.0;

        // Feedforward Gains
        public static final double kS = 0.1; // Volts
        public static final double kG = 0.7; // Volts
        public static final double kV = 0.2; // Volts * seconds / meter

        // Motion Profiling
        public static final double kMaxVelocity = 2.0; // meters / second
        public static final double kMaxAcceleration = 1.5; // meters / second^2
        public static final double kTolerance = 0.05; // meters

        // --- Simulation Constants ---
        public static final DCMotor kElevatorGearbox = DCMotor.getNEO(2);
        public static final double kElevatorGearing = 10.0; // 10:1 gearing
        public static final double kCarriageMass = 15.0; // kilograms
        public static final double kElevatorDrumRadius = Units.inchesToMeters(2); // 2 inch radius drum
        public static final double kMinElevatorHeightMeters = 0.0;
        public static final double kMaxElevatorHeightMeters = 1.5;
    }

    public static final class Arm {
        public static final int kMotorPort = 1;
        public static final int[] kEncoderPorts = {2, 3};
        public static final boolean kEncoderReversed = false;
        
        // This should be calculated based on your gear reduction and encoder resolution
        public static final double kDistancePerPulse = 360.0 / (2048.0 * 100.0); // Example: (360 deg) / (2048 pulses * 100:1 gearing)

        public static final double kArmLength = 0.75; // meters

        // Control Gains
        public static final double kP = 0.5;
        public static final double kI = 0.0;
        public static final double kD = 0.0;

        // Feedforward Gains
        public static final double kS = 0.1; // Volts
        public static final double kG = 0.5; // Volts
        public static final double kV = 0.2; // Volts * seconds / radian
        public static final double kTolerance = 2.0; // degrees

        // --- Simulation Constants ---
        public static final DCMotor kArmGearbox = DCMotor.getNEO(1);
        public static final double kArmGearing = 100.0; // 100:1 gearing
        public static final double kArmMass = 8.0; // kilograms
        // Moment of inertia for a single rod rotating around its end: 1/3 * m * L^2
        public static final double kArmMomentOfInertia = (1.0/3.0) * kArmMass * kArmLength * kArmLength;
        public static final double kMinAngleRads = Units.degreesToRadians(0);
        public static final double kMaxAngleRads = Units.degreesToRadians(120);
    }
}
