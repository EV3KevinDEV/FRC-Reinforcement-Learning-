// Copyright (c) FIRST and other WPILib contributors.
// Open Source Software; you can modify and/or share it under the terms of
// the WPILib BSD license file in the root directory of this project.

package frc.robot;

import edu.wpi.first.wpilibj.DriverStation.Alliance;
import edu.wpi.first.wpilibj.TimedRobot;
import edu.wpi.first.wpilibj2.command.Command;
import edu.wpi.first.wpilibj2.command.CommandScheduler;
import frc.robot.subsystems.ArmElevator;
import frc.robot.subsystems.Intake;
import edu.wpi.first.math.geometry.Pose2d;
import edu.wpi.first.math.geometry.Rotation2d;
import edu.wpi.first.networktables.NetworkTable;
import edu.wpi.first.networktables.NetworkTableInstance;

import java.util.Optional;

import org.ironmaple.simulation.SimulatedArena;
import org.ironmaple.simulation.seasonspecific.reefscape2025.ReefscapeCoralOnField;
import org.ironmaple.simulation.seasonspecific.reefscape2025.ReefscapeReefSimulation;
import org.littletonrobotics.junction.LogFileUtil;
import org.littletonrobotics.junction.LoggedRobot;
import org.littletonrobotics.junction.Logger;
import org.littletonrobotics.junction.networktables.NT4Publisher;
import org.littletonrobotics.junction.wpilog.WPILOGReader;
import org.littletonrobotics.junction.wpilog.WPILOGWriter;

/**
 * The methods in this class are called automatically corresponding to each
 * mode, as described in the TimedRobot
 * documentation. If you change the name of this class or the package after
 * creating this project, you must also update
 * the Main.java file in the project.
 */
public class Robot extends LoggedRobot {
  private Command m_autonomousCommand;

  private final RobotContainer m_robotContainer;
  
  // NetworkTable for RL state publishing
  private final NetworkTable m_rlStateTable;

  /**
   * This function is run when the robot is first started up and should be used
   * for any initialization code.
   */
  public Robot() {

    Logger.addDataReceiver(new NT4Publisher());

    // Start logging before creating RobotContainer
    Logger.start();
    
    // Initialize NetworkTable for RL state
    m_rlStateTable = NetworkTableInstance.getDefault().getTable("RL_State");

    // Instantiate our RobotContainer. This will perform all our button bindings,
    // and put our
    // autonomous chooser on the dashboard.
    m_robotContainer = new RobotContainer();
    // Obtains the default instance of the simulation world, which is a Crescendo
    // Arena.
    // Overrides the default simulation
  }

  /**
   * This function is called every 20 ms, no matter the mode. Use this for items
   * like diagnostics that you want ran
   * during disabled, autonomous, teleoperated and test.
   *
   * <p>
   * This runs after the mode specific periodic functions, but before LiveWindow
   * and SmartDashboard integrated
   * updating.
   */
  @Override
  public void robotPeriodic() {
    // Runs the Scheduler. This is responsible for polling buttons, adding
    // newly-scheduled
    // commands, running already-scheduled commands, removing finished or
    // interrupted commands,
    // and running subsystem periodic() methods. This must be called from the
    // robot's periodic
    // block in order for anything in the Command-based framework to work.
    CommandScheduler.getInstance().run();
    SimulatedArena.getInstance().simulationPeriodic();

    Logger.recordOutput("FieldSimulation/Algae",
        SimulatedArena.getInstance().getGamePiecesArrayByType("Algae"));
    Logger.recordOutput("FieldSimulation/Coral",
        SimulatedArena.getInstance().getGamePiecesArrayByType("Coral"));
    
    // Publish basic robot state for RL agent
    publishBasicRobotState();
    
    // Enhanced game piece and scoring tracking
    trackGamePiecesAndScoring();
  }

  /**
   * Track game pieces, scoring, and spatial relationships for RL training
   */
  private void trackGamePiecesAndScoring() {
    // Get robot pose for distance calculations
    var robotPose = m_robotContainer.drivetrain.getMapleSimSwerveDrivetrain().getMapleSimDrive().getSimulatedDriveTrainPose();
    
    // Track Algae game pieces
    var algaePieces = SimulatedArena.getInstance().getGamePiecesArrayByType("Algae");
    Logger.recordOutput("RL_State/algae_count", algaePieces.length);
    m_rlStateTable.getEntry("algae_count").setDouble(algaePieces.length);
    
    // Find closest algae piece
    double closestAlgaeDistance = Double.MAX_VALUE;
    double closestAlgaeX = 0.0, closestAlgaeY = 0.0;
    for (var algae : algaePieces) {
      // Convert Pose3d to Translation2d for distance calculation
      var algaeTranslation = new edu.wpi.first.math.geometry.Translation2d(algae.getX(), algae.getY());
      double distance = robotPose.getTranslation().getDistance(algaeTranslation);
      if (distance < closestAlgaeDistance) {
        closestAlgaeDistance = distance;
        closestAlgaeX = algae.getX();
        closestAlgaeY = algae.getY();
      }
    }
    Logger.recordOutput("RL_State/closest_algae_distance", closestAlgaeDistance);
    Logger.recordOutput("RL_State/closest_algae_x", closestAlgaeX);
    Logger.recordOutput("RL_State/closest_algae_y", closestAlgaeY);
    
    // Also publish to NetworkTable
    m_rlStateTable.getEntry("closest_algae_distance").setDouble(closestAlgaeDistance);
    m_rlStateTable.getEntry("closest_algae_x").setDouble(closestAlgaeX);
    m_rlStateTable.getEntry("closest_algae_y").setDouble(closestAlgaeY);
    
    // Track Coral game pieces
    var coralPieces = SimulatedArena.getInstance().getGamePiecesArrayByType("Coral");
    Logger.recordOutput("RL_State/coral_count", coralPieces.length);
    m_rlStateTable.getEntry("coral_count").setDouble(coralPieces.length);
    
    // Find closest coral piece
    double closestCoralDistance = Double.MAX_VALUE;
    double closestCoralX = 0.0, closestCoralY = 0.0;
    for (var coral : coralPieces) {
      // Convert Pose3d to Translation2d for distance calculation
      var coralTranslation = new edu.wpi.first.math.geometry.Translation2d(coral.getX(), coral.getY());
      double distance = robotPose.getTranslation().getDistance(coralTranslation);
      if (distance < closestCoralDistance) {
        closestCoralDistance = distance;
        closestCoralX = coral.getX();
        closestCoralY = coral.getY();
      }
    }
    Logger.recordOutput("RL_State/closest_coral_distance", closestCoralDistance);
    Logger.recordOutput("RL_State/closest_coral_x", closestCoralX);
    Logger.recordOutput("RL_State/closest_coral_y", closestCoralY);
    
    // Also publish to NetworkTable
    m_rlStateTable.getEntry("closest_coral_distance").setDouble(closestCoralDistance);
    m_rlStateTable.getEntry("closest_coral_x").setDouble(closestCoralX);
    m_rlStateTable.getEntry("closest_coral_y").setDouble(closestCoralY);
    
    // Track reef simulation and scoring
    Optional<ReefscapeReefSimulation> reefSimulation = ReefscapeReefSimulation.getInstance();
    if (reefSimulation.isPresent()) {
      var reef = reefSimulation.get();
      
      // Track scoring for both alliances
      int redScore = calculateAllianceScore(reef, Alliance.Red);
      int blueScore = calculateAllianceScore(reef, Alliance.Blue);
      Logger.recordOutput("RL_State/red_score", redScore);
      Logger.recordOutput("RL_State/blue_score", blueScore);
      Logger.recordOutput("RL_State/score_difference", redScore - blueScore);
      
      // Also publish to NetworkTable
      m_rlStateTable.getEntry("red_score").setDouble(redScore);
      m_rlStateTable.getEntry("blue_score").setDouble(blueScore);
      m_rlStateTable.getEntry("score_difference").setDouble(redScore - blueScore);
      
      // Track specific branch scoring (example: Branch G, Level 4)
      int amountOfCoralsOnBranch_G_L4 = reef.getBranches(Alliance.Red)[6][3]; // branch G, L4
      Logger.recordOutput("RL_State/Branch_G_L4_Corals", amountOfCoralsOnBranch_G_L4);
      
      // Track total pieces on reef
      int totalPiecesOnReef = calculateTotalPiecesOnReef(reef);
      Logger.recordOutput("RL_State/Total_Pieces_On_Reef", totalPiecesOnReef);
      
      // Calculate distances to scoring locations
      trackScoringLocationDistances(robotPose, reef);
    }
    
    // Track robot's relationship to field elements
    trackFieldElementDistances(robotPose);
  }
  
  /**
   * Calculate total score for an alliance
   */
  private int calculateAllianceScore(ReefscapeReefSimulation reef, Alliance alliance) {
    int score = 0;
    var branches = reef.getBranches(alliance);
    
    // Score calculation based on FRC 2025 Reefscape rules
    for (int branchIndex = 0; branchIndex < branches.length; branchIndex++) {
      var branch = branches[branchIndex];
      for (int level = 0; level < branch.length; level++) {
        int piecesAtLevel = branch[level];
        if (piecesAtLevel > 0) {
          // Basic scoring: higher levels worth more points
          score += piecesAtLevel * (level + 1) * 2; // Simplified scoring
        }
      }
    }
    return score;
  }
  
  /**
   * Calculate total pieces placed on the reef
   */
  private int calculateTotalPiecesOnReef(ReefscapeReefSimulation reef) {
    int total = 0;
    
    // Count pieces for both alliances
    for (Alliance alliance : new Alliance[]{Alliance.Red, Alliance.Blue}) {
      var branches = reef.getBranches(alliance);
      for (var branch : branches) {
        for (int piecesAtLevel : branch) {
          total += piecesAtLevel;
        }
      }
    }
    return total;
  }
  
  /**
   * Track distances to key scoring locations
   */
  private void trackScoringLocationDistances(edu.wpi.first.math.geometry.Pose2d robotPose, ReefscapeReefSimulation reef) {
    // Define key scoring locations (these would be the actual reef positions)
    // For now, using approximate locations - adjust based on actual field layout
    var reefCenterRed = new edu.wpi.first.math.geometry.Translation2d(13.0, 4.0); // Red reef center
    var reefCenterBlue = new edu.wpi.first.math.geometry.Translation2d(3.0, 4.0); // Blue reef center
    
    double distanceToRedReef = robotPose.getTranslation().getDistance(reefCenterRed);
    double distanceToBlueReef = robotPose.getTranslation().getDistance(reefCenterBlue);
    
    Logger.recordOutput("RL_State/distance_to_red_reef", distanceToRedReef);
    Logger.recordOutput("RL_State/distance_to_blue_reef", distanceToBlueReef);
    Logger.recordOutput("RL_State/distance_to_nearest_reef", Math.min(distanceToRedReef, distanceToBlueReef));
    
    // Also publish to NetworkTable
    m_rlStateTable.getEntry("distance_to_red_reef").setDouble(distanceToRedReef);
    m_rlStateTable.getEntry("distance_to_blue_reef").setDouble(distanceToBlueReef);
    m_rlStateTable.getEntry("distance_to_nearest_reef").setDouble(Math.min(distanceToRedReef, distanceToBlueReef));
  }
  
  /**
   * Track distances to important field elements
   */
  private void trackFieldElementDistances(edu.wpi.first.math.geometry.Pose2d robotPose) {
    // Define key field locations
    var fieldCenter = new edu.wpi.first.math.geometry.Translation2d(8.0, 4.0);
    var redAllianceStation = new edu.wpi.first.math.geometry.Translation2d(16.0, 4.0);
    var blueAllianceStation = new edu.wpi.first.math.geometry.Translation2d(0.0, 4.0);
    
    Logger.recordOutput("RL_GameState/Distance_To_Field_Center", 
                       robotPose.getTranslation().getDistance(fieldCenter));
    Logger.recordOutput("RL_GameState/Distance_To_Red_Station", 
                       robotPose.getTranslation().getDistance(redAllianceStation));
    Logger.recordOutput("RL_GameState/Distance_To_Blue_Station", 
                       robotPose.getTranslation().getDistance(blueAllianceStation));
    
    // Track robot's field position quadrant
    double x = robotPose.getX();
    double y = robotPose.getY();
    String quadrant = "";
    if (x < 8.0) quadrant += "Blue_";
    else quadrant += "Red_";
    if (y < 4.0) quadrant += "Bottom";
    else quadrant += "Top";
    
    Logger.recordOutput("RL_GameState/Robot_Field_Quadrant", quadrant);

    GamePieceSpawn();
  }

  // Drops a coral every 2 seconds
  private double lastCoralDropTime = 0.0;

  public void GamePieceSpawn() {
    double currentTime = edu.wpi.first.wpilibj.Timer.getFPGATimestamp();
    if (currentTime - lastCoralDropTime >= 5 && SimulatedArena.getInstance().getGamePiecesArrayByType("Coral").length < 15) {
      // Add some random variation to coral spawn position and rotation
      double x = 1.74 + Math.random() * 0.5; // x between 1.74 and 2.24
      double y = 7 + (Math.random() - 0.5) * 1.0; // y between 6.5 and 7.5
      double angle = 80 + Math.random() * 20; // angle between 80 and 100 degrees

      SimulatedArena.getInstance().addGamePiece(new ReefscapeCoralOnField(
        new Pose2d(x, y, Rotation2d.fromDegrees(angle))
      ));
      lastCoralDropTime = currentTime;
    }
  }

  /** This function is called once each time the robot enters Disabled mode. */
  @Override
  public void disabledInit() {
  }

  @Override
  public void disabledPeriodic() {
  }

  /**
   * This autonomous runs the autonomous command selected by your
   * {@link RobotContainer} class.
   */
  @Override
  public void autonomousInit() {
    m_autonomousCommand = m_robotContainer.getAutonomousCommand();

    // schedule the autonomous command (example)
    if (m_autonomousCommand != null) {
      m_autonomousCommand.schedule();
    }
  }

  /** This function is called periodically during autonomous. */
  @Override
  public void autonomousPeriodic() {
  }

  @Override
  public void teleopInit() {
    // This makes sure that the autonomous stops running when
    // teleop starts running. If you want the autonomous to
    // continue until interrupted by another command, remove
    // this line or comment it out.
    if (m_autonomousCommand != null) {
      m_autonomousCommand.cancel();
    }
  }

  /** This function is called periodically during operator control. */
  @Override
  public void teleopPeriodic() {
  }

  @Override
  public void testInit() {
    // Cancels all running commands at the start of test mode.
    CommandScheduler.getInstance().cancelAll();
  }

  /** This function is called periodically during test mode. */
  @Override
  public void testPeriodic() {
  }

  /** This function is called once when the robot is first started up. */
  @Override
  public void simulationInit() {
  }

  /** This function is called periodically whilst in simulation. */
  @Override
  public void simulationPeriodic() {
  }

  /**
   * Publish basic robot state data for RL agent
   */
  private void publishBasicRobotState() {
    // Get current robot pose using the consistent simulation pose
    var robotPose = m_robotContainer.drivetrain.getMapleSimSwerveDrivetrain().getMapleSimDrive().getSimulatedDriveTrainPose();
    
    // Publish basic robot pose and motion
    Logger.recordOutput("RL_State/pose_x", robotPose.getX());
    Logger.recordOutput("RL_State/pose_y", robotPose.getY());
    Logger.recordOutput("RL_State/pose_rotation", robotPose.getRotation().getRadians());
    
    // Also publish to NetworkTable
    m_rlStateTable.getEntry("pose_x").setDouble(robotPose.getX());
    m_rlStateTable.getEntry("pose_y").setDouble(robotPose.getY());
    m_rlStateTable.getEntry("pose_rotation").setDouble(robotPose.getRotation().getRadians());
    
    // Get drivetrain state for velocity information
    var drivetrainState = m_robotContainer.drivetrain.getState();
    
    // Publish robot velocities - using module speeds for now

    var chassisSpeed = RobotContainer.drivetrain.getMapleSimSwerveDrivetrain().getMapleSimDrive().getDriveTrainSimulatedChassisSpeedsFieldRelative();

    // Also publish to NetworkTable
    m_rlStateTable.getEntry("velocity_x").setDouble(chassisSpeed.vxMetersPerSecond);
    m_rlStateTable.getEntry("velocity_y").setDouble(chassisSpeed.vyMetersPerSecond);
    m_rlStateTable.getEntry("angular_velocity").setDouble(chassisSpeed.omegaRadiansPerSecond);


    
    ArmElevator armElevator = RobotContainer.m_armElevator;
    Intake intake = RobotContainer.m_intake;

    // Also publish to NetworkTable
    m_rlStateTable.getEntry("elevator_height").setDouble(armElevator.getElevatorHeight());
    m_rlStateTable.getEntry("arm_angle").setDouble(armElevator.getArmAngle());
    m_rlStateTable.getEntry("at_setpoint").setBoolean(armElevator.atSetpoint());
    m_rlStateTable.getEntry("has_game_piece").setBoolean(intake.hasGamePiece());
    m_rlStateTable.getEntry("timestamp").setDouble(edu.wpi.first.wpilibj.Timer.getFPGATimestamp());
  }
}
