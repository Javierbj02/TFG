from controller import Supervisor
from robot_controller import RobotController
import numpy as np
import pandas as pd

main_root = "C:/Users/javi2/Desktop/TFG - Webots/TFG/"
scenario = "ULA"
num_antennas = "8"
robot_controller = RobotController(Supervisor(), scenario, num_antennas)

# robot_controller.df = robot_controller.load_dataset(main_root, scenario, num_antennas)

print(robot_controller.get_real_position())
print("------------------------------------")

#robot_controller.df = robot_controller.load_dataset(main_root, scenario, num_antennas)


TIME_STEP = 16
MAX_SPEED = 2

epuck = robot_controller.robot.getFromDef("EPUCK")

distance_sensors = []

sensor_names = [
    'ps0', 'ps1', 'ps2', 'ps3',
    'ps4', 'ps5', 'ps6', 'ps7'
]

for i in range(8):
    distance_sensors.append(robot_controller.robot.getDevice(sensor_names[i]))
    distance_sensors[i].enable(TIME_STEP)


leftSpeed = 1.5 * MAX_SPEED
rightSpeed = 1.5 * MAX_SPEED

robot_controller.left_motor.setVelocity(leftSpeed)
robot_controller.right_motor.setVelocity(rightSpeed)

# df = robot_controller.load_dataset(main_root, scenario, num_antennas)

pos_array = []

while robot_controller.step(TIME_STEP) != -1:

    # robot_controller.left_motor.setVelocity(0)
    # robot_controller.right_motor.setVelocity(0)

    sensor_values = [sensor.getValue() for sensor in distance_sensors]

    # Detect Obstacles

    right_osbtacle = any(value > 80 for value in sensor_values[:3])
    left_obstacle = any(value > 80 for value in sensor_values[5:])

    leftSpeed = 1.5 * MAX_SPEED
    rightSpeed = 1.5 * MAX_SPEED

    if left_obstacle:
        leftSpeed = 0.5 * MAX_SPEED
        rightSpeed = -0.5 * MAX_SPEED
    elif right_osbtacle:
        leftSpeed = -0.5 * MAX_SPEED
        rightSpeed = 0.5 * MAX_SPEED

    robot_controller.left_motor.setVelocity(leftSpeed)
    robot_controller.right_motor.setVelocity(rightSpeed)


    print(" ")
    print("-- -- -- --")
    # posicion_estimada = robot_controller.getReading2(main_root, robot_controller.get_real_position())

    #results = robot_controller.get_real_position()
    results = robot_controller.readRoute(main_root, robot_controller.get_real_position())

    if results is not None:
        new_row = {
            "RoundedX": results[0],
            "RoundedY": results[1],
        }

        pos_array.append(new_row)

    print("-- -- -- --")
    print(" ")

    # robot_controller.left_motor.setVelocity(leftSpeed)
    # robot_controller.right_motor.setVelocity(rightSpeed)

    # robot_controller.step(TIME_STEP * 10)


    if robot_controller.getTime() > 40.0:
        robot_controller.left_motor.setVelocity(0.0)
        robot_controller.right_motor.setVelocity(0.0)
        break


# # Enter here exit cleanup code.
print("Exiting...")
df_positions = pd.DataFrame(pos_array)

df_positions.to_csv(main_root + "Test Case 2/Results/posiciones_ruta.csv", index=False)
# print(robot_controller.get_real_position())
