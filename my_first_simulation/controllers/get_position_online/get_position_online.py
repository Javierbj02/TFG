"""TestCase_ula8 controller."""

from controller import Supervisor
from TFG.robot_controller import RobotController
import numpy as np
import pandas as pd

main_root = "C:/Users/javi2/Desktop/TFG - Webots/TFG/"
scenario = "ULA"
num_antennas = "8"
robot_controller = RobotController(Supervisor(), scenario, num_antennas)

# robot_controller.df = robot_controller.load_dataset(main_root, scenario, num_antennas)

print(robot_controller.get_real_position())
print("------------------------------------")

# robot_controller.getReading(main_root, robot_controller.get_real_position())

# get the time step of the current world.
TIME_STEP = 16
MAX_SPEED = 2


# You should insert a getDevice-like function in order to get the
# instance of a device of the robot. Something like:
#  motor = robot.getDevice('motorname')
#  ds = robot.getDevice('dsname')
#  ds.enable(timestep)

# Main loop:
# - perform simulation steps until Webots is stopping the controller

# leftSpeed = 0.5 * MAX_SPEED
# rightSpeed = 0.5 * MAX_SPEED

leftSpeed = 1.5 * MAX_SPEED
rightSpeed = 1.5 * MAX_SPEED

robot_controller.left_motor.setVelocity(leftSpeed)
robot_controller.right_motor.setVelocity(rightSpeed)

# df = robot_controller.load_dataset(main_root, scenario, num_antennas)
# ! Probar pasando df como parámetro al método getReading2
# ! Probar también poniendolo como atributo de la clase RobotController

pos_array = []

while robot_controller.step(TIME_STEP) != -1:

    # robot_controller.left_motor.setVelocity(0)
    # robot_controller.right_motor.setVelocity(0)
    
    print(" ")
    print("-- -- -- --")

    results = robot_controller.readRoute(main_root, robot_controller.get_real_position())

    if results is not None:
        new_row = {
            "RoundedX": results[0],
            "RoundedY": results[1],
        }

        pos_array.append(new_row)

    print("-- -- -- --")
    print(" ")

    #robot_controller.step(1)


    if robot_controller.getTime() > 24.0:
        robot_controller.left_motor.setVelocity(0.0)
        robot_controller.right_motor.setVelocity(0.0)
        break


print("Exiting...")

df_positions = pd.DataFrame(pos_array)

df_positions.to_csv(main_root + "Test Case 1/Results/posiciones_ruta.csv", index=False)

