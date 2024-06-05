"""TestCase_ula8 controller."""

from controller import Supervisor
from robot_controller import RobotController
import numpy as np
import pandas as pd

main_root = "C:/Users/javi2/Desktop/TFG - Webots/TFG/"
scenario = "ULA"
num_antennas = "32"
radius = "5mm"
robot_controller = RobotController(Supervisor(), scenario, num_antennas)

# robot_controller.df = robot_controller.load_dataset(main_root, scenario, num_antennas)
robot_controller.df = robot_controller.load_dataset_testCases(main_root, scenario, num_antennas, radius)

print(robot_controller.get_real_position())
print("------------------------------------")

TIME_STEP = 16
MAX_SPEED = 2


leftSpeed = 0.5 * MAX_SPEED
rightSpeed = 0.5 * MAX_SPEED

robot_controller.left_motor.setVelocity(leftSpeed)
robot_controller.right_motor.setVelocity(rightSpeed)

pos_array = []

while robot_controller.step(TIME_STEP) != -1:

    
    print(" ")
    print("-- -- -- --")
    results = robot_controller.getReading2(main_root, robot_controller.get_real_position())
    print("-- -- -- --")
    print(" ")

    if results is not None:
        new_row = {
            "RoundedX": results[1][0],
            "RoundedY": results[1][1],
            "PredictedX": results[2][0],
            "PredictedY": results[2][1]
        }

        pos_array.append(new_row)

    robot_controller.step(TIME_STEP * 10)


    if robot_controller.getTime() > 70.0:
        robot_controller.left_motor.setVelocity(0.0)
        robot_controller.right_motor.setVelocity(0.0)
        break


print("Exiting...")
df_positions = pd.DataFrame(pos_array)

output_path = main_root + "Test Cases/Online positions/ULA 32/ruta_datos_5mm.csv"
df_positions.to_csv(output_path, index=False)

