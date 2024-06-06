"""TestCase_ula8 controller."""

from controller import Supervisor
from robot_controller import RobotController
import numpy as np

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

leftSpeed = 0.5 * MAX_SPEED
rightSpeed = 0.5 * MAX_SPEED

robot_controller.left_motor.setVelocity(leftSpeed)
robot_controller.right_motor.setVelocity(rightSpeed)

# df = robot_controller.load_dataset(main_root, scenario, num_antennas)
# ! Probar pasando df como parámetro al método getReading2
# ! Probar también poniendolo como atributo de la clase RobotController

position_array = []

while robot_controller.step(TIME_STEP) != -1:

    # robot_controller.left_motor.setVelocity(0)
    # robot_controller.right_motor.setVelocity(0)
    
    print(" ")
    print("-- -- -- --")
    # posicion_estimada = robot_controller.getReading2(main_root, robot_controller.get_real_position())
    # Quiero ir leyendo la posición actual del robot, y guardar cada lectura en un .npy
    pos = robot_controller.get_real_position()
    position_array.append(pos)
    print(pos)
    print("-- -- -- --")
    print(" ")

    # robot_controller.left_motor.setVelocity(leftSpeed)
    # robot_controller.right_motor.setVelocity(rightSpeed)

    robot_controller.step(TIME_STEP * 10)


    if robot_controller.getTime() > 70.0:
        robot_controller.left_motor.setVelocity(0.0)
        robot_controller.right_motor.setVelocity(0.0)
        break


# # Enter here exit cleanup code.
print("Exiting...")
np.save(main_root + "Test Case 1/Results/posiciones_reales_2.npy", position_array)
# print(robot_controller.get_real_position())
