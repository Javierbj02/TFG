from controller import Supervisor
import sys

from robot_controller import RobotController
import pickle

robot_controller = RobotController(Supervisor())

print(robot_controller.get_real_position())
print("------------------------------------")

main_root = "C:/Users/javi2/Desktop/TFG - Webots/TFG/"

robot_controller.getReading(main_root, "ULA", "32", robot_controller.get_real_position(), None)
