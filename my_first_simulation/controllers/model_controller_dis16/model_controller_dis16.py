from controller import Supervisor
from robot_controller import RobotController

main_root = "C:/Users/javi2/Desktop/TFG - Webots/TFG/"
scenario = "DIS"
num_antennas = "16"
robot_controller = RobotController(Supervisor(), scenario, num_antennas)

print(robot_controller.get_real_position())
print("------------------------------------")

robot_controller.getReading(main_root, robot_controller.get_real_position())