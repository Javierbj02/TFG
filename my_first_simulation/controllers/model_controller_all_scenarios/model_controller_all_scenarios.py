"""model_controller_all_scenarios controller."""
# World: my_first_simulation/worlds/escenario_segun_datos.wbt

from controller import Supervisor
from TFG.robot_controller import RobotController

main_root = "C:/Users/javi2/Desktop/TFG - Webots/TFG/"

scenarios = ["ULA", "URA", "DIS"]
nums_antennas = ["8", "16", "32", "64"]

# !! Change the scenario and the number of antennas to test different cases.
# !! This controller is used primarily to test each model in Webots.
scenario = "ULA"
num_antennas = "8"
robot_controller = RobotController(Supervisor(), scenario, num_antennas)

print(robot_controller.get_real_position())
print("------------------------------------")

robot_controller.getReading(main_root, robot_controller.get_real_position())