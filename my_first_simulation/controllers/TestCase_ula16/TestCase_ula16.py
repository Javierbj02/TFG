"""TestCase_ula8 controller."""

from controller import Supervisor
from robot_controller import RobotController
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

main_root = "C:/Users/javi2/Desktop/TFG - Webots/TFG/"

scenarios = ["ULA", "URA", "DIS"]
nums_antennas = ["8", "16", "32", "64"]
noises = ["No ruido", "Ruido bajo", "Ruido medio", "Ruido elevado"]

scenario = scenarios[0]
num_antennas = nums_antennas[1]
ruido = noises[0]

robot_controller = RobotController(Supervisor(), scenario, num_antennas)

# robot_controller.df = robot_controller.load_dataset(main_root, scenario, num_antennas)
robot_controller.df = robot_controller.load_dataset_testCases(main_root, scenario, num_antennas, ruido)

print(robot_controller.get_real_position())
print("------------------------------------")

TIME_STEP = 16
MAX_SPEED = 2


leftSpeed = 0.5 * MAX_SPEED
rightSpeed = 0.5 * MAX_SPEED

robot_controller.left_motor.setVelocity(leftSpeed)
robot_controller.right_motor.setVelocity(rightSpeed)

pos_array = []

t = time.time()
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
        tiempo = time.time() - t
        break


print("Exiting...")
df_positions = pd.DataFrame(pos_array)

output_path = main_root + "Test Cases/Results/" + ruido + "/" + scenario + " " + num_antennas + "/predicciones_" + scenario + num_antennas + ".csv"
df_positions.to_csv(output_path, index=False)

error_ruta = robot_controller.total_dist(df_positions[["PredictedX", "PredictedY"]].to_numpy(), df_positions[["RoundedX", "RoundedY"]].to_numpy())
print("Error ruta: ", error_ruta)

mae_x = mean_absolute_error(df_positions["RoundedX"], df_positions["PredictedX"])
mae_y = mean_absolute_error(df_positions["RoundedY"], df_positions["PredictedY"])

mse_x = mean_squared_error(df_positions["RoundedX"], df_positions["PredictedX"])
mse_y = mean_squared_error(df_positions["RoundedY"], df_positions["PredictedY"])

rmse_x = mean_squared_error(df_positions["RoundedX"], df_positions["PredictedX"], squared=False)
rmse_y = mean_squared_error(df_positions["RoundedY"], df_positions["PredictedY"], squared=False)

# Guardar en un txt el error de la ruta para ese caso de prueba (ruido, escenario, antena)
txt_path = main_root + "Test Cases/Results/" + ruido + "/" + scenario + " " + num_antennas + "/error_ruta_" + scenario + num_antennas + ".txt"
with open(txt_path, "w") as f:
    f.write(f"------------------------------------\n")
    f.write(f"Error ruta para el caso de prueba: {ruido}, {scenario}, {num_antennas}: {error_ruta} mm\n")
    f.write("\n")
    f.write(f"Mean Absolute Error (MAE) en X: {mae_x}\n")
    f.write(f"Mean Absolute Error (MAE) en Y: {mae_y}\n")
    f.write(f"Mean Squared Error (MSE) en X: {mse_x}\n")
    f.write(f"Mean Squared Error (MSE) en Y: {mse_y}\n")
    f.write(f"Root Mean Squared Error (RMSE) en X: {rmse_x}\n")
    f.write(f"Root Mean Squared Error (RMSE) en Y: {rmse_y}\n")
    f.write("\n")
    f.write(f"Tiempo de ejecucion: {tiempo} segundos\n")
    f.write(f"------------------------------------\n")
    f.write("\n")


plt.figure()
plt.scatter(df_positions["RoundedX"], df_positions["RoundedY"], label='Ground truth', s=3, alpha=0.7, c = 'blue')
plt.scatter(df_positions["PredictedX"], df_positions["PredictedY"], label='Prediction', s=3, alpha=0.7, c = 'red')
plt.xlabel('x position [mm]')
plt.ylabel('y position [mm]')
# plt.title('Comparación de Ruta Real y Ruta Predicha')
plt.legend()
# plt.xlim(-1500, 1500)
# plt.ylim(1500, 4500)
plt.grid(False)
plt.savefig(main_root + "Test Cases/Results/" + ruido + "/" + scenario + " " + num_antennas + "/graphic_s_" + scenario + num_antennas + ".png")


plt.figure()
plt.scatter(df_positions["RoundedX"], df_positions["RoundedY"], label='Ground truth', s=3, alpha=0.7, c = 'blue')
plt.scatter(df_positions["PredictedX"], df_positions["PredictedY"], label='Prediction', s=3, alpha=0.7, c = 'red')
plt.xlabel('x position [mm]')
plt.ylabel('y position [mm]')
# plt.title('Comparación de Ruta Real y Ruta Predicha')
plt.legend()
plt.xlim(-1500, 1500)
plt.ylim(1500, 4500)
plt.grid(False)
plt.savefig(main_root + "Test Cases/Results/" + ruido + "/" + scenario + " " + num_antennas + "/graphic_l_" + scenario + num_antennas + ".png")
