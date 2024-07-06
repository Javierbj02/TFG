#from controller import Robot, DistanceSensor, Supervisor
from std_msgs.msg import Float32
from rclpy.node import Node
from controller import Robot, Supervisor
import rclpy

import os
import cv2
import shutil
import pickle
import pandas as pd
import numpy as np
from joblib import Memory
from keras.models import load_model
from tensorflow_addons.metrics import RSquare
from TINTOlib.tinto import TINTO
import logging
import json

import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

from test_cases_kalman_filter.Scenario import Scenario
from test_cases_kalman_filter.generate_plots import GeneratePlots
from test_cases_kalman_filter.Kalman_filter import filtroKalman

# execution_times = []
cachedir = './cache'
memory = Memory(location=cachedir, verbose=0)

TIME_STEP = 16
MAX_SPEED = 2

logging.basicConfig(
    filename='robot_controller.log',
    level=logging.DEBUG,
    format='%(asctime)s:%(levelname)s:%(message)s'
)

class RobotController(Node):
    def __init__(self):
        super().__init__('robot_controller')

        self.robot = Supervisor()

        config_file = './test_cases_kalman_filter/config.json'
        with open(config_file, 'r') as f:
            config = json.load(f)

        self.epuck = self.robot.getFromDef("EPUCK")
        self.translation_field = self.epuck.getField("translation")
        self.rotation_field = self.epuck.getField("rotation")

        self.left_motor = self.robot.getDevice('left wheel motor')
        self.right_motor = self.robot.getDevice('right wheel motor')

        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))

        self.leftSpeed = 1.5 * MAX_SPEED
        self.rightSpeed = 1.5 * MAX_SPEED

        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)

        self.distance_sensors = []
        self.sensor_names = [
            'ps0', 'ps1', 'ps2', 'ps3',
            'ps4', 'ps5', 'ps6', 'ps7'
        ]

        for i in range(8):
            self.distance_sensors.append(self.robot.getDevice(self.sensor_names[i]))
            self.distance_sensors[i].enable(TIME_STEP)

        self.pos_array = []
        self.time = None

        self.scenario = self.buildScenario("ULA", "8", "No ruido", "TC1")
        # self.scenario = self.buildScenario("ULA", config["num_antennas"], config["noise_level"], config["test_case"])

        self.plots = self.buildPlots()
        # self.num_antennas = "8"
        # self.noise = "No ruido"

        self.modelX = None
        self.modelY = None
        self.df = None
        self.kidnapped = True

        # Parámetros del filtro de Kalman
        x0 = np.array([self.get_real_position()[0], self.get_real_position()[1], 0, 0]).reshape(-1, 1) # Posicion X, Posicion Y, Velocidad X, Velocidad Y
        P0 = np.eye(4) # Covarianza inicial = Matriz identidad
        A = np.array([ # Define cómo evolucionan los estados del sistema en el tiempo. En este caso, se utiliza un modelo de movimiento rectilíneo uniforme (MRU)
            [1, 0, TIME_STEP/1000, 0],
            [0, 1, 0, TIME_STEP/1000],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ]) # Este modelo asume que la velocidad es constante en cada intervalo de tiempo
        B = np.zeros((4, 2)) # Matriz de control = Matriz de ceros, no hay control
        H = np.array([ # Cómo se relacionan las mediciones observadas con el estado del sistema. 
            [1, 0, 0, 0], # La medida de la posición en X está relacionada directamente con la posición en X del estado
            [0, 1, 0, 0] # La medida de la posición en Y está relacionada directamente con la posición en Y del estado
        ])
        Q = np.eye(4) * 0.1 # Covarianza del ruido del proceso
        R = np.eye(2) * 0.1 # Covarianza del ruido de la medida

        self.kalman_filter = filtroKalman(x0, P0, A, B, H, Q, R)

        self.main_root = "/home/javi2002bj/Escritorio/TFG_Webots/TFG/"
        self.load_models_and_data()

        self.time = time.time()

        self.create_timer(TIME_STEP/1000, self.update)

    def buildScenario(self, config, num_antennas, noise, test_case):
        scenario = Scenario()

        scenario.config = config
        scenario.num_antennas = num_antennas
        scenario.noise = noise
        scenario.test_case = test_case

        return scenario
    
    
    def buildPlots(self):
        return GeneratePlots(self.scenario.config, self.scenario.num_antennas, self.scenario.noise, self.scenario.test_case)



    def load_models_and_data(self):
        # * Load the dataset that contains all the data, and the models that will be used to predict the position of the robot
        self.df = self.load_dataset_testCases(self.main_root, self.scenario.config, self.scenario.num_antennas, self.scenario.noise, self.scenario.test_case)

        #models_dir = self.main_root + "Models/" + self.scenario + "/" + self.scenario + " " + self.num_antennas + "/"
        models_dir = os.path.join(self.main_root, "Models", self.scenario.config, self.scenario.config + " " + self.scenario.num_antennas + "/")

        self.modelX = load_model(models_dir + "modelX.h5", custom_objects={'RSquare': RSquare})
        self.modelY = load_model(models_dir + "modelY.h5", custom_objects={'RSquare': RSquare})

    # @memory.cache
    def load_dataset_testCases(self, main_root, scenario, num_antennas, noise, test_case):
        # * Load the dataset that contains all the data
        root = os.path.join(main_root, "Data", test_case, noise, scenario + "_" + num_antennas + ".csv")
        return pd.read_csv(root)
    
    # @memory.cache
    def getMaxMin(self, df, columns):
        return df[columns].max(), df[columns].min()
    

    def round_position(self, position):
        # * Apply limits of the positions as was done in the study case 
        real_position = position.copy()
        if real_position[0] >= 0:
            real_position[0] = round(real_position[0] / 10) * 10 + (2 if real_position[0] % 10 < 5 else (7 if real_position[0] % 10 == 5 else -3))
        else:
            real_position[0] = round(real_position[0] / 10) * 10 + (3 if real_position[0] % 10 < 5 else (-7 if real_position[0] % 10 == 5 else -2))


        real_position[1] = round(real_position[1] / 5) * 5
        

        if real_position[1] > 2779:
            real_position[1] -= 1


        real_position[0] = max(-1437, min(1357, real_position[0]))
        if -187 < real_position[0] < 107:
            real_position[0] = -187 if real_position[0] < -40 else 107


        real_position[1] = max(1155, min(4029, real_position[1]))
        if 2405 < real_position[1] < 2779:
            real_position[1] = 2405 if real_position[1] < 2592 else 2779

        return real_position
    

    def get_real_position(self):
        return [self.epuck.getPosition()[0] * 1000, self.epuck.getPosition()[1] * 1000]

    
    def predict_position(self):
        real_position = self.get_real_position()
        results = self.get_reading(self.main_root, real_position)

        if results is not None:
            predicted_position = np.array([results[2][0], results[2][1]])

            self.kalman_filter.update(predicted_position)

            kalman_estimate = self.kalman_filter.x
            logging.info(f"KKKK: {kalman_estimate}")

            kalman_estimate_x = kalman_estimate[1, 0]  # Posición X predicha
            kalman_estimate_y = kalman_estimate[1, 1]  # Posición Y predicha

            new_row = {
                "RoundedX": results[0][0],
                "RoundedY": results[0][1],
                "PredictedX": kalman_estimate_x,
                "PredictedY": kalman_estimate_y
            }

            self.pos_array.append(new_row)


    def get_positions_along_path(self):
        position = self.get_real_position()
        position = self.round_position(position)
        return position
    
    def build_path(self):
        results = self.get_positions_along_path()
        if results is not None:
            new_row = {
                "RoundedX": results[0],
                "RoundedY": results[1],
            }

            self.pos_array.append(new_row)


    
    def move(self):
        sensor_values = [sensor.getValue() for sensor in self.distance_sensors]
        right_obstacle = any(value > 80 for value in sensor_values[:3])
        left_obstacle = any(value > 80 for value in sensor_values[5:])

        self.leftSpeed = 1.5 * MAX_SPEED
        self.rightSpeed = 1.5 * MAX_SPEED

        if left_obstacle:
            self.leftSpeed = 0.5 * MAX_SPEED
            self.rightSpeed = -0.5 * MAX_SPEED
        elif right_obstacle:
            self.leftSpeed = -0.5 * MAX_SPEED
            self.rightSpeed = 0.5 * MAX_SPEED

        u = np.array([[self.leftSpeed], [self.rightSpeed]])
        self.kalman_filter.predict(u)

        self.left_motor.setVelocity(self.leftSpeed)
        self.right_motor.setVelocity(self.rightSpeed)


    def get_reading(self, folder, real_position):
        # df = self.df

        position = self.round_position(real_position)

        mean_data = self.df.loc[(self.df['PosicionX'] == position[0]) & (self.df['PosicionY'] == position[1])]

        if mean_data.empty:
            print("No hay datos para esa posición")
            return None
        
        mean_data = mean_data.sample(n=1) # In case there are multiple rows with the same position, select one randomly
        mean_data = mean_data.iloc[[0]] # Select the first row

        columns_to_normalize = mean_data.columns[:-2]

        max_min_file = os.path.join(folder + "Models", self.scenario.config, self.scenario.config + " " + self.scenario.num_antennas, "max_min.npy")

        if os.path.exists(max_min_file):
            maximum, minimum = np.load(max_min_file)
        else:
            maximum, minimum = self.getMaxMin(self.df, columns_to_normalize)
            np.save(max_min_file, [maximum, minimum])

        data_normalized = (mean_data[columns_to_normalize] - minimum) / (maximum - minimum)
        data_normalized = pd.concat([data_normalized, mean_data[mean_data.columns[-2]], mean_data[mean_data.columns[-1]]], axis=1)


        # * Generate the image of the data

        pixel = 35

        images_dir = os.path.join(folder, "Images", "TestCases")
        loaded_model = pickle.load(open(os.path.join(folder, "Models", self.scenario.config, self.scenario.config + " " + self.scenario.num_antennas, "image_model.pkl"), "rb"))

        if os.path.exists(images_dir):
            shutil.rmtree(images_dir)

        loaded_model.generateImages_2(mean_data.iloc[:,:-1], images_dir)

        img_paths = os.path.join(images_dir, "regression.csv")

        imgs = pd.read_csv(img_paths)

        imgs["images"] = images_dir + "/" + imgs["images"]
        imgs["images"] = imgs["images"].str.replace("\\","/")

        x_num = data_normalized.drop("PosicionX",axis=1).drop("PosicionY",axis=1)

        image_path = imgs["images"].iloc[0]
        image = cv2.imread(image_path)
        
        if image is not None:
            try:
                x_img = np.array([cv2.resize(image, (pixel, pixel))])
            except Exception as e:
                print(e)
                print("Error resizing image")
        else:
            print("Error reading image")

        # Predict the position of the robot
        predictionX = self.modelX.predict([x_num, x_img], verbose = 0)
        predictionY = self.modelY.predict([x_num, x_img], verbose = 0)

        # print("Posición real: ", real_position)
        # print("Posición real (redondeada): ", position)
        # print("Posición estimada: ", [predictionX[0][0], predictionY[0][0]])

        return [real_position, position, [predictionX[0][0], predictionY[0][0]]]
    

    def total_dist(self, y_pred, y_true):
        return np.mean(np.sqrt(
            np.square(np.abs(y_pred[:,0] - y_true[:,0]))
            + np.square(np.abs(y_pred[:,1] - y_true[:,1]))
        ))

    
    def save_results(self):
        results = "/home/javi2002bj/Escritorio/TFG_Webots/TFG/TCs - Kalman Filter/Results/Results " + self.scenario.test_case

        df_positions = pd.DataFrame(self.pos_array)
        output_path = os.path.join(results, self.scenario.noise, self.scenario.config + " " + self.scenario.num_antennas, "predicciones_" + self.scenario.config + self.scenario.num_antennas + ".csv")

        df_positions.to_csv(output_path, index = False)

        error_path = self.total_dist(df_positions[["PredictedX", "PredictedY"]].to_numpy(), df_positions[["RoundedX", "RoundedY"]].to_numpy())
        logging.info(f"Average error of the path: {error_path}")

        mae_x = mean_absolute_error(df_positions["RoundedX"], df_positions["PredictedX"])
        mae_y = mean_absolute_error(df_positions["RoundedY"], df_positions["PredictedY"])

        mse_x = mean_squared_error(df_positions["RoundedX"], df_positions["PredictedX"])
        mse_y = mean_squared_error(df_positions["RoundedY"], df_positions["PredictedY"])

        rmse_x = mean_squared_error(df_positions["RoundedX"], df_positions["PredictedX"], squared=False)
        rmse_y = mean_squared_error(df_positions["RoundedY"], df_positions["PredictedY"], squared=False)

        txt_path = os.path.join(results, self.scenario.noise, self.scenario.config + " " + self.scenario.num_antennas, "error_" + self.scenario.config + self.scenario.num_antennas + ".txt")
        with open(txt_path, "w") as f:
            f.write(f"------------------------------------\n")
            f.write(f"Average error of the path: {self.scenario.noise}, {self.scenario.config}, {self.scenario.num_antennas}: {error_path} mm\n")
            f.write("\n")
            f.write(f"Mean Absolute Error (MAE) in X: {mae_x}\n")
            f.write(f"Mean Absolute Error (MAE) in Y: {mae_y}\n")
            f.write(f"Mean Squared Error (MSE) in X: {mse_x}\n")
            f.write(f"Mean Squared Error (MSE) in Y: {mse_y}\n")
            f.write(f"Root Mean Squared Error (RMSE) in X: {rmse_x}\n")
            f.write(f"Root Mean Squared Error (RMSE) in Y: {rmse_y}\n")
            f.write("\n")
            f.write(f"Execution time: {self.time} seconds\n")
            f.write(f"------------------------------------\n")
            f.write("\n")

        self.plots.df_positions = df_positions
        self.plots.generate_plots()

    def save_results_2(self):
        results = "/home/javi2002bj/Escritorio/TFG_Webots/TFG/TCs - Kalman Filter/Results/Results " + self.scenario.test_case
        df_positions = pd.DataFrame(self.pos_array)
        output_path = os.path.join(results, "posiciones_ruta.csv")
        df_positions.to_csv(output_path, index = False)

    def update_build_path(self):
        if self.robot.step(TIME_STEP) != -1:
            if self.scenario.test_case == 'TC1' and self.robot.getTime() > 24.0:
                self.left_motor.setVelocity(0.0)
                self.right_motor.setVelocity(0.0)
                self.time = time.time() - self.time
                self.save_results_2()
                logging.info("Results saved")
                self.robot.simulationQuit(0)
                self.finish()

            
            elif self.kidnapped and self.scenario.test_case == 'TC3' and self.robot.getTime() > 20.0:
                new_translation = [-0.513653, 3.7385, -2.95099e-05]
                self.translation_field.setSFVec3f(new_translation)

                new_rotation = [-1.09075e-09, 9.30002e-10, 1, -2.32828]
                self.rotation_field.setSFRotation(new_rotation)

                self.kidnapped = False

            elif self.robot.getTime() > 40.0:
                self.left_motor.setVelocity(0.0)
                self.right_motor.setVelocity(0.0)
                self.time = time.time() - self.time
                self.save_results_2()
                logging.info("Results saved")
                self.robot.simulationQuit(0)
                self.finish()

            else:
                self.move()
                self.build_path()

    def update(self):
        if self.robot.step(TIME_STEP) != -1:
            if self.scenario.test_case == 'TC1' and self.robot.getTime() > 24.0:
                self.left_motor.setVelocity(0.0)
                self.right_motor.setVelocity(0.0)
                self.time = time.time() - self.time
                self.save_results()
                logging.info("Results saved")
                self.robot.simulationQuit(0)
                self.finish()

            
            elif self.kidnapped and self.scenario.test_case == 'TC3' and self.robot.getTime() > 20.0:
                new_translation = [-0.513653, 3.7385, -2.95099e-05]
                self.translation_field.setSFVec3f(new_translation)

                new_rotation = [-1.09075e-09, 9.30002e-10, 1, -2.32828]
                self.rotation_field.setSFRotation(new_rotation)

                self.kidnapped = False

            elif self.robot.getTime() > 40.0:
                self.left_motor.setVelocity(0.0)
                self.right_motor.setVelocity(0.0)
                self.time = time.time() - self.time
                self.save_results()
                logging.info("Results saved")
                self.robot.simulationQuit(0)
                self.finish()

            else:
                self.move()
                self.predict_position()


    def finish(self):
        self.destroy_node()
        rclpy.shutdown()
        # Check that the node has been correctly destroyed


def main(args = None):
    rclpy.init(args=args)

    node = RobotController()
    rclpy.spin(node)

    node.finish()

    # node.destroy_node()
    # rclpy.shutdown()

if __name__ == '__main__':
    main()