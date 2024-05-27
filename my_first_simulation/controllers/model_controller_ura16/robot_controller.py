#from controller import Robot, DistanceSensor, Supervisor
import pandas as pd
import numpy as np
from joblib import Memory
import time
import os
import keras
from TINTOlib.tinto import TINTO
from tensorflow_addons.metrics import RSquare
import cv2
import shutil
import pickle

TIME_STEP = 16
MAX_SPEED = 2
cachedir = './cache'
memory = Memory(location=cachedir, verbose=0)

class RobotController:
    def __init__(self, robot, scenario, num_antennas):
        self.robot = robot
        self.left_motor = self.robot.getDevice('left wheel motor')
        self.right_motor = self.robot.getDevice('right wheel motor')
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)
        self.distance_sensors = self.initialize_distance_sensors()
        self.scenario = scenario
        self.num_antennas = num_antennas
        self.modelX = None
        self.modelY = None


    def initialize_distance_sensors(self):
        distance_sensors = []
        
        sensor_names = [
            'ps0', 'ps1', 'ps2', 'ps3',
            'ps4', 'ps5', 'ps6', 'ps7'
        ]
        
        for i in range(8):
            distance_sensors.append(self.robot.getDevice(sensor_names[i]))
            distance_sensors[i].enable(TIME_STEP)
        return distance_sensors
    
    def get_real_position(self):
        epuck = self.robot.getFromDef("EPUCK")
        return [epuck.getPosition()[0] * 1000, epuck.getPosition()[1] * 1000]
    
    def round_position(self, real_position):
        # Redondear la coordenada X al más cercano que termine en 2 o en 7
        if real_position[0] >= 0:
            real_position[0] = round(real_position[0] / 10) * 10 + (2 if real_position[0] % 10 < 5 else (7 if real_position[0] % 10 == 5 else -3))
        else:
            real_position[0] = round(real_position[0] / 10) * 10 + (3 if real_position[0] % 10 < 5 else (-7 if real_position[0] % 10 == 5 else -2))

        # Redondear la coordenada Y al más cercano que termine en 0 o en 5
        real_position[1] = round(real_position[1] / 5) * 5
        
        # Si la coordenada Y es mayor a 2779, restar 1
        if real_position[1] > 2779:
            real_position[1] -= 1

        # Aplicar límites de posición X
        real_position[0] = max(-1437, min(1357, real_position[0]))
        if -187 < real_position[0] < 107:
            real_position[0] = -187 if real_position[0] < -40 else 107

        # Aplicar límites de posición Y
        real_position[1] = max(1155, min(4029, real_position[1]))
        if 2405 < real_position[1] < 2779:
            real_position[1] = 2405 if real_position[1] < 2592 else 2779

        return real_position
    
    @memory.cache
    def load_dataset(folder, scenario, num_antennas):
        root = folder + "Data/" + scenario + "/" + scenario + "_" + num_antennas + ".csv"
        df = pd.read_csv(root) # ./Data/ULA/ULA_8.csv for example
        return df
    
    @memory.cache
    def getMaxMin(df, columns):
        return df[columns].max(), df[columns].min()
    
    def time(funcion):
        def function(*args, **kwargs):
            start = time.time()
            c = funcion(*args, **kwargs)
            print("TIEMPO DE EJECUCIÓN", time.time() - start)
            return c
        return function
    
    def true_dist(self, x_pred, x_true):
        uno = x_pred[0] - x_true[0]
        uno = uno ** 2
        
        dos = x_pred[1] - x_true[1]
        dos = dos ** 2
        
        return (uno + dos) ** 0.5
    
    @time
    def getReading(self, folder, real_position):
        """
        Reads a dataset file, obtains the data of the position and applies a noise.
        With that data, the model predicts the position of the robot.

        ! Args:
            * folder (str): The main folder where all the needed data is located.
            * real_position (list): The real position of the robot. Example: [-758.0246871437575, 1798.9217383115995]

        ! Returns:
            ? float: The predicted position of the robot.

        """
        # Read the dataset (the .csv format that we have generated with the measurements of the antennas in numpy) 
        t0 = time.time()
        df = self.load_dataset(folder, self.scenario, self.num_antennas)
        print("T0 - Tiempo de carga del dataset: ", time.time() - t0)
        
        
        # Obtain the adjusted/normalized/rounded position with respect to the nearest position of the measurements
        t1 = time.time()
        position = self.round_position(real_position) # Example: [-756.6850000000001, 1803.5]
        print("T1 - Tiempo de redondeo de la posición: ", time.time() - t1)

        print("POSICIÓN: ", position) # Example: [-757, 1805]
        
        
        
        # Obtain the data from the dataset that corresponds to the position
        t2 = time.time()
        data = df.loc[(df['PosicionX'] == position[0]) & (df['PosicionY'] == position[1])]
        print("T2 - Tiempo de obtención de los datos de la posición: ", time.time() - t2)
        
        # Apply noise to the data
        # TODO: Optimize this part

        # t3 = time.time()
        # data_2 = df.loc[(df['PosicionX'] == position[0] - 5) & (df['PosicionY'] == position[1] - 5)]
        # data_3 = df.loc[(df['PosicionX'] == position[0] + 5) & (df['PosicionY'] == position[1] + 5)]
        # data_4 = df.loc[(df['PosicionX'] == position[0] - 5) & (df['PosicionY'] == position[1] + 5)]
        # data_5 = df.loc[(df['PosicionX'] == position[0] + 5) & (df['PosicionY'] == position[1] - 5)]
        
        # selected_rows = pd.concat([data, data_2, data_3, data_4, data_5])
        # mean_data = selected_rows.iloc[:, :-2].mean()
        # mean_data = pd.DataFrame(mean_data).T
        
        # mean_data['PosicionX'] = data['PosicionX'].iloc[0]
        # mean_data['PosicionY'] = data['PosicionY'].iloc[0]

        # print("T3 - Tiempo de aplicación de ruido: ", time.time() - t3)
        
        mean_data = data
        # Normalize the data
        t41 = time.time()
        columns_to_normalize = mean_data.columns[:-2]
        
        # maximum = df[columns_to_normalize].max()
        # minimum = df[columns_to_normalize].min()

        max_min_file = os.path.join(folder + "Models/" + self.scenario + "/" + self.scenario + " " + self.num_antennas + "/max_min.npy")

        if os.path.exists(max_min_file):
            maximum, minimum = np.load(max_min_file)
        else:
            maximum, minimum = self.getMaxMin(df, columns_to_normalize)
            np.save(max_min_file, [maximum, minimum])

        print("T4.1 - Tiempo de get Max Min: ", time.time() - t41)
        
        t4 = time.time()
        data_normalized = (mean_data[columns_to_normalize] - minimum) / (maximum - minimum)
        data_normalized = pd.concat([data_normalized, mean_data[mean_data.columns[-2]], mean_data[mean_data.columns[-1]]], axis=1)
        
        
        # Generate image of the data
        print("T4 - Tiempo de normalización de datos: ", time.time() - t4)
        
        t5 = time.time()
        pixel = 35

        #image_model = TINTO(problem="regression",pixels=pixel,blur=True)

        #images_dir = "C:/Users/javi2/Desktop/TFG - Webots/TFG/Images/ULA8_Temp/"
        images_dir = folder + "Images/" + self.scenario + self.num_antennas + "_Temp/"
        loaded_model = pickle.load(open(folder + "Models/" + self.scenario + "/" + self.scenario + " " + self.num_antennas + "/image_model.pkl", "rb"))

        if os.path.exists(images_dir):
            shutil.rmtree(images_dir)

        loaded_model.generateImages_2(mean_data.iloc[:,:-1], images_dir)

    
        img_paths = os.path.join(images_dir + "/regression.csv")
        
        imgs = pd.read_csv(img_paths)
        
        imgs["images"] = images_dir + "/" + imgs["images"]
        imgs["images"] = imgs["images"].str.replace("\\","/")

        print("T5 - Tiempo de generación de imágenes: ", time.time() - t5)

        t6 = time.time()
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

        print("T6 - Tiempo de preparación de los datos para la predicción: ", time.time() - t6)
            
        # x_img = cv2.resize(img, (pixel, pixel))
        
        # TODO: Predict the position of the robot with the data
        

        t7 = time.time()

        models_dir = folder + "Models/" + self.scenario + "/" + self.scenario + " " + self.num_antennas + "/"

        if self.modelX is None or self.modelY is None:
            self.modelX = keras.models.load_model(models_dir + "modelX.h5", custom_objects={'RSquare': RSquare})
            self.modelY = keras.models.load_model(models_dir + "modelY.h5", custom_objects={'RSquare': RSquare})

        print("T7 - Tiempo de carga de los modelos: ", time.time() - t7)
        
        # # Predict the position of the robot
        t8 = time.time()
        predictionX = self.modelX.predict([x_num, x_img])
        predictionY = self.modelY.predict([x_num, x_img])
        print(predictionX[0])
        print(predictionY[0])
        print("T8 - Tiempo de predicción de la posición X e Y: ", time.time() - t8)
        
        t9 = time.time()
        error_test = self.true_dist([predictionX[0][0], predictionY[0][0]], [position[0], position[1]])
        print("Error test: ", error_test)
        print("T9 - Tiempo de cálculo del error: ", time.time() - t9)
        
        
        ######
        t10 = time.time()
        print("Imagen vacía")
        predictionX = self.modelX.predict([x_num, np.zeros((len(x_num), pixel, pixel, 3))])
        predictionY = self.modelY.predict([x_num, np.zeros((len(x_num), pixel, pixel, 3))])
        print(predictionX[0])
        print(predictionY[0])
        
        error_test = self.true_dist([predictionX[0][0], predictionY[0][0]], [position[0], position[1]])
        print("Error test: ", error_test)
        print("T10 - Tiempo de predicción +  cálculo del error con imagen vacía: ", time.time() - t10)