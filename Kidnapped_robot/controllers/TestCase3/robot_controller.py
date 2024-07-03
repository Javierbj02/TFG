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
        # self.distance_sensors = self.initialize_distance_sensors()
        self.scenario = scenario
        self.num_antennas = num_antennas
        self.modelX = None
        self.modelY = None
        self.df = None

    def step(self, time_step):
        return self.robot.step(time_step)
    
    def getTime(self):
        return self.robot.getTime()
    
    # def initialize_distance_sensors(self):
    #     distance_sensors = []
        
    #     sensor_names = [
    #         'ps0', 'ps1', 'ps2', 'ps3',
    #         'ps4', 'ps5', 'ps6', 'ps7'
    #     ]
        
    #     for i in range(8):
    #         distance_sensors.append(self.robot.getDevice(sensor_names[i]))
    #         distance_sensors[i].enable(TIME_STEP)
    #     return distance_sensors
    
    def get_real_position(self):
        epuck = self.robot.getFromDef("EPUCK")
        return [epuck.getPosition()[0] * 1000, epuck.getPosition()[1] * 1000]
    
    def get_real_position_coords(self):
        epuck = self.robot.getFromDef("EPUCK")
        return [epuck.getPosition()[0] * 1000, epuck.getPosition()[1] * 1000, epuck.getPosition()[2] * 1000]
    
    def round_position(self, position):
        # Redondear la coordenada X al más cercano que termine en 2 o en 7
        real_position = position.copy()
        if real_position[0] >= 0:
            real_position[0] = round(real_position[0] / 10) * 10 + (2 if real_position[0] % 10 < 5 else (7 if real_position[0] % 10 == 5 else -3))
        else:
            real_position[0] = round(real_position[0] / 10) * 10 + (3 if real_position[0] % 10 < 5 else (-7 if real_position[0] % 10 == 5 else -2))

        # Redondear la coordenada Y al más cercano que termine en 0 o en 5
        real_position[1] = round(real_position[1] / 5) * 5
        
        # Si la coordenada Y es mayor a 2779, restar 1
        if real_position[1] > 2779:
            real_position[1] -= 1

        # # Aplicar límites de posición X
        # real_position[0] = max(-1437, min(1357, real_position[0]))
        # if -187 < real_position[0] < 107:
        #     real_position[0] = -187 if real_position[0] < -40 else 107

        # # Aplicar límites de posición Y
        # real_position[1] = max(1155, min(4029, real_position[1]))
        # if 2405 < real_position[1] < 2779:
        #     real_position[1] = 2405 if real_position[1] < 2592 else 2779

        # Aplicar límites de posición
        real_position[0] = max(-1437, min(1357, real_position[0]))
        real_position[1] = max(1155, min(4029, real_position[1]))


        return real_position
    
    @memory.cache
    def load_dataset(folder, scenario, num_antennas):
        root = folder + "Data/" + scenario + "/" + scenario + "_" + num_antennas + ".csv"
        df = pd.read_csv(root) # ./Data/ULA/ULA_8.csv for example
        return df
    
    @memory.cache
    def load_dataset_testCases(folder, scenario, num_antennas, noise):
        root = folder + "Data/TC3/" + noise + "/"+ scenario + "_" + num_antennas  + ".csv"
        df = pd.read_csv(root)
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
    
    def total_dist(self, y_pred, y_true):
        return np.mean(np.sqrt(
            np.square(np.abs(y_pred[:,0] - y_true[:,0]))
            + np.square(np.abs(y_pred[:,1] - y_true[:,1]))
        ))

    @time
    def getReading2(self, folder, real_position):
        """
        Reads a dataset file, obtains the data of the position and applies a noise.
        With that data, the model predicts the position of the robot.

        ! Args:
            * folder (str): The main folder where all the needed data is located.
            * real_position (list): The real position of the robot. Example: [-758.0246871437575, 1798.9217383115995]

        ! Returns:
            ? float: The predicted position of the robot.

        """
        
        df = self.df
        # Obtain the adjusted/normalized/rounded position with respect to the nearest position of the measurements
        position = self.round_position(real_position) # Example: [-756.6850000000001, 1803.5]
        # Obtain the data from the dataset that corresponds to the position
        mean_data = df.loc[(df['PosicionX'] == position[0]) & (df['PosicionY'] == position[1])]

        if mean_data.empty:
            print("No hay datos para esa posición")
            return None
        
        mean_data = mean_data.sample(n=1) # Seleccionar un solo dato aleatorio
        mean_data = mean_data.iloc[[0]] # Seleccionar el primer dato

        
        
        # Normalize the data
        columns_to_normalize = mean_data.columns[:-2]
        
        # maximum = df[columns_to_normalize].max()
        # minimum = df[columns_to_normalize].min()

        max_min_file = os.path.join(folder + "Models/" + self.scenario + "/" + self.scenario + " " + self.num_antennas + "/max_min.npy")

        if os.path.exists(max_min_file):
            maximum, minimum = np.load(max_min_file)
        else:
            maximum, minimum = self.getMaxMin(df, columns_to_normalize)
            np.save(max_min_file, [maximum, minimum])
        
        data_normalized = (mean_data[columns_to_normalize] - minimum) / (maximum - minimum)
        data_normalized = pd.concat([data_normalized, mean_data[mean_data.columns[-2]], mean_data[mean_data.columns[-1]]], axis=1)
        
        # Generate image of the data

        pixel = 35

        #image_model = TINTO(problem="regression",pixels=pixel,blur=True)

        #images_dir = "C:/Users/javi2/Desktop/TFG - Webots/TFG/Images/ULA8_Temp/"
        # images_dir = folder + "Images/" + self.scenario + self.num_antennas + "TestCase1/"
        images_dir = folder + "Images/TestCase3/"
        loaded_model = pickle.load(open(folder + "Models/" + self.scenario + "/" + self.scenario + " " + self.num_antennas + "/image_model.pkl", "rb"))

        if os.path.exists(images_dir):
            shutil.rmtree(images_dir)

        loaded_model.generateImages_2(mean_data.iloc[:,:-1], images_dir)

    
        img_paths = os.path.join(images_dir + "/regression.csv")
        
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
            
        # x_img = cv2.resize(img, (pixel, pixel))
        
        # Predict the position of the robot with the data
        


        models_dir = folder + "Models/" + self.scenario + "/" + self.scenario + " " + self.num_antennas + "/"

        if self.modelX is None or self.modelY is None:
            self.modelX = keras.models.load_model(models_dir + "modelX.h5", custom_objects={'RSquare': RSquare})
            self.modelY = keras.models.load_model(models_dir + "modelY.h5", custom_objects={'RSquare': RSquare})

        
        # # Predict the position of the robot
        predictionX = self.modelX.predict([x_num, x_img], verbose=0)
        predictionY = self.modelY.predict([x_num, x_img], verbose=0)
        #print(predictionX[0])
        #print(predictionY[0])

        print("Posición real: ", real_position)
        print("Posición real (redondeada): ", position)
        print("Posición estimada: ", [predictionX[0][0], predictionY[0][0]])
        
        error_test = self.true_dist([predictionX[0][0], predictionY[0][0]], [position[0], position[1]])
        print("Error test: ", error_test)

        return [real_position, position, [predictionX[0][0], predictionY[0][0]]]
    

        

