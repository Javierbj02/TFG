import pandas as pd
import numpy as np

# cargar un numpy
# arr = np.load("C:/Users/javi2/Desktop/TFG - Webots/TFG/Data/channel_measurement_000000.npy")
# print(arr[0])
# df = pd.read_csv("C:/Users/javi2/Desktop/TFG - Webots/TFG/Data/ULA/ULA_8.csv")
# df = pd.read_csv("C:/Users/javi2/Desktop/TFG - Webots/TFG/Data/ULA/ULA_lab_LoS_8_100_v3.csv")


# fila = df.loc[(df['PosicionX'] == 842) & (df['PosicionY'] == 3379)]
# print(fila)

def true_dist(x_pred, x_true):
    uno = x_pred[0] - x_true[0]
    uno = uno ** 2
    
    dos = x_pred[1] - x_true[1]
    dos = dos ** 2
    
    return (uno + dos) ** 0.5

print(true_dist([827.7021, 3189.254], [842.0, 3379]))

