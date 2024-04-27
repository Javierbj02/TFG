import pandas as pd
import numpy as np

# cargar un numpy
arr = np.load("C:/Users/javi2/Desktop/TFG - Webots/TFG/Data/channel_measurement_000000.npy")
print(arr[0])
# df = pd.read_csv("C:/Users/javi2/Desktop/TFG - Webots/TFG/Data/ULA_8.csv")

# # print de df de la primera fila y dos primeras y dos ultimas columnas
# print(df.iloc[0, [0, 1, -2, -1]])

# print(df)
# print(df.shape)

# # df_except_1_and_2_columns = df.iloc[:, 2:] # Get all rows, and columns from 2 to end
# # print(df_except_1_and_2_columns)

# # df_except_1_and_2_rows = df.iloc[2:, :] # Get all columns, and rows from 2 to end
# # print(df_except_1_and_2_rows)

# fila = df.loc[(df['datos1'] > 2) & (df['datos2'] > 7)]

# print(fila)

# fila = df.loc[(df['posicionX'] == 300) & (df['posicionY'] == 3000)]
# print(fila)

# df = pd.read_csv("C:/Users/Medion/Desktop/WEBOTS PROYECTOS/TFG/Data/ULA_8.csv")  # ./Data/ULA_8.csv for example

# # Guardar un nuevo csv quitando todas las columnas menos las dos últimas y con las primeras 1000 filas
# df.iloc[:1000, -2:].to_csv("C:/Users/Medion/Desktop/WEBOTS PROYECTOS/TFG/Data/ULA_8_1000.csv", index=False)

# arr = np.load("C:/Users/Medion/Desktop/WEBOTS PROYECTOS/TFG/Data/ULA_lab_LoS/user_positions.npy")
# print(arr.shape)
# print("---")
# elemento = [-1052, 2110, 400]
# elemento_2 = [-1437, 1155, 400]
# elemento_3 = [-1437, 1160, 400]

# posicion = np.where((arr == elemento).all(axis=1))
# print(posicion)
# print("---")
# posicion = np.where(np.all(arr == elemento, axis=1))[0][0]
# print(posicion)
# print("---")
# posicion = np.where((arr == elemento_2).all(axis=1))
# print(posicion)
# print("---")
# posicion = np.where(np.all(arr == elemento_2, axis=1))[0][0]
# print(posicion)
# print("---")
# posicion = np.where((arr == elemento_3).all(axis=1))
# print(posicion)
# print("---")
# posicion = np.where(np.all(arr == elemento_3, axis=1))[0][0]
# print(posicion)

# data = {
#     'datos1': [1, 2, 3, 4, 5, 10],
#     'datos2': [6, 7, 8, 9, 10, 10],
#     'datos3': [11, 12, 13, 14, 15, 10],
#     'posicionX': [100, 200, 300, 400, 500, 600],
#     'posicionY': [1000, 2000, 3000, 4000, 5000, 6000]
# }

# arr = np.load("C:/Users/Medion/Desktop/WEBOTS PROYECTOS/TFG/Data/channel_measurement_034266.npy")

# print(arr[0])
# print("---")
# print(arr[28])