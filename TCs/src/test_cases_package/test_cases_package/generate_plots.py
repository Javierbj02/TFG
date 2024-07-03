import os 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import subprocess
import pickle
import matplotlib
matplotlib.use('Agg')

class GeneratePlots:
    def __init__(self, scenario, num_antennas, noise, test_case):
        self.path = "/home/javi2002bj/Escritorio/TFG_Webots/TFG/TCs/Results/Results " + test_case + "/" + noise + "/" + scenario + " " + num_antennas
        self.scenario = scenario
        self.num_antennas = num_antennas
        self.noise = noise
        self.df_positions = None
        self.test_case = test_case

    # PARA ESTO HACER UN STRATEGY PATTERN: Para ello, crear una clase abstracta GeneratePlots con un método abstracto generate_plots. 
    # Esta clase tendrá tres clases hijas, GeneratePlotsTC1, GeneratePlotsTC2 y GeneratePlotsTC3, que implementarán el método generate_plots de forma distinta.
    def generate_plots(self):
        if self.test_case == "TC1":
            self.generate_plots_tc1()
        elif self.test_case == "TC2":
            self.generate_plots_tc2()
        elif self.test_case == "TC3":
            self.generate_plots_tc3()
        else:
            print("Test case not found")
            return
        print("Plots generated in ", self.path)


    def generate_plots_tc1(self):
        plt.figure()
        plt.scatter(self.df_positions["RoundedX"], self.df_positions["RoundedY"], label='Ground truth', s=3, alpha=0.7, c = 'blue')
        plt.scatter(self.df_positions["PredictedX"], self.df_positions["PredictedY"], label='Prediction', s=3, alpha=0.7, c = 'red')
        plt.xlabel('x position [mm]')
        plt.ylabel('y position [mm]')
        # plt.title('Comparación de Ruta Real y Ruta Predicha')
        plt.legend()
        plt.xlim(-1400, -250)
        plt.ylim(2900, 4000)
        plt.grid(False)
        fig1_path = os.path.join(self.path, "graphic_s_" + self.scenario + self.num_antennas + ".png")
        plt.savefig(fig1_path)

        plt.figure()
        plt.scatter(self.df_positions["RoundedX"], self.df_positions["RoundedY"], label='Ground truth', s=3, alpha=0.7, c = 'blue')
        plt.scatter(self.df_positions["PredictedX"], self.df_positions["PredictedY"], label='Prediction', s=3, alpha=0.7, c = 'red')
        plt.xlabel('x position [mm]')
        plt.ylabel('y position [mm]')
        # plt.title('Comparación de Ruta Real y Ruta Predicha')
        plt.legend()
        plt.xlim(-1500, 1500)
        plt.ylim(1500, 4500)
        plt.grid(False)
        fig2_path = os.path.join(self.path, "graphic_l_" + self.scenario + self.num_antennas + ".png")
        plt.savefig(fig2_path)

    def generate_plots_tc2(self):
        plt.figure()
        plt.scatter(self.df_positions["RoundedX"], self.df_positions["RoundedY"], label='Ground truth', s=3, alpha=0.7, c = 'blue')
        plt.scatter(self.df_positions["PredictedX"], self.df_positions["PredictedY"], label='Prediction', s=3, alpha=0.7, c = 'red')
        plt.xlabel('x position [mm]')
        plt.ylabel('y position [mm]')
        # plt.title('Comparación de Ruta Real y Ruta Predicha')
        plt.legend()
        plt.xlim(-1400, -250)
        plt.ylim(2900, 4000)
        plt.grid(False)
        fig1_path = os.path.join(self.path, "graphic_s_" + self.scenario + self.num_antennas + ".png")
        plt.savefig(fig1_path)

        plt.figure()
        plt.scatter(self.df_positions["RoundedX"], self.df_positions["RoundedY"], label='Ground truth', s=3, alpha=0.7, c = 'blue')
        plt.scatter(self.df_positions["PredictedX"], self.df_positions["PredictedY"], label='Prediction', s=3, alpha=0.7, c = 'red')
        plt.xlabel('x position [mm]')
        plt.ylabel('y position [mm]')
        # plt.title('Comparación de Ruta Real y Ruta Predicha')
        plt.legend()
        plt.xlim(-1500, 1500)
        plt.ylim(1500, 4500)
        plt.grid(False)
        fig2_path = os.path.join(self.path, "graphic_l_" + self.scenario + self.num_antennas + ".png")
        plt.savefig(fig2_path)

    def generate_plots_tc3(self):
        plt.figure()
        plt.scatter(self.df_positions["RoundedX"], self.df_positions["RoundedY"], label='Ground truth', s=3, alpha=0.7, c = 'blue')
        plt.scatter(self.df_positions["PredictedX"], self.df_positions["PredictedY"], label='Prediction', s=3, alpha=0.7, c = 'red')
        plt.xlabel('x position [mm]')
        plt.ylabel('y position [mm]')
        # plt.title('Comparación de Ruta Real y Ruta Predicha')
        plt.legend()
        plt.xlim(-1250, -450)
        plt.ylim(2900, 3900)
        plt.grid(False)
        fig1_path = os.path.join(self.path, "graphic_s_" + self.scenario + self.num_antennas + ".png")
        plt.savefig(fig1_path)

        plt.figure()
        plt.scatter(self.df_positions["RoundedX"], self.df_positions["RoundedY"], label='Ground truth', s=3, alpha=0.7, c = 'blue')
        plt.scatter(self.df_positions["PredictedX"], self.df_positions["PredictedY"], label='Prediction', s=3, alpha=0.7, c = 'red')
        plt.xlabel('x position [mm]')
        plt.ylabel('y position [mm]')
        # plt.title('Comparación de Ruta Real y Ruta Predicha')
        plt.legend()
        plt.xlim(-1500, 1500)
        plt.ylim(1500, 4500)
        plt.grid(False)
        fig2_path = os.path.join(self.path, "graphic_l_" + self.scenario + self.num_antennas + ".png")
        plt.savefig(fig2_path)






