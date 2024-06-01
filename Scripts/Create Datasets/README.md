# Sobre la creación de los Datasets

El script "generate_scenarios.py" crea los datasets en formato .csv con los que trabajaremos. Para crear estos datasets, usamos los datos de las mediciones que se recogieron. Tenemos tres distintas carpetas de datos: DIS, ULA y URA, que se diferencian por la posición de las antenas. 

Dentro de cada carpeta, encontramos un .npy que contiene las posiciones en el punto de coordenadas de las antenas y otro .npy con la posición del "usuario". Esta posición del "usuario" hace refencia a las coordenadas en el punto de coordenadas en el que se hace cada medición. Además, dentro de cada configuración, encontramos la carpeta de "samples", que contiene los .npy con las mediciones CSI de cada posición. Al tener 252004 posiciones distintas, dentro de "samples" encontramos 252004 ficheros .npy con las mediciones CSI, cada uno definiendo una matriz de 64x100, de acuerdo con los datos en números complejos recogidos de las 100 subportadoras de cada una de las 64 antenas. El fichero de "antenna_positions.py" representa una matriz de 64x3, de acuerdo con las 64 antenas y la posición en coordenadas de cada una (x, y, z). El fichero de "user_positions.npy" representa una matriz de 252004x3, de acuerdo con las coordenadas (x, y, z) de cada una de las posiciones sobre las que se realizan mediciones.

En este repositorio, solo se ha incluido los .npy de los datos de las primeras posiciones, ya que los datos totales tienen demasiado tamaño.

En cuanto al fichero "generate_scenarios.py", este utiliza estos datos para crear los datasets .csv. En primer lugar, convertimos los datos de números complejos a forma polar. En segundo lugar, se seleccionan el número de antenas que se tendrán en cuenta para crear el dataset: 8, 16, 32 y 64 antenas, de forma que cuantas menos antenas, menos datos habrá y menos tamaño tendrá el .csv.

En definitiva, contamos con 12 datasets en total: 3 configuraciones x 4 números de antenas. Los tres datasets con 64 antenas cuentan con 252004 filas (todas las posiciones sobre las que se han realizado mediciones) x 12800 columnas (64 antenas x 100 subportadoras x 2 (forma polar)).

# Tamaño de los Datasets:
8 antenas: 7,7GB cada uno
16 antenas: 15,4GB cada uno
32 antenas: 30,8GB cada uno
64 antenas: 61.6GB cada uno

Debido a estos tamaños, no es posible guardar los datasets en el repositorio. No obstante, en /Data/Examples, se han guardado algunos .csv de ejemplo con las tres primeras posiciones nada más.

----------------------------------------------------------------------------------------------------------

# About the creation of the Datasets

The script "generate_scenarios.py" creates the datasets in .csv format that we will work with. To create these datasets, we use the measurement data that was collected. We have three different data folders: DIS, ULA and URA, which are differentiated by the position of the antennas. 

Within each folder, we find an .npy containing the coordinate point positions of the antennas and another .npy with the position of the "user". This "user" position refers to the coordinates at the coordinate point where each measurement is made. In addition, within each configuration, we find the "samples" folder, which contains the .npy with the CSI measurements of each position. As we have 252004 different positions, inside "samples" we find 252004 .npy files with the CSI measurements, each one defining a 64x100 matrix, according to the data in complex numbers collected from the 100 subcarriers of each of the 64 antennas. The "antenna_positions.py" file represents a 64x3 matrix, according to the 64 antennas and the coordinate position of each antenna (x, y, z). The "user_positions.npy" file represents a matrix of 252004x3, according to the coordinates (x, y, z) of each of the positions over which measurements are made.

In this repository, only the .npy of the data of the first positions has been included, as the total data is too big.

As for the "generate_scenarios.py" file, it uses this data to create the .csv datasets. First, we convert the data from complex numbers to polar form. Secondly, we select the number of antennas to be taken into account to create the dataset: 8, 16, 32 and 64 antennas, so the fewer antennas, the less data there will be and the smaller the .csv dataset will be.

In short, we have 12 datasets in total: 3 configurations x 4 numbers of antennas. The three datasets with 64 antennas have 252004 rows (all measured positions) x 12800 columns (64 antennas x 100 subcarriers x 2 (polar shape)).

# Size of Datasets:

8 antennae: 7.7GB each
16 antennas: 15.4GB each
32 antennas: 30.8GB each
64 antennas: 61.6GB each

Due to these sizes, it is not possible to store the datasets in the repository. However, in /Data/Examples, some example .csv have been saved with only the first three positions.
