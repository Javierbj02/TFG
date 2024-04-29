
In this folder, the data sets (.csv) containing the measurements for each of the 252004 positions are stored. We have 3 different scenarios (ULA, URA and DIS) in which the antenna distribution varies. For each scenario, we have created 4 datasets which differ in the number of antennas selected. Specifically, we have for each scenario a dataset with 8 antennas, 16 antennas, 32 antennas and finally all antennas: 64.

The measurements are originally expressed in complex numbers due to the naturalness of CSI. For our processing with the neural networks, these .csv have been created by passing the data from complex numbers to polar form.

For a data set of 64 antennas, therefore, we will have 252004 rows (each row represents a position) and 64 (antennas) x 100 (subcarriers per antenna) x 2 (polar form), plus labeled values, one for the X position and one for the Y position.

Due to the large size of these data sets, they have been included in the .gitignore. In the “Examples” folder, we can see some examples of these data sets with 3 rows.