import pandas as pd
import numpy as np

# Data frame should contain the water level for three days and delta values too

file_path = "D:\Revan\RhineWaterLevel-master\RhineWaterLevel-master"

prediction = pd.read_csv(file_path + "/team9submission.csv")

delta = prediction.delta
predictions = prediction.water_level

counter = 1
y_1,y_2,y_3 = [],[],[]

for delta_values in delta :
    if counter == 1:
        y_1.append(delta_values)
        counter += 1
    elif counter == 2:
        y_2.append(delta_values)
        counter += 1
    elif counter == 3:
        y_3.append(delta_values)
        counter += 1
    else:
        y_1.append(delta_values)
        counter = 2

y_1 = np.array(y_1)
y_2 = np.array(y_2)
y_3 = np.array(y_3)

no_of_rows = len(y_1) + len(y_2) + len(y_3)

y_mean = ( y_1.sum() + y_2.sum() + y_3.sum()) / no_of_rows

y_1_pred, y_2_pred, y_3_pred = [], [], []

counter = 1
for prediction in predictions :
    if counter == 1:
        y_1_pred.append(prediction)
        counter += 1
    elif counter == 2:
        y_2_pred.append(prediction)
        counter += 1
    elif counter == 3:
        y_3_pred.append(prediction)
        counter += 1
    else:
        y_1_pred.append(prediction)
        counter = 2

a, b, c, d, e, f = [], [], [], [], [], []

for i in range(len(y_1_pred)):
    a.append(( y_1[i] - y_1_pred[i]) ** 2)
    b.append((y_2[i] - y_2_pred[i]) ** 2)
    c.append((y_3[i] - y_3_pred[i]) ** 2)
    d.append((y_1[i] - y_mean) ** 2)
    e.append((y_2[i] - y_mean) ** 2)
    f.append((y_3[i] - y_mean) ** 2)

a = np.array(a)
b = np.array(b)
c = np.array(c)
d = np.array(d)
e = np.array(e)
f = np.array(f)

r_score = 1 - ((a.sum() + b.sum() + c.sum()) / (d.sum() + e.sum() + f.sum()))

print(r_score)


