import struct
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse

records = 0
parser = argparse.ArgumentParser()
parser.add_argument('--file','-f', help='archivos de datos crudos')
args = parser.parse_args()
with open("../datos/20201027_133329_data.raw","rb") as f:
    fecha, hora, resto = f.name.split('_')
    print(fecha,hora)
    dt = np.dtype('d')
    dt = dt.newbyteorder('>')
    data = f.read()
    np_data = np.frombuffer(data,dt)
    np_data.resize(5000,2048)
#    np_data.resize(2048,24358)
    np_data = np.transpose(np_data)
    df = pd.DataFrame(np_data)
#    print(np_data)
    print(df.size)
    print(df.shape)
    print(df.ndim)
    print(df[0])
    plt.plot(df[0])
    plt.yscale('log')
    plt.show()

print("End")
