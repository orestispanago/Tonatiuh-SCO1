import os
import glob
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

params = {
          'font.size': 14,
          'figure.constrained_layout.use': True,
           'savefig.dpi': 200.0,
          }
plt.rcParams.update(params)

def mkdir_if_not_exists(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)
        print("Created dir:", dirname)

def read_db_file(dbname):
    """ Reads .db files
    :type dbname: str
    :param str dbname: .db file path
    :returns: contents of Photons and Surfaces tables
    """
    conn = sqlite3.connect(dbname)
    df = pd.read_sql_query("SELECT * FROM Photons", con=conn)
    ids = pd.read_sql_query("SELECT * FROM Surfaces", con=conn)
    conn.close()
    return df, ids

def read_db_files(dbfiles):
    records = []
    for dbfile in tqdm(dbfiles):
        angle = float(os.path.basename(dbfile)[:-3])
        # pos = float(pos) * 1000  # converts abs position to mm
        photons, surfaces = read_db_file(dbfile)
        try:
            absorber_id = surfaces['id'][surfaces["Path"].str.contains("Cyl_abs")].values[0]  # Finds absorber id
            aux_id = surfaces['id'][surfaces["Path"].str.contains("aux")].values[0] # Finds auxiliary surface id
            abs_hits = photons['surfaceID'].value_counts()[absorber_id]
            aux_hits = photons['surfaceID'].value_counts()[aux_id]
            nj = abs_hits/aux_hits
            records.append({'angle': angle, 'intercept factor': nj})
        except IndexError as e:
            print(dbfile, e)
    df = pd.DataFrame.from_records(records)
    return df


def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    from math import factorial
    try:
        window_size = np.abs(int(window_size))
        order = np.abs(int(order))
    except ValueError:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')


def plot_equation_test(df):
    x = df["angle"].values
    y = df['intercept factor'].values
    plt.plot(x,y, linewidth=4)
    # yhat = savitzky_golay(y, 51, 3) # window size 51, polynomial order 3
    plt.plot(x[8:-8], smooth(y,6)[8:-8], linewidth=3, label="Discrete Linear Convolution")
    # plt.plot(x,yhat, color='red', linewidth=3, label="Savitzky-Golay filter")
    plt.legend()
    plt.xlabel('$\\theta_{az} \ (\degree)$')
    plt.ylabel('$\gamma$')
    plt.savefig("plots/equation_test.png")
    plt.show()

# dbfiles = glob.glob(os.getcwd() + '/raweq/*.db')

# df = read_db_files(dbfiles)
df = pd.read_csv("equation_test_10000rays.csv")
df = df.groupby(df.index // 10).mean()
plot_equation_test(df)

