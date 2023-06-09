"""
Reads .db files produced by ChangeAbsPositionAz.tnhs script
Counts hits on absorber for each angle and position
Plots heatmap
Plots yposition - angle linear regression
"""

import os
import glob
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from tqdm import tqdm
import json


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
        pos, angle = os.path.basename(dbfile)[:-3].split("_")
        # pos = float(pos) * 1000  # converts abs position to mm
        photons, surfaces = read_db_file(dbfile)
        try:
            absorber_id = surfaces['id'][surfaces["Path"].str.contains("Cyl_abs")].values[0]  # Finds absorber id
            aux_id = surfaces['id'][surfaces["Path"].str.contains("aux")].values[0] # Finds auxiliary surface id
            abs_hits = photons['surfaceID'].value_counts()[absorber_id]
            aux_hits = photons['surfaceID'].value_counts()[aux_id]
            nj = abs_hits/aux_hits
            records.append({'angle': angle, 'position': pos, 'intercept factor': nj})
        except IndexError as e:
            print(dbfile, e)
    df = pd.DataFrame.from_records(records)
    return df

def save_regresults(reg_results, fname="out/linregress_stats.json"):
    linregress_dict = {
        "slope": reg_results.slope,
        "slope_stderr": reg_results.stderr,
        "intercept":reg_results.intercept,
        "intercept_stderr":reg_results.intercept_stderr,
        "r" : reg_results.rvalue,
        "pvalue" : reg_results.pvalue,
    }
    with open(fname, "w") as f:
        json.dump(linregress_dict, f, indent=4)

def plot_regression(df, side="left"):
    if side=="left":
        df1 = df.loc[:180] # selects angles up to 180°
    else:
        df1 = df.loc[180:]
    x = df1.index.values
    y = df1['position'].values.tolist()
    reg_results = stats.linregress(x,y)
    slope = reg_results.slope
    intercept = reg_results.intercept
    save_regresults(reg_results, fname=f"out/linregress_stats_{side}.json")
    fig, axes = plt.subplots()
    plt.plot(x,y,".", markersize="8")
    plt.plot(x, x*slope+intercept, "r", linewidth=3, label="$y={0:.4f}x {1:+.2f}$".format(slope, intercept))
    plt.xlabel(r"$\theta_{az} \ (\degree)$")
    plt.ylabel("$y \ (m)$")
    plt.legend()
    fname = f"plots/linear_fit_{side}.png"
    mkdir_if_not_exists(os.path.dirname(fname))
    plt.savefig(fname)
    plt.show()



def plot_heatmap(df, fname="plots/heatmap.png", title="$\gamma$"):
    # df1.reset_index(inplace=True)
    df1 = df.pivot("position", "angle", "intercept factor")
    ax = sns.heatmap(df1)
    cbar = ax.collections[0].colorbar
    # cbar.set_label(cbar_label, labelpad=30)
    cbar.ax.set_title(title)
    ax.invert_yaxis()
    ax.set_xlabel(r"$\theta_{az} \ (\degree)$")
    ax.set_ylabel(r"$y \ (m)$")
    # ax.set_title(title)
    ax.set_yticks([0, 135, 270])
    ax.set_yticklabels(["0.030", "0.165", "0.300"])
    ax.set_xticks([0, 45, 90])
    ax.set_xticklabels(["135", "180", "225"], rotation=0)
    mkdir_if_not_exists(os.path.dirname(fname))
    plt.savefig(fname)
    plt.show()


# dbfiles = glob.glob(os.getcwd() + '/raw/*.db') # creates list of .db files in /raw1
# df = read_db_files(dbfiles)
# df.to_csv("linear.csv", index=False)

df = pd.read_csv("data/linear.csv")

plot_heatmap(df)

dfl = df.sort_values('intercept factor', ascending=False).drop_duplicates(['angle'])
dfl.set_index("angle", inplace=True)
dfl.sort_index(inplace=True)

plot_regression(dfl, side="left")
plot_regression(dfl, side="right")

