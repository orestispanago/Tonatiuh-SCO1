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
import statsmodels.formula.api as sm
from statsmodels.api import add_constant
from datareader import read_dir
import json
import scipy.stats as stats


params = {
          'font.size': 14,
          # 'figure.constrained_layout.use': True,
           'savefig.dpi': 200.0,
          }
plt.rcParams.update(params)


def mkdir_if_not_exists(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)
        print("Created dir:", dirname)
        
def plot_heatmap(df, fname="plots/heatmap.png", title="$\gamma$"):
    dfr = df.pivot("position", "angle", "intercept_factor")
    ax = sns.heatmap(dfr)
    cbar = ax.collections[0].colorbar
    # cbar.set_label(cbar_label, labelpad=30)
    cbar.ax.set_title(title)
    ax.invert_yaxis()
    ax.set_xlabel(r"$\theta_{az} \ (\degree)$")
    ax.set_ylabel(r"$y \ (m)$")
    ymin=dfr.index[0]
    ymax=dfr.index[-1]
    ymid = dfr.index[int(len(dfr)/2)]
    ax.set_yticks([0, len(dfr)/2, len(dfr)])
    ax.set_yticklabels([ymin, f"{ymid:.3f}", ymax])
    ax.set_xticks([0, 45, 90])
    ax.set_xticklabels(["135", "180", "225"], rotation=0)
    mkdir_if_not_exists(os.path.dirname(fname))
    plt.tight_layout()
    plt.savefig(fname)
    plt.show()

def save_regresults(reg_results, fname="out/linregress_stats.json"):
    linregress_dict = {
        "slope": reg_results.slope,
        "slope_stderr": reg_results.stderr,
        "intercept":reg_results.intercept,
        "intercept_stderr":reg_results.intercept_stderr,
        "r" : reg_results.rvalue,
        "pvalue" : reg_results.pvalue,
    }
    mkdir_if_not_exists(os.path.dirname(fname))
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
    plt.tight_layout()
    plt.savefig(fname)
    plt.show()



# dfr = read_dir("raw")
df = pd.read_csv("data/linear.csv")
plot_heatmap(df)


dfl = df.sort_values('intercept_factor', ascending=False).drop_duplicates(['angle'])
dfl.set_index("angle", inplace=True)
dfl.sort_index(inplace=True)

plot_regression(dfl, side="left")
plot_regression(dfl, side="right")

