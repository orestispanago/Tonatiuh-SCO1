"""
Reads .db files produced by ChangeAbsPositionAz.tnhs script
Calculates intercept factor for each angle and absorber position
Plots heatmap for each angle
"""

import os
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from datareader import read_dir
import matplotlib.patches as patches

#cwd = os.getcwd()
#expdir = cwd+'/out/'
#heatdir = expdir+'/out/heatmaps/'
#scatdir = expdir + '/scatters

#if not os.path.exists(heatdir):
#    os.makedirs(heatdir)


params = {
          'font.size': 14,
          # 'figure.constrained_layout.use': True,
           'savefig.dpi': 200.0,
          }
plt.rcParams.update(params)

def view(dbname):
    """ Reads .db files
    :type dbname: str
    :param str dbname: .db file path
    :returns: contents of Photons and Surfaces tables as a tuple of dataframes
    """
    conn = sqlite3.connect(dbname)
    df = pd.read_sql_query("SELECT * FROM Photons", con=conn)
    ids = pd.read_sql_query("SELECT * FROM Surfaces", con=conn)
    conn.close()
    return df, ids


def create_dir(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)
        print("Created directory:",dirname)
    return dirname


def plot_heatmaps(df,savefigs = False):
    for k in angles:
        dfang = df.loc[k].reset_index().pivot('y', 'x', 'g')
        ax = sns.heatmap(dfang,vmin=0, vmax=1)
        ax.invert_yaxis()
        ax.set_ylabel("$y \ (m)$")
        ax.set_xlabel("$x \ (m)$")
        ax.set_yticks([0, len(dfang)/2, len(dfang)])
        ax.set_yticklabels(["0.030", "0.165", "0.300"], rotation=0)
        ax.set_xticks([0, 50, 100])
        ax.set_xticklabels([-0.1, -0.05, 0], rotation=0)
        cbar = ax.collections[0].colorbar
        cbar.ax.set_title('$\gamma$')
        title=r'$\theta_{az} = $'+f'{k:.0f}'+'$\degree$'
        ax.set_title(title)
        plt.tight_layout()
        fname = f"plots/heatmaps/{k:.0f}.png"
        if savefigs is True:
            create_dir(os.path.dirname(fname))
            plt.savefig(fname)
        plt.show()
 
def plot_hist(df,gmin=0.5,savefig=False):
    gt = df[df["g"]>gmin]
    gt=gt.groupby(gt.index)['g'].count()
    plt.bar(gt.index, gt.values)
    plt.title(f'$\gamma > {gmin}$')
    plt.ylabel('Data points')
    plt.xlabel(r'$\theta_{az} \ (\degree)$')
    plt.tight_layout()
    fname = f"plots/hists/{gmin}.png"
    if savefig is True:
        create_dir(os.path.dirname(fname))
        plt.savefig(fname)
    plt.show()
    


def plot_best_loc(df,gmin=0.5,savefig=False):
    for i in angles:
        an = df.loc[i].reset_index()
        dfbest = an[an['g']>gmin]
        plt.scatter(dfbest['x'],dfbest['y'])
        plt.xlim(-0.150,0.150)
        plt.ylim(0,0.300)
        title=f'$\gamma > {gmin}, $'+r'$\theta_{az} = $'+f'${i:.0f}\degree$'
        plt.title(title)
        plt.grid(True)
        plt.ylabel("$y \ (m)$")
        plt.xlabel("$x \ (m)$")
        plt.tight_layout()
        fname = f"plots/scatters/loc/{gmin}/{i:.0f}.png"
        if savefig is True:
            create_dir(os.path.dirname(fname))
            plt.savefig(fname)
        plt.show()


def plot_best_loc_median(df,gmin=0.5,savefig=False):
#    df =  df.set_index(['angle','x','y'])
    for i in angles:
        an = df.loc[i].reset_index()
        dfbest = an[an['g']>gmin]
        xC = dfbest['x'].median()
        yC = dfbest['y'].median()
        plt.scatter(xC,yC,c='b')
        plt.title('$\\gamma >$'+str(gmin))
        plt.grid(True)
        plt.ylabel('y (m)')
        plt.xlabel('x (m)')
        plt.xlim(-0.150,0.150)
        plt.ylim(0,0.300)
    if savefig is True:
        folder = create_dir(os.getcwd()+'/plots/scatters/median/')
        fname = folder+str(gmin)+'.png'
        plt.savefig(fname)
    plt.show()
 
    
def plot_best_loc_mean(df,gmin=0.5,savefig=False):
#    df =  df.set_index(['angle','x','y'])
    for i in angles:
        an = df.loc[i].reset_index()
        dfbest = an[an['g']>gmin]
        xC = dfbest['x'].mean()
        yC = dfbest['y'].mean()
        plt.scatter(xC,yC,c='C0')
        plt.title(f'$\\gamma > {gmin}$')
        plt.grid(True)
        plt.ylabel("$y \ (m)$")
        plt.xlabel("$x \ (m)$")
        plt.xlim(-0.150,0.150)
        plt.ylim(0,0.300)
        plt.tight_layout()
    if savefig is True:
        fname = f"plots/scatters/mean/{gmin}.png"
        create_dir(os.path.dirname(fname))
        plt.savefig(fname)
    plt.show()




# dbfiles = glob.glob(os.getcwd() + '/raw/*.db') # creates list of .db files in /raw1
    
# dfr = read_dir("raw")
df = pd.read_csv("data.csv", index_col="azimuth")

# df_right = df_left.copy()
# df_right["x"]*=-1

# df = pd.concat([df_left, df_right])
max_g = df['g'].max()

angles = df.index.get_level_values('azimuth').unique().to_list()
angles.sort()


# plot_heatmaps(df,savefigs=True)
# plot_best_loc(df,gmin=0.7,savefig=True)

gmin_list = [p/10 for p in range(3, 9)] # gmin from 0.3 to 0.8
for i in gmin_list:
    plot_hist(df,gmin=i, savefig=True)
    # plot_best_loc(df,gmin=i,savefig=True)
    # plot_best_loc_median(df,gmin=i,savefig=True)
    # plot_best_loc_mean(df, gmin=i,savefig=True)
