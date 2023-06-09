"""
Reads .db files produced by ChangeAbsPositionAz.tnhs script
Calculates intercept factor for each angle and absorber position
Plots heatmap for each angle
"""

import os
import glob
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#cwd = os.getcwd()
#expdir = cwd+'/out/'
#heatdir = expdir+'/out/heatmaps/'
#scatdir = expdir + '/scatters

#if not os.path.exists(heatdir):
#    os.makedirs(heatdir)



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


def load_dataset():
    dfr = pd.DataFrame(columns=['angle', 'x', 'y', 'g'])
    for i in dbfiles:
        angle = float(os.path.basename(i)[-6:-3])  # extract angle from filename
        x = float(os.path.basename(i)[:-13])  # extracts abs x position from filename
        y = float(os.path.basename(i)[-12:-7])
        photons, surfaces = view(i)
        aux_id = surfaces['id'][surfaces['Path'].str.contains("aux")].values[0] # Finds auxiliary surface id
        aux_hits = photons['surfaceID'].value_counts()[aux_id]
        try:
            absorber_id = surfaces['id'][surfaces["Path"].str.contains("Cyl_abs")].values[0]  # Finds absorber surface id
            abs_hits = photons['surfaceID'].value_counts()[absorber_id]
    
            g = abs_hits/aux_hits
            dfr = dfr.append({'angle': angle, 'x': x, 'y': y,'g': g}, ignore_index=True)
        except IndexError:
            print('No absorber surface in:',os.path.basename(i), ', skipping...')
            pass # Skips files where absorber surfaces are not exported by Tonatiuh
    dfr =  dfr.set_index(['angle','x','y'])
    return dfr


def create_dir(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)
        print("Created directory:",dirname)
    return dirname


def plot_heatmaps(df,savefigs = False):
    for k in angles:
        dfang = df.loc[k].reset_index().pivot('y', 'x', 'g')
        dfang = dfang.sort_values(by='y', ascending=False)
        ax = plt.axes()
        ax = sns.heatmap(dfang,ax=ax,vmin=0, vmax=max_g, 
                         cbar_kws={'label':'$\gamma_{th}$'},
                         xticklabels=10,yticklabels=50)
        ax.set_title(str(k)+' $\degree$')
        plt.tight_layout()
        if savefigs is True:
            folder = create_dir(os.getcwd()+'/plots/heatmaps/')
            plt.savefig(folder+str(k)+'.png')
        plt.show()
 
def plot_num_datapoints(df,gmin=0.5,savefig=False):
    for i in angles:
        an = df.loc[i].reset_index()
        anbest = an[an['g']>gmin]
        plt.scatter(i,anbest['g'].count(),c='b')
        plt.title('$\\gamma_{th}>$'+str(gmin))
        plt.ylabel('Count')
        plt.xlabel('$\\theta_{az} (°)$')
    if savefig is True:
        folder = create_dir(os.getcwd()+'/plots/scatters/count/')
        fname = str(gmin)+'.png'
        plt.savefig(folder+fname,dpi=300)
    plt.show()
    
    
def plot_hists(df,gmin=0.5,savefig=False):
    for i in angles:
        an = df.loc[i].reset_index()
        anbest = an[an['g']>gmin]
        plt.hist(anbest['g'],bins=30)
        plt.xlabel('$\\gamma_{th}$')
        plt.title('$\\gamma_{th}>$'+str(gmin)+', '+ str(i)+' °')
        plt.xlim(gmin,0.8)
        if savefig is True:
            folder = create_dir(os.getcwd()+'/plots/hists/'+str(gmin)+'/')
            plt.savefig(folder+str(i)+'.png',dpi=300)
        plt.show()



def plot_best_loc(df,gmin=0.5,savefig=False):
    for i in angles:
        an = df.loc[i].reset_index()
        dfbest = an[an['g']>gmin]
        plt.scatter(dfbest['x'],dfbest['y'])
        plt.xlim(-0.150,0.150)
        plt.ylim(0,0.300)
        plt.title('$\\gamma_{th}>$'+str(gmin)+', '+ str(i)+' °')
        plt.grid(True)
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        if savefig is True:
            folder = create_dir(os.getcwd()+'/plots/scatters/loc/'+str(gmin)+'/')
            fname = str(i)+'.png'
            plt.savefig(folder+fname,dpi=300)
        plt.show()


def plot_best_loc_median(df,gmin=0.5,savefig=False):
#    df =  df.set_index(['angle','x','y'])
    for i in angles:
        an = df.loc[i].reset_index()
        dfbest = an[an['g']>gmin]
        xC = dfbest['x'].median()
        yC = dfbest['y'].median()
        plt.scatter(xC,yC,c='b')
        plt.title('$\\gamma_{th}>$'+str(gmin))
        plt.grid(True)
        plt.ylabel('y (m)')
        plt.xlabel('x (m)')
        plt.xlim(-0.150,0.150)
        plt.ylim(0,0.300)
    if savefig is True:
        folder = create_dir(os.getcwd()+'/plots/scatters/median/')
        fname = folder+str(gmin)+'.png'
        plt.savefig(fname,dpi=300)
    plt.show()
 
    
def plot_best_loc_mean(df,gmin=0.5,savefig=False):
#    df =  df.set_index(['angle','x','y'])
    for i in angles:
        an = df.loc[i].reset_index()
        dfbest = an[an['g']>gmin]
        xC = dfbest['x'].mean()
        yC = dfbest['y'].mean()
        plt.scatter(xC,yC,c='b')
        plt.title('$\\gamma_{th}>$'+str(gmin))
        plt.grid(True)
        plt.ylabel('y (m)')
        plt.xlabel('x (m)')
        plt.xlim(-0.150,0.150)
        plt.ylim(0,0.300)
    if savefig is True:
        folder = create_dir(os.getcwd()+'/plots/scatters/mean/')
        fname = folder+str(gmin)+'.png'
        plt.savefig(fname,dpi=300)
    plt.show()




dbfiles = glob.glob(os.getcwd() + '/raw/*.db') # creates list of .db files in /raw1
    
dfr = load_dataset()


max_g = dfr['g'].max()

angles = dfr.index.get_level_values('angle').unique().to_list()
angles.sort()


plot_heatmaps(dfr,savefigs=True)

gmin_list = [p/10 for p in range(3, 9)] # gmin from 0.3 to 0.8
for i in gmin_list:
#    plot_num_datapoints(dfr,gmin = i,savefig=True)
#    plot_hists(dfr,gmin=i,savefig=True)
#    plot_best_loc(dfr,gmin=i,savefig=True)
#    plot_best_loc_median(dfr,gmin=i,savefig=True)
    plot_best_loc_mean(dfr, gmin=i,savefig=True)




    
