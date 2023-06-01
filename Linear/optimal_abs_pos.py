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


params = {
          'font.size': 14,
          'figure.constrained_layout.use': True,
           'savefig.dpi': 200.0,
          }
plt.rcParams.update(params)

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

def read_db_files(files):
    dfr = pd.DataFrame(columns=['angle', 'position', 'intercept factor'])
    for i in dbfiles:
        angle = os.path.basename(i)[5:-3]  # extract angle from filename
        pos = os.path.basename(i)[:4]  # extracts abs position from filename
        pos = float(pos) * 1000  # converts abs position to mm
        photons, surfaces = read_db_file(i)
        try:
            absorber_id = surfaces['id'][surfaces["Path"].str.contains("Cyl_abs")].values[0]  # Finds absorber id
            aux_id = surfaces['id'][surfaces["Path"].str.contains("aux")].values[0] # Finds auxiliary surface id
            abs_hits = photons['surfaceID'].value_counts()[absorber_id]
            aux_hits = photons['surfaceID'].value_counts()[aux_id]
            nj = 100*abs_hits/aux_hits
            dfr = dfr.append({'angle': angle, 'position': pos, 'intercept factor': nj},
                             ignore_index=True)
        except IndexError as e:
            print(e)
    dfr = dfr.astype(int)
    dfr = dfr.pivot('position', 'angle', 'intercept factor')
    dfr = dfr.sort_values(by='position', ascending=False)
    return dfr

def plot_regression(df, side="left"):
    if side=="left":
        df1 = df.loc[:180] # selects angles up to 180°
    else:
        df1 = df.loc[180:]
    x = df1.index.values
    y = df1['pos'].values.tolist()
    slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
    
    fig, axes = plt.subplots()
    plt.plot(x,y,".", markersize="8")
    plt.plot(x, x*slope+intercept, "r", linewidth=3, label="$y={0:.3f}x {1:+.1f}$".format(slope, intercept))
    plt.xlabel(r"$\theta_{az} \ (\degree)$")
    plt.ylabel("$y \ (m)$")
    # g = sns.regplot(x = x,y=y,line_kws={'label': "$y={0:.3f}x+{1:.1f}$".format(slope, intercept)})
    # plt.legend()
    plt.savefig(f"linear_fit_{side}.png")
    plt.show()

dbfiles = glob.glob(os.getcwd() + '/raw1/*.db') # creates list of .db files in /raw1

dfr = read_db_files(dbfiles)

ax = sns.heatmap(dfr,cbar_kws={'label': '$\gamma$'})

# Selects max intercept factor values
dfl = pd.DataFrame(columns=['angle','pos','maxnj'])
for j in list(dfr):
    max_nj = dfr[j].nlargest(1).values[0]
    bestpos = dfr[j].nlargest(1).index.values[0]/1000
    dfl = dfl.append({'angle':j,'pos':bestpos, 'maxnj': max_nj},ignore_index=True)
dfl = dfl.set_index('angle',drop=True)


plot_regression(dfl, side="left")
plot_regression(dfl, side="right")
