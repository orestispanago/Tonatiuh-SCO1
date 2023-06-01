import os
import glob
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt


dbfiles = glob.glob(os.getcwd() + '/raw1/*.db')


def view(dbname):
    conn = sqlite3.connect(dbname)
    df = pd.read_sql_query("SELECT * FROM Photons",con=conn)
    ids = pd.read_sql_query("SELECT * FROM Surfaces",con=conn)
    conn.close()
    return df,ids

hitlist = []
for i in dbfiles:
    photons,surfaces=view(i)
    angle = int(os.path.split(i)[1][:-3])
    hit = photons['surfaceID'].value_counts().rename(angle)
    absorber_id= surfaces['id'][surfaces["Path"].str.contains("Cyl_abs")].values[0]
    aux_id = surfaces['id'][surfaces["Path"].str.contains("aux")].values[0]
    hit['AbsID'] = absorber_id
    hit['AuxID'] = aux_id
    hit['nj'] = int((hit[absorber_id] / hit[aux_id])*100)
    hitlist.append(hit)

hits = pd.concat(hitlist,axis=1,sort=True)

hits = hits.T # Transposes dataframe


plt.plot(hits['nj'])
plt.xlabel('$\\theta_{az}(\degree)$')
#plt.title('Compound parabolic - cyl abs')
plt.ylabel('$\gamma_{th}$ (%)')
plt.ylim(0,110)
plt.tight_layout()
plt.savefig('linregress_equation_test2.png',dpi=300)
plt.show()


