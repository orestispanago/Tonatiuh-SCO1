import os
import glob
import sqlite3
import pandas as pd
from tqdm import tqdm

def read_sql_script(script_name):
    with open(script_name, 'r') as f:
        sql_script = f.read()
    return sql_script

def select_fluxes(dbname, sql_script):
    """ Gets flux on surface (in Watts) from .db file"""
    conn = sqlite3.connect(dbname)
    cursor = conn.cursor()
    cursor.execute(sql_script)
    values = cursor.fetchall()[0]
    conn.close()
    return values


def fluxes_to_dict(fname, sql_script, data_dict):
    sun, absorber, aux = select_fluxes(fname, sql_script)
    data_dict["sun"].append(sun),
    data_dict["absorber"].append(absorber)
    data_dict["aux"].append(aux)


def parse_filename(fname):
    x, y, az = os.path.basename(fname)[:-3].split("_")
    return float(x), float(y), float(az)


def angles_to_dict(fname, data_dict):
    x, y, az = parse_filename(fname)
    data_dict["azimuth"].append(az)
    data_dict["x"].append(x)
    data_dict["y"].append(y)

def read_dir(folder):
    dbfiles = glob.glob(os.getcwd() + f'/{folder}/*.db')
    dbfiles.sort()
    data_dict = {
        "sun": [],
        "absorber": [],
        "aux": [],
        "azimuth": [],
        "x" : [],
        "y" : []
    }
    sql_script = read_sql_script('select_fluxes.sql')
    for fname in tqdm(dbfiles):
        fluxes_to_dict(fname, sql_script, data_dict)
        angles_to_dict(fname, data_dict)
    df = pd.DataFrame(data_dict)
    df["g"] = df["absorber"] / df["aux"]
    df = df.set_index("azimuth")
    df.sort_index(inplace=True)
    return df
