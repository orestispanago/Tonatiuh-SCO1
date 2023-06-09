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
    base_name = os.path.basename(fname)[:-3]
    pos, angle = base_name.split("_")[:]
    return float(pos), float(angle)


def angles_to_dict(fname, data_dict):
    pos, angle = parse_filename(fname)
    data_dict["angle"].append(angle)
    data_dict["position"].append(pos)

def read_dir(folder):
    dbfiles = glob.glob(os.getcwd() + f'/{folder}/*.db')
    dbfiles.sort()
    data_dict = {
        "sun": [],
        "absorber": [],
        "aux": [],
        "angle": [],
        "position":[]
    }
    sql_script = read_sql_script('select_fluxes.sql')
    for fname in tqdm(dbfiles):
        fluxes_to_dict(fname, sql_script, data_dict)
        angles_to_dict(fname, data_dict)
    df = pd.DataFrame(data_dict)
    df["intercept_factor"] = df["absorber"] / df["aux"]
    df = df.set_index("angle")
    df.sort_index(inplace=True)
    return df
