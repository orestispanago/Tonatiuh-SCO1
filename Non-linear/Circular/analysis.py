import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import datareader
import lmfit
import statsmodels.api as sm
from statsmodels.api import add_constant
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
import os
import json


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

def select_semicircles(df):
    df = df.reset_index()
    az = df["azimuth"]
    phi = df["phi"]
    left_semicircle = (az < 180) & (phi < 180)
    right_semicircle = (az >= 180) & (phi >= 180)
    dfout = df.loc[left_semicircle | right_semicircle]
    dfout = dfout.set_index("azimuth")
    return dfout

def plot_heatmap(df, fname="pics/heatmap.png", title="$\gamma$"):
    df1 = df[["phi", "intercept_factor"]]
    df1.reset_index(inplace=True)
    df1 = df1.pivot("phi", "azimuth", "intercept_factor")
    ax = sns.heatmap(df1)
    cbar = ax.collections[0].colorbar
    # cbar.set_label(cbar_label, labelpad=30)
    cbar.ax.set_title(title)
    ax.invert_yaxis()
    ax.set_xlabel(r"$\theta_{az} \ (\degree)$")
    ax.set_ylabel(r"$\phi \ (\degree)$")
    ax.set_yticks([0, 180, 360])
    ax.set_yticklabels(["0", "180", "360"])
    ax.set_xticks([0, 45, 90])
    ax.set_xticklabels(["135", "180", "225"], rotation=0)
    mkdir_if_not_exists(os.path.dirname(fname))
    plt.tight_layout()
    plt.savefig(fname)
    plt.show()

def regression_results(x,y):
    X = add_constant(x)  # include constant (intercept) in ols model
    mod = sm.OLS(y, X)
    return mod.fit()

def polynomial(x, a, b, c,d):
    return a * x** 3 + b * x**2 + c*x +d 

def lmfit_polynomial(x,y, func):
    model = lmfit.models.Model(func)
    params = model.make_params(a=0, b=0, c=40,  d=-3000)
    return model.fit(y, params, x=x)

def lmfit_polynomial_to_txt(lmfit_result, fname="lmfit_polynomial_results.txt"):
    with open(fname, "w") as f:
        f.write(lmfit_result.fit_report())
        
def statsmodels_polynomial(x,y, degree):
    polynomial_features= PolynomialFeatures(degree=degree)
    xp = polynomial_features.fit_transform(x.values.reshape(-1,1))
    model = sm.OLS(y, xp)
    result = model.fit()
    return result, result.predict(xp)

def logit_function(x, a, b, c,d):
    """ logit function is the inverse sigmoid: log(x/(1-x)), 0<x<1
    added c.d to normalize data """
    x_norm = (x-d)/c
    logit =  np.log(x_norm/(1-x_norm))
    return a*logit +b

def lmfit_logit(x,y, func):    
    model = lmfit.models.Model(func)
    params = model.make_params(a=35.4, b=170, c=91, d=134.4)
    return model.fit(y, params, x=x)

def select_greater_than(df, value):    
    filtered = df.loc[df['intercept_factor'] >= value]
    selected = select_semicircles(filtered)
    return filtered, selected

def select_max(df):
    df1 = df.reset_index()
    maxes = df1.loc[df1.groupby('azimuth')['intercept_factor'].idxmax()][
        ['azimuth','phi','intercept_factor']]
    return maxes.set_index("azimuth")


def save_regresults(results, fname="linregress_stats.json"):
    linregress_dict = {
        "slope": results.params[1],
        "slope_stderr": results.bse.values[1],
        "intercept":results.params[0],
        "intercept_stderr":results.bse.values[0],
        "r2" : results.rsquared,
        "pvalue" : results.pvalues.values[0],
    }
    print(json.dumps(linregress_dict, indent=4))
    with open(fname, "w") as f:
        json.dump(linregress_dict, f, indent=4)
        
def plot_fits(x,y, threshold = "threshold"): 
    linregres_results = regression_results(x, y)
    save_regresults(linregres_results, fname="out/linregress_stats.json")
    slope = linregres_results.params[1]
    intercept = linregres_results.params[0]
    
    polynomial_result = lmfit_polynomial(x,y,polynomial)
    lmfit_polynomial_to_txt(polynomial_result, fname="out/lmfit_polynomial_results.txt")
    
    sm_poly, ypred = statsmodels_polynomial(x,y,20)
    logit_result = lmfit_logit(x,y, logit_function)
    
    plt.plot(x,y,'.')
    plt.plot(x, intercept + slope*x, 'r', linewidth=3,label="Linear fit")
    # plt.plot(x,ypred, label="20-degree polyunomial")
    # plt.plot(x, logit_result.init_fit, 'b--', label="initial fit")
    # plt.plot(x, logit_result.best_fit, label="logit best fit")
    plt.plot(x, polynomial_result.best_fit, 'k--',linewidth=2, label="3rd degree polynomial")
    plt.xlabel(r"$\theta_{az} \ (\degree)$")
    plt.ylabel("$\phi \ (\degree)$")
    plt.title(f"$\gamma > {threshold}$")
    plt.legend()
    pic_path = "pics/fits.png"
    mkdir_if_not_exists(os.path.dirname(pic_path))
    plt.tight_layout()
    plt.savefig(pic_path)
    plt.show()
    
    
df = pd.read_csv("data/equation-calculation/Radius_0.0825_shift_y_0.165.csv", index_col="azimuth")
# df = datareader.read_dir("Radius_0.0825_shift_y_0.165")
plot_heatmap(df, fname="pics/heatmap_intercept_factor_all_phi_az.png")

threshold = 0.7
filtered, selected = select_greater_than(df, threshold)
# plot_heatmap(filtered)
# plot_heatmap(selected, fname="pics/filter07_simple.png", title="$\gamma > 0.7$")
selected.reset_index(inplace=True)
x = selected["azimuth"]
y = selected["phi"]

plot_fits(x,y, threshold = 0.7)

# lin_linear = pd.read_csv("data/equation-validation/linear_linear_regression_10000rays.csv", index_col="angle")
# lin_linear = lin_linear.groupby(lin_linear.index // 1).mean()


# # circ_linear = datareader.read_dir("data/equation-validation/raw/linear_regression")
# circ_linear = pd.read_csv("data/equation-validation/circular_linear_regression_1000rays.csv", index_col="azimuth")
# circ_linear = circ_linear.groupby(circ_linear.index // 1).mean()

# # circ_polyn = datareader.read_dir("data/equation-validation/raw/polynomial")
# circ_polyn = pd.read_csv("data/equation-validation/circular_polynomial_regression_1000rays.csv", index_col="azimuth")
# circ_polyn = circ_polyn.groupby(circ_polyn.index // 1).mean()


# fig = plt.figure(figsize=(6.4, 4.8))
# plt.plot(lin_linear.index, lin_linear["intercept factor"],"--", linewidth=3, label="Linear path - linear equation")
# plt.plot(circ_linear.index, circ_linear["intercept_factor"], linewidth=3, label="Circular path - linear equation")
# plt.plot(circ_polyn.index, circ_polyn["intercept_factor"], linewidth=3, label="Circular path - 3rd degree polynomial")
# plt.xlabel(r"$\theta_{az} \ (\degree)$")
# plt.ylabel("$\gamma$")
# plt.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
#                 mode="expand", borderaxespad=0)
# pic_path = "pics/all_paths_and_equations.png"
# mkdir_if_not_exists(os.path.dirname(pic_path))
# plt.tight_layout()
# plt.savefig(pic_path)
# plt.show()
