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
from matplotlib import ticker

params = {
          'font.size': 14,
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
    ax.invert_yaxis()
    ax.set_xlabel(r"$\theta_{az}$")
    ax.set_ylabel(r"$\phi$")
    ax.set_title(title)
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
        f.write(polynomial_result.fit_report())
        
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

df = pd.read_csv("Radius_0.0825_shift_y_0.165.csv", index_col="azimuth")
# df = datareader.read_dir("Radius_0.0825_shift_y_0.165")
plot_heatmap(df, fname="pics/heatmap_intercept_factor_all_phi_az.png")

threshold = 0.7
filtered, selected = select_greater_than(df, threshold)
# plot_heatmap(filtered)
plot_heatmap(selected, fname="pics/filter07_simple.png", title="$\gamma > 0.7$")

# sns.scatterplot(data=selected, x='azimuth', y='phi')
# plt.plot(selected.index, selected["phi"], ".")
# plt.xlabel(r"$\theta_{az}")
# plt.ylabel("$\phi$")
# plt.title(f"$\gamma > {threshold}$")
# pic_path = "pics/intercept_07.png"
# mkdir_if_not_exists(os.path.dirname(pic_path))
# plt.savefig(pic_path)
# plt.show()

selected.reset_index(inplace=True)
x = selected["azimuth"]
y = selected["phi"]

linregres_results = regression_results(x, y)
slope = linregres_results.params[1]
intercept = linregres_results.params[0]

polynomial_result = lmfit_polynomial(x,y,polynomial)
lmfit_polynomial_to_txt(polynomial_result, fname="lmfit_polynomial_results.txt")
a,b,c,d = polynomial_result.values.values()

sm_poly, ypred = statsmodels_polynomial(x,y,20)

logit_result = lmfit_logit(x,y, logit_function)

plt.plot(x,y,'.')
plt.plot(x, intercept + slope*x, 'r', linewidth=3,label="Linear fit")
# plt.plot(x,ypred, label="20-degree polyunomial")
# plt.plot(x, logit_result.init_fit, 'b--', label="initial fit")
# plt.plot(x, logit_result.best_fit, label="logit best fit")
plt.plot(x, polynomial_result.best_fit, 'k--',linewidth=2, label="3rd degree polynomial")
plt.xlabel(r"$\theta_{az}$")
plt.ylabel("$\phi$")
plt.title(f"$\gamma > {threshold}$")
plt.legend()
pic_path = "pics/fits.png"
mkdir_if_not_exists(os.path.dirname(pic_path))
plt.tight_layout()
plt.savefig(pic_path)
plt.show()


# linreg = datareader.read_dir("linear_regression")
# sns.lineplot(data=linreg, x="azimuth", y="intercept_factor")
# plt.title("Linear regression")
# pic_path = "pics/linear_regression_validation.png"
# mkdir_if_not_exists(os.path.dirname(pic_path))
# plt.savefig(pic_path)
# plt.show()

# polyn = datareader.read_dir("polynomial")
# sns.lineplot(data=polyn, x="azimuth", y="intercept_factor")
# plt.title("3rd degree polynomial")
# pic_path = "pics/3deg_polynomial_validation.png"
# mkdir_if_not_exists(os.path.dirname(pic_path))
# plt.savefig(pic_path)
# plt.show()
