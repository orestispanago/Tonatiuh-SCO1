import matplotlib.pyplot as plt
import seaborn as sns
import datareader
import lmfit
import statsmodels.api as sm
from statsmodels.api import add_constant
from sklearn.preprocessing import PolynomialFeatures


def select_semicircles(df):
    df = df.reset_index()
    az = df["azimuth"]
    phi = df["phi"]
    left_semicircle = (az < 180) & (phi < 180)
    right_semicircle = (az >= 180) & (phi >= 180)
    dfout = df.loc[left_semicircle | right_semicircle]
    dfout = dfout.set_index("azimuth")
    return dfout

def plot_heatmap(df):
    df1 = df[["phi", "intercept_factor"]]
    df1.reset_index(inplace=True)
    df1 = df1.pivot("phi", "azimuth", "intercept_factor")
    sns.heatmap(df1)
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

def statsmodels_polynomial(x,y, degree):
    polynomial_features= PolynomialFeatures(degree=degree)
    xp = polynomial_features.fit_transform(x.values.reshape(-1,1))
    model = sm.OLS(y, xp)
    result = model.fit()
    return result, result.predict(xp)

def select_greater_than(df, value):    
    filtered = df.loc[df['intercept_factor'] >= value]
    selected = select_semicircles(filtered)
    return filtered, selected

def select_max(df):
    df1 = df.reset_index()
    maxes = df1.loc[df1.groupby('azimuth')['intercept_factor'].idxmax()][
        ['azimuth','phi','intercept_factor']]
    return maxes.set_index("azimuth")

# sns.set_theme()

df = datareader.read_dir("Radius_0.0825_shift_y_0.165")

threshold = 0.7
filtered, selected = select_greater_than(df, threshold)
plot_heatmap(filtered)
plot_heatmap(selected)

sns.scatterplot(data=selected, x='azimuth', y='phi')
plt.title(f"Intercept factor > {threshold}")
plt.savefig("pics/intercept_07.png")
plt.show()


selected.reset_index(inplace=True)
x = selected["azimuth"]
y = selected["phi"]

linregres_results = regression_results(x, y)
slope = linregres_results.params[1]
intercept = linregres_results.params[0]

polynomial_result = lmfit_polynomial(x,y,polynomial)
a,b,c,d = polynomial_result.values.values()

sm_poly, ypred = statsmodels_polynomial(x,y,20)

plt.plot(x,y,'.', alpha=0.3)
plt.plot(x, intercept + slope*x, 'r', label=f'{slope:.2f} x {intercept:.2f}')
plt.plot(x, polynomial_result.best_fit, 'k--', label=f"{a:.4f}x3 {b:.2f}x2 + {c:.2f}x {d:.2f}")
# plt.plot(x,ypred, label="20-degree polyunomial")
plt.xlabel("azimuth")
plt.ylabel("phi")
plt.title(f"Intercept factor > {threshold}")
plt.legend()
plt.savefig("pics/fits.png")
plt.show()


linreg = datareader.read_dir("linear_regression")
sns.lineplot(data=linreg, x="azimuth", y="intercept_factor")
plt.title("Linear regression")
plt.savefig("pics/linear_regression_validation.png")
plt.show()

polyn = datareader.read_dir("polynomial")
sns.lineplot(data=polyn, x="azimuth", y="intercept_factor")
plt.title("3rd degree polynomial")
plt.savefig("pics/3deg_polynomial_validation.png")
plt.show()
