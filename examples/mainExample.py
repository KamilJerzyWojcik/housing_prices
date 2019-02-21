import numpy
import pandas
import sklearn
from sklearn import *
from sklearn.externals import joblib
import prepare_data
import fit_data


print("-------------------------start-------------------------")
# Read data
BLI_OECD = pandas.read_csv("data/oecd_bli_2018.csv", thousands=',')
GDP_per_capita_IMF = pandas.read_csv("data/imf_pkb_per_capita_2018.csv", sep=';')

# prepare data
country_stats_all = prepare_data.join_data(BLI_OECD, GDP_per_capita_IMF)
test_rows = [0, 1, 6, 8, 33, 34, 35]
country_stats_test = prepare_data.get_test_data(country_stats_all, test_rows)
country_stats_learn = prepare_data.get_learn_data(country_stats_all, test_rows)

# training model
model_oecd_gdp = fit_data.linear_regresion(country_stats_learn, 'PKB per capita', 'Life satisfaction')

# save model
joblib.dump(model_oecd_gdp, 'filename.joblib')

# load model
model_oecd_gdp = joblib.load('filename.joblib')

# value estimation
GDP_test = prepare_data.get_test_GDP(country_stats_test)
BLI_real = prepare_data.get_real_BLI(country_stats_test)

#  regresion
BLI_predict = model_oecd_gdp.predict(GDP_test)

# show results in console
prepare_data.show_results(BLI_predict, BLI_real, GDP_test)

# Plot results
prepare_data.plot_data([country_stats_learn, country_stats_test], ['Learn data', 'Test data'])

print("-------------------------end---------------------------")



