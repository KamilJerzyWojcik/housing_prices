import pandas as pd
import matplotlib.pyplot as pyplot


def join_data(bli, gdp):
    o_bli = pd.DataFrame(bli, columns=['Country', 'Value', 'Inequality', 'Indicator'])
    o_bli = o_bli[o_bli["Inequality"] == 'Total']
    o_bli = o_bli.pivot(index="Country", columns="Indicator", values="Value")
    g_per_capita = pd.DataFrame(gdp, columns=['Country', 'PKB per capita'])
    g_per_capita.set_index("Country", inplace=True)
    full_country_stats = pd.merge(left=o_bli, right=g_per_capita, left_index=True, right_index=True)
    full_country_stats.sort_values(by="Life satisfaction", inplace=True)
    full_country_stats['PKB per capita'] = pd.to_numeric(full_country_stats['PKB per capita'])
    return full_country_stats


def get_learn_data(data, test_rows):
    keep_indices = list(set(range(len(data.index))) - set(test_rows))
    return data.iloc[keep_indices]


def get_test_data(data, test_rows):
    keep_indices = list(set(test_rows))
    return data.iloc[keep_indices]


def plot_data(data, titles = ''):
    for d in enumerate(data):
        try:
            d[1].plot(kind='scatter', x='PKB per capita', y='Life satisfaction', figsize=(8, 5), title=titles[d[0]])
        except:
            d[1].plot(kind='scatter', x='PKB per capita', y='Life satisfaction', figsize=(8, 5))
        pyplot.axis([0, 100000, 0, 10])
    pyplot.show()


def get_test_GDP(test_data):
    items = test_data['PKB per capita'].values
    test_GDP = []
    for item in items:
        test_GDP.append([item])
    return test_GDP


def get_real_BLI(test_data):
    return test_data['Life satisfaction'].values


def show_results(BLI_predict, BLI_real, GDG_test):
    for result in enumerate(BLI_predict):
        print("predicted BLI: {0} | real BLI: {1} | GDP: {2}"
              .format(round(result[1][0], 2), BLI_real[result[0]], round(GDG_test[result[0]][0], 2)))
    return 0
