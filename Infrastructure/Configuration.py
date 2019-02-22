import os

#ImportData

#GetHousingDataFromUrl
DOWNLOAD_ROOT = "https://github.com/KamilJerzyWojcik/housing_prices/tree/master/" #blokada github
HOUSING_PATH = os.path.join("data", "house")
HOUSING_URL = DOWNLOAD_ROOT + "housing_data/housing.tgz"
FILE_PATH = os.path.join("housing_data", "housing.tgz")


