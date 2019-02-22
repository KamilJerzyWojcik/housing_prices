from Infrastructure import Configuration as CONFIG
from os import path, makedirs
from tarfile import open
from pandas import read_csv
from six.moves import urllib


def GetHousingDataFromUrl(housingUrl = CONFIG.HOUSING_URL, housingPath = CONFIG.HOUSING_PATH, filePath = CONFIG.FILE_PATH):
    if not path.isdir(housingPath):
        makedirs(housingPath)
    #tgzPath = os.path.join(housingPath, "housing.jpg")
    #urllib.request.urlretrieve(housingUrl, tgzPath)
    housingTgz = open(filePath)
    housingTgz.extractall(path=housingPath)
    housingTgz.close()

def LoadHousingDataFromPath(housingPath = CONFIG.HOUSING_PATH):
    csvPath = path.join(housingPath, "housing.csv")
    return read_csv(csvPath)
