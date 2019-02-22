import numpy as np
import hashlib

def SplitRandomTrainTestData(data, testRatio):
    shuffledIndicates = np.random.permutation(len(data))
    testSizeSet = int(len(data) * testRatio)
    testIndices = shuffledIndicates[:testSizeSet]
    trainIndices = shuffledIndicates[testSizeSet:]
    return data.iloc[trainIndices], data.iloc[testIndices]

def _testSetCheck(identifier, testRatio, hash):
    return hash(np.int64(identifier)).digest()[-1] < 256 * testRatio

def SplitTrainTestDataById(data, testRatio, idColumn, hash=hashlib.md5):
    ids = data[idColumn]
    inTestSet = ids.apply(lambda id_: _testSetCheck(id_, testRatio, hash))
    return data.loc[~inTestSet], data.loc[inTestSet]

def GetComputedId(data, col1, col2):
    return data[col1] * 1000 + data[col2]