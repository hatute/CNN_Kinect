import os
import shutil
import numpy as np


if __name__ == '__main__':
    sample_of_each_kind = 125

    path_list = [['../TrainingSamples/Ges_0', '../TestSamples/Ges_0'],
                 ['../TrainingSamples/Ges_1', '../TestSamples/Ges_1'],
                 ['../TrainingSamples/Ges_2', '../TestSamples/Ges_2'],
                 ['../TrainingSamples/Ges_3', '../TestSamples/Ges_3'],
                 ['../TrainingSamples/Ges_3-A', '../TestSamples/Ges_3-A'],
                 ['../TrainingSamples/Ges_3-B', '../TestSamples/Ges_3-B'],
                 ['../TrainingSamples/Ges_4', '../TestSamples/Ges_4'],
                 ['../TrainingSamples/Ges_5', '../TestSamples/Ges_5']
                 ]

    for path in path_list:
        files = os.listdir(path[0])
        count = len(files)
        idx = np.random.choice(count, sample_of_each_kind, replace=False)
        for i in idx:
            src = path[0] + '/' + files[i]
            dst = path[1] + '/' + files[i]
            shutil.move(src, dst)
