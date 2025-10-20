# Authors: M. Gupta, B. Gallamoza, N. Cutrona, P. Dhakal, R. Poulain, and R. Beheshti,
# “An extensive data processing pipeline for mimic-iv,” in Machine Learning for Health, 2022, pp. 311–325.

import pandas as pd
import numpy as np


def drop_wrong_uom(data, cut_off):
#     count=0
    grouped = data.groupby(['itemid'])['valueuom']
    for id_number, uom in grouped:
        value_counts = uom.value_counts()
        num_observations = len(uom)
        if(value_counts.size >1):
#             count+=1
            most_frequent_measurement = value_counts.index[0]
            frequency = value_counts[0]
#             print(id_number,value_counts.size,frequency/num_observations)
            if(frequency/num_observations > cut_off):
                values = uom
                index_to_drop = values[values != most_frequent_measurement].index
                data.drop(index_to_drop, axis=0, inplace=True)
    data = data.reset_index(drop=True)
#     print(count)
    return data

