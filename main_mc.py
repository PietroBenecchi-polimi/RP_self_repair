from sklearn.base import clone
from typing import Dict
import pandas as pd
import utils.datacleaner as dc
import model_checker.hri_designtime.src.hmt_factors as hmtf

import pandas as pd
from self_repair.configuration_validation import validate_configurations
import warnings
from sklearn.exceptions import InconsistentVersionWarning
import multiprocessing
import time
from sklearn.model_selection import train_test_split
import numpy as np
import utils as ut
import main 

warnings.simplefilter("ignore", InconsistentVersionWarning)
from utils.rp_logger import logger
from self_repair.mc_opt_interface import *

def process_data(data):
    # Convert categorical columns to numeric
    data = dc.numeric_to_categorical(data)
    # Combine AGE and STAT into a new column
    data['AGE/STAT 1'] = data['HUM_1_AGE'].astype(str) + '/' + data['HUM_1_STA'].astype(str)
    data['AGE/STAT 2'] = data['HUM_2_AGE'].astype(str) + '/' + data['HUM_2_STA'].astype(str)

    data['Position 1'] = data['HUM_1_POS_X'].astype(str) + ', ' + data['HUM_1_POS_Y'].astype(str)
    data['Position 2'] = data['HUM_2_POS_X'].astype(str) + ', ' + data['HUM_2_POS_Y'].astype(str) 

    data.drop(columns=['HUM_1_POS_X', 'HUM_1_POS_Y', 'HUM_2_POS_X', 'HUM_2_POS_Y', 'HUM_1_AGE', 'HUM_2_AGE',
                      'HUM_1_STA', 'HUM_2_STA'], inplace=True) 
    #Predicted columns drop
    #data.drop(columns=['PRSCS_LB', 'PRSCS_UB', 'SCS', 'FTG_HUM_1_LB', 'FTG_HUM_1_UB', 'FTG_HUM_1',
    #                   'FTG_HUM_2_LB', 'FTG_HUM_2_UB', 'FTG_HUM_2'], inplace=True)
    # Extract new columns
    age_stat_1 = data.pop('AGE/STAT 1')
    age_stat_2 = data.pop('AGE/STAT 2')
    vel_1 = data.pop('Position 1')
    vel_2 = data.pop('Position 2')

    # Desidered position
    data.insert(9, 'AGE/STAT 1', age_stat_1)
    data.insert(11, 'AGE/STAT 2', age_stat_2)
    data.insert(12, 'Position 1', vel_1)
    data.insert(13, 'Position 2', vel_2)

    data = data.rename(columns={
        'PRSCS_LB': 'PRSCS_LOWER_BOUND',
        'PRSCS_UB': 'PRSCS_UPPER_BOUND',
        'FTG_HUM_1': 'FTG_HUM_1',
        'FTG_HUM_2': 'FTG_HUM_2',
    })
    data.drop(columns=['SCS', 'FTG_HUM_1_LB','FTG_HUM_1_UB',
                      'FTG_HUM_2_LB','FTG_HUM_2_UB',], inplace=True)
    return data

def mc_results_from_configs(data):
    data = process_data(data)
    data = hmtf.run_mc_simulations(data)

    data.to_csv("data/processed_dataset.csv", index=False)

if __name__ == "__main__":
    data = pd.read_csv("data/initial_configurations_to_improve.csv")
    data_frame = mc_results_from_configs(data)
    data_frame.to_csv("data/processed_dataset.csv", index=False)