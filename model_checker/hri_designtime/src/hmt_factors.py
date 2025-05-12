import csv
import json
import sys
from typing import List
import pandas as pd

from model_checker.hri_designtime.src.domain.hmtfactor import Configuration
from model_checker.hri_designtime.src.domain.query import Query, Query_Type
from model_checker.hri_designtime.src.logging.logger import Logger
from model_checker.hri_designtime.src.mgr.factor_mgr import Factor_Mgr
from model_checker.hri_designtime.src.mgr.json_mgr import Json_Mgr
from model_checker.hri_designtime.src.mgr.param_mgr import Param_Mgr
from model_checker.hri_designtime.src.mgr.query_mgr import Query_Mgr
from model_checker.hri_designtime.src.mgr.tplt_mgr import Template_Mgr
from model_checker.hri_designtime.src.mgr.upp_mgr import Upp_Mgr

def read_configurations_from_df(df: pd.DataFrame, config_json_path: str) -> List[Configuration]:
    configurations = []
    for _, row in df.iterrows():
        configurations.append(Configuration.parse(config_json_path, row.tolist()))
    return configurations

def configurations_to_df(configurations: List[Configuration]) -> pd.DataFrame:
    header = configurations[0].get_header()
    rows = [[f.value for f in c.factors] + [m.value for m in c.metrics] for c in configurations]
    return pd.DataFrame(rows, columns=header)

def run_mc_simulations(dataframe: pd.DataFrame = None):
    SCENARIO = "Dpa"
    CONFIG_JSON = "hmtfactor_config"

    CONFIG_JSON_PATH = './model_checker/hri_designtime/resources/config/{}.json'.format(CONFIG_JSON)

    with open(CONFIG_JSON_PATH) as json_file:
        data = json.load(json_file)
        N = int(data['N_BOUND'])

    LOGGER = Logger('EASE MAIN')

    LOGGER.info('Reading scenario template...')

    json_mgr = Json_Mgr()
    json_mgr.load_json()

    upp_mgr = Upp_Mgr()

    CSV_FILE = upp_mgr.UPPAAL_OUT_PATH.format(SCENARIO).replace('.txt', '.csv')

    SIM_PATH = upp_mgr.UPPAAL_OUT_PATH.replace('/{}.txt', '')

    configurations: List[Configuration] = []

    factor_mgr = Factor_Mgr(json_mgr)

    # Parsing function: Dataframe -> Configuration
    for _, row in dataframe.iterrows():
        str_row = [str(value) for value in row.tolist()]
        configurations.append(Configuration.parse(CONFIG_JSON_PATH, str_row))

    LOGGER.info('{} Configurations to process.'.format(len(configurations)))

    N = N if N >= 0 else len(configurations)
    queries_copy = json_mgr.queries.copy()

    for i, conf in enumerate(configurations[:N]):
        LOGGER.info('Processing conf {}...'.format(i))

#        to_be_processed = [m.m_id for j, m in enumerate(conf.metrics) if j not in conf.processed()]
#        LOGGER.info('Configuration {}: {} to be estimated.'.format(i, ','.join(to_be_processed)))
#        to_be_processed = [x.split('_')[0] for x in to_be_processed]
#        factor_mgr.filter_queries(to_be_processed, queries_copy)

        SCENARIO_NAME = '{}_{}'.format(SCENARIO, i)

        factor_mgr.apply(conf)

        if conf.get_checkpoint() == len(json_mgr.hums):
            LOGGER.warn('Discarding configuration (empty mission).')
            continue

        # Replaces PARAM keywords within main template file with scenario parameters
        param_mgr = Param_Mgr(json_mgr.rescale_hums(conf.get_checkpoint()), json_mgr.robots, json_mgr.layout,
                            json_mgr.params)
        param_mgr.replace_params(SCENARIO_NAME)

        # Replaces TPLT keywords within main template file with individual automata templates
        tplt_mgr = Template_Mgr(param_mgr)
        tplt_mgr.replace_tplt(SCENARIO_NAME)

        factor_mgr.fix_orch_params(conf, SCENARIO_NAME)

        # Generate query file
        query_mg = Query_Mgr(json_mgr.queries)
        query_mg.hums = json_mgr.rescale_hums(conf.get_checkpoint())
        query_mg.gen_q_file(SCENARIO_NAME)

        # Run Uppaal Experiment
        out_file = upp_mgr.run_exp(SCENARIO_NAME)
        
        try:
            factor_mgr.save_metrics(conf, out_file)
        except IndexError:
            LOGGER.error('Verification unsuccessful.')

    LOGGER.info('Done.')
    
    return configurations_to_df(configurations)
