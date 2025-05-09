import configparser
import os
from datetime import datetime

from model_checker.hri_designtime.src.logging.logger import Logger

config = configparser.ConfigParser()
config.read('./model_checker/hri_designtime/resources/config/config.ini')
config.sections()


class Upp_Mgr:
    SCRIPT_PATH = config['UPPAAL SETTINGS']['UPPAAL_SCRIPT_PATH']
    UPPAAL_PATH = config['UPPAAL SETTINGS']['UPPAAL_PATH']
    UPPAAL_XML_PATH = config['TEMPLATES SETTING']['MODEL_PATH']
    MODEL_EXT = config['TEMPLATES SETTING']['TEMPLATES_EXT']
    UPPAAL_Q_PATH = config['TEMPLATES SETTING']['QUERY_PATH']
    QUERY_EXT = config['TEMPLATES SETTING']['QUERY_EXT']
    UPPAAL_OUT_PATH = config['UPPAAL SETTINGS']['UPPAAL_OUT_PATH']

    def __init__(self):
        self.LOGGER = Logger('Uppaal_Mgr')

    def get_ts(self):
        ts = datetime.now()
        ts_split = str(ts).split('.')[0]
        ts_str = ts_split.replace('-', '_')
        ts_str = ts_str.replace(' ', '_').replace(':', '_')
        return ts_str

    def run_exp(self, scen_name):
        self.LOGGER.info('Starting verification...')
        res_name = scen_name + '_' + self.get_ts()
        os.system('{} {} {} {} {}'.format(self.SCRIPT_PATH, self.UPPAAL_PATH,
                                          self.UPPAAL_XML_PATH + scen_name + self.MODEL_EXT,
                                          self.UPPAAL_XML_PATH + scen_name + self.QUERY_EXT,
                                          self.UPPAAL_OUT_PATH.format(res_name)))
        self.LOGGER.info('Verification complete.')
        return self.UPPAAL_OUT_PATH.format(res_name)
