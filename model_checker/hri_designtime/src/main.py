import configparser
import sys

from src.logging.logger import Logger
from src.mgr.json_mgr import Json_Mgr
from src.mgr.param_mgr import Param_Mgr
from src.mgr.query_mgr import Query_Mgr
from src.mgr.tplt_gen import generate_templates
from src.mgr.tplt_mgr import Template_Mgr
from src.mgr.upp_mgr import Upp_Mgr

config = configparser.ConfigParser()
config.read('./resources/config/config.ini')
config.sections()

LOGGER = Logger('main')

if len(sys.argv) < 2:
    LOGGER.error('USAGE: main.py $JSON_FILE')
else:
    LOGGER.info('Starting conf of HRI scenario...')

    SCENARIO_NAME = sys.argv[1]

    json_mgr = Json_Mgr()
    json_mgr.load_json()

    # Generate Templates (if necessary)
    if json_mgr.params['behavioral_model'] not in ['random', 'errors']:
        generate_templates(json_mgr.params['behavioral_model'])

    # Replaces PARAM keywords within main template file with scenario parameters
    param_mgr = Param_Mgr(json_mgr.hums, json_mgr.robots, json_mgr.layout, json_mgr.params)
    param_mgr.replace_params(SCENARIO_NAME)

    # Replaces TPLT keywords within main template file with individual automata templates
    tplt_mgr = Template_Mgr(param_mgr)
    tplt_mgr.replace_tplt(SCENARIO_NAME)

    # Generate query file
    query_mg = Query_Mgr(json_mgr.queries)
    query_mg.gen_q_file(SCENARIO_NAME)

    # Run Uppaal Experiment
    upp_mgr = Upp_Mgr()
    upp_mgr.run_exp(SCENARIO_NAME)
