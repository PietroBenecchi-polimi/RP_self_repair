import configparser
import os

from src.logging.logger import Logger

config = configparser.ConfigParser()
config.read('./resources/config/config.ini')
config.sections()

PCKGS_PATH = config['TEMPLATES SETTING']['TEMPLATE_GEN']
TPLT_PATH = config['TEMPLATES SETTING']['TEMPLATES_PATH']
LOGGER = Logger('Template Generator')


def generate_templates(model_type: str):
    LOGGER.info('Template generation starting...')
    if model_type == 'cognitive_v1':
        file_path = PCKGS_PATH.replace('./', '') + 'cognitive_v1/'
        args = 'argv1.csv argv2.xml argv3.xml argv4.csv argv5.xml argv6.xml argv7.xml DPA XML main.xml'.split(' ')
        args = ' '.join([args[0]] + [file_path + a for a in args[1:-1]] + [args[-1]])
        os.system('{}cognitive_v1/cognitive_v1 {}'.format(PCKGS_PATH, args))
        os.system('mv {} {}'.format(args.split(' ')[-2] + '/' + args.split(' ')[-1],
                                    args.split(' ')[-2] + '/main_v2.xml'))
        os.system('cp {} {}'.format(args.split(' ')[-2] + '/main_v2.xml', TPLT_PATH))
    LOGGER.info('Template generation done.')
    return
