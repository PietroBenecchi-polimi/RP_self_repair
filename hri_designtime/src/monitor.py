import csv
import os
import sys
from typing import List, Tuple

from src.domain.hmtfactor import Configuration
from src.domain.layout import Point
from src.logging.logger import Logger
from src.mgr.upp_mgr import Upp_Mgr

SIM_PATH = './resources/sim_logs'
LOGS = ['humanFatigue', 'humanPosition', 'robotBattery']

SCENARIO = sys.argv[1]
FILE_PATH = './resources/input_params/ease_exp/{}.json'.format(SCENARIO)

CONFIG_JSON = sys.argv[2]
CONFIG_JSON_PATH = './resources/config/{}.json'.format(CONFIG_JSON)

LOGGER = Logger('Monitor')

CSV_FILE = Upp_Mgr.UPPAAL_OUT_PATH.replace('.txt', '.csv')


def sim_to_conf(config_json: str, sims_path: str, sim: str):
    LOGGER.info('Converting simulation {}...'.format(sim))

    VREP_X_OFFSET = 7.725
    VREP_Y_OFFSET = 11.4
    REAL_X_OFFSET = 0.4
    REAL_Y_OFFSET = 1.55

    signals: List[List[int, Point, float, float, float, Point, Point]] = []

    with open(sims_path + '/' + sim + '/humanFatigue.log') as h_ftg:
        lines = h_ftg.readlines()[1:]
        for line in lines:
            fields = line.split(':')
            if fields[1] == 'hum1':
                signals.append([int(float(fields[0])), None, float(fields[2]), None, None, None, None])
            else:
                signals[-1] = [signals[-1][0], None, signals[-1][2], float(fields[2]), None, None, None]

    with open(sims_path + '/' + sim + '/humanPosition.log') as h_pos:
        lines = h_pos.readlines()[1:]
        for line in lines:
            fields = line.split(':')
            pos = fields[2].split('#')
            if fields[1] == 'hum2':
                for i, pt in enumerate(signals):
                    if pt[0] == int(float(fields[0])):
                        signals[i][1] = Point((float(pos[0]) + VREP_X_OFFSET) * 100,
                                              (float(pos[1]) + VREP_Y_OFFSET) * 100)
            elif fields[1] == 'hum1':
                for i, pt in enumerate(signals):
                    if pt[0] == int(float(fields[0])):
                        signals[i][5] = Point((float(pos[0]) + VREP_X_OFFSET) * 100,
                                              (float(pos[1]) + VREP_Y_OFFSET) * 100)

    with open(sims_path + '/' + sim + '/robotBattery.log') as r_btr:
        lines = r_btr.readlines()[1:]
        start_t = int(float(lines[0].split(':')[0]))
        for line in lines:
            fields = line.split(':')
            for i, pt in enumerate(signals):
                if pt[0] == int(float(fields[0])) - start_t:
                    signals[i][4] = float(fields[1])

    with open(sims_path + '/' + sim + '/robotPosition.log') as r_pos:
        lines = r_pos.readlines()[1:]
        start_t = int(float(lines[0].split(':')[0]))
        for line in lines:
            fields = line.split(':')
            pos = fields[1].split('#')
            for i, pt in enumerate(signals):
                if pt[0] == int(float(fields[0])) - start_t:
                    signals[i][6] = Point((float(pos[0]) + REAL_X_OFFSET) * 100, (float(pos[1]) + REAL_Y_OFFSET) * 100)

    # Fill in missing data.
    for i, sig in enumerate(signals):
        if sig[4] is None:
            if i == 0:
                sig[4] = 100.0
            else:
                sig[4] = signals[i - 1][6]
        if sig[6] is None:
            if i == 0:
                sig[6] = Point(0.0, 0.0)
            else:
                sig[6] = signals[i - 1][6]

    exp_ftg: List[Tuple[float, float]] = []
    max_ftg: Tuple[float, float] = (0.0, 0.0)
    for sig in signals:
        if sig[2] > max_ftg[0]:
            max_ftg = (sig[2], max_ftg[1])
        if sig[3] > max_ftg[1]:
            max_ftg = (max_ftg[0], sig[3])
        exp_ftg.append(max_ftg)

    rows = [["5.0", "2.0", "0.7", "0.3", str(500 - signals[0][0]), "100.0", "100.0",
             "free", "y/s", "foc", "e/h", str(signals[0][1]), "26.0", str(signals[0][4] * 0.9 / 100 + 11.1),
             None, None, None, None]]
    rows += [["5.0", "2.0", "0.7", "0.3", str(500 - sig[0]), str(sig[5].dist_from(signals[i - 1][5])),
              str(sig[1].dist_from(signals[i - 1][1])), "free", "y/s", "foc", "e/h", str(sig[1]),
              str(sig[6].dist_from(signals[i - 1][6])), str(sig[4] * 0.9 / 100 + 11.1),
              None, None, None, None] for i, sig in enumerate(signals) if i > 0]
    confs = [Configuration.parse(config_json, row) for row in rows]

    return confs


def dump_csv(configurations: List[Configuration], csv_path: str):
    LOGGER.info('Dumping HMT factors to {} ...'.format(csv_path))

    with open(csv_path, 'w') as out_csv:
        write = csv.writer(out_csv)
        write.writerow(configurations[0].get_header())
        for conf_towrite in configurations:
            write.writerow([f.value for f in conf_towrite.factors] + [m.value for m in conf_towrite.metrics])


sims = [x for x in os.listdir(SIM_PATH) if x.startswith('SIM')]
[dump_csv(sim_to_conf(CONFIG_JSON_PATH, SIM_PATH, sim), CSV_FILE.format(sim)) for sim in sims]
# [sim_to_conf(CONFIG_JSON_PATH, SIM_PATH, sim) for sim in sims]
LOGGER.info('Done.')
