import json
import math
import random
from typing import List, Tuple, Any, Dict

import numpy as np

from src.domain.human import Fatigue_Profile, FreeWill_Profile
from src.domain.layout import Point


class HMTFactor:
    def __init__(self, hmt_id: str, hmt_type: str, ranges: List[Tuple[float, float]] = None, value: Any = None):
        self.hmt_id = hmt_id
        self.hmt_type = hmt_type
        self.ranges = ranges
        self.value = value

    def set_value(self, s: str = None):
        if s is None or len(s) == 0:
            self.value = None

        if self.hmt_type == 'float':
            self.value = float(s)
        elif self.hmt_type == 'int':
            self.value = int(s)
        elif self.hmt_type == 'str':
            self.value = s
        elif self.hmt_type == 'FreeWillProfile':
            self.value = FreeWill_Profile.parse_fw_profile(s)
        elif self.hmt_type == 'FatigueProfile':
            self.value = Fatigue_Profile.parse_ftg_profile(s)
        elif self.hmt_type == 'Point':
            self.value = Point.parse(s)

    def sample(self):
        if self.hmt_type == 'float':
            self.set_value(str(self.ranges[0][0] + ((self.ranges[0][1] - self.ranges[0][0]) * np.random.rand(1)[0])))
        elif self.hmt_type == 'int':
            self.set_value(str(random.randint(int(self.ranges[0][0]), int(self.ranges[0][1]))))
        elif self.hmt_type == 'str':
            # FIXME
            self.set_value()
        elif self.hmt_type == 'FreeWillProfile':
            fw_values = [FreeWill_Profile.FOCUSED, FreeWill_Profile.FREE, FreeWill_Profile.DISTRACTED]
            self.set_value(random.choice(fw_values).value)
        elif self.hmt_type == 'FatigueProfile':
            ftg_values = [Fatigue_Profile.YOUNG_HEALTHY, Fatigue_Profile.YOUNG_SICK,
                          Fatigue_Profile.ELDERLY_HEALTHY, Fatigue_Profile.ELDERLY_SICK,
                          Fatigue_Profile.YOUNG_UNSTEADY, Fatigue_Profile.ELDERLY_UNSTEADY]
            self.set_value(random.choice(ftg_values).value)
        elif self.hmt_type == 'Point':
            x = self.ranges[0][0] + ((self.ranges[0][1] - self.ranges[0][0]) * np.random.rand(1)[0])
            y = self.ranges[1][0] + ((self.ranges[1][1] - self.ranges[1][0]) * np.random.rand(1)[0])
            self.set_value('{:.2f},{:.2f}'.format(x, y))


class Metric:
    def __init__(self, m_id: str, m_type: str, value: Any = None):
        self.m_id = m_id
        self.m_type = m_type
        self.value = value

    def set_value(self, s: str):
        if s is None or len(s) == 0:
            self.value = None
        elif self.m_type == 'float':
            self.value = float(s)
        elif self.m_type == 'str':
            self.value = s

    def processed(self):
        return self.value is not None


class Configuration:
    def __init__(self, factors: List[HMTFactor], metrics: List[Metric]):
        self.factors = factors
        self.metrics = metrics

    def __eq__(self, other):
        return all([f in other.factors for f in self.factors])

    def __len__(self):
        return len(self.factors) + len(self.metrics)

    def __getitem__(self, item):
        if item < len(self.factors):
            return self.factors[item].value
        else:
            return self.metrics[item - len(self.factors)].value

    def set_checkpoint(self, chk):
        self.factors.append(HMTFactor('PROGRESS', 'int', value=chk))

    def lookup(self, key: str):
        for i, factor in enumerate(self.factors):
            if factor.hmt_id == key:
                return i
        for i, metric in enumerate(self.metrics):
            if metric.m_id == key:
                return i + len(self.factors)

    def get_checkpoint(self):
        return self.factors[self.lookup('PROGRESS')].value

    def get_header(self):
        return [f.hmt_id for f in self.factors] + [m.m_id for m in self.metrics]

    def processed(self):
        return [i for i, m in enumerate(self.metrics) if m.processed()]

    @staticmethod
    def config(config_json: str):
        new_factors: List[HMTFactor] = []
        new_metrics: List[Metric] = []
        with open(config_json, 'r') as json_file:
            data = json.load(json_file)

            if data['sample_sim'] == 'true':
                new_factors.append(HMTFactor('PROGRESS', 'int', value=0))

            for f in data['factors']:
                ranges: List[Tuple[float, float]] = []
                try:
                    ranges.append((float(f["min"]), float(f["max"])))
                    new_factors.append(HMTFactor(f['id'], f['type'], ranges))
                except KeyError:
                    try:
                        ranges.append((float(f["min_x"]), float(f["max_x"])))
                        ranges.append((float(f["min_y"]), float(f["max_y"])))
                        new_factors.append(HMTFactor(f['id'], f['type'], ranges))
                    except KeyError:
                        new_factors.append(HMTFactor(f['id'], f['type']))

            for m in data['metrics']:
                new_metrics.append(Metric(m['id'], m['type']))

        return Configuration(new_factors, new_metrics)

    @staticmethod
    def sample(config_json: str):
        new_conf = Configuration.config(config_json)

        for factor in new_conf.factors:
            factor.sample()

        return new_conf

    @staticmethod
    def parse(config_json: str, fields: List[str]):
        new_conf = Configuration.config(config_json)

        for i, factor in enumerate(new_conf.factors):
            factor.set_value(fields[i])
        for i, metric in enumerate(new_conf.metrics):
            metric.set_value(fields[len(new_conf.factors) + i])

        return new_conf

    @staticmethod
    def parse_from_sim(config_json: str, N_SAMPLE: int, sim_file: str, T: int):
        configurations: List[Configuration] = []

        name_2_factor = {'opchk_1.stopDistance': 'ORCH_1_Dstop', 'opchk_1.restartDistance': 'ORCH_1_Drestart',
                         'opchk_1.stopFatigue': 'ORCH_1_Fstop', 'opchk_1.resumeFatigue': 'ORCH_1_Frestart',
                         'TAU': 'PSCS__TAU', 'h_1.v': 'HUM_1_VEL', 'h_2.v': 'HUM_2_VEL',
                         'h_1.p_fw': 'HUM_1_FW', 'h_1.p_f': 'HUM_1_FTG', 'h_2.p_fw': 'HUM_2_FW',
                         'h_2.p_f': 'HUM_2_FTG', 'humanPositionX[0]': 'HUM_1_POS_X', 'humanPositionY[0]': 'HUM_1_POS_Y',
                         'humanPositionX[1]': 'HUM_2_POS_X', 'humanPositionY[1]': 'HUM_2_POS_Y',
                         'r_1.v_max': 'ROB_1_VEL', 'b_r_1.C': 'ROB_1_CHG'}

        signals: Dict[str, Dict[int, Any]] = {}

        signals['HUM_1_POS'] = {}
        signals['HUM_2_POS'] = {}

        with open(sim_file) as sim_file:
            lines = sim_file.readlines()
            i_start = lines.index('\x1b[2K -- Formula is satisfied.\n')
            i_stop = [x for x, l in enumerate(lines) if l.startswith(' -- States explored')][0]
            lines = lines[i_start + 1:i_stop]

            new_variables_indexes = [i for i, l in enumerate(lines) if not l.startswith('[')][1:]

            for i, index in enumerate(new_variables_indexes):
                if i >= len(new_variables_indexes) - 1:
                    continue

                if lines[index].split(':')[0] in name_2_factor:
                    sig_name = name_2_factor[lines[index].split(':')[0]]
                else:
                    sig_name = lines[index].split(':')[0]

                points_str = lines[index + 1].split(':')[1].split(' ')[1:]
                points = []
                for pt in points_str:
                    pt = pt.replace('(', '').replace(')', '')
                    t_str = pt.split(',')[0]
                    v_str = pt.split(',')[1]
                    if not '.' in t_str and not '.' in v_str:
                        points.append((int(t_str), float(int(v_str))))
                    elif not '.' in t_str:
                        points.append((int(t_str), float(v_str)))
                    elif not '.' in v_str:
                        points.append((int(float(t_str)), float(int(v_str))))
                    else:
                        points.append((int(float(t_str)), float(v_str)))
                points = {pt[0]: pt[1] for pt in points}

                for t in range(T):
                    if t not in points:
                        last_t = max([ts for ts in list(points.keys()) if ts < t])
                        points[t] = points[last_t]

                    if 'FTG' in sig_name:
                        try:
                            points[t] = Fatigue_Profile.parse_from_int(int(points[t]))
                        except TypeError:
                            pass
                    elif 'FW' in sig_name:
                        try:
                            points[t] = FreeWill_Profile.parse_from_int(int(points[t]))
                        except TypeError:
                            pass

                signals[sig_name] = points

        signals['PSCS__TAU'] = {t: T - t for t in range(T)}

        for sig_name in signals:
            if 'POS_X' in sig_name:
                for t in signals[sig_name]:
                    signals[sig_name.replace('_X', '')][t] = "{},{}".format(signals[sig_name][t],
                                                                            signals[sig_name.replace('_X', '_Y')][t])

        for i in range(1, T, int(math.floor(T / N_SAMPLE))):
            served = 0
            for sig in signals:
                if sig.startswith('served'):
                    if signals[sig][i] > 0:
                        served += 1

            new_conf = Configuration.config(config_json)

            for factor in new_conf.factors:
                if factor.hmt_id == 'PROGRESS':
                    factor.set_value(str(served))
                else:
                    factor.set_value(str(signals[factor.hmt_id][i]))

            configurations.append(new_conf)

        return configurations
