import configparser
from typing import List, Tuple

from model_checker.hri_designtime.src.domain.hmtfactor import Configuration, HMTFactor, Metric
from model_checker.hri_designtime.src.domain.query import Query
from model_checker.hri_designtime.src.logging.logger import Logger
from model_checker.hri_designtime.src.mgr.json_mgr import Json_Mgr

config = configparser.ConfigParser()
config.read('./model_checker/hri_designtime/resources/config/config.ini')
config.sections()

DEST_PATH = config['TEMPLATES SETTING']['MODEL_PATH']
TPLT_EXT = config['TEMPLATES SETTING']['TEMPLATES_EXT']


class Factor_Mgr:
    def __init__(self, scenario: Json_Mgr):
        self.LOGGER = Logger('Factor_Mgr')
        self.scenario = scenario

    def get_agent_list(self, s: str):
        agent_category = s.split('_')[0]
        if agent_category == 'HUM':
            return self.scenario.hums
        elif agent_category == 'ROB':
            return self.scenario.robots

    def get_agents(self, s: str):
        agent_id = int(s.split('_')[1])
        if s.split('_')[0] == 'ROB':
            return [self.get_agent_list(s)[agent_id - 1]]
        else:
            return [agent for agent in self.get_agent_list(s) if agent_id in [agent.h_id, agent.same_as]]

    def set_factor_value(self, factor: HMTFactor):
        if factor.hmt_id == 'PROGRESS':
            return

        factor_name = factor.hmt_id.split('_')[2]
        if factor_name == 'TAU':
            for q in self.scenario.queries:
                if (q.t.value == 'pscs' and factor.hmt_id.split('_')[0] == 'PSCS') or \
                        (q.t.value == 'eftg' and factor.hmt_id.split('_')[0] == 'FTG'):
                    q.tau = factor.value
        elif factor_name == 'FW':
            for h in self.get_agents(factor.hmt_id):
                h.p_fw = factor.value
        elif factor_name == 'FTG':
            for h in self.get_agents(factor.hmt_id):
                h.p_f = factor.value
        elif factor_name == 'CHG':
            for r in self.get_agents(factor.hmt_id):
                r.chg = factor.value
        elif factor_name == 'POS':
            self.get_agents(factor.hmt_id)[0].start = factor.value
        elif factor_name == 'VEL':
            for a in self.get_agents(factor.hmt_id):
                a.v = factor.value
                if factor.hmt_id.split('_')[0] == 'ROB':
                    a.a = factor.value
        elif factor.hmt_id.split('_')[0] == 'ORCH':
            self.scenario.params[factor.hmt_id] = factor.value

    def apply(self, conf: Configuration):
        for factor in conf.factors:
            self.set_factor_value(factor)

    def validate(self, conf: Configuration):
        for f in conf.factors:
            if f.hmt_id.endswith('POS') and not any([a.contains_point(f.value) for a in self.scenario.layout.areas]):
                return False
        return True

    def filter_queries(self, to_process: List[str], all_queries: List[Query]):
        filtered_queries: List[Query] = []
        for q in all_queries:
            if (q.t.value == 'pscs' and 'PRSCS' in to_process) or (q.t.value == 'eftg' and 'FTG' in to_process):
                filtered_queries.append(q)
        self.scenario.queries = filtered_queries

    def parse_upp_results(self, m: Metric, out_file: str):
        with open(out_file) as upp_res:
            lines = upp_res.readlines()
            if m.m_id in ['PRSCS_LOWER_BOUND', 'PRSCS_UPPER_BOUND']:
                pr_range = \
                    [line.split(' Pr(<> …) in ')[1].replace('[', '').replace(']', '').replace('\n', '').split(',')
                     for line in lines if line.__contains__('Pr(<> …)')][0]
                if m.m_id == 'PRSCS_LOWER_BOUND':
                    m.value = float(pr_range[0])
                else:
                    m.value = float(pr_range[1].replace(' (95% CI)', ''))
            elif m.m_id.startswith('FTG'):
                ftg_ranges = [line.split('Values in [')[1].split(']')[0].split(',') for l_i, line in enumerate(lines)
                              if line.__contains__('Values in ') and not lines[l_i - 2].__contains__('Pr(<> …)')]
                ftg_ranges = [(float(r[0]), float(r[1])) for r in ftg_ranges]
                ftg_ranges = [(r[0] + (r[1] - r[0]) / 2, (r[1] - r[0]) / 2) for r in ftg_ranges]
                h_ids = [h.h_id - 1 for h in self.get_agents(m.m_id.replace('FTG_', ''))]
                ftg_range: Tuple[float, float] = tuple()
                for r_i, r in enumerate(ftg_ranges):
                    if r_i not in h_ids:
                        continue
                    if len(ftg_range) == 0 or r[0] > ftg_range[0]:
                        ftg_range = (r[0], r[1])
                m.value = '{:.4f}+-{:.4f}'.format(ftg_range[0], ftg_range[1])

    def save_metrics(self, conf: Configuration, out_file: str):
        for m in conf.metrics:
            self.parse_upp_results(m, out_file)

    def fix_orch_params(self, conf: Configuration, scen_name: str):
        lines = []
        with open(DEST_PATH + scen_name + TPLT_EXT, 'r') as main_tplt:
            lines = main_tplt.readlines()
            for f in conf.factors:
                if f.hmt_id == 'ORCH_1_Dstop':
                    i = lines.index('const double stopDistance = 6.0;\n')
                    lines[i] = 'const double stopDistance = {:.2f};\n'.format(f.value)
                elif f.hmt_id == 'ORCH_1_Drestart':
                    i = lines.index('const double restartDistance = 3.0;\n')
                    lines[i] = 'const double restartDistance = {:.2f};\n'.format(f.value)
                elif f.hmt_id == 'ORCH_1_Fstop':
                    i = lines.index('const double stopFatigue = 0.6;\n')
                    lines[i] = 'const double stopFatigue = {:.2f};\n'.format(f.value)
                elif f.hmt_id == 'ORCH_1_Frestart':
                    i = lines.index('const double resumeFatigue = 0.3;\n')
                    lines[i] = 'const double resumeFatigue = {:.2f};\n'.format(f.value)
        with open(DEST_PATH + scen_name + TPLT_EXT, 'w') as main_tplt:
            main_tplt.truncate(0)
            main_tplt.writelines(lines)
