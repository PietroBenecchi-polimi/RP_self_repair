import configparser
from typing import List, Dict

from model_checker.hri_designtime.src.domain.hri_const import Constants as const
from model_checker.hri_designtime.src.domain.human import Human
from model_checker.hri_designtime.src.domain.layout import Layout
from model_checker.hri_designtime.src.domain.robot import Robot
from model_checker.hri_designtime.src.logging.logger import Logger

config = configparser.ConfigParser()
config.read('./model_checker/hri_designtime/resources/config/config.ini')
config.sections()


class Param_Mgr:
    TPLT_PATH = config['TEMPLATES SETTING']['TEMPLATES_PATH']
    DEST_PATH = config['TEMPLATES SETTING']['MODEL_PATH']
    TPLT_EXT = config['TEMPLATES SETTING']['TEMPLATES_EXT']
    MAIN = const.MAIN_TPLT.value
    TRAJ_TEMPLATES = [const.HF_TPLT.value, const.ROB_TPLT.value, const.HL_TPLT.value, const.HRec_TPLT.value,
                      const.HA_TPLT.value, const.HC_TPLT.value, const.HRes_TPLT.value]
    HUM_KEYWORDS = [const.N_H.value, const.N_H_bool.value, const.N_H_double.value, const.N_H_double_2.value,
                    const.N_H_int.value, const.START_X.value, const.START_Y.value, const.PATTERNS.value,
                    const.DEST_X.value, const.DEST_Y.value, const.SAME_IDs_MAT.value]
    ROB_KEYWORDS = [const.N_R.value, const.N_R_bool.value, const.N_R_int.value, const.N_R_neg.value,
                    const.N_R_double.value, const.N_R_act.value]
    TRAJ_KEYWORDS = [const.N_P_double.value, const.MAX_NEIGH.value, const.MAX_NEIGH_int.value, const.N_I_false.value]
    LAYOUT_KEYWORDS = [const.N_AREAS.value, const.N_POINTS.value, const.N_INTERSECT.value, const.LAYOUT.value,
                       const.INTERSECT.value]
    INST_KEYWORDS = [const.ROB_INST.value, const.ORCH_INST.value, const.HUM_INST.value, const.ALL_INST.value]
    QUERY_KEYWORDS = [const.TAU.value]

    def __init__(self, hums: List[Human], robs: List[Robot], layout: Layout, params: Dict[str, str]):
        self.LOGGER = Logger('Param_Mgr')
        self.hums = hums
        self.N_H = len(hums)
        self.robs = robs
        self.N_R = len(robs)
        self.layout = layout
        self.N_A = len(layout.areas)
        self.N_I = len(layout.inter_pts)
        self.MAX_NEIGH = layout.max_neigh
        self.N_P = self.N_I + 1
        self.inst = [h.name for h in hums] + [r.name for r in robs] + ['b_{}'.format(r.name) for r in robs] + \
                    ['r_pub_{}'.format(r.r_id) for r in robs] + ['o_{}'.format(r.r_id) for r in robs] + \
                    ['opchk_{}'.format(r.r_id) for r in robs] + ['c_pub', 'h_pub_pos', 'h_pub_ftg']
        self.params = params

    def replace_hum_keys(self, scen_name):
        if self.params['behavioral_model'] == 'cognitive_v1':
            self.MAIN += '_v2'
        elif self.params['behavioral_model'] == 'cognitive_v2':
            self.MAIN += '_v3'

        with open(self.TPLT_PATH + self.MAIN + self.TPLT_EXT, 'r') as main_tplt:
            self.LOGGER.debug('Replacing Human-related parameters...')
            main_content = main_tplt.read()
            for key in self.HUM_KEYWORDS:
                value = None
                if key == const.N_H.value:
                    exclude = len([h for h in self.hums if h.path == 2])
                    value = str(self.N_H - exclude) + ";\n"
                elif key == const.N_H_bool.value:
                    value = "{"
                    for x in range(self.N_H):
                        if self.hums[x].path in [1, -1]:
                            value += 'false'
                        if self.hums[x].path in [1, -1] and x < self.N_H - 1:
                            value += ','
                    value += '};'
                elif key == const.N_H_double.value:
                    value = "{"
                    for x in range(self.N_H):
                        if self.hums[x].path in [1, -1]:
                            value += '0.0'
                        if self.hums[x].path in [1, -1] and x < self.N_H - 1:
                            value += ','
                    value += '};'
                elif key == const.N_H_double_2.value:
                    value = "{"
                    for x in range(self.N_H):
                        if self.hums[x].path in [1, -1]:
                            value += '1.0'
                        if self.hums[x].path in [1, -1] and x < self.N_H - 1:
                            value += ','
                    value += '};'
                elif key == const.N_H_int.value:
                    value = "{"
                    for x in range(self.N_H):
                        if self.hums[x].path in [1, -1]:
                            value += '0'
                        if self.hums[x].path in [1, -1] and x < self.N_H - 1:
                            value += ','
                    value += '};'
                elif key == const.PATTERNS.value:
                    value = "{"
                    for x in range(self.N_H):
                        if self.hums[x].path == 1:
                            value += 'ND'
                        elif self.hums[x].path == -1:
                            value += str(self.hums[x].ptrn.to_int())
                        if self.hums[x].path in [1, -1] and x < self.N_H - 1:
                            value += ','
                    value += '};'
                elif key == const.START_X.value:
                    value = "{"
                    for x in range(self.N_H):
                        if self.hums[x].path == -1:
                            value += str(self.hums[x].start.x)
                        elif self.hums[x].path == 1:
                            value += '0.0'
                        if self.hums[x].path in [1, -1] and x < self.N_H - 1:
                            value += ','
                    value += '};'
                elif key == const.START_Y.value:
                    value = "{"
                    for x in range(self.N_H):
                        if self.hums[x].path == -1:
                            value += str(self.hums[x].start.y)
                        elif self.hums[x].path == 1:
                            value += '0.0'
                        if self.hums[x].path in [1, -1] and x < self.N_H - 1:
                            value += ','
                    value += '};'
                elif key == const.DEST_X.value:
                    value = "{"
                    for x in range(self.N_H):
                        if self.hums[x].path == -1:
                            value += str(self.hums[x].dest.x)
                        elif self.hums[x].path == 1:
                            value += '0.0'
                        if self.hums[x].path in [1, -1] and x < self.N_H - 1:
                            value += ','
                    value += '};'
                elif key == const.DEST_Y.value:
                    value = "{"
                    for x in range(self.N_H):
                        if self.hums[x].path == -1:
                            value += str(self.hums[x].dest.y)
                        elif self.hums[x].path == 1:
                            value += '0.0'
                        if self.hums[x].path in [1, -1] and x < self.N_H - 1:
                            value += ','
                    value += '};'
                elif key == const.SAME_IDs_MAT.value:
                    value = '{'

                    same_ids = []
                    for h_i, h in enumerate(self.hums):
                        same_ids.append([])
                        same_ids[-1].append(h.h_id)
                        with_same_ids = list(filter(lambda h2: h.h_id != h2.h_id and
                                                               ((h.same_as != -1 and h2.same_as == h.same_as) or
                                                                h.h_id == h2.same_as or h2.h_id == h.same_as),
                                                    self.hums))
                        same_ids[-1].extend([h2.h_id for h2 in with_same_ids])
                        while len(same_ids[-1]) < len(self.hums):
                            same_ids[-1].append(-1)

                    for r_i, row in enumerate(same_ids):
                        value += '{'
                        value += ','.join([str(x) for x in row])
                        value += '}'
                        if r_i < len(self.hums) - 1:
                            value += ','

                    value += '};'
                main_content = main_content.replace(key, str(value))
            dest_model = open(self.DEST_PATH + scen_name + self.TPLT_EXT, 'w')
            dest_model.write(main_content)
            dest_model.close()
            self.LOGGER.info('{} model successfully saved in {}.'.format(scen_name, self.DEST_PATH))

    def replace_layout_keys(self, scen_name):
        with open(self.DEST_PATH + scen_name + self.TPLT_EXT, 'r') as main_tplt:
            self.LOGGER.debug('Replacing Layout-related parameters...')
            main_content = main_tplt.read()
            for key in self.LAYOUT_KEYWORDS:
                value = None
                if key == const.N_AREAS.value:
                    value = str(self.N_A) + ';\n'
                elif key == const.N_POINTS.value:
                    value = str(self.N_P) + ';\n'
                elif key == const.N_INTERSECT.value:
                    value = str(self.N_I) + ';\n'
                elif key == const.LAYOUT.value:
                    value = '{'
                    for (i, a) in enumerate(self.layout.areas):
                        value += '{'
                        for (j, p) in enumerate(a.corners):
                            value += '{' + str(p.x) + ', ' + str(p.y) + '}'
                            if j <= 2:
                                value += ','
                        value += '}'
                        if i <= self.N_A - 2:
                            value += ','
                    value += '};\n'
                elif key == const.INTERSECT.value:
                    value = '{'
                    for (i, pt) in enumerate(self.layout.inter_pts):
                        value += '{' + str(pt.x) + ', ' + str(pt.y) + '}'
                        if i <= self.N_I - 2:
                            value += ','
                    value += '};\n'
                main_content = main_content.replace(key, str(value))
        dest_model = open(self.DEST_PATH + scen_name + self.TPLT_EXT, 'w')
        dest_model.write(main_content)
        dest_model.close()
        self.LOGGER.info('{} model successfully saved in {}.'.format(scen_name, self.DEST_PATH))

    def replace_inst_keys(self, scen_name):
        with open(self.DEST_PATH + scen_name + self.TPLT_EXT, 'r') as main_tplt:
            self.LOGGER.debug('Replacing Instances-related parameters...')
            main_content = main_tplt.read()
            for key in self.INST_KEYWORDS:
                value = None
                if key == const.ROB_INST.value:
                    value = ''.join([r.get_constructor() for r in self.robs])
                    value += 'c_pub = ROS_SensPub(0, 0.5, 0.01);\n'
                elif key == const.HUM_INST.value:
                    value = ''.join([h.get_constructor() for h in self.hums])
                    value += 'h_pub_pos = ROS_SensPub(2, 0.5, 0.01);\nh_pub_ftg = ROS_SensPub(3, 0.5, 0.01);\n'
                elif key == const.ORCH_INST.value:
                    value = ''.join([r.get_orch_constructor() for r in self.robs])
                elif key == const.ALL_INST.value:
                    value = ''
                    for (i, ins) in enumerate(self.inst):
                        value += ins
                        if i <= len(self.inst) - 2:
                            value += ",\n"
                main_content = main_content.replace(key, str(value))
        dest_model = open(self.DEST_PATH + scen_name + self.TPLT_EXT, 'w')
        dest_model.write(main_content)
        dest_model.close()
        self.LOGGER.info('{} model successfully saved in {}.'.format(scen_name, self.DEST_PATH))

    def replace_query_keys(self, scen_name):
        with open(self.DEST_PATH + scen_name + self.TPLT_EXT, 'r') as main_tplt:
            self.LOGGER.debug('Replacing Query-related parameters...')
            main_content = main_tplt.read()
            for key in self.QUERY_KEYWORDS:
                value = None
                # TODO
                main_content = main_content.replace(key, str(value))
        dest_model = open(self.DEST_PATH + scen_name + self.TPLT_EXT, 'w')
        dest_model.write(main_content)
        dest_model.close()
        self.LOGGER.info('{} model successfully saved in {}.'.format(scen_name, self.DEST_PATH))

    def replace_traj_keys(self, tplt):
        is_cog_v1 = self.params['behavioral_model'] == 'cognitive_v1' and tplt.replace('_v2', '') in self.TRAJ_TEMPLATES
        is_cog_v2 = self.params['behavioral_model'] == 'cognitive_v2' and tplt.replace('_v3', '') in self.TRAJ_TEMPLATES
        if tplt in self.TRAJ_TEMPLATES or is_cog_v1 or is_cog_v2:
            with open(self.TPLT_PATH + tplt + self.TPLT_EXT, 'r') as main_tplt:
                self.LOGGER.debug('Replacing Trajectory-related parameters...')
                main_content = main_tplt.read()
                for key in self.TRAJ_KEYWORDS:
                    # const.N_P_double, const.MAX_NEIGH, const.MAX_NEIGH_int, const.N_I_false
                    if key == const.N_P_double.value:
                        value = '{'
                        for x in range(self.N_P):
                            value += '{0.0, 0.0}'
                            if x < self.N_P - 1:
                                value += ','
                        value += '};'
                    elif key == const.MAX_NEIGH.value:
                        value = self.MAX_NEIGH
                    elif key == const.MAX_NEIGH_int.value:
                        value = '{'
                        for x in range(self.MAX_NEIGH):
                            value += '-1'
                            if x < self.MAX_NEIGH - 1:
                                value += ','
                        value += '};'
                    else:
                        value = '{'
                        for x in range(self.N_I):
                            value += 'false'
                            if x < self.N_I - 1:
                                value += ','
                        value += '};'
                    main_content = main_content.replace(key, str(value))
        else:
            f = open(self.TPLT_PATH + tplt + self.TPLT_EXT, 'r')
            main_content = f.read()
        return main_content

    def replace_rob_keys(self, scen_name):
        with open(self.DEST_PATH + scen_name + self.TPLT_EXT, 'r') as main_tplt:
            self.LOGGER.debug('Replacing Robot-related parameters...')
            main_content = main_tplt.read()
            for key in self.ROB_KEYWORDS:
                value = None
                if key == const.N_R.value:
                    value = str(self.N_R) + ";\n"
                elif key == const.N_R_bool.value:
                    value = '{'
                    for x in range(self.N_R):
                        value += 'false'
                        if x < self.N_R - 1:
                            value += ','
                    value += '};'
                elif key == const.N_R_act.value:
                    value = '{true'
                    if self.N_R > 1:
                        value += ','
                    for x in range(self.N_R - 1):
                        value += 'false'
                        if x < self.N_R - 2:
                            value += ','
                    value += '};'
                elif key == const.N_R_neg.value:
                    value = '{'
                    for x in range(self.N_R):
                        value += '-1.0'
                        if x < self.N_R - 1:
                            value += ','
                    value += '};'
                elif key == const.N_R_int.value:
                    value = '{'
                    for x in range(self.N_R):
                        value += '1'
                        if x < self.N_R - 1:
                            value += ','
                    value += '};'
                elif key == const.N_R_double.value:
                    value = '{'
                    for x in range(self.N_R):
                        value += '100.0'
                        if x < self.N_R - 1:
                            value += ','
                    value += '};'
                main_content = main_content.replace(key, str(value))
            dest_model = open(self.DEST_PATH + scen_name + self.TPLT_EXT, 'w')
            dest_model.write(main_content)
            dest_model.close()
            self.LOGGER.info('{} model successfully saved in {}.'.format(scen_name, self.DEST_PATH))

    def replace_params(self, scen_name):
        self.replace_hum_keys(scen_name)
        self.replace_rob_keys(scen_name)
        self.replace_layout_keys(scen_name)
        self.replace_inst_keys(scen_name)
