import configparser
import json
import sys
from typing import List, Dict

from model_checker.hri_designtime.src.domain.human import Human, Interaction_Pattern, Fatigue_Profile, FreeWill_Profile
from model_checker.hri_designtime.src.domain.layout import Layout, Point, Area
from model_checker.hri_designtime.src.domain.query import Query, Query_Type
from model_checker.hri_designtime.src.domain.robot import Robot
from model_checker.hri_designtime.src.logging.logger import Logger

config = configparser.ConfigParser()
config.read('./model_checker/hri_designtime/resources/config/config.ini')
config.sections()

SCENARIO_NAME = "hmtfactor_config"

class Json_Mgr:
    JSON_PATH = config['PARAMS SETTINGS']['PARAMS_PATH']
    JSON_EXT = config['PARAMS SETTINGS']['PARAMS_EXT']
    SCENARIO_NAME = "hmtfactor_config"
    PARAMS_FILE = "DPa"

    def __init__(self):
        self.LOGGER = Logger('Json_Mgr')
        self.hums: List[Human] = []
        self.robots: List[Robot] = []
        self.layout: Layout = Layout([], [], 0)
        self.queries: List[Query] = []
        self.params: Dict[str, str] = dict()

    def load_json(self):
        with open(self.JSON_PATH + self.PARAMS_FILE + self.JSON_EXT, 'r') as json_file:
            data = json.load(json_file)
            # parse global params
            try:
                for d in data['global_params']:
                    self.params.update(d)
            except KeyError:
                self.LOGGER.warn("No global params specified.")
                self.params['behavioral_model'] = 'random'
            # parse humans
            humans_data = data['humans']
            self.LOGGER.info("Loading human-related data...")
            for h in humans_data:
                if SCENARIO_NAME.startswith('DP2') and h['h_id'] >= 6:
                    h['v'] = 26.0
                self.hums.append(Human(h['name'], h['h_id'], h['v'], Interaction_Pattern.parse_ptrn(h['ptrn']),
                                       Fatigue_Profile.parse_ftg_profile(h['p_f']),
                                       FreeWill_Profile.parse_fw_profile(h['p_fw']),
                                       Point(h['start'][0], h['start'][1]), Point(h['dest'][0], h['dest'][1]),
                                       h['dext'], h['same_as'], h['path'], self.params['behavioral_model']))
            self.LOGGER.info("Successfully loaded.")
            # parse robots
            robots_data = data['robots']
            self.LOGGER.info("Loading robot-related data...")
            for r in robots_data:
                if SCENARIO_NAME.startswith('DP1'):
                    r['v'] = 36.0
                    r['a'] = 36.0
                self.robots.append(
                    Robot(r['name'], r['r_id'], r['v'], r['a'], Point(r['start'][0], r['start'][1]), r['chg']))
            self.LOGGER.info("Successfully loaded.")
            # parse layout
            areas_data = data['areas']
            inters_pts_data = data['intersect']
            self.LOGGER.info("Loading layout-related data...")
            for a in areas_data:
                self.layout.areas.append(
                    Area(Point(a['p1'][0], a['p1'][1]), Point(a['p2'][0], a['p2'][1]), Point(a['p3'][0], a['p3'][1]),
                         Point(a['p4'][0], a['p4'][1])))
            for pt in inters_pts_data:
                self.layout.inter_pts.append(Point(pt['p'][0], pt['p'][1]))
            self.layout.max_neigh = data['max_neigh']
            self.LOGGER.info("Successfully loaded.")
            # parse queries
            queries_data = data['queries']
            self.LOGGER.info("Loading query-related data...")
            for q in queries_data:
                self.queries.append(Query(Query_Type.parse_query(q['type']), q['tau'], q['n'], self.hums, self.robots))
            self.LOGGER.info("Successfully loaded.")

    def rescale_hums(self, served: int):
        if served == 0:
            return self.hums

        new_hums: List[Human] = []

        dependencies = {h.h_id: [h_2.h_id for h_2 in self.hums if h_2.same_as == h.h_id] for h in self.hums}
        new_ids = {}

        for i, h in enumerate(self.hums):
            if h.h_id <= served:
                continue
            new_ids[h.h_id] = h.h_id - served
            new_hums.append(
                Human(h.name, h.h_id, h.v, h.ptrn, h.p_f, h.p_fw, h.start, h.dest, h.dext, h.same_as, h.path,
                      h.fw_model))

        for i, h in enumerate(new_hums):
            for old_ids in dependencies:
                if h.h_id in dependencies[old_ids]:
                    if len(dependencies[old_ids]) == 1 or h.h_id == min(dependencies[old_ids]):
                        h.same_as = -1
                    else:
                        if min(dependencies[old_ids]) in new_ids:
                            h.same_as = new_ids[min(dependencies[old_ids])]
                        else:
                            h.same_as = -1

            h.h_id -= served

        return new_hums
