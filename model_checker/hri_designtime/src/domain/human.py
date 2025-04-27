from enum import Enum

from src.domain.layout import Point


class Interaction_Pattern(Enum):
    APPLICANT = "APPLICANT"
    COMPETITOR = "COMPETITOR"
    FOLLOWER = "FOLLOWER"
    LEADER = "LEADER"
    RECIPIENT = "RECIPIENT"
    RESCUER = "RESCUER"

    @staticmethod
    def parse_ptrn(p: str):
        if p == "APPLICANT":
            return Interaction_Pattern.APPLICANT
        elif p == "COMPETITOR":
            return Interaction_Pattern.COMPETITOR
        elif p == "FOLLOWER":
            return Interaction_Pattern.FOLLOWER
        elif p == "LEADER":
            return Interaction_Pattern.LEADER
        elif p == "RECIPIENT":
            return Interaction_Pattern.RECIPIENT
        else:
            return Interaction_Pattern.RESCUER

    def to_int(self):
        if self == Interaction_Pattern.APPLICANT:
            return 3
        elif self == Interaction_Pattern.COMPETITOR:
            return 12
        elif self == Interaction_Pattern.FOLLOWER:
            return 0
        elif self == Interaction_Pattern.LEADER:
            return 1
        elif self == Interaction_Pattern.RECIPIENT:
            return 2
        elif self == Interaction_Pattern.RESCUER:
            return 10

    def __str__(self):
        return self.value


class Fatigue_Profile(Enum):
    YOUNG_HEALTHY = "y/h"
    YOUNG_SICK = "y/s"
    ELDERLY_HEALTHY = "e/h"
    ELDERLY_SICK = "e/s"
    YOUNG_UNSTEADY = "y/u"
    ELDERLY_UNSTEADY = "e/u"

    def to_int(self):
        if self == Fatigue_Profile.YOUNG_HEALTHY:
            return 1
        elif self == Fatigue_Profile.YOUNG_SICK:
            return 2
        elif self == Fatigue_Profile.ELDERLY_HEALTHY:
            return 3
        elif self == Fatigue_Profile.ELDERLY_SICK:
            return 4
        elif self == Fatigue_Profile.YOUNG_UNSTEADY:
            return 5
        else:
            return 6

    @staticmethod
    def parse_ftg_profile(s: str):
        if s == "y/h":
            return Fatigue_Profile.YOUNG_HEALTHY
        elif s == "y/s":
            return Fatigue_Profile.YOUNG_SICK
        elif s == "e/h":
            return Fatigue_Profile.ELDERLY_HEALTHY
        elif s == "e/s":
            return Fatigue_Profile.ELDERLY_SICK
        elif s == "y/u":
            return Fatigue_Profile.YOUNG_UNSTEADY
        else:
            return Fatigue_Profile.ELDERLY_UNSTEADY

    @staticmethod
    def parse_from_int(x: int):
        if x == 1:
            return Fatigue_Profile.YOUNG_HEALTHY
        elif x == 2:
            return Fatigue_Profile.YOUNG_SICK
        elif x == 3:
            return Fatigue_Profile.ELDERLY_HEALTHY
        elif x == 4:
            return Fatigue_Profile.ELDERLY_SICK
        elif x == 5:
            return Fatigue_Profile.YOUNG_UNSTEADY
        else:
            return Fatigue_Profile.ELDERLY_UNSTEADY

    def __str__(self):
        return self.value


class FreeWill_Profile(Enum):
    DISABLED = 'd'
    BUSY = 'busy'
    FREE = 'free'
    UNEXPERIENCED = 'unexp'
    EXPERIENCED = 'exp'
    CRITICAL = 'crit'
    STABLE = 'stable'
    DISTRACTED = 'distr'
    FOCUSED = 'foc'

    def to_int(self):
        if self == FreeWill_Profile.BUSY:
            return 1
        elif self == FreeWill_Profile.FREE:
            return 2
        elif self == FreeWill_Profile.UNEXPERIENCED:
            return 3
        elif self == FreeWill_Profile.EXPERIENCED:
            return 4
        elif self == FreeWill_Profile.CRITICAL:
            return 5
        elif self == FreeWill_Profile.STABLE:
            return 6
        elif self == FreeWill_Profile.DISTRACTED:
            return 7
        elif self == FreeWill_Profile.FOCUSED:
            return 8
        else:
            return -1

    @staticmethod
    def parse_fw_profile(s: str):
        if s == 'busy':
            return FreeWill_Profile.BUSY
        elif s == 'free':
            return FreeWill_Profile.FREE
        elif s == 'unexp':
            return FreeWill_Profile.UNEXPERIENCED
        elif s == 'exp':
            return FreeWill_Profile.EXPERIENCED
        elif s == 'crit':
            return FreeWill_Profile.CRITICAL
        elif s == 'stable':
            return FreeWill_Profile.STABLE
        elif s == 'distr':
            return FreeWill_Profile.DISTRACTED
        elif s == 'foc':
            return FreeWill_Profile.FOCUSED
        else:
            return FreeWill_Profile.DISABLED

    @staticmethod
    def parse_from_int(x: int):
        if x == 1:
            return FreeWill_Profile.BUSY
        elif x == 2:
            return FreeWill_Profile.FREE
        elif x == 3:
            return FreeWill_Profile.UNEXPERIENCED
        elif x == 4:
            return FreeWill_Profile.EXPERIENCED
        elif x == 5:
            return FreeWill_Profile.CRITICAL
        elif x == 6:
            return FreeWill_Profile.STABLE
        elif x == 7:
            return FreeWill_Profile.DISTRACTED
        elif x == 8:
            return FreeWill_Profile.FOCUSED
        else:
            return FreeWill_Profile.DISABLED

    def __str__(self):
        return self.value


class Human:
    def __init__(self, name: str, h_id: int, v: int, ptrn: Interaction_Pattern, p_f: Fatigue_Profile,
                 p_fw: FreeWill_Profile, start: Point, dest: Point, dext: int, same_as: int, path: int,
                 fw_model: str = None):
        self.name = name
        self.h_id = h_id
        self.v = v
        self.ptrn = ptrn
        self.p_f = p_f
        self.p_fw = p_fw
        self.start = start
        self.dest = dest
        self.dext = dext
        self.same_as = same_as
        self.path = path
        if fw_model is not None:
            self.fw_model = fw_model
        else:
            self.fw_model = 'random'

    def get_constructor(self):
        if self.ptrn == Interaction_Pattern.APPLICANT:
            return "{} = Human_Applicant({}, {}, {}, {}, {}, {});\n".format(self.name, self.h_id, self.v,
                                                                            self.p_f.to_int(),
                                                                            self.p_fw.to_int(), self.dext, self.path)
        elif self.ptrn == Interaction_Pattern.COMPETITOR:
            return "{} = Human_Competitor({}, {}, {}, {}, {});\n".format(self.name, self.h_id, self.v,
                                                                         self.p_f.to_int(),
                                                                         self.p_fw.to_int(), self.path)
        elif self.ptrn == Interaction_Pattern.FOLLOWER:
            if self.fw_model in ['random', 'errors']:
                return "{} = Human_Follower({}, {}, {}, {}, {}, {});\n".format(self.name, self.h_id, self.v,
                                                                               self.p_f.to_int(),
                                                                               self.p_fw.to_int(), self.same_as,
                                                                               self.path)
            elif self.fw_model == 'cognitive_v1':
                args = "{}_{}, {}_{}, {}_{}, {}_{}, {}_{}, {}_{}".format('arg_us_HF', self.h_id, 'arg_ad_HF',
                                                                         self.h_id, 'SEEAO_HF', self.h_id,
                                                                         'importance_entity_HF', self.h_id,
                                                                         'importance_sense_HF', self.h_id,
                                                                         'obj_xye_HF', self.h_id)
                return "{} = Human_Follower({}, {}, {}, {}, {}, {}, {});\n".format(self.name, self.h_id, self.v,
                                                                                   self.p_f.to_int(),
                                                                                   self.p_fw.to_int(), self.same_as,
                                                                                   self.path, args)
            elif self.fw_model == 'cognitive_v2':
                # TODO: should be configurable
                args = "{}, {}, {}, {}".format(0.01, 0.05, 13.0, -1.9)
                return "{} = Human_Follower({}, {}, {}, {}, {}, {}, {});\n".format(self.name, self.h_id, self.v,
                                                                                   self.p_f.to_int(),
                                                                                   self.p_fw.to_int(), self.same_as,
                                                                                   self.path, args)

        elif self.ptrn == Interaction_Pattern.LEADER:
            if self.fw_model in ['random', 'errors']:
                return "{} = Human_Leader({}, {}, {}, {}, {}, {});\n".format(self.name, self.h_id, self.v,
                                                                             self.p_f.to_int(), self.p_fw.to_int(),
                                                                             self.same_as, self.path)
            elif self.fw_model == 'cognitive_v1':
                args = "{}_{}, {}_{}, {}_{}, {}_{}, {}_{}, {}_{}".format('arg_us_HL', self.h_id, 'arg_ad_HL',
                                                                         self.h_id, 'SEEAO_HL', self.h_id,
                                                                         'importance_entity_HL', self.h_id,
                                                                         'importance_sense_HL', self.h_id,
                                                                         'obj_xye_HL', self.h_id)
                return "{} = Human_Leader({}, {}, {}, {}, {}, {}, {});\n".format(self.name, self.h_id, self.v,
                                                                                 self.p_f.to_int(), self.p_fw.to_int(),
                                                                                 self.same_as, self.path, args)
            elif self.fw_model == 'cognitive_v2':
                # TODO: should be configurable
                args = "{}, {}, {}, {}".format(0.01, 0.05, 13.0, -1.9)
                return "{} = Human_Leader({}, {}, {}, {}, {}, {}, {});\n".format(self.name, self.h_id, self.v,
                                                                                 self.p_f.to_int(), self.p_fw.to_int(),
                                                                                 self.same_as, self.path, args)

        elif self.ptrn == Interaction_Pattern.RECIPIENT:
            if self.fw_model in ['random', 'errors', 'cognitive_v2']:
                return "{} = Human_Recipient({}, {}, {}, {}, {});\n".format(self.name, self.h_id, self.v,
                                                                            self.p_f.to_int(),
                                                                            self.p_fw.to_int(), self.path)
            elif self.fw_model == 'cognitive_v1':
                args = "{}_{}, {}_{}, {}_{}, {}_{}, {}_{}, {}_{}".format('arg_us_HRec', self.h_id, 'arg_ad_HRec',
                                                                         self.h_id, 'SEEAO_HRec', self.h_id,
                                                                         'importance_entity_HRec', self.h_id,
                                                                         'importance_sense_HRec', self.h_id,
                                                                         'obj_xye_HRec', self.h_id)
                return "{} = Human_Recipient({}, {}, {}, {}, {}, {});\n".format(self.name, self.h_id, self.v,
                                                                                self.p_f.to_int(),
                                                                                self.p_fw.to_int(), self.path, args)
        else:
            return "{} = Human_Rescuer({}, {}, {}, {}, {}, {});\n".format(self.name, self.h_id, self.v,
                                                                          self.p_f.to_int(), self.p_fw.to_int(),
                                                                          self.dext, self.path)
