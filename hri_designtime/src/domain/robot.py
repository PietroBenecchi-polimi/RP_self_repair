from src.domain.layout import Point


class Robot:
    def __init__(self, name: str, r_id: int, v: int, a: int, start: Point, chg: int):
        self.name = name
        self.r_id = r_id
        self.v = v
        self.a = a
        self.start = start
        self.chg = chg

    def get_constructor(self):
        return "{} = Robot({}, {}, {}, {:.2f}, {:.2f});\n{} = Battery({}, {});\nr_pub_{} = ROS_SensPub({}, 0.0, 0.00);\n" \
            .format(self.name, self.r_id, self.v, self.a, self.start.x,
                    self.start.y, "b_{}".format(self.name), self.r_id, self.chg, self.r_id, self.r_id)

    def get_orch_constructor(self):
        return "o_{} = Orchestrator({});\nopchk_{} = OpChk({}, 1, 0);\n".format(self.r_id, self.r_id, self.r_id,
                                                                                self.r_id)
