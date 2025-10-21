
class fitness():
    def __init__(self):
        pass

    def error_accumulation(self):
        # ranking individuals based on how much error they accumulate through their gait.
        # e.g. legs going into body, legs over lapping
        pass

    def leg_angles(self):
        # rank individuals based on two leg pair angles,
        # where one leg moves clockwise and the other anti clockwise 
        # and the pair of legs behind are flipped
        pass

    def pose(self):
        # rank individuals based on keyframe pose
        # every x keyframes check how close indivudal to reaching rotation of keyframe.
        # the closer the better fitness
        pass

