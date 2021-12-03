import random
def get_random_exploration_sequence():
    rotation = int(round(random.random()))
    looks = ["LookUp_15", "LookDown_15"]
    rotations = ["RotateRight_90", "RotateLeft_90"]
    moves = random.randrange(4)
    return looks + [rotations[rotation]] + ['MoveAhead']*moves + looks

