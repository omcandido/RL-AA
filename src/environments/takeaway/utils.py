from enum import IntEnum

# Takeaway Actions.
class Actions(IntEnum):
    TackleBall   = 0
    MarkKeeper_2 = 1
    MarkKeeper_3 = 2
    MarkKeeper_4 = 3

arg_actions = {
    # Arguments T1
    'TackleBall1': Actions.TackleBall,
    'OpenKeeper1,2': Actions.MarkKeeper_2,
    'FarKeeper1,2': Actions.MarkKeeper_2,
    'MinAngle1,2': Actions.MarkKeeper_2,
    'MinDist1,2': Actions.MarkKeeper_2,
    'OpenKeeper1,3': Actions.MarkKeeper_3,
    'FarKeeper1,3': Actions.MarkKeeper_3,
    'MinAngle1,3': Actions.MarkKeeper_3,
    'MinDist1,3': Actions.MarkKeeper_3,
    'OpenKeeper1,4': Actions.MarkKeeper_4,
    'FarKeeper1,4': Actions.MarkKeeper_4,
    'MinAngle1,4': Actions.MarkKeeper_4,
    'MinDist1,4': Actions.MarkKeeper_4,
    # Arguments T2
    'TackleBall2': Actions.TackleBall,
    'OpenKeeper2,2': Actions.MarkKeeper_2,
    'FarKeeper2,2': Actions.MarkKeeper_2,
    'MinAngle2,2': Actions.MarkKeeper_2,
    'MinDist2,2': Actions.MarkKeeper_2,
    'OpenKeeper2,3': Actions.MarkKeeper_3,
    'FarKeeper2,3': Actions.MarkKeeper_3,
    'MinAngle2,3': Actions.MarkKeeper_3,
    'MinDist2,3': Actions.MarkKeeper_3,
    'OpenKeeper2,4': Actions.MarkKeeper_4,
    'FarKeeper2,4': Actions.MarkKeeper_4,
    'MinAngle2,4': Actions.MarkKeeper_4,
    'MinDist2,4': Actions.MarkKeeper_4,
    # Arguments T3
    'TackleBall3': Actions.TackleBall,
    'OpenKeeper3,2': Actions.MarkKeeper_2,
    'FarKeeper3,2': Actions.MarkKeeper_2,
    'MinAngle3,2': Actions.MarkKeeper_2,
    'MinDist3,2': Actions.MarkKeeper_2,
    'OpenKeeper3,3': Actions.MarkKeeper_3,
    'FarKeeper3,3': Actions.MarkKeeper_3,
    'MinAngle3,3': Actions.MarkKeeper_3,
    'MinDist3,3': Actions.MarkKeeper_3,
    'OpenKeeper3,4': Actions.MarkKeeper_4,
    'FarKeeper3,4': Actions.MarkKeeper_4,
    'MinAngle3,4': Actions.MarkKeeper_4,
    'MinDist3,4': Actions.MarkKeeper_4,
}