ALL_GAMES = [
    'Alien',
    'Amidar',
    'Assault',
    'Asterix',
    'Atlantis',
    'BankHeist',
    'BattleZone',
    'BeamRider',
    'Boxing',
    'Breakout',
    'Carnival',  
    'Centipede',
    'ChopperCommand',
    'CrazyClimber',
    'DemonAttack',
    'DoubleDunk',
    'Enduro',
    'FishingDerby',
    'Freeway',
    'Frostbite',
    'Gopher',
    'Gravitar',
    'Hero',
    'IceHockey',
    'Jamesbond',
    'Kangaroo',
    'Krull',
    'KungFuMaster',
    'MsPacman',
    'NameThisGame',
    'Phoenix',
    'Pong',
    'Pooyan',  
    'Qbert',
    'Riverraid',
    'Robotank',
    'Seaquest',
    'SpaceInvaders',
    'StarGunner',
    'TimePilot',
    'UpNDown',
    'VideoPinball',
    'WizardOfWor',
    'YarsRevenge',
    'Zaxxon'
]

# Test games used in Scaled-QL
TEST_GAMES = [
    'Alien',
    'MsPacman',
    'Pong',
    'SpaceInvaders',
    'StarGunner'
]

TRAIN_GAMES = [game for game in ALL_GAMES if game not in TEST_GAMES]

assert len(TRAIN_GAMES) == 40
assert len(TEST_GAMES) == 5
assert len(ALL_GAMES) == 45