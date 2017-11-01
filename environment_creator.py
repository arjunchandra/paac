class EnvironmentCreator(object):

    def __init__(self, args):
        """
        Creates an object from which new environments can be created
        :param args:
        """
        if args.experiment_type == 'atari':
            from atari_emulator import AtariEmulator
            from ale_python_interface import ALEInterface
            filename = args.rom_path + "/" + args.game + ".bin"
            ale_int = ALEInterface()
            ale_int.loadROM(str.encode(filename))
            self.num_actions = len(ale_int.getMinimalActionSet())
            self.create_environment = lambda i: AtariEmulator(i, args)
        
        elif args.experiment_type == 'corridor':
            corridor_envs = {
                    1:('FrozenLake-v0', None),
                    2:('FrozenLakeNonskid4x4-v0', None),
                    3:('FrozenLakeNonskid8x8-v0', None),
                    4:('CorridorSmall-v1', CorridorEnv),
                    5:('CorridorSmall-v2', CorridorEnv),
                    6:('CorridorActionTest-v0', CorridorEnv),
                    7:('CorridorActionTest-v1', ComplexActionSetCorridorEnv),
                    8:('CorridorBig-v0', CorridorEnv),
                    9:('CorridorFLNonSkid-v1', CorridorEnv)
                }
    
            from corridor_emulator import *
            corridor_game_id = args.game
            corridor_class = corridor_envs[args.game]
            self.num_actions = GymEnvironment(-1, corridor_game_id, corridor_class).gym_actions
            self.create_environment = lambda i: GymEnvironment(i, args)
