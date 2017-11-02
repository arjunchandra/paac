from corridor_emulator import *

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
                    'FrozenLake-v0': None,
                    'FrozenLakeNonskid4x4-v0': None,
                    'FrozenLakeNonskid8x8-v0': None,
                    'CorridorSmall-v1': CorridorEnv,
                    'CorridorSmall-v2': CorridorEnv,
                    'CorridorActionTest-v0': CorridorEnv,
                    'CorridorActionTest-v1': ComplexActionSetCorridorEnv,
                    'CorridorBig-v0': CorridorEnv,
                    'CorridorFLNonSkid-v1': CorridorEnv
                }
            
            corridor_game_id = args.game
            corridor_class = corridor_envs[args.game]
            self.num_actions = GymEnvironment(-1, corridor_game_id, corridor_class).num_actions
            self.create_environment = lambda i: GymEnvironment(i, corridor_game_id, corridor_class)
