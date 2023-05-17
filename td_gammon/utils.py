import os
import time

import gym
import sys
from agents import TDAgent, HumanAgent, TDAgentGNU, RandomAgent, evaluate_agents
from gnubg.gnubg_backgammon import GnubgInterface, GnubgEnv, evaluate_vs_gnubg
from gym_backgammon.envs.backgammon import WHITE, BLACK
from model import TDGammon, TDGammonCNN
from web_gui.gui import GUI
from torch.utils.tensorboard import SummaryWriter

#  tensorboard --logdir=runs/ --host localhost --port 8001


def write_file(path, **kwargs):
    with open('{}/parameters.txt'.format(path), 'w+') as file:
        print("Parameters:")
        for key, value in kwargs.items():
            file.write("{}={}\n".format(key, value))
            print("{}={}".format(key, value))
        print()


def path_exists(path):
    if os.path.exists(path):
        return True
    else:
        print("The path {} doesn't exists".format(path))
        sys.exit()



# ==================================== TRAINING PARAMETERS ===================================
def launch_train(args, net, num_episodes, eval_step, barrier):
    if args.type == 'nn':
        env = gym.make('gym_backgammon:backgammon-v0')
    else:
        env = gym.make('gym_backgammon:backgammon-pixel-v0')

    net.train_agent(env=env, n_episodes=num_episodes, eval_step=eval_step, eval_opponent=args.eval_opponent, eval_hidden_units=args.eval_hidden_units,
                    eligibility=True, name_experiment=args.name, barrier=barrier)

def args_train(args):
    n_episodes = args.episodes
    init_weights = args.init_weights
    lr = args.lr
    hidden_units = args.hidden_units
    lamda = args.lamda
    name = args.name
    model_type = args.type
    seed = args.seed

    eligibility = False
    optimizer = None

    if model_type == 'nn':
        net = TDGammon(hidden_units=hidden_units, lr=lr, lamda=lamda, init_weights=init_weights, seed=seed, save_path=args.save_path)
        eligibility = True
        env = gym.make('gym_backgammon:backgammon-v0')

    else:
        net = TDGammonCNN(lr=lr, seed=seed)
        optimizer = True
        env = gym.make('gym_backgammon:backgammon-pixel-v0')

    if args.model and path_exists(args.model):
        # assert os.path.exists(args.model), print("The path {} doesn't exists".format(args.model))
        net.load(checkpoint_path=args.model, optimizer=optimizer, eligibility_traces=eligibility)

    if args.save_path and path_exists(args.save_path):
        # assert os.path.exists(args.save_path), print("The path {} doesn't exists".format(args.save_path))

        write_file(
            args.save_path, save_path=args.save_path, command_line_args=args, type=model_type, hidden_units=hidden_units, init_weights=init_weights, alpha=net.lr, lamda=net.lamda,
            n_episodes=n_episodes, start_episode=net.start_episode, name_experiment=name, env=env.spec.id, restored_model=args.model, seed=seed,
            eligibility=eligibility, optimizer=optimizer, modules=[module for module in net.modules()]
        )

    if args.processes == 1:
        net.train_agent(env=env, n_episodes=n_episodes, eval_step=args.eval_step,
                        eligibility=eligibility, name_experiment=name)
    else:
        net.share_memory()

        barrier = mp.Barrier(args.processes)
        start = time.time()
        processes = []

        for process_num in range(args.processes):
            # this evenly distributes the episodes across the processes
            num_train_eps_per_process = n_episodes // args.processes + (process_num < n_episodes % args.processes)

            # we also evenly distribute the eval steps across the processes
            eval_step = args.eval_step // args.processes

            p = mp.Process(target=launch_train, args=(args, net, num_train_eps_per_process, eval_step, barrier))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        end = time.time()

        if args.save_path:
            net.checkpoint(checkpoint_path=net.save_path, step=n_episodes - 1, name_experiment=name)
        print("Total training time: {}".format(end - start))


# ==================================== WEB GUI PARAMETERS ====================================
def args_gui(args):
    if path_exists(args.model):
        # assert os.path.exists(args.model), print("The path {} doesn't exists".format(args.model))

        if args.type == 'nn':
            net = TDGammon(hidden_units=args.hidden_units, lr=0.1, lamda=None, init_weights=False)
            env = gym.make('gym_backgammon:backgammon-v0')
        else:
            net = TDGammonCNN(lr=0.0001)
            env = gym.make('gym_backgammon:backgammon-pixel-v0')

        net.load(checkpoint_path=args.model, optimizer=None, eligibility_traces=False)

        agents = {BLACK: TDAgent(BLACK, net=net), WHITE: HumanAgent(WHITE)}
        gui = GUI(env=env, host=args.host, port=args.port, agents=agents)
        gui.run()


# =================================== EVALUATE PARAMETERS ====================================
def args_evaluate(args):
    model_agent0 = args.model_agent0
    model_agent1 = args.model_agent1
    model_type = args.type
    hidden_units_agent0 = args.hidden_units_agent0
    hidden_units_agent1 = args.hidden_units_agent1
    n_episodes = args.episodes

    if path_exists(model_agent0) and path_exists(model_agent1):
        # assert os.path.exists(model_agent0), print("The path {} doesn't exists".format(model_agent0))
        # assert os.path.exists(model_agent1), print("The path {} doesn't exists".format(model_agent1))

        if model_type == 'nn':
            net0 = TDGammon(hidden_units=hidden_units_agent0, lr=0.1, lamda=None, init_weights=False)
            net1 = TDGammon(hidden_units=hidden_units_agent1, lr=0.1, lamda=None, init_weights=False)
            env = gym.make('gym_backgammon:backgammon-v0')
        else:
            net0 = TDGammonCNN(lr=0.0001)
            net1 = TDGammonCNN(lr=0.0001)
            env = gym.make('gym_backgammon:backgammon-pixel-v0')

        net0.load(checkpoint_path=model_agent0, optimizer=None, eligibility_traces=False)
        net1.load(checkpoint_path=model_agent1, optimizer=None, eligibility_traces=False)

        agents = {WHITE: TDAgent(WHITE, net=net1), BLACK: TDAgent(BLACK, net=net0)}

        evaluate_agents(agents, env, n_episodes)


# ===================================== GNUBG PARAMETERS =====================================
import torch.multiprocessing as mp
import threading

def launch_gnubg_eval(args, model, difficulty, proc, num_episodes):
    port = args.port + proc
    gnubg_interface = GnubgInterface(host=args.host, port=port)
    gnubg_env = GnubgEnv(gnubg_interface, difficulty=difficulty, model_type=args.type)
    wins = evaluate_vs_gnubg(agent=TDAgentGNU(WHITE, net=model, gnubg_interface=gnubg_interface), env=gnubg_env,
                      n_episodes=num_episodes)

    args.queue.put(wins[WHITE])

def args_gnubg(args):
    model_agent0 = args.model_agent0
    model_type = args.type
    hidden_units_agent0 = args.hidden_units_agent0
    n_episodes = args.episodes
    host = args.host
    port = args.port
    difficulty = args.difficulty

    args.queue = mp.Queue()

    if path_exists(model_agent0):
        # assert os.path.exists(model_agent0), print("The path {} doesn't exists".format(model_agent0))
        if model_type == 'nn':
            net0 = TDGammon(hidden_units=hidden_units_agent0, lr=0.1, lamda=None, init_weights=False)
        else:
            net0 = TDGammonCNN(lr=0.0001)

        net0.load(checkpoint_path=model_agent0, optimizer=None, eligibility_traces=False)
        net0.share_memory()

        start = time.time()

        processes = []
        num_processes = args.num_servers
        for proc in range(num_processes):
            num_eval_eps = n_episodes // num_processes + (proc < n_episodes % num_processes) # I don't remember how this works, but it makes sure that the total number of episodes is correct even if n_episodes is not divisible by num_processes
            p = threading.Thread(target=launch_gnubg_eval, args=(args, net0, args.difficulty, proc, num_eval_eps))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        end = time.time()
        print(f'It took {end - start} seconds to complete all {n_episodes} episodes')

        wins = 0
        for _ in range(num_processes):
            wins += args.queue.get()

        print(f'Agent won {wins} out of {n_episodes} episodes for a win rate of {wins / n_episodes}')


# ===================================== PLOT PARAMETERS ======================================
def args_plot(args, parser):
    '''
    This method is used to plot the number of time an agent wins when it plays against an opponent.
    Instead of evaluating the agent during training (it can require some time and slow down the training), I decided to plot the wins separately, loading the different
    model saved during training.
    For example, suppose I run the training for 100 games and save my model every 10 games.
    Later I will load these 10 models, and for each of them, I will compute how many times the agent would win against an opponent.
    :return: None
    '''

    src = args.save_path
    hidden_units = args.hidden_units
    n_episodes = args.episodes
    opponents = args.opponent.split(',')
    host = args.host
    port = args.port
    difficulties = args.difficulty.split(',')
    model_type = args.type

    args.queue = mp.Queue()

    if path_exists(src):
        # assert os.path.exists(src), print("The path {} doesn't exists".format(src))

        for d in difficulties:
            if d not in ['beginner', 'intermediate', 'advanced', 'world_class']:
                parser.error("--difficulty should be (one or more of) 'beginner','intermediate', 'advanced' ,'world_class'")

        dst = args.dst

        if 'gnubg' in opponents and (not host or not port):
            parser.error("--host and --port are required when 'gnubg' is specified in --opponent")

        for root, dirs, files in os.walk(src):
            global_step = 0
            files = sorted(files)

            writer = SummaryWriter(dst)

            for file in files:
                if ".tar" in file:
                    print("\nLoad {}".format(os.path.join(root, file)))

                    if model_type == 'nn':
                        net = TDGammon(hidden_units=hidden_units, lr=0.1, lamda=None, init_weights=False)
                        env = gym.make('gym_backgammon:backgammon-v0')
                    else:
                        net = TDGammonCNN(lr=0.0001)
                        env = gym.make('gym_backgammon:backgammon-pixel-v0')

                    net.load(checkpoint_path=os.path.join(root, file), optimizer=None, eligibility_traces=False)
                    net.share_memory()

                    if 'gnubg' in opponents:
                        tag_scalar_dict = {}

                        for difficulty in difficulties:

                            start = time.time()

                            processes = []
                            num_processes = args.num_servers
                            for proc in range(num_processes):
                                num_eval_eps = n_episodes // num_processes + (
                                            proc < n_episodes % num_processes)  # I don't remember how this works, but it makes sure that the total number of episodes is correct even if n_episodes is not divisible by num_processes
                                p = threading.Thread(target=launch_gnubg_eval, args=(args, net, difficulty, proc, num_eval_eps))
                                p.start()
                                processes.append(p)

                            for p in processes:
                                p.join()

                            wins = 0
                            for _ in range(num_processes):
                                wins += args.queue.get()

                            end = time.time()
                            print(f'It took {end - start} seconds to complete all {n_episodes} episodes for {difficulty} difficulty')

                            tag_scalar_dict[difficulty] = wins

                        writer.add_scalars('wins_vs_gnubg/', tag_scalar_dict, global_step)

                        with open(root + '/results.txt', 'a') as f:
                            print("{};".format(file) + str(tag_scalar_dict), file=f)

                    if 'random' in opponents:
                        tag_scalar_dict = {}
                        agents = {WHITE: TDAgent(WHITE, net=net), BLACK: RandomAgent(BLACK)}
                        wins = evaluate_agents(agents, env, n_episodes)

                        tag_scalar_dict['random'] = wins[WHITE]

                        writer.add_scalars('wins_vs_random/', tag_scalar_dict, global_step)

                    global_step += 1

                    writer.close()
