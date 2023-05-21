import random
import time
from itertools import count
from random import randint, choice

import numpy as np
import torch

from gym_backgammon.envs import BackgammonEnv
from gym_backgammon.envs.backgammon import WHITE, BLACK, COLORS

random.seed(0)


# AGENT ============================================================================================


class Agent:
    def __init__(self, color):
        self.color = color
        self.name = 'Agent({})'.format(COLORS[color])

    def roll_dice(self):
        return (-randint(1, 6), -randint(1, 6)) if self.color == WHITE else (randint(1, 6), randint(1, 6))

    def choose_best_action(self, actions, env):
        raise NotImplementedError


# RANDOM AGENT =======================================================================================


class RandomAgent(Agent):
    def __init__(self, color):
        super().__init__(color)
        self.name = 'RandomAgent({})'.format(COLORS[color])

    def choose_best_action(self, actions, env):
        return choice(list(actions)) if actions else None


# HUMAN AGENT =======================================================================================


class HumanAgent(Agent):
    def __init__(self, color):
        super().__init__(color)
        self.name = 'HumanAgent({})'.format(COLORS[color])

    def choose_best_action(self, actions=None, env=None):
        pass


# TD-GAMMON AGENT =====================================================================================


class TDAgent(Agent):
    def __init__(self, color, net):
        super().__init__(color)
        self.net = net
        self.name = 'TDAgent({})'.format(COLORS[color])
        self.color = color
        self.opp_color = BLACK if color == WHITE else WHITE

    def choose_best_action(self, actions, env):
        best_action = None

        if actions:
            #values = [0.0] * len(actions)
            obs = []
            tmp_counter = env.counter
            env.counter = 0
            state = env.game.save_state()

            # Iterate over all the legal moves and pick the best action
            for i, action in enumerate(actions):
                observation, reward, done, winner = env.step(action)
                #values[i] = self.net(observation)
                obs.append(observation)
                # restore the board and other variables (undo the action)
                env.game.restore_state(state)

            values = self.net(obs)

            # practical-issues-in-temporal-difference-learning, pag.3
            # ... the network's output P_t is an estimate of White's probability of winning from board position x_t.
            # ... the move which is selected at each time step is the move which maximizes P_t when White is to play and minimizes P_t when Black is to play.
            best_action_index = int(torch.argmax(values)) if self.color == WHITE else int(torch.argmin(values))
            best_action = list(actions)[best_action_index]
            env.counter = tmp_counter

        return best_action

class TDPlyAgent(TDAgent):
    def __init__(self, color, net, ply, top_n_moves=8):
        super().__init__(color, net)
        self.ply = ply
        self.top_n_moves = top_n_moves # TODO: we currently check all moves, but that's inefficient, change it

        self.all_white_rolls = []
        self.all_black_rolls = []

        for first_die in range(1,7):
            for second_die in range(first_die, 7):
                self.all_white_rolls.append((-first_die, -second_die))
                self.all_black_rolls.append((first_die, second_die))

    def choose_best_action(self, actions, env : BackgammonEnv):
        value, action = self._choose_best_action(actions, env, self.ply, self.color)

        return action

    def _choose_best_action(self, actions, env, ply, color):
        best_action = None
        best_action_value = .5

        if not actions:
            return best_action_value, best_action

        if ply == 0:
            obs = []
            tmp_counter = env.counter
            env.counter = 0
            state = env.game.save_state()

            # Iterate over all the legal moves and pick the best action
            for i, action in enumerate(actions):
                observation, reward, done, winner = env.step(action)

                obs.append(observation)
                # restore the board and other variables (undo the action)
                env.game.restore_state(state)

            values = self.net(obs)

            best_action_value = float(torch.max(values)) if color == WHITE else float(torch.min(values))
            best_action_index = int(torch.argmax(values)) if color == WHITE else int(torch.argmin(values))

            best_action = list(actions)[best_action_index]
            env.counter = tmp_counter
        else:
            obs = []
            future_values = []
            tmp_counter = env.counter
            env.counter = 0
            state = env.game.save_state()

            # Iterate over all the legal moves, and then recursively calculate the value of the board after taking that move
            for i, action in enumerate(actions):
                observation, reward, done, winner = env.step(action)

                if done:
                    # if the game is over, we don't need to explore down this action any further
                    future_values.append(winner * ply)
                else:
                    avg_future_value = 0
                    opp_color = env.get_opponent_agent()
                    all_rolls = self.all_white_rolls if opp_color == WHITE else self.all_black_rolls

                    for roll in all_rolls:
                        next_valid_actions = env.get_valid_actions(roll)
                        future_value, future_action = self._choose_best_action(next_valid_actions, env, ply - 1, opp_color)
                        avg_future_value += future_value

                    avg_future_value /= len(all_rolls)
                    future_values.append(avg_future_value)

                obs.append(observation)
                # restore the board and other variables (undo the action)
                env.game.restore_state(state)
                env.current_agent = color


            values = self.net(obs)
            values = values.flatten() + torch.tensor(future_values)

            best_action_value = float(torch.max(values)) if color == WHITE else float(torch.min(values))
            best_action_index = int(torch.argmax(values)) if color == WHITE else int(torch.argmin(values))

            best_action = list(actions)[best_action_index]
            env.counter = tmp_counter

        return best_action_value, best_action

# TD-GAMMON AGENT (play against gnubg) ================================================================


class TDAgentGNU(TDAgent):

    def __init__(self, color, net, gnubg_interface):
        super().__init__(color, net)
        self.gnubg_interface = gnubg_interface

    def roll_dice(self):
        gnubg = self.gnubg_interface.send_command("roll")
        return self.handle_opponent_move(gnubg)

    def choose_best_action(self, actions, env):
        best_action = None

        if actions:
            game = env.game
            values = [0.0] * len(actions)
            state = game.save_state()

            for i, action in enumerate(actions):
                game.execute_play(self.color, action)
                opponent = game.get_opponent(self.color)
                observation = game.get_board_features(opponent) if env.model_type == 'nn' else env.render(mode='state_pixels')
                values[i] = self.net(observation)
                game.restore_state(state)

            best_action_index = int(np.argmax(values)) if self.color == WHITE else int(np.argmin(values))
            best_action = list(actions)[best_action_index]

        return best_action

    def handle_opponent_move(self, gnubg):
        # Once I roll the dice, 2 possible situations can happen:
        # 1) I can move (the value gnubg.roll is not None)
        # 2) I cannot move, so my opponent rolls the dice and makes its move, and eventually ask for doubling, so I have to roll the dice again

        # One way to distinguish between the above cases, is to check the color of the player that performs the last move in gnubg:
        # - if the player's color is the same as the TD Agent, it means I can send the 'move' command (no other moves have been performed after the 'roll' command) - case 1);
        # - if the player's color is not the same as the TD Agent, this means that the last move performed after the 'roll' is not of the TD agent - case 2)
        previous_agent = gnubg.agent
        if previous_agent == self.color:  # case 1)
            return gnubg
        else:  # case 2)
            while previous_agent != self.color and gnubg.winner is None:
                # check if my opponent asks for doubling
                if gnubg.double:
                    # default action if the opponent asks for doubling is 'take'
                    gnubg = self.gnubg_interface.send_command("take")
                else:
                    gnubg = self.gnubg_interface.send_command("roll")
                previous_agent = gnubg.agent
            return gnubg


def evaluate_agents(agents, env, n_episodes):
    wins = {WHITE: 0, BLACK: 0}

    for episode in range(n_episodes):

        agent_color, first_roll, observation = env.reset()
        agent = agents[agent_color]

        t = time.time()

        for i in count():

            if first_roll:
                roll = first_roll
                first_roll = None
            else:
                roll = agent.roll_dice()

            actions = env.get_valid_actions(roll)
            action = agent.choose_best_action(actions, env)
            observation_next, reward, done, winner = env.step(action)

            if done:
                if winner is not None:
                    wins[agent.color] += 1
                tot = wins[WHITE] + wins[BLACK]
                tot = tot if tot > 0 else 1

                print("EVAL => Game={:<6d} | Winner={} | after {:<4} plays || Wins: {}={:<6}({:<5.1f}%) | {}={:<6}({:<5.1f}%) | Duration={:<.3f} sec".format(episode + 1, winner, i,
                    agents[WHITE].name, wins[WHITE], (wins[WHITE] / tot) * 100,
                    agents[BLACK].name, wins[BLACK], (wins[BLACK] / tot) * 100, time.time() - t))
                break

            # one run 34.085 sec, 2nd = 33.850 sec, 3rd 33.988 sec, 34.197 sec 4th
            # w/ batching: 12.83 secs
            agent_color = env.get_opponent_agent()
            agent = agents[agent_color]

            observation = observation_next
    return wins
