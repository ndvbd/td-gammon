#
import os
import sys
import time
from collections import namedtuple
from itertools import count
import requests
from gym_backgammon.envs.backgammon import Backgammon as Game, WHITE, BLACK, NUM_POINTS, COLORS, assert_board
from gym_backgammon.envs.backgammon_env import STATE_W, STATE_H, SCREEN_W, SCREEN_H
from gym_backgammon.envs.rendering import Viewer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

gnubgState = namedtuple('GNUState', ['agent', 'roll', 'move', 'board', 'double', 'winner',
                                     'n_moves', 'action', 'resigned', 'history'])


class GnubgInterface:
	def __init__(self, host, port):
		self.url = "http://{}:{}".format(host, port)
		# Mapping from gnu board representation to representation used by the environment
		self.gnu_to_idx = {23 - k: k for k in range(NUM_POINTS)}
		# In GNU Backgammon, position 25 (here 24 because I start from 0-24) represents the 'bar' move
		self.gnu_to_idx[24] = 'bar'
		self.gnu_to_idx[-1] = -1

	def send_command(self, command):
		try:
			resp = requests.post(url=self.url, data={"command": command})
			if "match()" in command or "nbspecial" in command or "hint" in command:
				return resp.json()
			elif "set board" in command:
				return resp.json()
			else:
				return self.parse_response(resp.json())
		except Exception as e:
			print("Error during connection to {}: {} (Remember to run gnubg -t -p bridge.py)".format(self.url, e))
			raise
		

	def parse_response(self, response):
		gnubg_board = response["board"]
		action = response["last_move"][-1] if response["last_move"] else None

		info = response["info"][-1] if response["info"] else None

		winner = None
		n_moves = 0
		resigned = False
		double = False
		move = ()
		roll = ()
		agent = None

		if info:
			winner = info['winner']
			n_moves = info['n_moves']
			resigned = info['resigned']

		if action:

			agent = WHITE if action['player'] == 'O' else BLACK

			if action['action'] == "double":
				double = True
			elif 'dice' in action:
				roll = tuple(action['dice'])
				roll = (-roll[0], -roll[1]) if agent == WHITE else (roll[0], roll[1])

			if action['action'] == 'move':
				move = tuple(tuple([self.gnu_to_idx[a - 1], self.gnu_to_idx[b - 1]]) for (a, b) in action['move'])

		return gnubgState(agent=agent, roll=roll, move=move, board=gnubg_board[:], double=double, winner=winner,
		                  n_moves=n_moves, action=action, resigned=resigned, history=response["info"])

	def parse_action(self, action):
		result = ""
		if action:
			for move in action:
				src, target = move
				if src == 'bar':
					result += "bar/{},".format(target + 1)
				elif target == -1:
					result += "{}/off,".format(src + 1)
				else:
					result += "{}/{},".format(src + 1, target + 1)

		return result[:-1]  # remove the last semicolon


class GnubgEnv:
	DIFFICULTIES = ['beginner', 'intermediate', 'advanced', 'world_class']  # I dont see it is used anywhere

	def __init__(self, gnubg_interface, difficulty='beginner', model_type='nn'):
		self.game = Game()
		self.current_agent = WHITE
		self.gnubg_interface = gnubg_interface
		self.gnubg = None
		self.difficulty = difficulty
		self.is_difficulty_set = False
		self.model_type = model_type
		self.viewer = None

	def step(self, action):
		reward = 0
		done = False

		# We are always the white = 0 , playing against gnubg which is black, unless gnubg won, and only then the agent can be BLACK
		assert(self.gnubg.agent == WHITE or (self.gnubg.agent == BLACK and self.gnubg.winner == 'X'))
		
		# if self.gnubg.action['board'] == 'DwAAwLp9IIAAAA':
		# 	delme = 5  # for debug
			
		if action and self.gnubg.winner is None:
			# WHITE (we) makes a move
			action = self.gnubg_interface.parse_action(action)
			self.gnubg = self.gnubg_interface.send_command(action)  # After we (white) move, the agent becomes black (gnu)

		if self.gnubg.double and self.gnubg.winner is None:  # If we are offereed to double cube
			raise 'Should not occur'  # as we turned cube off in the settings
			self.gnubg = self.gnubg_interface.send_command("take")

		if self.gnubg.agent == WHITE and self.gnubg.action['action'] == 'move' and self.gnubg.winner is None:
			if self.gnubg.winner != 'O':  # gnubg.winnder 'O' means 'X' won, which is white, but appear in render as X
				# Since the agent is WHITE again, it means we probably received a resignation request from gnubg, which we
				# can accept or decline
				
				
				if False:  # if we always ACCEPT his resignation requests
					self.gnubg = self.gnubg_interface.send_command("accept")
					assert self.gnubg.winner == 'O', print(self.gnubg)
					assert self.gnubg.action['action'] == 'resign' and self.gnubg.agent == 1 and self.gnubg.action['player'] == 'X'
					print('gnubg resigns')
					assert self.gnubg.resigned
					
				else:  # If we ALWAYS DECLINE gnubg resignations requests
					self.gnubg = self.gnubg_interface.send_command("decline")
					assert self.gnubg.agent == BLACK
					assert self.gnubg.winner is None
					assert self.gnubg.resigned is None

		self.update_game_board(self.gnubg.board)

		observation = self.game.get_board_features(self.current_agent) if self.model_type == 'nn' else self.render(mode='state_pixels')

		winner = self.gnubg.winner  # # gnubg.winnder 'O' means 'X' won, which is white agent, but appear in render as X
		
		if False:  # If we only care about win/lose and not the points
			if winner is not None:
				winner = WHITE if winner == 'O' else BLACK
				if winner == WHITE:
					reward = 1
				done = True
			return observation, reward, done, winner
		else:  # If we care about the points
			if winner is not None:
				winner = WHITE if winner == 'O' else BLACK
				
			reward = self.game.get_reward()
			
			# gnugb_backgammon, updates the state in opposite colors for some reason. so when white wins, the reward
			# can be something like [0,1]... This is I guess a bug in update_game_board()
			# and it is not consistent... so sometimes need to fix it...
			if reward != [0, 0]:
				if winner == WHITE:
					if reward[1] > reward[0]:
						reward = [reward[1], reward[0]]
					else:
						delme = 2  # for debugging
				elif winner == BLACK:
					if reward[0] > reward[1]:
						reward = [reward[1], reward[0]]
					else:
						delme = 2  # for debugging
			
			if reward == [0, 0] and winner is not None:  #  if opponent resigns
				raise 'Should not occur'
				reward = [1, 0] if winner == WHITE else [0, 1]
				
			return observation, reward

	def reset(self):
		# Start a new session in gnubg simulator
		self.gnubg = self.gnubg_interface.send_command("new session")

		if not self.is_difficulty_set:
			self.set_difficulty()

		roll = None if self.gnubg.agent == BLACK else self.gnubg.roll

		self.current_agent = WHITE
		self.game = Game()
		self.update_game_board(self.gnubg.board)

		observation = self.game.get_board_features(self.current_agent) if self.model_type == 'nn' else self.render(mode='state_pixels')
		return observation, roll

	def update_game_board(self, gnu_board):
		# Update the internal board representation with the representation of the gnubg program
		# The gnubg board is represented with two list of 25 elements each, one for each player
		
		gnu_positions_black = gnu_board[0]
		gnu_positions_white = gnu_board[1]
		
		# NB may 2024, I think there's a bug here, the internal representation should be be WHITE as zero
		# gnu_positions_white = gnu_board[WHITE]
		# gnu_positions_black = gnu_board[BLACK]
		
		
		board = [(0, None)] * NUM_POINTS

		for src, checkers in enumerate(gnu_positions_white[:-1]):
			if checkers > 0:
				board[src] = (checkers, WHITE)

		for src, checkers in enumerate(reversed(gnu_positions_black[:-1])):
			if checkers > 0:
				board[src] = (checkers, BLACK)

		self.game.board = board
		# the last element represent the checkers on the bar
		self.game.bar = [gnu_positions_white[-1], gnu_positions_black[-1]]
		# update the players position
		self.game.players_positions = self.game.get_players_positions()
		# off bar
		self.game.off = [15 - sum(gnu_positions_white), 15 - sum(gnu_positions_black)]
		# Just for debugging
		# self.render()
		assert_board(None, self.game.board, self.game.bar, self.game.off)

	def get_valid_actions(self, roll):
		return self.game.get_valid_plays(self.current_agent, roll)

	def set_difficulty(self):
		self.is_difficulty_set = True

		self.gnubg_interface.send_command('set automatic roll off')
		self.gnubg_interface.send_command('set automatic game off')
		
		# NB May 2024, turn off cube doubling
		self.gnubg_interface.send_command('set cube use off')

		if self.difficulty == 'beginner':
			self.gnubg_interface.send_command('set player gnubg chequer evaluation plies 0')
			self.gnubg_interface.send_command('set player gnubg chequer evaluation prune off')
			self.gnubg_interface.send_command('set player gnubg chequer evaluation noise 0.060')
			self.gnubg_interface.send_command('set player gnubg cube evaluation plies 0')
			self.gnubg_interface.send_command('set player gnubg cube evaluation prune off')
			self.gnubg_interface.send_command('set player gnubg cube evaluation noise 0.060')

		elif self.difficulty == 'intermediate':
			self.gnubg_interface.send_command('set player gnubg chequer evaluation plies 0')
			self.gnubg_interface.send_command('set player gnubg chequer evaluation prune off')
			self.gnubg_interface.send_command('set player gnubg cube evaluation plies 0')
			self.gnubg_interface.send_command('set player gnubg chequer evaluation noise 0.040')
			self.gnubg_interface.send_command('set player gnubg cube evaluation prune off')
			self.gnubg_interface.send_command('set player gnubg cube evaluation noise 0.040')

		elif self.difficulty == 'advanced':
			self.gnubg_interface.send_command('set player gnubg chequer evaluation plies 0')
			self.gnubg_interface.send_command('set player gnubg chequer evaluation prune off')
			self.gnubg_interface.send_command('set player gnubg chequer evaluation noise 0.015')
			self.gnubg_interface.send_command('set player gnubg cube evaluation plies 0')
			self.gnubg_interface.send_command('set player gnubg cube evaluation prune off')
			self.gnubg_interface.send_command('set player gnubg cube evaluation noise 0.015')

		elif 'world_class' in self.difficulty:
			if '2plies' in self.difficulty:
				plies = 2
			elif '1plies' in self.difficulty:
				plies = 1
			else:
				raise NotImplementedError
			
			# NB change number of rollout plies here
			# print('WARNING FIX ME - GNUBG_.PY')
			# self.gnubg_interface.send_command('set player gnubg chequer evaluation plies 2')
			self.gnubg_interface.send_command(f'set player gnubg chequer evaluation plies {plies}')  # for some reason got better gnu here 0.2
			
			self.gnubg_interface.send_command('set player gnubg chequer evaluation prune on')
			self.gnubg_interface.send_command('set player gnubg chequer evaluation noise 0.000')
			
			# some explanation about the rollout move filter:
			# https://www.gnu.org/software/gnubg/manual/html_node/The-depth-to-search-and-plies.html
			# https://www.gnu.org/software/gnubg/manual/html_node/Introduction-to-move-filters.html#Introduction-to-move-filters
			# To query the status run command: show player gnubg movefilter
			
			# This means that for a 1-ply checker play, for 0-ply we accept 0 moves, and add 8 moves with 0.160 noise, and we do no pruning for 1-ply.
			self.gnubg_interface.send_command('set player gnubg movefilter 1 0 0 8 0.160')
			
			# means that for 2-ply checker play decision, for 0-ply we accept 0 moves, and add 8 moves with 0.160 noise, and we do no pruning for 1-ply.
			self.gnubg_interface.send_command('set player gnubg movefilter 2 0 0 8 0.160')
			
			self.gnubg_interface.send_command('set player gnubg movefilter 3 0 0 8 0.160')
			self.gnubg_interface.send_command('set player gnubg movefilter 3 2 0 2 0.040')
			
			# This means, that for 4-ply checker play decision, for 0-ply we accept 0 moves, and add 8 moves with 0.160 noise
			# for 1-ply we do no pruning (because there's no line for it), and for ply-2 moves we accept 0 moves, and add 2 moves with 0.040 noise
			self.gnubg_interface.send_command('set player gnubg movefilter 4 0 0 8 0.160')
			self.gnubg_interface.send_command('set player gnubg movefilter 4 2 0 2 0.040')
			
			# print('WARNING FIX ME - GNUBG_.PY')
			# self.gnubg_interface.send_command('set player gnubg cube evaluation plies 2')
			self.gnubg_interface.send_command(f'set player gnubg cube evaluation plies {plies}')
			
			
			self.gnubg_interface.send_command('set player gnubg cube evaluation prune on')
			self.gnubg_interface.send_command('set player gnubg cube evaluation noise 0.000')
		
		elif self.difficulty == 'grandmaster':
			self.gnubg_interface.send_command('set player gnubg chequer evaluation plies 3')
			self.gnubg_interface.send_command('set player gnubg chequer evaluation prune off')
			self.gnubg_interface.send_command('set player gnubg chequer evaluation noise 0.000')
			
			self.gnubg_interface.send_command('set player gnubg cube evaluation plies 3')
			self.gnubg_interface.send_command('set player gnubg cube evaluation prune off')
			self.gnubg_interface.send_command('set player gnubg cube evaluation noise 0.000')
			
			self.gnubg_interface.send_command('set evaluation chequer filter large')
		
		else:
			raise NotImplementedError("unknown gnubg level?")
		
		self.gnubg_interface.send_command('save setting')

	def render(self, mode='human'):
		assert mode in ['human', 'rgb_array', 'state_pixels'], print(mode)

		if mode == 'human':
			self.game.render()
			return True
		else:
			if self.viewer is None:
				self.viewer = Viewer(SCREEN_W, SCREEN_H)

			if mode == 'rgb_array':
				width = SCREEN_W
				height = SCREEN_H

			else:
				assert mode == 'state_pixels', print(mode)
				width = STATE_W
				height = STATE_H

			return self.viewer.render(board=self.game.board, bar=self.game.bar, off=self.game.off, state_w=width, state_h=height)


def evaluate_vs_gnubg(agent, env, n_episodes):
	wins = {WHITE: 0, BLACK: 0}

	for episode in range(n_episodes):
		observation, first_roll = env.reset()
		t = time.time()
		for i in count():
			if first_roll:
				roll = first_roll
				first_roll = None
			else:
				env.gnubg = agent.roll_dice()
				env.update_game_board(env.gnubg.board)
				roll = env.gnubg.roll

			actions = env.get_valid_actions(roll)
			action = agent.choose_best_action(actions, env)

			observation_next, reward, done, info = env.step(action)
			# env.render(mode='rgb_array')

			if done:
				winner = WHITE if env.gnubg.winner == 'O' else BLACK
				wins[winner] += 1
				tot = wins[WHITE] + wins[BLACK]

				print("EVAL => Game={:<6} {:>15} | Winner={} | after {:<4} plays || Wins: {}={:<6}({:<5.1f}%) | gnubg={:<6}({:<5.1f}%) | Duration={:<.3f} sec".format(
					episode + 1, '('+env.difficulty+')', info, env.gnubg.n_moves, agent.name, wins[WHITE], (wins[WHITE] / tot) * 100, wins[BLACK], (wins[BLACK] / tot) * 100, time.time() - t))
				break
			observation = observation_next

	env.gnubg_interface.send_command("new session")
	return wins
