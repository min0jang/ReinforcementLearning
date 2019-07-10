import math
import numpy as np
import gym
from gym import spaces
from gym.utils import seeding

class MountainCarEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self):
        self.counter = 0
        self.min_position = -1.2
        self.max_position = 0.6

        self.min_position_y = 0.35
        self.max_position_y = 0.7

        self.max_speed = 0.01      # need to multiply by 50
        self.goal_position = 0
        self.goal_position_y = 0.25
        self.goal_velocity = 0

        self.record_x = -10
        self.record_y = -10

        self.prev_rel_y = 100

        self.low = np.array([self.min_position, -self.max_speed])
        self.high = np.array([self.max_position, self.max_speed])

        self.viewer = None
        self.action_space = spaces.Discrete(9)

        self.observation_space = spaces.Box(self.low, self.high, dtype=np.float32)

        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        self.counter += 1

        if self.counter == 5000:
            self.counter = 0
            reward = -10000
            done = 1
            return np.array(self.state), reward, done, {}

        #assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        #assert self.action_space_theta.contains(action_theta), "%r (%s) invalid" % (action_theta, type(action_theta))

        position, position_y, velocity, theta, goal_position, goal_position_y  = self.state

        if action == 0:
            d_velocity = -1
            d_theta = -1
        elif action == 1:
            d_velocity = 0
            d_theta = -1
        elif action == 2:
            d_velocity = 1
            d_theta = -1
        elif action == 3:
            d_velocity = -1
            d_theta = 0
        elif action == 4:
            d_velocity = 0
            d_theta = 0
        elif action == 5:
            d_velocity = 1
            d_theta = 0
        elif action == 6:
            d_velocity = -1
            d_theta = 1
        elif action == 7:
            d_velocity = 0
            d_theta = 1
        elif action == 8:
            d_velocity = 1
            d_theta = 1


        velocity += d_velocity*0.001
        velocity = np.clip(velocity, -self.max_speed, self.max_speed)

        theta += (d_theta)*(math.pi/180)

        position += velocity * math.cos(theta)
        position = np.clip(position, self.min_position, self.max_position)

        if (position==self.min_position and velocity<0): velocity = 0

        position_y += velocity * math.sin(theta)
        position_y = np.clip(position_y, self.min_position_y, self.max_position_y)

        # threshold
        thr_position = 0.2
        thr_position_y_up = 0.15
        thr_position_y_down = 0.11
        thr_velocity = 0.03     # abs(velocity) ~ [0.0 ~ 0.005]

        done_pos = bool((position >= self.goal_position-thr_position) & (position <= self.goal_position+thr_position))
        done_pos_y = bool((position_y >= self.goal_position_y-thr_position_y_down) & (position_y <= self.goal_position_y+(thr_position_y_up)))

        done = done_pos# & done_pos_y# & done_vel

        if done:
            self.counter = 0

        rel_y = abs(self.goal_position_y - position_y)

        if rel_y < self.prev_rel_y:
            reward = 0
        else:
            reward = -1 - abs(self.goal_position_y - position_y)*50 #- abs(rel_position_y)*100

        self.state = (position, position_y, velocity, theta, self.goal_position, self.goal_position_y)
        self.prev_rel_y = rel_y
        return np.array(self.state), reward, done, {}

    def reset(self):
        #state : [position, position_y, velocity, theta, goal_position, goal_position_y]
        x = -1.0#self.np_random.uniform(low=-1.0, high=-0.8)
        y = 0.5#self.np_random.uniform(low=0.3, high=0.75)
        theta = 0#-math.pi/3
        self.state = np.array([x, y, 0, theta, self.goal_position, self.goal_position_y])
        return np.array(self.state)

    def _height(self, xs):
        return xs*0+0.5

    def render(self, mode='human'):
        screen_width = 600
        screen_height = 400

        world_width = self.max_position - self.min_position
        scale = screen_width/world_width
        carwidth=40
        carheight=20


        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)

            clearance = 10

            l,r,t,b = -carwidth/2, carwidth/2, carheight, 0
            car = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            car.add_attr(rendering.Transform(translation=(0, clearance)))
            self.cartrans = rendering.Transform()
            car.add_attr(self.cartrans)
            self.viewer.add_geom(car)

            driver = rendering.make_circle(carheight/2.5)
            driver.set_color(.8,.8,0)
            driver.add_attr(rendering.Transform(translation=(-carwidth/4,clearance+10)))
            driver.add_attr(self.cartrans)
            self.viewer.add_geom(driver)

            z,x,c,v = l/3, r/3, t/3, b/3
            topfront_wheel = rendering.FilledPolygon([(z,v), (z,c), (x,c), (x,v)])
            topfront_wheel.set_color(.5, .5, .5)
            topfront_wheel.add_attr(rendering.Transform(translation=(carwidth/4,clearance+carwidth/2)))
            topfront_wheel.add_attr(self.cartrans)
            self.viewer.add_geom(topfront_wheel)

            toprear_wheel = rendering.FilledPolygon([(z,v), (z,c), (x,c), (x,v)])
            toprear_wheel.set_color(.5, .5, .5)
            toprear_wheel.add_attr(rendering.Transform(translation=(-carwidth/4-3,clearance+carwidth/2)))
            toprear_wheel.add_attr(self.cartrans)
            self.viewer.add_geom(toprear_wheel)

            bottomrear_wheel = rendering.FilledPolygon([(z,v), (z,c), (x,c), (x,v)])
            bottomrear_wheel.add_attr(rendering.Transform(translation=(-carwidth/4-3,clearance-6)))
            bottomrear_wheel.add_attr(self.cartrans)
            bottomrear_wheel.set_color(.5, .5, .5)
            self.viewer.add_geom(bottomrear_wheel)

            bottomfront_wheel = rendering.FilledPolygon([(z,v), (z,c), (x,c), (x,v)])
            bottomfront_wheel.add_attr(rendering.Transform(translation=(carwidth/4,clearance-6)))
            bottomfront_wheel.add_attr(self.cartrans)
            bottomfront_wheel.set_color(.5, .5, .5)
            self.viewer.add_geom(bottomfront_wheel)

            flagx = (self.goal_position-self.min_position)*scale
            flagy1 = self._height(self.goal_position)*scale
            flagy2 = flagy1+50
            flagpole = rendering.Line((flagx, flagy1+15), (flagx, flagy2))
            self.viewer.add_geom(flagpole)
            flag = rendering.FilledPolygon([(flagx, flagy2), (flagx, flagy2-10), (flagx+25, flagy2-5)])
            flag.set_color(.8,.3,0)
            self.viewer.add_geom(flag)

            road_top = rendering.Line((self.min_position*scale, self.max_position_y*scale), (self.max_position*scale*3, self.max_position_y*scale))
            self.viewer.add_geom(road_top)
            road_bot = rendering.Line((self.min_position*scale, self.min_position_y*scale), (self.max_position*scale*3, self.min_position_y*scale))
            self.viewer.add_geom(road_bot)

        pos_x = self.state[0]
        pos_y = self.state[1]
        vel = self.state[2]
        theta = self.state[3]

        self.cartrans.set_translation((pos_x-self.min_position)*scale, pos_y*scale)
        self.cartrans.set_rotation(theta)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
