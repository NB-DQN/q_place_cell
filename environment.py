import math
import numpy as np
import maze_generator
import maze

class Environment:
    def __init__(self, size, goal_location):
        self.size = size
        self.goal = goal_location
        self.maze = maze.Maze(self.size, self.goal)
        self.current_coordinate = (0, 0)
        self.move_count = 0

    def coordinate_id(self):
        if self.current_coordinate[1] == 0:
            cid = self.current_coordinate[0]
        elif self.current_coordinate[0] == self.size[1] - 1:
            cid = self.current_coordinate[1] + self.size[0] -1
        elif self.current_coordinate[1] == self.size[1] -1:
            cid = (self.size[0] * 2 + self.size[1] - 3) - self.current_coordinate[0]
        elif self.current_coordinate[0] == 0:
            cid = (self.size[0] + self.size[1] -2) * 2  - self.current_coordinate[1]
        return cid

    def get_coordinate_from_id(self, cid):
        if 0 <= cid and cid < self.size[0]:
            x = cid
            y = 0
        elif self.size[0] <= cid and cid < self.size[0] + self.size[1] -1:
            x = self.size[0] -1
            y = cid - x
        elif self.size[0] + self.size[1] -1 <= cid and cid < self.size[0] * 2 + self.size[1] -2:
            x = self.size[0] * 2 + self.size[1] -3 - cid
            y = self.size[1] -1
        elif self.size[0] * 2 + self.size[1] -2 <= cid:
            x = 0
            y = (self.size[0] + self.size[1] -2) * 2 - cid

        return (x, y)

    def wall(self, cid=None):
        if cid is None:
            cid = self.coordinate_id()
        return self.maze.wall(self.get_coordinate_from_id(cid))

    def move(self, direction):
        neighbor = [ \
            (self.current_coordinate[0] + 1, self.current_coordinate[1]    ), \
            (self.current_coordinate[0] - 1, self.current_coordinate[1]    ), \
            (self.current_coordinate[0]    , self.current_coordinate[1] + 1), \
            (self.current_coordinate[0]    , self.current_coordinate[1] - 1)]
        if self.wall()[direction] == 0:
            self.current_coordinate = neighbor[direction]
            self.move_count += 1

    def get_goal(self):
        if self.move_count > 200:
            return self.maze.is_goal(self.current_coordinate)
        else:
            return 0

    def visual_targets(self):
        return [ \
            (self.size[0], self.size[1]), \
            (          -1, self.size[1]), \
            (          -1,           -1), \
            (self.size[0],           -1)]

    def visual_distance(self):
        distance = []
        for target in self.visual_targets():
            distance.append(math.sqrt( \
                (self.current_coordinate[0] - target[0]) ** 2 + \
                (self.current_coordinate[1] - target[1]) ** 2))
        return distance

    def visual_image(self, cid=None):
        if cid is None:
            cid = self.coordinate_id()
        coordinate = self.get_coordinate_from_id(cid)

        DEGREE_PER_DOT = 6

        image = np.zeros(360 / DEGREE_PER_DOT)
        for target in self.visual_targets():
            distance = math.sqrt( \
                (coordinate[0] - target[0]) ** 2 + \
                (coordinate[1] - target[1]) ** 2)
            visual_width = math.degrees(math.atan(0.5 / distance))
            angle = math.degrees(math.atan2(target[1] - coordinate[1], target[0] - coordinate[0]))
            if angle < 0:
                angle += 360

            visual_range = [round(i / DEGREE_PER_DOT) for i in [angle - visual_width, angle + visual_width]]
            image[visual_range[0]:(visual_range[1] + 1)] = 1
        return image

    def reset(self):
        self.current_coordinate = (0, 0)
        self.move_count = 0