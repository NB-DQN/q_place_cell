import random
import numpy as np

class MazeGenerator:
    def __init__(self, size, goal):
        self.size = size
        self.cell_size = (self.size[0] * 2 + 3, self.size[1] * 2 + 3)
        self.goal = goal

    def maze_generate(self):
        field = np.zeros(self.cell_size)
        for x in range(self.cell_size[0]):
            if x == 1 or x == self.cell_size[0] - 2:
                for y in range(1, self.cell_size[1] - 1):
                    field[x, y] = 9
            if x == 2 or x == self.cell_size[0] - 3:
                field[x, 1] = 9
                field[x, self.cell_size[1] -2] = 9
            if x == 3 or x == self.cell_size[0] - 4:
                for y in range(3, self.cell_size[1] - 3):
                    field[x,y] = 9
            elif x > 3 and x < self.cell_size[0] - 4:
                for y in [1, 3, self.cell_size[1] - 2, self.cell_size[1] - 4]:
                    field[x, y] = 9

        # goal
        field[(self.goal[0] * 2, self.goal[1] * 2)] = 1


        return self.rearrange(field)
        
    def rearrange(self, raw_field):
        # each cell in the field containes 5-element array (all elemtns are binary).
        # first 4 bits indicate if there are wall around the cell (1 means wall).
        # last bit indicate if the cell is the exit (1 means exit).
        # i.e. { wall_1, wall_2, wall_3, wall_4, exit }
        field = np.zeros(self.size + (5, ))

        for x in range(0, self.size[0]):
            for y in range(0, self.size[1]):
                current_cell = (x * 2 + 2, y * 2 + 2)
                neighbor = [ \
                    (current_cell[0] + 1, current_cell[1]    ), \
                    (current_cell[0] - 1, current_cell[1]    ), \
                    (current_cell[0]    , current_cell[1] + 1), \
                    (current_cell[0]    , current_cell[1] - 1)]
                for direction in range(0, 4):
                    if raw_field[neighbor[direction]] == 9:
                        field[x][y][direction] = 1
                if raw_field[current_cell] == 1:
                    field[x][y][4] = 1

        return field

if __name__ == '__main__':
    print(MazeGenerator((9, 9)).maze_generate())
