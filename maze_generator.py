import random
import numpy as np

class MazeGenerator:
    def __init__(self, size):
        self.size = size
        self.cell_size = (self.size[0] * 2 + 3, self.size[1] * 2 + 3)

    def maze_generate(self):
        field = np.zeros(self.cell_size)
        field[1:(self.cell_size[0] - 1), 1:(self.cell_size[1] - 1)] = 9

        # main backtracking loop
        # start from coordinate(0, 0) => field_cell(2, 2)
        active_cells = [(2, 2)]
        field[active_cells[0]] = 0
        while len(active_cells) > 0:
            current_cell = active_cells[-1]

            candidate_cells = [ \
                ((current_cell[0] + 2, current_cell[1]    ), (current_cell[0] + 1, current_cell[1]    )), \
                ((current_cell[0] - 2, current_cell[1]    ), (current_cell[0] - 1, current_cell[1]    )), \
                ((current_cell[0]    , current_cell[1] + 2), (current_cell[0]    , current_cell[1] + 1)), \
                ((current_cell[0]    , current_cell[1] - 2), (current_cell[0]    , current_cell[1] - 1))]

            candidate_directions = []
            for direction in range(0, 4):
                if field[candidate_cells[direction][0]] == 9:
                    candidate_directions.append(direction)

            if len(candidate_directions) > 0:
                direction = candidate_directions[random.randint(0, len(candidate_directions) - 1)]
                next_cell = candidate_cells[direction][0]
                field[next_cell] = 0
                active_cells.append(next_cell)
                field[candidate_cells[direction][1]] = 0
            else:
                active_cells.pop(-1)

        # create exit -- may be refactorable
        periphery = []
        for x in range(2, self.cell_size[0] - 2, 2):
            periphery.append(((x, self.cell_size[1] - 3), (x, self.cell_size[1] - 2)))
        for y in range(4, self.cell_size[1] - 4, 2):
            periphery.append(((self.cell_size[0] - 3, y), (self.cell_size[0] - 2, y)))

        exit = periphery[random.randint(0, len(periphery) - 1)]
        field[exit[0]] = 1
        field[exit[1]] = 0

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
