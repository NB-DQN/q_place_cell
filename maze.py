import maze_generator

class Maze:
    def __init__(self, size, goal):
        self.field = maze_generator.MazeGenerator(size, goal).maze_generate()
        self.size = size

    def check_coordinate(self, coordinate):
        if not (0 <= coordinate[0] < self.size[0] and 0 <= coordinate[1] < self.size[1]):
            raise Exception("wrong coordinate")

    def wall(self, coordinate):
        self.check_coordinate(coordinate)
        return self.field[coordinate][0:4]

    def is_goal(self, coordinate):
        self.check_coordinate(coordinate)
        return self.field[coordinate][4]

    def display_cui(self):
        display_str = "  "
        for x in range(0, self.size[0]):
            display_str += "  " + str(x) + " "
        display_str += " \n"

        for y in range(self.size[1] - 1, -1, -1):
            display_str += "  "
            for x in range(0, self.size[0]):
                display_str += "8"
                if self.field[(x, y, 2)] == 1:
                    display_str += "888"
                else:
                    display_str += "   "
            display_str += "8  \n"

            display_str += str(y) + " "
            for x in range(0, self.size[0]):
                if self.field[(x, y, 1)] == 1:
                    display_str += "8"
                else:
                    display_str += " "
                if self.field[(x, y, 4)] == 1:
                    display_str += " G "
                else:
                    display_str += "   "
            if self.field[(self.size[0] - 1, y, 0)] == 1:
                display_str += "8"
            else:
                display_str += " "
            display_str += " " + str(y) + "\n"

        display_str += "  "
        for x in range(0, self.size[0]):
            display_str += "8"
            if self.field[(x, 0, 3)] == 1:
                display_str += "888"
            else:
                display_str += "   "
        display_str += "8\n"

        display_str += "  "
        for x in range(0, self.size[0]):
            display_str += "  " + str(x) + " "
        display_str += " \n"

        print(display_str)

    def dump_params(self):
        csvstr = ""
        for x in range(self.size[0]):
            for y in range(self.size[1]):
                csvstr += ",".join(str(int(i)) for i in self.field[(x, y)].tolist())
                csvstr += ","
            csvstr += "\n"
        return csvstr

if __name__ == "__main__":
    Maze((9, 9), (9, 5)).display_cui()
