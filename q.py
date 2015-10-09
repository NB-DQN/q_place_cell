import modules

size = (9, 9)
environment = modules.environment.Environment(size)
agent = modules.agent.QAgent(environment)

environment.maze.display_cui()

for i in range(0, 1000):
    while not environment.exit():
        agent.choose_action()
    environment.reset()

# test in trained environment
f = open('q_trained.log', 'w')

f.write(",".join(str(i) for i in size))
f.write("\n")
f.write(environment.maze.dump_params())
f.write("\n")
f.write("0,0")
f.write("\n")

while not environment.exit():
    agent.choose_action()
    f.write(",".join(str(i) for i in environment.current_coordinate))
    f.write('\n')

f.close()


# test in untrained environment
environment = modules.environment.Environment(size)
agent.environment = environment

f = open('q_untrained.log', 'w')

f.write(",".join(str(i) for i in size))
f.write("\n")
f.write(environment.maze.dump_params())
f.write("\n")
f.write("0,0")
f.write("\n")

while not environment.exit():
    agent.choose_action()
    f.write(",".join(str(i) for i in environment.current_coordinate))
    f.write('\n')

f.close()
