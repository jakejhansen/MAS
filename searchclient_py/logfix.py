import os
from tabulate import tabulate
logNames = os.listdir("./logs")


for log_name in sorted(logNames):

    if "_planner" not in log_name and ".DS_Store" not in log_name:

        cond3 = None
        with open("logs/" + log_name, "r") as f:
            first_line = f.readline()
            cond3 = ".lvl" in first_line

        if cond3:
            loglines = []
            total_moves = 0
            total_time = 0
            for line in open("logs/" + log_name):
                lvl, moves, time = line.split()
                total_moves += int(moves)
                total_time += float(time)
                loglines.append([lvl, moves, time])

            loglines = sorted(loglines, key=str.lower)

            with open("logs/" + log_name, "w") as myfile:
                myfile.write(tabulate(loglines, ["lvl", "len", "time"], tablefmt="plain"))
                myfile.write("\n\nTotal moves: {}\nTotal time:  {:.2f}".format(total_moves, total_time))

            print()
            print(tabulate(loglines, ["lvl", "len", "time"], tablefmt="plain"))
            print("\n\nTotal moves: {}\nTotal time:  {:.2f}\n".format(total_moves, total_time))