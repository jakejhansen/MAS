import os
from tabulate import tabulate
logNames = os.listdir("./logs")


for logName in sorted(logNames):
    if "_strategy" not in logName and ".DS_Store" not in logName:

        res = []
        for line in open("logs/" + logName):
            res.append(line.split())



        with open("logs/" + logName, "w") as myfile:
            myfile.write(tabulate(res, ["lvl", "len", "time"], tablefmt="plain"))


        print(tabulate(res, ["lvl", "len", "time"], tablefmt="plain"))