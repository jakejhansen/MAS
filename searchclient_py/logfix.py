import os
from tabulate import tabulate
logName = os.listdir("./logs")
logName = sorted(logName)[-2]
res = []
for line in open("logs/" + logName):
    res.append(line.split())


    
with open("logs/" + logName, "w") as myfile:
    myfile.write(tabulate(res, ["lvl", "len", "time"], tablefmt="plain"))

    
print(tabulate(res, ["lvl", "len", "time"], tablefmt="plain"))