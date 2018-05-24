import os
import zipfile

from tabulate import tabulate

# java -jar cserver.jar -d levels/competition_levels/ -c "python searchclient_py/searchclient.py --max_memory 8192 --strategy custom"

zip_ref = zipfile.ZipFile("../output.out", 'r')
zip_ref.extractall("../output")
zip_ref.close()

logs = os.listdir("../output")
logs = sorted(logs, key=str.casefold)

loglines = []
passed = 0
passed_hash = ""
total_moves = 0
total_time = 0
for log in logs:
    if ".DS_Store" not in log:
        with open("../output/" + log, "r") as f:
            lines = f.readlines()
        lines = [line.rstrip() for line in lines]
        status = lines[-5]  # either 'successful' or 'unsuccessful'
        time = lines[-4]
        moves = lines[-3]
        lvl = log.split('.')[0] + ".lvl"

        if lvl[0:2] != "MA":
            if status == "successful":
                loglines.append([lvl, moves, time])
                passed += 1
                total_moves += int(moves)
                total_time += int(time)
                passed_hash += lvl
            else:
                loglines.append([lvl, "N/A", "N/A"])

passed_hash = hash(passed_hash)

with open("../output.log", "w") as myfile:
    myfile.write(tabulate(loglines, ["lvl", "moves", "time"], tablefmt="plain"))
    myfile.write("\n\nPassed:      {}\nTotal moves: {}\nTotal time:  {:.2f}\n\nPassed hash: {}".format(passed, total_moves, total_time,passed_hash))