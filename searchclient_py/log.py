import os
import sys
from time import localtime, strftime

directory = "logs/"
filename = strftime("%Y-%m-%d-%H-%M-%S", localtime())
log_name = directory + filename


if not os.path.exists(directory):
    os.makedirs(directory)

open(log_name, 'a').close()

sys.exit(log_name)