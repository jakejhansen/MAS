#!/usr/bin/env bash

# COMPETITION LEVELS
log_name=$(python3 log.py 2>&1)
max_time=180

timeout $max_time python3 searchclient.py --max_memory 8192 --lvl "competition_levels/SAAIFather.lvl" --strategy "custom" --log $log_name &     # PASSED
timeout $max_time python3 searchclient.py --max_memory 8192 --lvl "competition_levels/SAAiAiCap.lvl" --strategy "custom" --log $log_name &      # PASSED
#timeout $max_time python3 searchclient.py --max_memory 8192 --lvl "competition_levels/SAAiMasTers.lvl" --strategy "custom" --log $log_name &    # HANG
timeout $max_time python3 searchclient.py --max_memory 8192 --lvl "competition_levels/SAAlphaOne.lvl" --strategy "custom" --log $log_name &     # PASSED
#timeout $max_time python3 searchclient.py --max_memory 8192 --lvl "competition_levels/SAAntsStar.lvl" --strategy "custom" --log $log_name &     # TIME
timeout $max_time python3 searchclient.py --max_memory 8192 --lvl "competition_levels/SABahaMAS.lvl" --strategy "custom" --log $log_name &      # PASSED
timeout $max_time python3 searchclient.py --max_memory 8192 --lvl "competition_levels/SABeTrayEd.lvl" --strategy "custom" --log $log_name &     # PASSED
#timeout $max_time python3 searchclient.py --max_memory 8192 --lvl "competition_levels/SAByteMe.lvl" --strategy "custom" --log $log_name &       # TIME
#timeout $max_time python3 searchclient.py --max_memory 8192 --lvl "competition_levels/SACybot.lvl" --strategy "custom" --log $log_name &        # TIME
#timeout $max_time python3 searchclient.py --max_memory 8192 --lvl "competition_levels/SADaVinci.lvl" --strategy "custom" --log $log_name &      # TIME
#timeout $max_time python3 searchclient.py --max_memory 8192 --lvl "competition_levels/SAEasyPeasy.lvl" --strategy "custom" --log $log_name &    # TIME
#timeout $max_time python3 searchclient.py --max_memory 8192 --lvl "competition_levels/SAGreenDots.lvl" --strategy "custom" --log $log_name &    # NOT PASSED ANYMORE
timeout $max_time python3 searchclient.py --max_memory 8192 --lvl "competition_levels/SAJMAI.lvl" --strategy "custom" --log $log_name &         # PASSED
timeout $max_time python3 searchclient.py --max_memory 8192 --lvl "competition_levels/SAKJFWAOL.lvl" --strategy "custom" --log $log_name &      # PASSED
#timeout $max_time python3 searchclient.py --max_memory 8192 --lvl "competition_levels/SAKaldi.lvl" --strategy "custom" --log $log_name &        # HANG
#timeout $max_time python3 searchclient.py --max_memory 8192 --lvl "competition_levels/SAKarlMarx.lvl" --strategy "custom" --log $log_name &     # HANG
timeout $max_time python3 searchclient.py --max_memory 8192 --lvl "competition_levels/SALobot.lvl" --strategy "custom" --log $log_name &        # PASSED
#timeout $max_time python3 searchclient.py --max_memory 8192 --lvl "competition_levels/SAMagicians.lvl" --strategy "custom" --log $log_name &    # TIME
#timeout $max_time python3 searchclient.py --max_memory 8192 --lvl "competition_levels/SANavy.lvl" --strategy "custom" --log $log_name &         # TIME
timeout $max_time python3 searchclient.py --max_memory 8192 --lvl "competition_levels/SANikrima.lvl" --strategy "custom" --log $log_name &      # PASSED
timeout $max_time python3 searchclient.py --max_memory 8192 --lvl "competition_levels/SANotHard.lvl" --strategy "custom" --log $log_name &      # PASSED
timeout $max_time python3 searchclient.py --max_memory 8192 --lvl "competition_levels/SAPushPush.lvl" --strategy "custom" --log $log_name &     # TIME
#timeout $max_time python3 searchclient.py --max_memory 8192 --lvl "competition_levels/SAZEROagent.lvl" --strategy "custom" --log $log_name &    # TIME
timeout $max_time python3 searchclient.py --max_memory 8192 --lvl "competition_levels/SAbAnAnA.lvl" --strategy "custom" --log $log_name &       # PASSED
timeout $max_time python3 searchclient.py --max_memory 8192 --lvl "competition_levels/SAbongu.lvl" --strategy "custom" --log $log_name &        # PASSED
timeout $max_time python3 searchclient.py --max_memory 8192 --lvl "competition_levels/SAdashen.lvl" --strategy "custom" --log $log_name &       # HANG
timeout $max_time python3 searchclient.py --max_memory 8192 --lvl "competition_levels/SAora.lvl" --strategy "custom" --log $log_name            # PASSED

# MISC LEVELS
python3 searchclient.py --max_memory 8192 --lvl "SAboxesOfHanoi5.lvl" --strategy "custom" --log $log_name                     # PASSED
python3 searchclient.py --max_memory 8192 --lvl "SAboxesOfHanoi10.lvl" --strategy "custom" --log $log_name                    # PASSED

wait
python3 logfix.py
