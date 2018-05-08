python3 searchclient.py --lvl "SAsoko3_08.lvl" --strategy "custom" --log True &
python3 searchclient.py --lvl "SAsoko3_07.lvl" --strategy "custom" --log True &
python3 searchclient.py --lvl "SAsoko3_12.lvl" --strategy "custom" --log True &
python3 searchclient.py --lvl "SAsoko3_06.lvl" --strategy "custom" --log True

wait
python3 logfix.py