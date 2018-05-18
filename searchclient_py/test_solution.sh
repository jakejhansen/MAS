python3 searchclient.py --lvl competition_levels/SAAiAiCap.lvl --strategy custom --max_memory 3500 --log True &
python3 searchclient.py --lvl "SAsoko3_07.lvl" --strategy "custom" --log True &
python3 searchclient.py --lvl "SAsoko3_12.lvl" --strategy "custom" --log True &
python3 searchclient.py --lvl "SAsoko3_06.lvl" --strategy "custom" --log True &

wait
echo "done"

python3 logfix.py