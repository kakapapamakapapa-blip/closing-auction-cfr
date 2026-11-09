#!/bin/bash
cd /home/claude/nasdaq
export THREADS=2 LOGT_A=25 LOGT_L=24 LOGT_BR=23 CONTBUY=1
for cfg in "0 0" "8 1" "1 1" "2 2" "4 4"; do set -- $cfg
  ./ncc3 run $1 $2 330 1 4e6,16e6,64e6 16e6 3e5 3 res4/v$1 >> res4/scale.jsonl
done
./ncc3 cross 330 3e5 res4/v0:0:0 res4/v8:8:1 res4/v1:1:1 res4/v2:2:2 res4/v4:4:4 > res4/cross.jsonl
echo DONE > res4/done
