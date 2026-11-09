#!/bin/bash
# real resolution: a decision at every one of the 330 NOII publications
cd /home/claude/nasdaq
export THREADS=2 LOGT_A=25 LOGT_L=24 LOGT_BR=23
for v in 0 1 2 3 4; do
  ./ncc2 run $v $v 330 1 4e6,16e6,64e6 16e6 3e5 3 res3/v$v >> res3/scale.jsonl
done
./ncc2 run 5 5 330 1 4e6,16e6 16e6 3e5 3 res3/v5 >> res3/scale.jsonl
./ncc2 cross 330 3e5 res3/v0:0 res3/v1:1 res3/v2:2 res3/v3:3 res3/v4:4 res3/v5:5 > res3/cross.jsonl
echo DONE > res3/done
