#!/bin/zsh

script_directory="$(dirname "$0")"
cd $script_directory/../build

PARAMETERS="--offline"
time ./voly-labeller ../scenes/heidelberg_motorcycle.xml $1 $PARAMETERS\
  --video="S0,S1,S2"
time ./voly-labeller ../scenes/heidelberg_schuss_cal45.xml $1 $PARAMETERS\
  --video="P0,P1,P2,P3,P4,P5,P6,P7,S2,S3,S4,S5"
time ./voly-labeller ../scenes/jet-engine.xml $1 $PARAMETERS\
  --video="S0,S1,S2,S4"
time ./voly-labeller ../scenes/pedestrian.xml $1 $PARAMETERS\
  --video="S0,S1,S2,S3"
time ./voly-labeller ../scenes/heidelberg_delikt_messer2.xml $1 $PARAMETERS\
  --video="S0,S1,S2,S3_7,S4,S5,S6,S7_7,S8,S9"
time ./voly-labeller ../scenes/heidelberg_sturz2.xml $1 $PARAMETERS\
  --video="S0,S1,S2,S3,S4,S5"
time ./voly-labeller ../scenes/grch.xml $1 $PARAMETERS\
  --video="S0,S1,S2_6,S3"
time ./voly-labeller ../scenes/sponza.xml $1 $PARAMETERS\
  --video="S0,S1,S2,S3,S4"
time ./voly-labeller ../scenes/plane.xml $1 $PARAMETERS\
  --video="S0,S1,S2,S3"
time ./voly-labeller ../scenes/human.xml $1 $PARAMETERS\
  --video="P0,P1,P2"
time ./voly-labeller ../scenes/100-labels.xml $1 $PARAMETERS\
  --video="S0,S1,S15,S2"

LUNG=""
for ((i=0; i<=15; i++)); do
  LUNG+="S$i,"
done
LUNG_ALL=${LUNG:0:${#LUNG}-1}
time ./voly-labeller ../scenes/LIDC-IDRI_0469.xml $1 $PARAMETERS\
  --video="$LUNG_ALL"
