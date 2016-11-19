#!/bin/zsh

script_directory="$(dirname "$0")"
cd $script_directory/../build

PARAMETERS="$1 --offline"
echo $PARAMETERS
./voly-labeller ../scenes/heidelberg_motorcycle.xml "$PARAMETERS" --video="S0,S1,S2"
./voly-labeller ../scenes/heidelberg_schuss_cal45.xml "$PARAMETERS" --video="P0,P1,P2,P3,P4,P5,P6,P7,S2,S3,S4,S5"
./voly-labeller ../scenes/jet-engine.xml "$PARAMETERS" "S0,S1,S2,S4"
./voly-labeller ../scenes/pedestrian.xml "$PARAMETERS" --video="S0,S1,S2,S3"
./voly-labeller ../scenes/heidelberg_delikt_messer.xml "$PARAMETERS" --video="Screenshot"
./voly-labeller ../scenes/heidelberg_sturz2.xml "$PARAMETERS" --video="S0,S1,S2,S3,S4,S5"
./voly-labeller ../scenes/grch.xml "$PARAMETERS" --video="S0,S1,S2_6,S3"
./voly-labeller ../scenes/sponza.xml "$PARAMETERS" "S0,S1,S2,S3,S4"
./voly-labeller ../scenes/plane.xml "$PARAMETERS" --video="S0,S1,S2,S3"
./voly-labeller ../scenes/human.xml "$PARAMETERS" --video="P0,P1,P2"

LUNG=""
for ((i=0; i<=15; i++)); do
  LUNG+="S$i,"
done
LUNG_ALL=${LUNG:0:${#LUNG}-1}
./voly-labeller ../scenes/LIDC-IDRI_0469.xml $PARAMETERS --video="$LUNG_ALL"
