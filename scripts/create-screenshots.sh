#!/bin/zsh

script_directory="$(dirname "$0")"
cd $script_directory

./take-screenshots-in-modes.sh ../scenes/heidelberg_motorcycle.xml "S0,S1,S2"
./take-screenshots-in-modes.sh ../scenes/heidelberg_schuss_cal45.xml "S1,S2,S3,S4,S5"
./take-screenshots-in-modes.sh ../scenes/jet-engine.xml "S0,S1,S2,S4"
./take-screenshots-in-modes.sh ../scenes/pedestrian.xml "S0,S1,S2,S3"
./take-screenshots-in-modes.sh ../scenes/heidelberg_delikt_messer.xml "Screenshot"
./take-screenshots-in-modes.sh ../scenes/heidelberg_sturz2.xml "S0,S1,S2,S3,S4,s5"
#./take-screenshots-in-modes.sh ../scenes/heidelberg_motorcycle.xml "Screenshot"
./take-screenshots-in-modes.sh ../scenes/sponza.xml "S0,S1,S2,S3,S4"


LUNG=""
for ((i=0; i<=15; i++)); do
  LUNG+="S$i,"
done
LUNG_ALL=${LUNG:0:${#LUNG}-1}
./take-screenshots-in-modes.sh ../scenes/LIDC-IDRI_0469.xml "$LUNG_ALL"