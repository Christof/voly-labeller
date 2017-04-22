#!/bin/zsh

script_directory="$(dirname "$0")"
cd $script_directory

./take-screenshots-in-modes.sh ../scenes/heidelberg_delikt_messer3.xml "Y00,Y1"
./take-screenshots-in-modes.sh ../scenes/heidelberg_schuss_cal45.xml "S2"
./take-screenshots-in-modes.sh ../scenes/LIDC-IDRI_0469.xml "L2"
./take-screenshots-in-modes.sh ../scenes/heidelberg_delikt_messer2.xml "S5"
./take-screenshots-in-modes.sh ../scenes/100-labels.xml "S0,S1,S2"

