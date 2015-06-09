#!/bin/sh

find ../src/ -name "*.h" > fileList.log
if [ "$#" -ne 1 ]; then
  ../scripts/complete_coverage.sh fileList.log
  exit 0
fi

dir=$(pwd | sed 's,/*[^/]\+/*$,,')
file="coverage.info.complete"

#exclude some header files with no source code
echo "" > $file

excludeList="utils/project_root.h"

while read line; do
  found=0
  for excludeFile in $excludeList; do
    case $line in
      *$excludeFile*)
        echo "exclude file $line";
        found=1
        break;
        ;;
    esac
  done
  if [ $found -eq 0 ]; then
    echo $line >> $file;
  fi
done < "coverage.info.cleaned"

while read line; do
  #generate path to .cpp and .h file
  line2="${line%.h}.cpp"
  if [ -f $line2 ]; then
    file1=$line
    file2=$line2
  else
    file1=$line
    file2=$line
  fi
  path1="${file1##*../}"
  path2="${file2##*../}"

  #test if file is in ignore list
  if [ $(grep $path1 $0 -c) -eq 1 ]; then
    echo "skip $path1"
    continue
  fi

  #test if .cpp file is already in list
  grep $path2 coverage.info.cleaned -c >> /dev/null
  if [ $? -eq 1 ]; then

    #test if .h file is already in list
    grep $path1 coverage.info.cleaned -c >> /dev/null
    if [ $? -eq 1 ]; then

      #add file to list if both files are not contained
      #echo "add $path2"
      echo "TN:" >> $file
      echo "SF:$dir/$path2" >> $file
      #echo "DA:0,0" >> $file

      loc=$(wc -l < $file2)
      for i in $(seq 1 $loc); do
        echo "DA:$i,0" >> $file
      done

      echo "end_of_record" >> $file
    fi
  fi

done < $1

rm fileList.log


exit
# IGNORE LIST:
src/utils/project_root.h
