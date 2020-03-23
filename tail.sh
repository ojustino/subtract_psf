#! /bin/bash
# follow live output from a condor job
# example usage: sh tail.sh logs/ JOBNUMBER
# logs/ ($1 below) is that path to the directory that holds job log files
# JOBNUMBER ($2 below) is the assigned number for the job in question

half=0;
full=0;

tail -f $1/*$2* |
while IFS= read line
 do
   echo $line
   if [[ $line == *"0 to go in ref"* ]]; then
     ((half+=1))
   elif [[ $line == *"0 to go in sci"* ]]; then
     ((full+=1))
   fi

   #echo $((half == full)), $((full))
   #echo "$2"
   if [[ $half == $full ]] && [[ $full > 0 ]]; then
	   echo "COMPLETE!!"
	   pkill tail
	   break
   fi

   if [[ $line == *"005 ($2"* ]]; then
   #if [[ $line == *"Run Bytes Sent"* ]]; then
     echo "(might be an) ERROR!! check ' cat $1*$2* ' for output"
	   pkill tail
	   break
   fi
done
