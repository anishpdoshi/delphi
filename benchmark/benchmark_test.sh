#!bin/sh
for file in ./*
  do
    {
      start=$(date "+%s")
      # delphi question1f.oracle.smt2
      (delphi $file) & pid=$!
      echo delphi $file
      ( sleep 600 && kill -HUP $pid ) 2>/dev/null & watcher=$!
      if wait $pid 2>/dev/null; then
          echo "Slow task finished"
          pkill -HUP -P $watcher
          wait $watcher
      else
          echo "Slow task interrupted"
      fi
      now=$(date "+%s")
      time=$((now-start))
      echo "$file time used:$time seconds" >> result
    } &
  done
  wait
  echo finish all test