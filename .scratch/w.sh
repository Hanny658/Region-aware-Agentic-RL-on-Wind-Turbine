for i in $(seq 1 45); do
  n=$(grep -cE '^\[tune/r0\] +[0-9]+/' $HOME/wtrl/exp/rosco_native.log 2>/dev/null)
  if [ "${n:-0}" -ge 2 ]; then break; fi
  sleep 20
done
tail -7 $HOME/wtrl/exp/rosco_native.log
