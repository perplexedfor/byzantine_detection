#!/bin/sh

echo "Simulating malicious execution from ${EXEC_PATH}..."

TARGETS="1.1.1.1 8.8.8.8 9.9.9.9"

mkdir -p ${EXEC_PATH}

while true; do

  # Random filename (polymorphic payload)
  RAND=$RANDOM
  SCRIPT="${EXEC_PATH}/bad_${RAND}.sh"

  echo "#!/bin/sh" > $SCRIPT
  echo "echo Malicious execution id $RAND" >> $SCRIPT
  echo "echo RANDOM_VALUE=$RANDOM" >> $SCRIPT

  chmod +x $SCRIPT

  # Execute payload and burn a little CPU so the pod isn't "idle"
  $SCRIPT
  # spin for a short time to make it show up in metrics
  dd if=/dev/zero of=/dev/null bs=1M count=50 &

  ###################################
  # Occasional network beacon/payload
  ###################################
  if [ $((RANDOM % 4)) -eq 0 ]; then
      TARGET=$(echo $TARGETS | tr ' ' '\n' | shuf -n1)
      echo "Fetching remote payload from $TARGET"
      wget -q --timeout=2 http://$TARGET/payload.sh -O /dev/null || true
  fi

  # Random execution interval
  sleep $((RANDOM % EXEC_INTERVAL + 1))

done