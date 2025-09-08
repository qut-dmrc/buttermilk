#!/bin/bash
# Returns the most recent buttermilk log, preferring the _debug.log if available.

# THIS HAS NOT BEEN UPDATED TO WORK WITH STRUCTURED LOGGING YET.

# Find the latest log file, preferring debug if it is one of the last two created.
BMLOG=$( (ls -t /tmp/buttermilk_*.log 2>/dev/null | head -n 2 | grep '_debug.log' || ls -t /tmp/buttermilk_*.log 2>/dev/null | head -n 1) | head -n 1 )

if [ -z "$BMLOG" ]; then
  echo "No buttermilk logs found in /tmp."
  exit 1
fi

echo $BMLOG
