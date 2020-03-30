#!/bin/sh
PRG_KEY="FWebSSD"

RUN_PATH=$(cd `dirname $0`;pwd)   # Get the absolute path of the work
cd $RUN_PATH
echo "$RUN_PATH"

case "$1" in
    start)
        nohup python3 -u $RUN_PATH/test_web.py > nohup.log 2>&1 &

        echo "$PRG_KEY started, please check log."

        ;;

    stop)
        pid=$(pgrep -f test_web.py)
        echo "$pid"
        kill -9 $pid
            ##killall python3
        echo "$PRG_KEY stoped!"

        ;;

    restart)
        $0 stop
        sleep 1
        $0 start

        ;;
    *)
        echo "Usage: $0 {start|stop|restart}"
        exit 1
esac

exit 0