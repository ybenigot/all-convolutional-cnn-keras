timestamp=`date +%d%m%y-%H%M%S`
filename=record/test$timestamp.log
touch $filename
pidfile=`hostname`.pid
kill `cat $pidfile`
python convnet3.py 200 $timestamp ${1:-NO} >> $filename 2>&1 & 
echo $! > $pidfile
tail -f $filename

