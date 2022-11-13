INCLUDE_CBF=False
K=3
CURR_ITER=0
N_ITERS=10
# python3 train.py -r 4
for (( c=0; c<=$N_ITERS; c++ ))
do
    echo "Running $c th iteration"
    ../../CarlaUE4.sh /Game/Maps/RaceTrack -windowed -carla-server -benchmark -fps=30 -quality-level=Low -l & pid=$!
    sleep 25
    python3 run_iter.py -r $c
    a=$c
    a=$((a-1))
    kill -9 $(ps -ef | grep carla | awk '{print $2}')
    echo $a
    F1="run$a""_images/*"
    F2="run$c""_images/"
    cp $F1 $F2 -r
    python3 train.py -r $c
    echo "run$c""_images/*" 
done
python3 create_video.py -n $N_ITERS
python3 path_plot.py -n $N_ITERS