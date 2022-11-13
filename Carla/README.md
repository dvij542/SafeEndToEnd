## Installation instructions

Download pre-compiled Carla with the RaceTrack from : ![Link]()

Clone and place this repository in CarlaSimulator/PythonClient

Copy the CarlaSimulator/PythonClient/carla folder to CarlaSimulator/PythonClient/Carla/carla

Install dependencies from requirements.txt :-

```
pip install -r requirements.txt
```

## Change parameters

Modify parameters in run_all.sh, run_iter.py, train.py 

## Running instructions

To run with CBF guard included, set SAFEGUARD=True in run_iter.py, else set it to False

Run :-

```
bash run_all.sh
```

You should get the image datasets, videos, trained models, state model predictions comparision with ground truth, visualizations for trajectories followed, speeds, etc, all the figures used in the paper for all the iterations
