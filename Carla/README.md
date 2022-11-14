## Installation instructions

Download pre-compiled Carla with the RaceTrack from : ![Link](https://d18ky98rnyall9.cloudfront.net/3dXfty7_EemFOA6Hm29iNA_de05a1c02eff11e9821ed19f5bd73b7b_CarlaUE4Ubuntu.tar.gz?Expires=1668556800&Signature=W5Sg3DM5OxRluFH-~kt0fMySIRVdQdmRv4iK8RNaTspWYpcBXCLtDT-QFZJNwB-WDH~9BxG44DUIzC7eQfMNhEFmUE7S7n9M0~w5jQG3PQhA9rTIyDL9B0HUHVfaUE8dci8MIYc~wZXUyisEiWpKVmMKRLEEc1vTsgh4ZHwxD74_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A)

Follow these instructions for setup : ![Link](https://d18ky98rnyall9.cloudfront.net/IFfK-Ce8Eem3Cw5hhdQCGg_210f0c4027bc11e9ae95c9d2c8ddb796_CARLA-Setup-Guide-_Ubuntu_.pdf?Expires=1668556800&Signature=bwdhFdbjajiQzIHlj3qEFz-AuUDpsvLkG~bX32A-66-T1AyhhcPESE5CvXebsugWlXESqbOIpbMj4nkzmwiOkXzTU5mwVhrW7ov8qakZpZhxccg1BEHcJ-gX8PasGQrUwbmeqUYvAh-xvxU9v6mP9aE6mgcE~9dLniRksouNT9g_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A)

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
