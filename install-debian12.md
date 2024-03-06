on host debian 12, one time only
```
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sed -i -e '/experimental/ s/^#//g' /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

1. start the container
replace the mount volume with the path of this repository
```
docker run --rm --gpus all -it -v <host path to the StarTrader>:/app/StarTrader debian:12
```

2. this step and the subsequent steps, all done in container. installation build packages
```
# apt-get update && apt-get install -y cmake openmpi-bin libopenmpi-dev python3 python3-venv python3-dev git zlib1g-dev vim libgl1 libglib2.0-0 
```

3. create venv
```
python3 -m venv /app-env
source /app-env/bin/activate
pip install --upgrade pip
pip install numpy==1.23.5 opencv-python mujoco-py==0.5.7 lockfile graphviz
```

4. ready the apps
```
cd /app
```

5. install gym
```
git clone https://github.com/openai/gym.git
cd gym
pip install -e .
```

6. install tensorflow
https://www.tensorflow.org/install/pip#linux 
```
pip install tensorflow[and-cuda]
```

7. install openai baselines
```
cd /app
git clone https://github.com/openai/baselines.git
cd baselines
pip install -e .
```

8. startrader and startradertest installation
```
cd /app/StarTrader
cp -r gym/envs/StarTrader /app/gym/gym/envs
cp -r gym/envs/StarTraderTest/ /app/gym/gym/envs
cp gym/envs/__init__.py ../gym/gym/envs/__init__.py

cp /app/baselines/baselines/run.py /app/StarTrader/run.py
cp -r ./gym/envs/StarTrader/data/  /app/StarTrader/
# cp -r ./gym/envs/StarTraderTest/data/* /app/StarTrader/data/
cp /app/StarTrader/baselines/baselines/ddpg/ddpg.py /app/baselines/baselines/ddpg/ddpg.py
cp /app/StarTrader/baselines/baselines/ddpg/ddpg_learner.py /app/baselines/baselines/ddpg/ddpg_learner.py
```

9. train agent
```
# test 
$ # python -m run --alg=ddpg --env=StarTrader-v0 --network=mlp --num_timesteps=2e4
$ python -m run --alg=ddpg --env=StarTrader:v0 --network=mlp --num_timesteps=2e4
$ python -m run --alg=ddpg --env=StarTraderTest-v0 --network=mlp --num_timesteps=2e3 --load_path='./model/DDPG_trained_model_8'
$ python compare.py
```
