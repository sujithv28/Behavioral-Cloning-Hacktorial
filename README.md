# Behavioral-Cloning-Hacktorial-Files
All the files needed for the Terrapin Hackers Hacktorial on Behavioral Cloning (End to End Learning) for Self Driving Cars. 

The presentation is available to view [**here**](http://goo.gl/XtIBNg).

A written explanation going over the code and the concepts behind it is available on Medium [**here**](https://medium.com/decode-ways/hacktorial-self-driving-car-df81dde2bc25).

## Downloading Training Data
To download the data onto your computer run
```
wget https://www.dropbox.com/s/3cwc2atg1qorzg4/data.zip?dl=0
unzip -a data.zip?dl=0
rm data.zip?dl=0
```

## Train the Model
To train the model run
```
python model.py
```

## Run Model on Simulator
To run your trained model on the simulator, open up the simulator application and start an autonomous session on either track. Then run in the env
```
python drive.py model.json
```
