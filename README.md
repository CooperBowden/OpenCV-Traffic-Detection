# Vehicle Speed Estimation using OpenCV

This project is a proof of concept using OpenCV with Ultralytics Object Detection to estimate vehicle speed by 
creating a birds eye view abstraction of highways. 

The associated Paper describing OpenCV's history, API quality, and project implementation can be found here: 
https://drive.google.com/file/d/1PCKXY0Dd-EtAT3gWmPXNF9ny5EdGzMDH/view?usp=sharing

The associated presentation on OpenCV and the projecct implementation can be found here:
https://docs.google.com/presentation/d/1yWclhNMFJSK7WHIh2EgWyXcRGWDn5_8g_gecZKst_vI/edit?usp=drive_link

## Usage

1. Install dependencies by running ```pip install -r requirements.txt``` in the project directory
2. Edit default project settings in [settings.ini](settings.ini)
3. To add a new video or stream, add a new entry to the json file in the following format:
   ```
   "unique ID": {
        "name": "display name",
        "url": "URL to stream or path to local video",}
   ```
3. Run the program via the command line with ```python main.py```

## Features

- All project scripts are located in the [Source Folder](src/)
- Screenshots of results and function outputs are located in the [Screenshots Folder](screenshots/)
- Can use either local videos or livestreams
- Detects vehicle speed within bounding boxes
- Sends alerts when vehicles speed is greater than 130% the average speed
- Can detect traffic going in both directions
- Estimates how busy traffic is based on the number of vehicles detected