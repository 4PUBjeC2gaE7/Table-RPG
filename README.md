# Table-RPG
Experimenting with OpenCV for RPG projector table. This applciation first calibrates
a camera against a projector screen to generate a remap solution. Then, it scans for
placed miniatures, which it highlights with selected auras.

## Contents
 Filepath           | Description 
 ------------------ | -----------------------------------------------
 background.png     | Sample background used to test image tracking.
 Camera Test.ipynb  | Jupyter test file which provides.
 LICENSE            | license file; GNU GPL v3
 README.md          | This file.
 RPG GUI.py         | Main test application.
 ScreenRemap.py     | Python subroutine for calibrating the camera-projector system.
 auras/             | Folder of sample "auras" projected around minatures

 ## How to Run
 1. Connect a quality webcam to your computer. It really can be any quality.
 1. Setup a projector or seconds screen.
    - Note: if the software only detects one screen, it will pop-up a test screen in a new window.
 1. Start with [Camera Test.ipynb](./Camera%20Test.ipynb). This Jupyter notebook will walk you through the whole thing.