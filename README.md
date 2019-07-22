# Object detection inference

This application performs inference from an object detection model from a given input (images, video stream, etc.)

## Dependencies

Running the executable avoids worrying about dependencies. See the **Use** section below for more information.

It's also possible to run the source script in Python. If you're a conda user, you can create an environment from the ```environment.yml``` file using the Terminal or an Anaconda Prompt for the following steps:

1. Create the environment from the ```environment.yml``` file:

    ```conda env create -f environment.yml```
2. Activate the new environment:
    > * Windows: ```activate obj-det```
    > * macOS and Linux: ```source activate obj-det``` 

3. Verify that the new environment was installed correctly:

    ```conda env list```
    
You can also clone the environment through the environment manager of Anaconda Navigator.

Finally, an already trained model must be available in the *trained_model* folder for inference. For a quick start, have a look at the models available in the [TF Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md).

## Use

### Executable

The simplest way to use the program is to run the executable. A floating window with the processed video stream of the main camera attached to the computer will show up.

It's possible to alter the configuration variables from the *config.ini* file to modify the behavior of the program. This are the modifiable entries:
> * **source:** Integer from 0 to N, where 0 is the main camera connected to your computer and so on. Default 0.
> * **num_classes:** Integer from 0 to N, where N is the total of classes the model's been trained on. With the default model, *num_classes* is set to 3.
> * **path_to_labels:** Path where the labels of the classes to detect by the model are defined. Default *./label_map.pbtxt*.

To stop the program, select the floating window and press *'q'* or '*Ctrl+C'* in the console.

### Source

After having available the *obj-det* environment, launch *main.py* to see the processed input video signal from a wired cam. It's also possible to modify the *config.ini* file as explained above. The result is the same as in the executable version. Press the *'q'* button in the resulting floating window or *Ctrl+C* in the console to close the program.
