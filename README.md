# Image Colorization

The Image Colorization program reconstructs a color image using three separate, successively recorded and partially shifted intensity images. The color information, represented by three color channels (red, green, and blue), is interpreted as coordinates in the color space, providing a clear definition of each pixel's color.

## Example Pictures
The provided example pictures follow Sergei Produkin-Gorski's method from the early 20th century. Three intensity images were captured in quick succession through red, blue, and green color filters. To recreate a color image, three projectors — each equipped with a red, green, or blue color filter — were used to project the individual images onto one another.

## Procedure
The colorization process involves the following steps:
* Edge detection 
    * Gradient calculation (Sobel)
    * Binary image calculation
    * Edge detection
* Translation of an image
* Calculation and realisation of the shift
* Combining the colour channels
* Cropping the result
* Creation and applying of a cartoon filter

## Requirements
* OpenCV 4.2.0
* g++ 9.3.0
* CMake 3.16.3

## Installation Instructions
The easiest way to set up the project is to build a Docker image based on the provided Dockerfile and to use it within an interactive session. Input data is automatically read from JSON files, streamlining the calculation process.

```shell
$ docker build -t cvtask1 .
```

To list the JSON files:
```shell
$ ls src/cv/task1/*.json | xargs -n1 basename
boats.json
chalice.json
emir.json
train.json
```

To generate the output pictures out of JSON files above, use following command:

```Shell
$ docker run --rm -it -v "./output:/app-data/src/cv/task1/output" cvtask1 <picture>.json
```
