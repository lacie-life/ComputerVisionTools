
## How to compile Linux Library (Assumed that CMake setup done already)
* cd linux_build
* mkdir build
* cd build
* cmake ..
* make

## How to execute test
* cd ../bin (Assuming the current directorty is build directory, created above)
* ./ublox_f9p_test /dev/ttyACM0
* ./ublox_f9p_i2c_test /dev/ublox_i2c 0x42 (or without command line arguments)

## License
This repository consists files from different other repositories such as Arduino and SparkFun_u-blox_GNSS_Arduino_Library. Please consider the licenses according to the files where it is from.

## Linux Examples
Feel free to port the examples from Sparkfun_Ublox_Arduino_Library to linux.

## Future Work
Port some important examples from Arduino to Linux

