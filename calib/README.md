# Using numpy in Jetson Series

```
export OPENBLAS_CORETYPE=ARMV8
```

```
python3 stereo_calibration.py --left_file=left_camera.yaml --right_file=right_camera.yaml --left_prefix=left --right_prefix=right --left_dir=images --right_dir=images --image_format=jpg --save_file=results.yaml --square_size=0.025
```


