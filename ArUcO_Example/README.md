
# Generate ArUcO marker
```
python gen_tag.py --id 24 --type DICT_5X5_100 -o tags/
```

# Pose estimate
```
python main.py --K_Matrix calibration_matrix.npy --D_Coeff distortion_coefficients.npy --type DICT_5X5_100
```
