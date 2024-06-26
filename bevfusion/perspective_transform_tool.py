import cv2
import numpy as np
import glob
from keymousehandler import KeyMouseHandler
import json
import os

# create key and mouse handler
kmh = KeyMouseHandler()

# load config
f = open('cfg.json', "r")
cfg_file = json.load(f)

img = cv2.imread(cfg_file['file_path'])

if img is None:
    print("Image file not found")
    os._exit(0)

wimg = None

while True:
    oimg = img.copy()

    k = cv2.waitKey(10)
    if k == 27:
        cv2.destroyAllWindows()
        os._exit(0)

    # next frame
    if k == ord('n'):
        break

    # load
    if k == ord('l'):
        kmh.iw_obj.set_param(np.array(cfg_file['dst']), param_index=1)
        kmh.iw_obj.set_param(np.array(cfg_file['src']), param_index=0)

    # save
    if k == ord('s'):
        cfg_file['src'] = kmh.iw_obj.get_params(param_index=0).tolist()
        cfg_file['dst'] = kmh.iw_obj.get_params(param_index=1).tolist()

        j_obj = json.dumps(cfg_file, indent=4)
        with open("cfg.json", "w") as f:
            f.write(j_obj)

        # save transformation matrix
        np.save("transform.npy", kmh.iw_obj.wmat)
        np.save("transform_inv.npy", kmh.iw_obj.wmat_inv)

        # save sample files
        cv2.imwrite('orig.jpg', oimg)
        cv2.imwrite('warped.jpg', wimg)

    wimg = kmh.run(key=k, in_img=oimg)

    # cobine input with output
    concat_img1 = cv2.hconcat([oimg, wimg])

    # menu items
    cv2.putText(concat_img1, 'n-next frame, s-save, l-load, esc-exit', (20, 24), cv2.FONT_HERSHEY_SIMPLEX,
                cfg_file['sizef'] * 1.2, cfg_file['color'], cfg_file['sizeb'])
    cv2.putText(concat_img1, 'o-input points, p-output points, t-transform, e-edit points location, ', (20, 36),
                cv2.FONT_HERSHEY_SIMPLEX, cfg_file['sizef'] * 1.2, cfg_file['color'], cfg_file['sizeb'])
    cv2.imshow("Test_detections", concat_img1)
