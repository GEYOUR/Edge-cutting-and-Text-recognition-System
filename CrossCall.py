import matlab.engine
import matlab
from  skimage import io
import numpy as np
import cv2
import time
from Identify import CnOcr
from cnstd import CnStd
import os
import os
import random
from skimage import io, color, img_as_ubyte
import logging
import logging.config

ocr = CnOcr()
std = CnStd()
eng = matlab.engine.start_matlab()
org_path = r".\Candidate_imgs"
save_path = r".\Pcut_imgs"
catalog = ['车牌','瓶盖','硬币']

numeric_level = logging.DEBUG
logging.basicConfig(format="%(levelname)s [%(asctime)s] :%(message)s", filename='./logs/test.log', level=numeric_level, filemode='w')
for i in range(5):
    logging.debug(f"calling logging.debug({i})")
    logging.info(f"calling logging.debug({i})")
    logging.warning(f"calling warning({i})")

for i in range(5):
    logging.info("gogo")

def get_identity(_file_path):
    contain = []
    box_infos = std.detect(_file_path)
    for box_info in box_infos['detected_texts']:
        cropped_img = box_info['cropped_img']
        cv2.imshow("c", cropped_img)
        result = ocr.ocr_for_single_line(cropped_img)
        if result[1] >= 0.4:
            contain.extend(result[0])
    if len(contain)==0:
        contain = ["Unable to recognise "]
    return contain

# detect and save images
for cata in catalog:
    for filename in os.listdir(os.path.join(org_path,cata)):

        file_path = os.path.join(org_path, cata, filename)
        if cata=="车牌":
            img = eng.Pcut(file_path,"2")
            print("正在进行矩形目标提取")
            logging.info("矩形目标识别：")
        else:
            img = eng.Pcut(file_path,"1")
            print("正在进行圆形目标提取")
            logging.info("圆形目标识别：")

        img =  np.array(img).astype(float)*255
        cv2.imshow("output", img)
        # cv2.waitKey(0)
        print("read successfully")
        # save as .bmp
        ext = os.path.splitext(filename)
        sf_name = ext[0]+".bmp"
        save_name = os.path.join(save_path, sf_name)
        io.imsave(save_name, img)
        # "identify"
        identity_text = get_identity(file_path)
        logging.debug(f"IMage: {sf_name}"
                      f"\t     {identity_text}")
        print(identity_text)

eng.exit()



# box_infos = std.detect(file_path)
# for box_info in box_infos['detected_texts']:
#     cropped_img = box_info['cropped_img']
#     cv2.imshow("c", cropped_img)
#     time.sleep(1)
#     result = ocr.ocr_for_single_line(cropped_img)
#     print(result)
#

# try with ocr_for_single_line
def main():
    # loglevel=input()
    # numeric_level = getattr(logging, loglevel.upper(), None)
    # if not isinstance(numeric_level, int):
    #     raise ValueError('Invalid log level: %s' % loglevel)

    # logging.config.fileConfig('cjdLogger.conf')

    numeric_level = logging.DEBUG
    logging.basicConfig(format="%(levelname)s [%(asctime)s] :%(message)s", filename='./logs/test.log', level=numeric_level, filemode='w')
    for i in range(5):
        logging.debug(f"calling logging.debug({i})")
        logging.info(f"calling logging.debug({i})")
        logging.warning(f"calling warning({i})")

if __name__ == '__main__':
    main()

