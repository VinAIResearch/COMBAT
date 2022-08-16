import cv2
import numpy as np

src_imgs = ['clean_trigger.png', 'nonoise_trigger.png', 'finale_trigger.png']
gap = 1

ims = []
for src_img in src_imgs:
   im = cv2.imread(src_img)
   ims.append(im)
   ims.append(im[:,:gap,:] * 0 + 255)

ims = np.concatenate(ims, 1)
cv2.imwrite('nonoise.png', ims)

