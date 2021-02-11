# import scipy.misc
# from scipy import misc
# from scipy.misc.pilutil import Image

# im = Image.open(r"C:\Users\LINTA\Downloads\Dataset\pan\pan_37.jpg")
# im_array = scipy.misc.fromimage(im)
# im_inverse = 255 - im_array
# im_result = scipy.misc.toimage(im_inverse)
# misc.imageio.imwrite('result.jpg', im_result)
import numpy as np
from PIL import Image
import cv2
from matplotlib import pyplot as plt, cm

import numpy as np
import math
import cv2
from pytesseract import pytesseract, Output

img = Image.open(r"C:\Users\LINTA\Downloads\Dataset\pan\pan_37.jpg").convert('LA')
img.save('greyscale.png')

img = cv2.imread('greyscale.png', 2)

ret, bw_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# converting to its binary form
bw = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

cv2.imwrite("Binary.png", bw_img)

dst = cv2.fastNlMeansDenoising(bw_img, None, 1, 7, 21)
cv2.imwrite('denoised.png', dst)
plt.subplot(122), plt.imshow(dst)
plt.show()
plt.imsave('newimg.png', np.array(dst), cmap=cm.gray)

# configuring parameters for tesseract
custom_config = r'--oem 3 --psm 6'
# now feeding image to tesseract
details = pytesseract.image_to_data('newimg.png', output_type=Output.DICT, config=custom_config, lang='eng')
print(details.keys())
total_boxes = len(details['text'])
for sequence_number in range(total_boxes):
    if int(details['conf'][sequence_number]) > 30:
        (x, y, w, h) = (
            details['left'][sequence_number], details['top'][sequence_number], details['width'][sequence_number],
            details['height'][sequence_number])
        threshold_img = cv2.rectangle(threshold_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
# display image
cv2.imshow("captured_txt", threshold_img)
# Maintain output window until user presses a key
cv2.waitKey(0)
# Destroying present windows on screen
cv2.destroyAllWindows()
