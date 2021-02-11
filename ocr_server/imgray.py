import scipy.misc
from scipy import misc
from scipy.misc.pilutil import Image

im = Image.open("/static/uploads/p_4.JPG")
im_array = scipy.misc.fromimage(im)
im_inverse = 255 - im_array
im_result = scipy.misc.toimage(im_inverse)
misc.imsave('result.jpg', im_result)