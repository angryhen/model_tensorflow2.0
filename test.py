import matplotlib.pyplot as plt
from utils.randaugment import Rand_Augment
from PIL import Image

img_augment = Rand_Augment()
img_origal = Image.open('/home/du/Desktop/dataset/ibox/cls/ibox_c15/tdr-tdrsslltgm-tz-yw-90g/D_00000001_00000001_0000000010000358_0a53e1ba52034ee0b334021f45bfef5c_1.jpg')
img_final = img_augment(img_origal)
plt.imshow(img_final)
plt.show()
print('how to call')