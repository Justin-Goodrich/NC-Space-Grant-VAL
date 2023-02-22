import tensorflow as tf
import numpy as np

def get_image(path, n_channels):
    img = tf.io.read_file(path)
    return tf.io.decode_image(img, channels = n_channels)

def get_image_set(path):
    return (get_image(os.path.join(DATASET_DIR,'output/{}.jpg'.format(path)),3), get_image(os.path.join(dir,'{}_m.png'.format(path)),1),4)

class Image_Generator(tf.keras.utils.Sequence):
  def __init__(self, _dir,img_list,l,split,train=False,batch_size=1):
        """
        @param dir: directory containing file lists and output dir
        @param img_list: file containing list of training images provided by dataset
        @param l: length of training/validation set, found on https://landcover.ai.linuxpolska.com/
        """
        self.dir = _dir
        self.l = l
        self.img_list = os.path.join(_dir,img_list)
        self.batch_size = batch_size
        self.train = train
        self._file = open(self.img_list)
        self.split = split

  def on_epoch_end(self):
        self._file.seek(0, 0)
    
  def __getitem__(self, index):
        features = []
        masks = []
        
        for i in range(self.batch_size):
          img = self._file.readline()
          jpgtensor = get_image(os.path.join(self.dir,'output','{}.jpg'.format(img.strip())),3).numpy()
          pngtensor = get_image(os.path.join(self.dir,'output','{}_m.png'.format(img.strip())),1).numpy()

          if self.train:
            transformed = transform(image=jpgtensor, mask=pngtensor)
            jpgtensor = transformed['image']
            pngtensor = transformed['mask']

          pngtensor = keras.utils.to_categorical(pngtensor,5)
          features.append(jpgtensor)
          masks.append(pngtensor)

        return np.array(features), np.array(masks)
  def __len__(self):
        return int(self.l * self.split)// self.batch_size