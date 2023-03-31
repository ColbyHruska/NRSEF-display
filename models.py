import keras
from tensorflow import convert_to_tensor
import os
import tensorflow as tf
import tensorflow_hub as hub
os.environ["TFHUB_DOWNLOAD_PROGRESS"] = "True"

ESR_PATH = "https://tfhub.dev/captain-pool/esrgan-tf2/1"
esr_model = hub.load(ESR_PATH)

generator = keras.models.load_model("./openimages2paintings_g_AB0.h5")

def filt(imgs):
    return generator.call(convert_to_tensor(imgs))

def esr(imgs):
    out = preprocess_image(imgs)
    out = esr_model(out)
    return out

def preprocess_image(imgs):
    if imgs.shape[-1] == 4:
        imgs = imgs[...,:-1]
    size = (tf.convert_to_tensor(imgs.shape[:-1]) // 4) * 4
    imgs = tf.image.crop_to_bounding_box(imgs, 0, 0, size[0], size[1])
    imgs = imgs * 127.5 + 127.5
    return tf.expand_dims(imgs, 0)