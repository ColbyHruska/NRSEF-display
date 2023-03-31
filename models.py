import keras
from tensorflow import convert_to_tensor

encoder = keras.models.load_model("D:/nrsef/g100.encoder.h5")
generator = keras.models.load_model("D:/nrsef/g100.generator.h5")

def encode(imgs):
    return encoder.call(convert_to_tensor(imgs))

def decode(imgs):
    return generator.call(convert_to_tensor(imgs))

def filt(imgs):
    return decode(encode(imgs))