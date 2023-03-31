import cv2
from models import filt, encode, decode
import numpy as np
import pygame
from time import sleep

class Viewer:
    def __init__(self, update_func, display_size):
        self.update_func = update_func
        pygame.init()
        self.display = pygame.display.set_mode(display_size, pygame.RESIZABLE)
        self.vc = cv2.VideoCapture(0)
    
    def set_title(self, title):
        pygame.display.set_caption(title)
    
    def start(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            Z = self.update_func(self.vc)

            surf = pygame.surfarray.make_surface(Z)
            surf = pygame.transform.scale(surf, self.display.get_size())
            self.display.blit(surf, (0, 0))

            sleep(1)
            pygame.display.update()

        self.vc.release()
        pygame.quit()
def update(vc):
    _, cam = vc.read()
    resized = cv2.resize(cam, (64, 64))

    image = resized / 127.5 - 1
    image = np.expand_dims(image, 0)
    image = np.array(filt(image)[0])
    image = image * 127.5 + 127.5
    return image.astype('uint8')
viewer = Viewer(update, (64, 64))
viewer.start()
