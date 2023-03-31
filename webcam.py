import cv2
from models import filt, esr
import numpy as np
import pygame
from time import sleep
import keyboard

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
        self.update()
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            try:
                if keyboard.is_pressed('enter'):
                    print("hi")
                    self.update()
                    sleep(5)
                    continue
            except:
                continue
        self.vc.release()
        pygame.quit()
    
    def update(self):
        img = self.update_func(self.vc)

        surf = pygame.surfarray.make_surface(img)
        surf = pygame.transform.scale(surf, self.display.get_size())
        self.display.blit(surf, (0, 0))

        pygame.display.update()
def update(vc):
    _, cam = vc.read()
    resized = cv2.resize(cam, (64, 64))

    image = resized / 127.5 - 1
    image = np.expand_dims(image, 0)
    image = np.array(esr(filt(image)[0])[0])
    return image.astype('uint8')
viewer = Viewer(update, (256, 256))
viewer.start()