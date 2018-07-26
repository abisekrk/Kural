import pygame
import pygame.camera
from pygame.locals import *

DEVICE = '/dev/video0'
SIZE = (640, 480)
FILENAME = '/media/abisek/Important Files/R.K/++/Machine Learning/Projects/K/captures/capturetest'

def camstream():
    pygame.init()
    pygame.camera.init()
    display = pygame.display.set_mode(SIZE, 0)
    camera = pygame.camera.Camera(DEVICE, SIZE)
    camera.start()
    screen = pygame.surface.Surface(SIZE, 0, display)
    capture = True
    count=0
    while capture:
        screen = camera.get_image(screen)
        display.blit(screen, (0,0))
        pygame.display.flip()
        for event in pygame.event.get():
            if event.type == KEYDOWN and event.key == 113:
                capture = False
            elif event.type == KEYDOWN and event.key == 97:
                pygame.image.save(screen, FILENAME+str(count)+'.jpg')
                count+=1
    camera.stop()
    pygame.quit()
    return

if __name__ == '__main__':
    camstream()