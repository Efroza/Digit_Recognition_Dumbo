import pygame
import sys
from ia import *

# Initialize Pygame
pygame.init()

# Set up display
width, height = 800, 600
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Simple Drawing App")

# Set up colors
black = (0, 0, 0)
white = (255, 255, 255)

# Set up drawing variables
drawing = False
radius = 20
color = black

screen.fill(white)
# pygame.time.Clock().tick(230)
# Main game loop
last_pos = None
launch()
# Linear interpolation function
def lerp(start, end, alpha):
    return int((1 - alpha) * start + alpha * end)

# Main game loop
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                drawing = True
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                drawing = False
                pygame.image.save(screen, "drawing.png")
                recognize_digits()
                last_pos = None
        elif event.type == pygame.MOUSEMOTION:
            if drawing:
                if last_pos:
                    # Interpolate between the last position and the current position
                    for alpha in range(0, 101, 5):
                        x = lerp(last_pos[0], event.pos[0], alpha / 100)
                        y = lerp(last_pos[1], event.pos[1], alpha / 100)
                        pygame.draw.circle(screen, color, (x, y), radius)
                else:
                    pygame.draw.circle(screen, color, event.pos, radius)
                last_pos = event.pos

    pygame.display.flip()

    pygame.time.Clock().tick(230)
    if drawing == False:
        screen.fill(white)

        # Limit frames per second

    # Limit frames per second
