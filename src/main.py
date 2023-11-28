import pygame
import sys
from ia import *

# Initialize Pygame
pygame.init()

# Set up display
width, height = 800, 600
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Simple Drawing App")

num = 0

# Set up colors
black = (0, 0, 0)
white = (255, 255, 255)

tick = 0

# Set up drawing variables
drawing = False
radius = 10
color = white

screen.fill(black)

# Linear interpolation function
def lerp(start, end, alpha):
    return int((1 - alpha) * start + alpha * end)

font = pygame.font.Font(None, 36)  

def draw_circle(start, end):
    for alpha in range(0, 101, 5):
        x = lerp(start[0], end[0], alpha / 100)
        y = lerp(start[1], end[1], alpha / 100)
        pygame.draw.circle(screen, color, (x, y), radius)

# Main game loop
last_pos = None
launch()

# Main game loop
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            drawing = True
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            drawing = False
            pygame.image.save(screen, "drawing.png")
            num = recognize_digits()
            print(num)
            last_pos = None
        elif event.type == pygame.MOUSEMOTION and drawing:
            if last_pos:
                tick = 1
                draw_circle(last_pos, event.pos)
            else:
                tick = 1
                pygame.draw.circle(screen, color, event.pos, radius)
            last_pos = event.pos

    pygame.display.flip()

    pygame.time.Clock().tick(230)
    if not drawing and tick == 1:
        print("clean")
        tick = 0
        screen.fill(black)
        number_text = font.render(str(num), True, white)
        text_rect = number_text.get_rect(center=(width // 2, height - 20))
        screen.blit(number_text, text_rect)
