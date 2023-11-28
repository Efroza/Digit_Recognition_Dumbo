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

font = pygame.font.Font(None, 36)  

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
                num = recognize_digits()
                print(num)
                last_pos = None
        elif event.type == pygame.MOUSEMOTION:
            if drawing:
                tick = 1
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
    if drawing == False and tick == 1:
        print("clean")
        tick = 0
        screen.fill(white)
        number_text = font.render(str(num), True, black)  # Replace "42" with your desired number
        text_rect = number_text.get_rect(center=(width // 2, height - 20))  # Adjust the Y-coordinate as needed
        screen.blit(number_text, text_rect)
