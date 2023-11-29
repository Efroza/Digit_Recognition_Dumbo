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
red = (255, 0, 0)
green = (0,255,0)

# check button
button_rect = pygame.Rect(100, 300, 100, 100)  # x, y, width, height
button_color = green

# reset button
button_rect2 = pygame.Rect(500, 300, 100, 100)  # x, y, width, height
button_color2 = red



tick = 0

# Set up drawing variables
drawing = False
radius = 20
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
    
def is_mouse_over_button(pos, button_rect):
    return button_rect.collidepoint(pos)

def is_on_buttons(pos):
    if button_rect.collidepoint(pos) or button_rect2.collidepoint(pos):
        return True
    else:
        return False

# Main game loop
last_pos = None
launch()

# Main game loop
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == pygame.MOUSEBUTTONDOWN and is_on_buttons(event.pos) :
            if event.button == 1:
                if is_mouse_over_button(event.pos, button_rect):
                    pygame.image.save(screen, "drawing.png")
                    num = recognize_digits()
                    tick = 0
                    screen.fill(black)
                    number_text = font.render(str(num), True, white)
                    text_rect = number_text.get_rect(center=(width // 2, height - 20))
                    screen.blit(number_text, text_rect)
                    last_pos = None
                if is_mouse_over_button(event.pos, button_rect2):
                    screen.fill(black)
                    last_pos = None
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            drawing = False
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            drawing = True
        elif event.type == pygame.MOUSEMOTION and drawing:
            if last_pos:
                tick = 1
                draw_circle(last_pos, event.pos)
            else:
                tick = 1
                pygame.draw.circle(screen, color, event.pos, radius)
            last_pos = event.pos

    pygame.display.flip()

    # Draw button check
    pygame.draw.rect(screen, button_color, button_rect)

    # Draw button reset
    pygame.draw.rect(screen, button_color2, button_rect2)

    # Draw button check text
    button_text = font.render("check", True, white)
    text_rect = button_text.get_rect(center=button_rect.center)
    screen.blit(button_text, text_rect)

    # Draw button reset text
    button_text2 = font.render("reset", True, white)
    text_rect2 = button_text2.get_rect(center=button_rect2.center)
    screen.blit(button_text2, text_rect2)

    pygame.time.Clock().tick(230)

        
