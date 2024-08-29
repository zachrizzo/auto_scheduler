import pygame
import neat
import random

# Initialize Pygame
pygame.init()

# Constants
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
BACKGROUND_COLOR = (30, 30, 30)
ROOM_COLOR = (200, 200, 200)
STUDENT_COLOR = (0, 128, 255)

# Set up the display
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Classroom Scheduler")

# Set up font
font = pygame.font.Font(None, 36)

# Define rooms (classrooms)
rooms = [
    pygame.Rect(50, 50, 200, 200),
    pygame.Rect(300, 50, 200, 200),
    pygame.Rect(550, 50, 200, 200),
    pygame.Rect(50, 300, 200, 200),
    pygame.Rect(300, 300, 200, 200),
    pygame.Rect(550, 300, 200, 200),
]

# Example student data
students = [
    {"name": "Alice", "grade": 9, "position": [100, 100]},
    {"name": "Bob", "grade": 10, "position": [350, 100]},
    {"name": "Charlie", "grade": 11, "position": [600, 100]},
    {"name": "David", "grade": 12, "position": [100, 350]},
]

# Main game loop
running = True
clock = pygame.time.Clock()

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Fill the background
    screen.fill(BACKGROUND_COLOR)

    # Draw the rooms
    for room in rooms:
        pygame.draw.rect(screen, ROOM_COLOR, room)

    # Draw the students
    for student in students:
        pygame.draw.circle(screen, STUDENT_COLOR, student["position"], 20)
        text = font.render(student["name"], True, (255, 255, 255))
        screen.blit(text, (student["position"][0] - 20, student["position"][1] - 10))

    # Update the display
    pygame.display.flip()

    # Cap the frame rate
    clock.tick(60)

# Clean up
pygame.quit()
