import neat
import pygame
import random

pygame.init()
SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 1000
BACKGROUND_COLOR = (30, 30, 30)
ROOM_COLOR = (200, 200, 200)
OBSTACLE_COLOR = (255, 0, 0)
NODE_COLOR = (0, 255, 0)
CONNECTION_POSITIVE_COLOR = (0, 255, 0)
CONNECTION_NEGATIVE_COLOR = (255, 0, 0)

screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Advanced Classroom Scheduler")
font = pygame.font.Font(None, 24)

schools = ["School A", "School B"]
time_slots = ["9:00", "10:00", "11:00", "14:00", "15:00"]

rooms = [
    pygame.Rect(50, 50, 200, 200),
    pygame.Rect(300, 50, 200, 200),
    pygame.Rect(550, 50, 200, 200),
    pygame.Rect(50, 300, 200, 200),
    pygame.Rect(300, 300, 200, 200),
    pygame.Rect(550, 300, 200, 200),
]

student_colors = [
    (255, 0, 0),  # Red
    (0, 255, 0),  # Green
    (0, 0, 255),  # Blue
    (255, 255, 0),  # Yellow
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Cyan
    (255, 165, 0),  # Orange
    (128, 0, 128),  # Purple
]

students = [
    {"name": "Alice", "grade": 9, "school": "School A", "preferred_time": "9:00", "color": student_colors[0]},
    {"name": "Bob", "grade": 10, "school": "School A", "preferred_time": "10:00", "color": student_colors[1]},
    {"name": "Charlie", "grade": 11, "school": "School B", "preferred_time": "11:00", "color": student_colors[2]},
    {"name": "David", "grade": 12, "school": "School B", "preferred_time": "14:00", "color": student_colors[3]},
    {"name": "Eve", "grade": 9, "school": "School A", "preferred_time": "15:00", "color": student_colors[4]},
    {"name": "Frank", "grade": 10, "school": "School B", "preferred_time": "9:00", "color": student_colors[5]},
    {"name": "Grace", "grade": 11, "school": "School A", "preferred_time": "10:00", "color": student_colors[6]},
    {"name": "Henry", "grade": 12, "school": "School B", "preferred_time": "11:00", "color": student_colors[7]},
]

teachers = [
    {"name": "Mr. Smith", "school": "School A", "obstacle": "9:00"},
    {"name": "Ms. Johnson", "school": "School B", "obstacle": "14:00"},
]

def move_student(student, target_x, target_y):
    dx = target_x - student['x']
    dy = target_y - student['y']
    distance = (dx**2 + dy**2)**0.5
    speed = 100

    if distance > speed:
        student['x'] += speed * (dx / distance)
        student['y'] += speed * (dy / distance)
    else:
        student['x'] = target_x
        student['y'] = target_y

def evaluate_grouping(genomes, config):
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)

        fitness = 1000
        groups = [[] for _ in range(len(rooms))]

        for student in students:
            inputs = (
                student["grade"] / 12,
                schools.index(student["school"]) / len(schools),
                time_slots.index(student["preferred_time"]) / len(time_slots)
            )
            output = net.activate(inputs)
            group_index = min(int(output[0] * len(groups)), len(groups) - 1)

            if all(abs(student["grade"] - existing_student["grade"]) <= 1 for existing_student in groups[group_index]):
                groups[group_index].append(student)
            else:
                for i, group in enumerate(groups):
                    if all(abs(student["grade"] - existing_student["grade"]) <= 1 for existing_student in group):
                        groups[i].append(student)
                        break
                else:
                    empty_groups = [i for i, g in enumerate(groups) if not g]
                    if empty_groups:
                        groups[empty_groups[0]].append(student)
                    else:
                        groups[min(range(len(groups)), key=lambda i: len(groups[i]))].append(student)

        for i, group in enumerate(groups):
            if len(group) == 0:
                fitness -= 100
            elif len(group) > 4:
                fitness -= 50 * (len(group) - 4)

            if len(group) > 0:
                school = group[0]["school"]
                time_slot = time_slots[i % len(time_slots)]

                if not all(student["school"] == school for student in group):
                    fitness -= 200

                for teacher in teachers:
                    if teacher["school"] == school and teacher["obstacle"] == time_slot:
                        fitness -= 300

                for student in group:
                    if student["preferred_time"] == time_slot:
                        fitness += 50

            for i, student1 in enumerate(group):
                for student2 in group[i+1:]:
                    if abs(student1["grade"] - student2["grade"]) > 1:
                        fitness -= 1000

        genome.fitness = max(0, fitness)

        draw_groups(groups, genome_id, fitness)
        draw_neural_network(net, genome_id, fitness)  # Corrected drawing function call

        pygame.display.flip()
        pygame.time.delay(0)

def draw_groups(groups, genome_id, fitness):
    screen.fill(BACKGROUND_COLOR)
    for i, (room, group) in enumerate(zip(rooms, groups)):
        pygame.draw.rect(screen, ROOM_COLOR, room)
        time_slot = time_slots[i % len(time_slots)]
        text = font.render(f"Time: {time_slot}", True, (255, 255, 255))
        screen.blit(text, (room.x + 10, room.y + 10))

        for j, student in enumerate(group):
            target_x = room.x + 40 + (j % 2) * 80
            target_y = room.y + 60 + (j // 2) * 60
            move_student(student, target_x, target_y)
            pygame.draw.circle(screen, student['color'], (int(student['x']), int(student['y'])), 15)
            text = font.render(f"{student['name']} ({student['grade']})", True, (255, 255, 255))
            screen.blit(text, (int(student['x']) - 30, int(student['y']) + 20))
            text = font.render(f"{student['school']}", True, (255, 255, 255))
            screen.blit(text, (int(student['x']) - 30, int(student['y']) + 40))

        for teacher in teachers:
            if teacher["school"] == (group[0]["school"] if group else "") and teacher["obstacle"] == time_slot:
                pygame.draw.rect(screen, OBSTACLE_COLOR, (room.x, room.y, 10, room.height))

    gen_text = font.render(f"Genome ID: {genome_id}", True, (255, 255, 255))
    fit_text = font.render(f"Fitness: {fitness:.2f}", True, (255, 255, 255))
    screen.blit(gen_text, (10, SCREEN_HEIGHT - 60))
    screen.blit(fit_text, (10, SCREEN_HEIGHT - 30))


def draw_neural_network(net, genome_id, fitness):
    start_x = SCREEN_WIDTH / 4 + 100
    start_y = 700  # Adjust this value to position below the scheduler
    layer_width = 150
    node_radius = 15

    node_positions = {}
    layer_positions = {}
    node_layer = {}

    def make_hashable(nid):
        if isinstance(nid, (int, str)):
            return nid
        elif isinstance(nid, list):
            return tuple(nid)
        else:
            return str(nid)


    # Determine positions for input, hidden, and output nodes
    input_nodes = [make_hashable(nid) for nid in net.input_nodes]
    output_nodes = [make_hashable(nid) for nid in net.output_nodes]
    hidden_nodes = [make_hashable(node[0]) for node in net.node_evals if make_hashable(node[0]) not in input_nodes and make_hashable(node[0]) not in output_nodes]

    all_layers = [input_nodes, hidden_nodes, output_nodes]

    # Assign positions to each node
    for layer_index, layer in enumerate(all_layers):
        layer_positions[layer_index] = []
        y_offset = start_y
        for node_id in layer:
            node_positions[node_id] = (start_x + layer_index * layer_width, y_offset)
            layer_positions[layer_index].append(node_id)
            y_offset += 50
            node_layer[node_id] = layer_index

    # Draw connections (edges) between nodes
    for node in net.node_evals:
        node_id = make_hashable(node[0])
        x, y = node_positions[node_id]
        incoming = node[5]  # The connections are in the 6th element of the tuple
        for conn_id, weight in incoming:
            conn_id = make_hashable(conn_id)
            if conn_id in node_positions:
                x1, y1 = node_positions[conn_id]
                color = CONNECTION_POSITIVE_COLOR if weight > 0 else CONNECTION_NEGATIVE_COLOR
                pygame.draw.line(screen, color, (x1, y1), (x, y), 2)

    # Draw nodes
    for node_id, (x, y) in node_positions.items():
        pygame.draw.circle(screen, NODE_COLOR, (x, y), node_radius)

    # Display genome ID and fitness
    gen_text = font.render(f"Genome ID: {genome_id}", True, (255, 255, 255))
    fit_text = font.render(f"Fitness: {fitness:.2f}", True, (255, 255, 255))
    screen.blit(gen_text, (start_x - 100, SCREEN_HEIGHT - 60))
    screen.blit(fit_text, (start_x - 100, SCREEN_HEIGHT - 30))




def main():
    config_path = "config-feedforward"
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    for student in students:
        student['x'] = random.randint(0, SCREEN_WIDTH)
        student['y'] = random.randint(0, SCREEN_HEIGHT)

    population = neat.Population(config)
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)

    running = True
    clock = pygame.time.Clock()
    generation = 0

      # Define the fitness threshold for considering the problem solved
    fitness_threshold = 1500  # Adjust this threshold as per your criteria
    print("Starting NEAT algorithm...")


    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        winner = population.run(evaluate_grouping, 1)
        generation += 1
        clock.tick(1200)

        # Debugging output to monitor fitness and generation
        print(f"Generation {generation} completed. Best fitness: {winner.fitness}")

        if winner.fitness >= fitness_threshold:
            print(f"Solution found in generation {generation} with fitness {winner.fitness}")

    pygame.quit()

if __name__ == "__main__":
    main()
