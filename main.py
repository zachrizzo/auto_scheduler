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
interests = ["Math", "Science", "History", "Arts"]
locations = ["Classroom", "Recess", "Speech Therapy Room", "Library", "Gym"]  # New locations added

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
    {"name": "Alice", "grade": 9, "school": "School A", "preferred_time": "9:00", "color": student_colors[0], "interest": "Math", "location": "Classroom"},
    {"name": "Bob", "grade": 10, "school": "School A", "preferred_time": "10:00", "color": student_colors[1], "interest": "Science", "location": "Recess"},
    {"name": "Charlie", "grade": 11, "school": "School B", "preferred_time": "11:00", "color": student_colors[2], "interest": "History", "location": "Speech Therapy Room"},
    {"name": "David", "grade": 12, "school": "School B", "preferred_time": "14:00", "color": student_colors[3], "interest": "Arts", "location": "Library"},
    {"name": "Eve", "grade": 9, "school": "School A", "preferred_time": "15:00", "color": student_colors[4], "interest": "Math", "location": "Gym"},
    {"name": "Frank", "grade": 10, "school": "School B", "preferred_time": "9:00", "color": student_colors[5], "interest": "Science", "location": "Classroom"},
    {"name": "Grace", "grade": 11, "school": "School A", "preferred_time": "10:00", "color": student_colors[6], "interest": "History", "location": "Recess"},
    {"name": "Henry", "grade": 12, "school": "School B", "preferred_time": "11:00", "color": student_colors[7], "interest": "Arts", "location": "Speech Therapy Room"},
    {"name": "Ivy", "grade": 9, "school": "School A", "preferred_time": "14:00", "color": student_colors[0], "interest": "Math", "location": "Library"},
    {"name": "Jack", "grade": 10, "school": "School B", "preferred_time": "15:00", "color": student_colors[1], "interest": "Science", "location": "Gym"},
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

        fitness = 1000  # Initial fitness can be adjusted

        # Reset groups
        groups = [[] for _ in range(len(rooms))]

        for student in students:
            # Normalize inputs
            inputs = (
                student["grade"] / 12,
                schools.index(student["school"]) / len(schools),
                time_slots.index(student["preferred_time"]) / len(time_slots),
                interests.index(student["interest"]) / len(interests),  # Interest input
                int(any(teacher["obstacle"] == student["preferred_time"] for teacher in teachers)),  # Obstacle input
                locations.index(student["location"]) / len(locations)  # Location input
            )
            output = net.activate(inputs)

            # Determine group index
            group_index = min(int(output[0] * len(groups)), len(groups) - 1)

            # Check group constraints and assign students
            if all(abs(student["grade"] - existing_student["grade"]) <= 1 for existing_student in groups[group_index]) and \
               all(student["location"] == existing_student["location"] for existing_student in groups[group_index]):  # Check location match
                groups[group_index].append(student)
            else:
                # Find alternative group or penalize heavily
                for i, group in enumerate(groups):
                    if all(abs(student["grade"] - existing_student["grade"]) <= 1 for existing_student in group) and \
                       all(student["location"] == existing_student["location"] for existing_student in group):  # Check location match
                        groups[i].append(student)
                        break
                else:
                    fitness -= 500  # Heavier penalty for no valid group found

        # Re-evaluate fitness based on updated criteria
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

        # Assign calculated fitness
        genome.fitness = max(0, fitness)

        # Optional visualization and delay
        draw_groups(groups, genome_id, fitness)
        draw_neural_network(net, genome_id, fitness)
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
    start_x = SCREEN_WIDTH / 6
    end_x = SCREEN_WIDTH * 5 / 6  # Fixed position for output nodes
    start_y = 600
    node_radius = 10
    vertical_spacing = 60

    node_positions = {}
    node_layer = {}

    input_node_names = {
        net.input_nodes[0]: "Grade",
        net.input_nodes[1]: "School",
        net.input_nodes[2]: "Preferred Time",
        net.input_nodes[3]: "Interest",
        net.input_nodes[4]: "Obstacle",
        net.input_nodes[5]: "Location"
    }

    def make_hashable(nid):
        return tuple(nid) if isinstance(nid, list) else nid

    input_nodes = [make_hashable(nid) for nid in net.input_nodes]
    output_nodes = [make_hashable(nid) for nid in net.output_nodes]
    hidden_nodes = [make_hashable(node[0]) for node in net.node_evals if make_hashable(node[0]) not in input_nodes and make_hashable(node[0]) not in output_nodes]

    # Position input nodes
    for i, node_id in enumerate(input_nodes):
        y = start_y + i * vertical_spacing
        node_positions[node_id] = (start_x, y)
        node_layer[node_id] = 0

    # Position output nodes
    for i, node_id in enumerate(output_nodes):
        y = start_y + i * vertical_spacing
        node_positions[node_id] = (end_x, y)
        node_layer[node_id] = float('inf')  # Use infinity to denote output layer

    # Calculate depths for hidden nodes
    depth_cache = {node: 0 for node in input_nodes}

    def calculate_node_depth(node_id, node_evals, depth_cache):
        if node_id in depth_cache:
            return depth_cache[node_id]
        node = next((n for n in node_evals if make_hashable(n[0]) == node_id), None)
        if node is None:
            depth_cache[node_id] = 0
            return 0
        incoming_connections = node[5]
        max_depth = max(calculate_node_depth(make_hashable(conn_id), node_evals, depth_cache) for conn_id, _ in incoming_connections) + 1
        depth_cache[node_id] = max_depth
        return max_depth

    for node_id in hidden_nodes:
        calculate_node_depth(node_id, net.node_evals, depth_cache)

    max_depth = max(depth_cache.values()) if depth_cache else 0

    # Position hidden nodes
    if max_depth > 0:
        layer_width = (end_x - start_x) / (max_depth + 1)
        hidden_layers = [[] for _ in range(max_depth)]
        for node_id in hidden_nodes:
            depth = depth_cache[node_id]
            hidden_layers[depth - 1].append(node_id)

        for layer_index, layer in enumerate(hidden_layers, start=1):
            x = start_x + layer_index * layer_width
            y_positions = [start_y + i * vertical_spacing for i in range(len(layer))]
            for node_id, y in zip(layer, y_positions):
                node_positions[node_id] = (x, y)
                node_layer[node_id] = layer_index

    # Draw connections
    for node in net.node_evals:
        node_id = make_hashable(node[0])
        x, y = node_positions[node_id]
        incoming = node[5]
        for conn_id, weight in incoming:
            conn_id = make_hashable(conn_id)
            if conn_id in node_positions:
                x1, y1 = node_positions[conn_id]
                color = CONNECTION_POSITIVE_COLOR if weight > 0 else CONNECTION_NEGATIVE_COLOR
                pygame.draw.line(screen, color, (x1, y1), (x, y), 2)

    # Draw nodes and display information
    for node_id, (x, y) in node_positions.items():
        pygame.draw.circle(screen, NODE_COLOR, (x, y), node_radius)

        if node_id in input_nodes:
            text = input_node_names.get(node_id, f"Input: {node_id}")
        elif node_id in output_nodes:
            text = f"Output: {node_id}"
        else:
            text = f"Hidden: {node_id}"

        node_text = font.render(text, True, (255, 255, 255))
        screen.blit(node_text, (x + node_radius + 5, y - node_radius))

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
    fitness_threshold = 1000  # Adjust this threshold as per your criteria
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

            # Enter a loop to display the winning group continuously
            display_winner = True
            while display_winner:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        display_winner = False
                        running = False

                # Display the final solution
                evaluate_grouping([(1, winner)], config)

                pygame.display.flip()
                clock.tick(60)  # Control the frame rate for display

    pygame.quit()

if __name__ == "__main__":
    main()
