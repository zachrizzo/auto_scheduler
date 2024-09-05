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
locations = ["Classroom", "Recess", "Speech Therapy Room", "Library", "Gym"]

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
    {"name": "Kate", "grade": 11, "school": "School A", "preferred_time": "9:00", "color": student_colors[2], "interest": "History", "location": "Classroom"},
    {"name": "Liam", "grade": 12, "school": "School B", "preferred_time": "10:00", "color": student_colors[3], "interest": "Arts", "location": "Recess"},
    {"name": "Mia", "grade": 9, "school": "School A", "preferred_time": "11:00", "color": student_colors[4], "interest": "Math", "location": "Speech Therapy Room"},
    {"name": "Noah", "grade": 10, "school": "School B", "preferred_time": "14:00", "color": student_colors[5], "interest": "Science", "location": "Library"},
    {"name": "Olivia", "grade": 11, "school": "School A", "preferred_time": "15:00", "color": student_colors[6], "interest": "History", "location": "Gym"},
    {"name": "Peter", "grade": 12, "school": "School B", "preferred_time": "9:00", "color": student_colors[7], "interest": "Arts", "location": "Classroom"},
    {"name": "Quinn", "grade": 9, "school": "School A", "preferred_time": "10:00", "color": student_colors[0], "interest": "Math", "location": "Recess"},
    {"name": "Rachel", "grade": 10, "school": "School B", "preferred_time": "11:00", "color": student_colors[1], "interest": "Science", "location": "Speech Therapy Room"},
    {"name": "Sam", "grade": 11, "school": "School A", "preferred_time": "14:00", "color": student_colors[2], "interest": "History", "location": "Library"},
    {"name": "Thomas", "grade": 12, "school": "School B", "preferred_time": "15:00", "color": student_colors[3], "interest": "Arts", "location": "Gym"},
    {"name": "Ursula", "grade": 9, "school": "School A", "preferred_time": "9:00", "color": student_colors[4], "interest": "Math", "location": "Classroom"},
    {"name": "Victor", "grade": 10, "school": "School B", "preferred_time": "10:00", "color": student_colors[5], "interest": "Science", "location": "Recess"},
    {"name": "Wendy", "grade": 11, "school": "School A", "preferred_time": "11:00", "color": student_colors[6], "interest": "History", "location": "Speech Therapy Room"},
    {"name": "Xavier", "grade": 12, "school": "School B", "preferred_time": "14:00", "color": student_colors[7], "interest": "Arts", "location": "Library"},
    {"name": "Yara", "grade": 9, "school": "School A", "preferred_time": "15:00", "color": student_colors[0], "interest": "Math", "location": "Gym"},
    {"name": "Zane", "grade": 10, "school": "School B", "preferred_time": "9:00", "color": student_colors[1], "interest": "Science", "location": "Classroom"},
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
    best_fitness = float('-inf')
    best_groups = None
    best_unscheduled = None
    best_genome_id = None
    best_net = None
    genome_counter = 0  # Counter to track every 10 genomes

    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)

        fitness = 1000  # Start with a baseline fitness
        groups = [[] for _ in range(len(rooms))]
        unscheduled_students = students.copy()

        # Add randomness in group assignment for exploration
        random.shuffle(unscheduled_students)

        # First pass: Use neural network to make initial assignments
        for student in unscheduled_students[:]:
            inputs = (
                student["grade"] / 12,
                schools.index(student["school"]) / len(schools),
                time_slots.index(student["preferred_time"]) / len(time_slots),
                interests.index(student["interest"]) / len(interests),
                int(any(teacher["obstacle"] == student["preferred_time"] for teacher in teachers)),
                locations.index(student["location"]) / len(locations)
            )
            output = net.activate(inputs)
            group_index = min(int(output[0] * len(groups)), len(groups) - 1)

            # Add flexibility by allowing students into groups with slight mismatches
            if len(groups[group_index]) < 4:
                if not groups[group_index] or (
                    abs(student["grade"] - groups[group_index][0]["grade"]) <= 2 and  # Allow larger grade difference
                    student["school"] == groups[group_index][0]["school"] and
                    student["location"] == groups[group_index][0]["location"]
                ):
                    groups[group_index].append(student)
                    unscheduled_students.remove(student)

        # Second pass: Try to fit remaining students with flexibility
        for student in unscheduled_students[:]:
            for group in groups:
                if len(group) < 4 and (not group or (
                    abs(student["grade"] - group[0]["grade"]) <= 2 and  # Allow more mismatch
                    student["school"] == group[0]["school"] and
                    student["location"] == group[0]["location"]
                )):
                    group.append(student)
                    unscheduled_students.remove(student)
                    break

        # Calculate fitness
        for group in groups:
            if len(group) == 0:
                fitness -= 50  # Less harsh penalty for empty groups
            elif len(group) == 1:
                fitness += 25  # Lower reward for small groups
            elif 2 <= len(group) <= 4:
                fitness += 100 * len(group)  # Higher reward for balanced groups
            else:
                fitness -= 25 * (len(group) - 4)  # Lower penalty for oversized groups

            if len(group) > 0:
                time_slot = time_slots[groups.index(group) % len(time_slots)]
                if all(student["preferred_time"] == time_slot for student in group):
                    fitness += 50 * len(group)
                if all(student["interest"] == group[0]["interest"] for student in group):
                    fitness += 50 * len(group)

        # Reduce heavy penalties for unscheduled students
        fitness -= 100 * len(unscheduled_students)

        # Add bonus for solutions with fewer unscheduled students, even if not perfect
        if len(unscheduled_students) < len(students) / 2:
            fitness += 200  # Bonus for scheduling more than half

        genome.fitness = max(0, fitness)

        if fitness > best_fitness:
            best_fitness = fitness
            best_groups = groups
            best_unscheduled = unscheduled_students
            best_genome_id = genome_id
            best_net = net

        # Increment the counter
        genome_counter += 1

        # Draw every 10 genomes
        if genome_counter % 100 == 0 and best_groups:
            draw_groups(best_groups, best_unscheduled, best_genome_id, best_fitness)
            draw_neural_network(best_net, best_genome_id, best_fitness)
            pygame.display.flip()

    return best_fitness




def draw_groups(groups, unscheduled_students, genome_id, fitness):
    screen.fill(BACKGROUND_COLOR)

    # Calculate number of scheduled and unscheduled students
    scheduled_count = sum(len(group) for group in groups)
    unscheduled_count = len(unscheduled_students)

    # Draw the counts at the top of the screen
    count_text = font.render(f"Scheduled: {scheduled_count} | Unscheduled: {unscheduled_count}", True, (255, 255, 255))
    screen.blit(count_text, (10, 10))

    # Draw scheduled groups in rooms
    for i, (room, group) in enumerate(zip(rooms, groups)):
        pygame.draw.rect(screen, ROOM_COLOR, room)
        time_slot = time_slots[i % len(time_slots)]
        text = font.render(f"Time: {time_slot}", True, (255, 255, 255))
        screen.blit(text, (room.x + 10, room.y + 10))

        for teacher in teachers:
            if group and teacher["school"] == group[0]["school"] and teacher["obstacle"] == time_slot:
                pygame.draw.rect(screen, OBSTACLE_COLOR, (room.x, room.y, 10, room.height))

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

    # Draw unscheduled students list at the bottom
    unscheduled_y = 600
    pygame.draw.rect(screen, (50, 50, 50), (0, unscheduled_y, SCREEN_WIDTH, SCREEN_HEIGHT - unscheduled_y))
    unscheduled_text = font.render("Unscheduled Students:", True, (255, 255, 255))
    screen.blit(unscheduled_text, (10, unscheduled_y + 10))

    for i, student in enumerate(unscheduled_students):
        text = font.render(f"{student['name']} ({student['grade']} - {student['school']})", True, student['color'])
        screen.blit(text, (10, unscheduled_y + 40 + i * 30))

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

    def check_extinction(population):
        fitnesses = [genome.fitness for genome in population.population.values() if genome.fitness is not None]
        if not fitnesses:
            print("Warning: No valid fitness values found.")
            return False
        best_fitness = max(fitnesses)
        if best_fitness < config.genome_config.fitness_threshold:
            print(f"Extinction event triggered! Best fitness: {best_fitness}")
            return True
        return False

    # Check for extinction at the start
    if check_extinction(population):
        population = neat.Population(config)  # Reset the population

    # Add a custom extinction handler
    def on_extinction():
        print("Total extinction occurred! Resetting population...")

    # Add custom extinction handler to NEAT
    population.config.reset_on_extinction = True
    population.config.extinction_callback = on_extinction

    running = True
    clock = pygame.time.Clock()
    generation = 0

    # Define the fitness threshold for considering the problem solved
    fitness_threshold = config.fitness_threshold  # Or set it directly if not using from config
    print("Starting NEAT algorithm...", fitness_threshold)

    # Initialize variables for fitness plateau detection
    best_fitness = None
    stagnant_generations = 0
    max_stagnant_generations = 200  # Number of generations to allow without improvement

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Run NEAT for one generation
        winner = population.run(evaluate_grouping, 1)
        generation += 1
        clock.tick(1200)

        # Debugging output to monitor fitness and generation
        print(f"Generation {generation} completed. Best fitness: {winner.fitness}")

        # Check if fitness has plateaued
        if best_fitness is None or winner.fitness > best_fitness:
            best_fitness = winner.fitness
            stagnant_generations = 0
        else:
            stagnant_generations += 1

        if stagnant_generations >= max_stagnant_generations:
            print(f"No improvement for {max_stagnant_generations} generations. Best fitness achieved: {best_fitness}")
            break

        if winner.fitness >= fitness_threshold:
            print(f"Solution found in generation {generation} with fitness {winner.fitness}")
            break

    # Display the final solution continuously
    print("Displaying final schedule...")
    display_winner = True
    while display_winner:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                display_winner = False

        # Display the final solution
        evaluate_grouping([(1, winner)], config)

        pygame.display.flip()
        clock.tick(60)  # Control the frame rate for display

    pygame.quit()

    # Return the best fitness and generation count for further analysis or logging
    return best_fitness, generation


if __name__ == "__main__":
    best_fitness, total_generations = main()
    print(f"Best fitness: {best_fitness}, achieved in {total_generations} generations.")
