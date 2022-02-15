import pygame
import time
import os
import random
import neat
pygame.font.init()

window_height = 600
window_width = 800

space_ship_img = pygame.transform.scale((pygame.image.load(os.path.join("spaceship.png"))), (150, 150))
pip_img = pygame.transform.scale(pygame.image.load(os.path.join("pipe_n.png")), (300, 300))
background_img = pygame.image.load(os.path.join("background.jpg"))
base_img = pygame.transform.scale(pygame.image.load(os.path.join("base.png")), (300, 300))
font = pygame.font.SysFont("comicsans", 50)
class Space_Ship:
    def __init__(self,x,y):
        self.x = x
        self.y = y
        self.t_count = 0
        self.speed = 0
        self.h = self.y
        self.image_count = 0
        self.space_ship_img = space_ship_img

    def jump(self):
        self.speed = -10.5
        self.t_count = 0
        self.h = self.y

    def move(self):
        self.t_count = self.t_count + 1
        distance = self.speed*self.t_count + 0.5*self.t_count**2

        if distance >= 16:
            distance = 16
        if distance < 0:
            distance = distance - 2

        self.y = self.y + distance


    def draw(self, win):
        self.image_count += 1
        win.blit(space_ship_img, (self.x, self.y))

    def get_mask(self):
       return pygame.mask.from_surface(self.space_ship_img)
    
class Pipe:
    diff = 200
    speed = 5

    def __init__(self, x):
        self.x = x
        self.height = 0
        self.top = 0
        self.bottom = 0
        self.pipe_top = pygame.transform.flip(pip_img, False, True)
        self.pipe_bottom = pip_img
        self.passed = False
        self.set_height()

    def set_height(self):
        self.height = random.randrange(50, 450)
        self.top = self.height - self.pipe_top.get_height()
        self.bottom = self.height + self.diff

    def move(self):
        self.x -= self.speed

    def draw(self, win):
        win.blit(self.pipe_top, (self.x, self.top))
        win.blit(self.pipe_bottom, (self.x, self.bottom))

    def handle_collision(self, spaceship):
        spaceship_mask = spaceship.get_mask()
        top_mask = pygame.mask.from_surface(self.pipe_top)
        bottom_mask = pygame.mask.from_surface(self.pipe_bottom)
        top_offset = (self.x-spaceship.x, self.top-round(spaceship.y))
        bottom_offset = (self.x - spaceship.x, self.bottom - round(spaceship.y))
        b_point = spaceship_mask.overlap(bottom_mask, bottom_offset)
        t_point = spaceship_mask.overlap(top_mask, top_offset)
        if t_point or b_point:
            return True

        return False


class Base:
    speed = 10
    width = base_img.get_width()
    img = base_img

    def __init__(self, y):
        self.y = y
        self.x1 = 0
        self.x2 = self.width

    def move(self):
        self.x1 -= self.speed
        self.x2 -= self.speed

        if self.x1 + self.width < 0:
            self.x1 = self.x2 + self.width

        if self.x2 + self.width < 0:
            self.x2 = self.x1 + self.width

    def draw(self, win):
        win.blit(self.img, (self.x1, self.y))
        win.blit(self.img, (self.x2, self.y))

def draw_window(win, spaceships, pipes, base, score):
    win.blit(background_img, (0, 0))
    for pipe in pipes:
        pipe.draw(win)
    text = font.render("Score: " + str(score), 1, (255, 255, 255))
    win.blit(text, (window_width-10-text.get_width(), 10))
    base.draw(win)
    for spaceship in spaceships:
        spaceship.draw(win)
    pygame.display.update()

def main(genomes, config):
    nets = []
    genome_ = []
    spaceships = []
    for _, g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        spaceships.append(Space_Ship(20, 200))
        g.fitness = 0
        genome_.append(g)

    base = Base(400)
    pipes = [Pipe(400)]
    win = pygame.display.set_mode((window_width, window_height))
    clock = pygame.time.Clock()
    score = 0
    run = True
    while run:
        clock.tick(20)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()
                quit()
        pipe_ind = 0
        if len(spaceships) > 0:
            if len(pipes) > 1 and spaceships[0].x > pipes[0].x + pipes[0].pipe_top.get_width():
                pipe_ind = 1
        else:
            run = False
            break
        for x, spaceship, in enumerate(spaceships):
            spaceship.move()
            #Choose low fitness to not reach complexity threshold
            genome_[x].fitness += 0.1
            output = nets[x].activate((spaceship.y, abs(spaceship.y-pipes[pipe_ind].height), abs(spaceship.y-pipes[pipe_ind].bottom)))
            if output[0] > 0.5:
                spaceship.jump()
        spaceship.move()
        append_pipe = False
        rem = []
        
        for pipe in pipes:
            for x, spaceship in enumerate(spaceships):
                if pipe.handle_collision(spaceship):
                    genome_[x].fitness -= 1
                    spaceships.pop(spaceship)
                    nets.pop(x)
                    genome_.pop(x)

                if not pipe.passed and pipe.x < spaceship.x:
                    pipe.passed = True
                    append_pipe = True
            if pipe.x + pipe.pipe_top.get_width() < 0:
                rem.append(pipe)

            
            pipe.move()
        if append_pipe:
            score += 1
            for g in genome_:
                g.fitness += 5
            pipes.append(Pipe(400))
        for r in rem:
            pipes.remove(r)
        base.move()
        draw_window(win, spaceships, pipes, base, score)
        for x, spacehip in enumerate(spaceships):
            if spaceship.y + space_ship_img.get_width() >= 400 or spaceship.y<-50:
                spaceships.pop(x)
                nets.pop(x)
                genome_.pop(x)




def run(l_path):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, 
    neat.DefaultStagnation, l_path)

    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    success = p.run(main, 50)

if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    l_path = os.path.join(local_dir, "config.txt")
    run(l_path)