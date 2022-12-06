"""
    Computational Intelligence Coursework
        
    Wish, Taimoor, Ionut

"""

import torch 
import torch.nn
from deap import base
from deap import benchmarks
from deap import creator
from deap import tools
import random
import operator
import numpy as np
import math


TYPE = ['global','local','social','competitive']

# softmax activation function
softmax = torch.nn.Softmax(dim=1)

"""
************************
Particle Swarm Optimizer
************************
"""
class ParticleSwarm():
    
    def __init__ (self, 
                  objective :object, 
                  population_size: int, 
                  model :object ,
                  device : str,
                  vmaxInit: float =2.0 ,
                  vminInit : float=0.2,
                  c1 : float =2,
                  c2 : float =2,
                  wmax : float =0.9,
                  wmin : float =0.4,
                  posMinInit : float =-.1,
                  posMaxInit : float =+.1, 
                  pso_type : str ='global',
                  num_neighbours : int = None,
                ):
        
        # Neural Network model 
        self.model = model  
        # nunmber of Swarm 
        self.population_size = population_size 
        # Dimension of decision variable
        self.particle_size = sum(params.numel() for params in self.model.parameters() if params.requires_grad)
        # Max and Min initialize velocity 
        self.vmaxInit = vmaxInit
        self.vminInit = vminInit
        # Accelerate coeffienct
        self.c1 = c1
        self.c2 = c2    
        # upper and lower bound of inertia weight
        self.wmax = wmax
        self.wmin = wmin
        # Max and Min initialize position
        self.posMinInit = posMinInit
        self.posMaxInit = posMaxInit
        # Objective Function 
        self.objective = objective 
        # Global Best
        self.best = None 
         
        # device (CPU or GPU)
        self.device = device
        # type of particle swarm
        self.pso_type = pso_type
        # number of neighbour for local pso
        self.num_neighbours = num_neighbours
        
        # probabilites for Social Learning Swarm
        self.prob = [0]*self.population_size
        for i  in range((self.population_size)):
            self.prob[self.population_size - i - 1] = 1 - i/(self.population_size - 1)
            self.prob[self.population_size - i - 1] = pow(self.prob[self.population_size - i - 1], math.log(math.sqrt(math.ceil(self.particle_size/100.0))))
            
        
        """
        
        Genetate a swarm particle
        
        
        return : swarm particle
        
        """
        def generate(self):

            particles = self.creator.Particle(random.uniform(self.posMinInit, self.posMaxInit) 
                                     for _ in range(self.particle_size))
            particles.speed = [random.uniform(self.vminInit, self.vmaxInit) for _ in range(self.particle_size)]

            return particles
        
        """
        Update velocity of particle  for global and local best
        """ 
        def updateParticle(self ,particle, best, weight):
            
            
            r1 = (random.uniform(0,1) for _ in range(len(particle)))
            r2 = (random.uniform(0,1) for _ in range(len(particle)))

            v_r0 = [weight*x for x in particle.speed]
            v_r1 = [self.c1*x for x in map(operator.mul, r1, map(operator.sub, particle.best, particle))]
            v_r2 = [self.c2*x for x in map(operator.mul, r2, map(operator.sub, best, particle))]

            particle.speed = [0.7*x for x in map(operator.add , v_r0, map(operator.add, v_r1, v_r2))]

            particle[:] = list(map(operator.add, particle, particle.speed))
        
        """
        Update velocity of particle for social learning pso
        """  
        def updateParticle_sl(self,particle,population,center,idx):
            
            r1 = random.uniform(0,1)
            r2 = random.uniform(0,1)
            r3 = random.uniform(0,1)
            
            demonstrator = random.choice(list(self.population[0:idx]))
            epsilon =self.particle_size/100.0 * 0.01
             
            for i in range(self.particle_size):
                
                particle.speed[i] = r1*particle.speed[i] + r2*(demonstrator[i]-particle[i]) + r3*epsilon*(center[i]-particle[i]) 
                particle[i] = particle[i] + particle.speed[i]
                
                
        
        
        """
        Objective funtion (Loss function) 
        """ 
        def objective_function(self, x, y):
            
            with torch.no_grad():
                # set model to training stage 
                
                self.model.train()
                pred = self.model(x).detach()
                #print(pred)
                
                loss = self.objective(pred,y).item()
                
                 # get probabilites of each label
                proba = softmax(pred).cpu().detach().numpy()
                # get predicted label
                pred_labels = [np.argmax(i) for i in proba]
                pred_labels = np.array(pred_labels)

                
                # Calculated accuracy 
                correct = 0
                accuracy = 0
                
                # allocate label to cpu
                gt_labels = y.cpu().detach().numpy()

                for p ,g in zip(pred_labels,gt_labels):

                    if p == g:
                        correct+=1

                accuracy = 100 * (correct/len(gt_labels))
            
            
            return (loss,) , accuracy  # ****return in tuple****
        
        
        """
        
        Deap library section
        
        """
             
        # Deap Creator 
        self.creator  = creator
        self.creator.create("FitnessMin", base.Fitness, weights=(-1.0,)) 
        self.creator.create("Particle", list, fitness=creator.FitnessMin, speed=list,acc=list, smin=None, smax=None, best=None)
        # Deap Toolbox
        self.toolbox = base.Toolbox()
        self.toolbox.register("particle", generate, self)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.particle)
        self.toolbox.register("update", updateParticle)
        self.toolbox.register("update_sl",updateParticle_sl)
        self.toolbox.register("evaluate",objective_function)
        
        # Swarm population
        self.population = self.toolbox.population(n=self.population_size)

    
    
    
    """
    Assign decision variable to a model's parameter weight 
    """ 
    def weight_assign(self, particle):
       
        # convert a list of decision variable to numpy array 
        part = np.array(particle) 
        
        # count the number of paramters in model 
        params_no = sum(params.numel() for params in self.model.parameters())
       
        
        # *** The number of parameter and decision variable should be the same*** 
        assert params_no == len(part)
       
        #  counter 
        params_count = 0
        with torch.no_grad():
            for layer in self.model.parameters():
                
                if layer.requires_grad:
                    #  select a weight from numpy array and reshape it into the same shape as the tensor 
                    weight =part[params_count:params_count+layer.numel()].reshape(layer.data.shape)
                    # convert weight to t ensor and assign a weight to model 
                    layer.data = torch.nn.parameter.Parameter(torch.FloatTensor(weight).to(self.device))
                    # increment counter
                    
                    params_count += layer.numel()
                
    
           
    """
    Calculate Euclidean distance  
    
    """
    def euclidean_distance(self, part1, part2):
        
        return math.sqrt(sum([(q+p)**2 for q,p  in zip(part1, part2)]))

    # Locate the best neighbour
    def neighbourBest(self, pop, individual):
        distances = list()
        #get all distances
        for sample in pop:
            dist = self.euclidean_distance(individual, sample)
            distances.append((sample, dist))
        #sort them by distance
        distances.sort(key=lambda tup: tup[1])
        
        bestndist=distances[0][0].fitness
        bestn=distances[0][0]
        for i in range(self.num_neighbours):
            if distances[i][0].fitness>bestndist:  # is overloaded
                bestndist=distances[i][0].fitness
                bestn=distances[i][0]
        return bestn
    
    # function to get the mean positions of the inviduals (swarm centre)
    def getcenter(self, pop):
        center=list()
        for j in range(self.particle_size): # count through dimensions
            centerj = 0
            for i in pop: # for each particle
                centerj += i[j] # sum up position in dimention j
            centerj /= self.population_size # Average
            center.append(centerj)
        return center

             
    
    
    """
    
    Optimize weight of neural network 
    
    
    """ 
    def search(self,iter_no,nepoch,x,y):
        
        #inertia
        w = self.wmax - ((self.wmax-self.wmin) * iter_no/nepoch)
        for particle in self.population:
            # assign decision variable of a current particle in model's parameter 
            self.weight_assign(particle)
            
            # Calculate a fitness
            particle.fitness.values, particle.acc = self.toolbox.evaluate(self, x, y)
            
            if (not particle.best) or (particle.best.fitness < particle.fitness):
                
                particle.best = creator.Particle(particle)
                particle.best.fitness.values = particle.fitness.values
                particle.best.acc = particle.acc
                
            if (not self.best) or self.best.fitness < particle.fitness:
                
                self.best = self.creator.Particle(particle)
                self.best.fitness.values = particle.fitness.values
                self.best.acc = particle.acc
                
        if self.pso_type == "social":
            
            self.population.sort(key=lambda x: x.fitness, reverse=True) 
            social_best = self.population[0]
            temp = [x.fitness for x in self.population]
            
            center = self.getcenter(self.population)
            for i in reversed(range(len(self.population)-1)):
                
                
                if random.uniform(0,1)<self.prob[i+1]: 
                        
                    self.toolbox.update_sl(self, self.population[i+1], self.population, center, i+1) 
           
        else :
        
            # update a velocity for all particles 
            for particle in self.population:
                
                if self.pso_type == "global":
                    
                    self.toolbox.update(self, particle, self.best, w) 
                
                elif self.pso_type == "local":
                
                    neighbour = self.neighbourBest(self.population,particle) 
                    self.toolbox.update(self,particle,neighbour,w)
        
        # Assing the decision varible of Global best to model 
        if self.pso_type == "social":
            
            self.weight_assign(social_best)
            loss = social_best.fitness.values[0]
            acc  = self.best.acc
            
        else:
            self.weight_assign(self.best)
            loss = self.best.fitness.values[0]
            acc = self.best.acc

        
        return loss, acc
            








        
        

        

        




