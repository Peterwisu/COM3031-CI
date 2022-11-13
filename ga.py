
import random
import numpy as np
from sympy.combinatorics.graycode import GrayCode
from sympy.combinatorics.graycode import gray_to_bin
from deap import creator 
from deap import base 
from deap import tools
from deap import algorithms
from deap import benchmarks
import torch
from tqdm import tqdm


"""
Genetic algorithms
"""
class GA():
    
    
    def __init__ (self, objective, population_size, dimension,
                  numOfBits = 10, nElitists=1, crossPoint=2, crossProb=0.6,
                  flipProb =1, mutateProb=0.1, lower_bound = -2, upper_bound = 2):
    
        self.population_size = population_size 
        self.dimension = dimension  
        self.numOfBits = numOfBits 
        self.nElistists = nElitists 
        self.crossPoint = crossPoint 
        self.crossProb = crossProb 
        self.flipProb = flipProb 
        self.mutateProb = mutateProb 
        self.lower_bound = lower_bound 
        self.upper_bound = upper_bound 
        self.maxnum = 2**self.numOfBits
        
        self.population = None 
        self.model = None
        self.device =None
        self.objective = objective

     
        
        def eval_sphere(individual):
            
            sep = separatevariables(individual)
            
            f = (sep[0]-1)**2 + (sep[1]-2)**2
            return  benchmarks.sphere(sep)
        
        
        def chrom2real(c):
        
            indasstring = ''.join(map(str,c))
            degray = gray_to_bin(indasstring)
            numasint = int(degray, 2)
            numinrange= self.lower_bound + ( (self.upper_bound-self.lower_bound)*(numasint/(self.maxnum-1)))
            
            return numinrange
        
        def separatevariables(v):
            
            variable = []
            num_bits = self.numOfBits
            
            bit_counter = 0 
            for i in range(dimension):
                
                #print(bit_counter)
                #print(num_bits*(1+i))
                
                #chromosome = v[bit_counter:num_bits*(1+i)]
                #print(chromosome)
                variable.append(chrom2real(v[bit_counter:num_bits*(1+i)]))
                bit_counter+=num_bits
            
            
            #print(variable)
            
    
            return variable
            #return chrom2real(v[0:self.numOfBits]), chrom2real(v[self.numOfBits:])
        
        def objective_nn(data, labels):
            
            with torch.no_grad():
                
                self.model.train()
                pred = self.model(data)
                loss = self.objective(pred,labels).item()
                
            return (loss,)
        
        
        """
        Deap Libraries section 
        """
        
        # Deap Creator
        self.creator = creator
        self.creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        self.creator.create("Individual", list, fitness=creator.FitnessMin)
        
        # Deap Toolbox
        self.toolbox = base.Toolbox()
        self.toolbox.register("attr_bool", random.randint, 0, 1)
        self.toolbox.register("individual", tools.initRepeat, self.creator.Individual, self.toolbox.attr_bool, self.numOfBits*self.dimension)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate",tools.mutFlipBit, indpb= self.flipProb)
        self.toolbox.register("select",tools.selTournament, fit_attr='fitness')
        
        self.toolbox.register("evaluate_nn",objective_nn)
        self.toolbox.register("evaluate", eval_sphere)
    
        
    def initInd(self):
    
        self.population = self.toolbox.population(n=self.population_size)
        
        fitnesses = list(map(self.toolbox.evaluate, self.population))
        
        for ind, fit in zip(self.population, fitnesses):
            
            ind.fitness.values = fit
            
        print("Selecting Parent of %i" %self.population_size)
        
    def initNN(self, model, device, data):
        
        self.population = self.toolbox.population(n=self.population_size)
        
        self.model = model
        
        self.device = device
        
        fitnesses = [] 
        
        
        print("Calculating fitness of individual")
        for individual in tqdm(self.population):
            
            weight = self.separatevariables(individual)
            
            self.weight_assign(weight)
            
            for images, labels in data:
                
                images = images.to(device)
                labels = labels.to(device)
                fitness = self.toolbox.evaluate_nn(images, labels)
                
                fitnesses.append(fitness)
        
        
        for individual , fitness in zip(self.population,fitnesses):
            
            individual.fitness.values = fitness
            
 
    def separatevariables(self, v):
            
        variable = []
        num_bits = self.numOfBits
            
        bit_counter = 0 
        for i in range(self.dimension):
                
                #print(bit_counter)
                #print(num_bits*(1+i))
                
            #chromosome = v[bit_counter:num_bits*(1+i)]
                #print(chromosome)
            variable.append(self.chrom2real(v[bit_counter:num_bits*(1+i)]))
            bit_counter+=num_bits
            
            
            #print(variable)
            
    
        return variable
            #return chrom2real(v[0:self.numOfBits]), chrom2real(v[self.numOfBits:])    

    
    def chrom2real(self, c):
        
            indasstring = ''.join(map(str,c))
            degray = gray_to_bin(indasstring)
            numasint = int(degray, 2)
            numinrange= self.lower_bound + ( (self.upper_bound-self.lower_bound)*(numasint/(self.maxnum-1)))
            
            return numinrange
        
    def weight_assign(self, individual):
        
        ind = np.array(individual)
        
        params_no = sum( params.numel() for params in self.model.parameters())
        
        assert params_no == len(ind)
        
        params_count = 0 
        
        for layer in self.model.parameters():
            
            weight = ind[params_count: params_count+layer.numel()].reshape(layer.data.shape)
            print(layer)
            
            layer.data = torch.nn.parameter.Parameter(torch.FloatTensor(weight).to(self.device))
            
            params_count += layer.numel()
             
        
    def optimize(self,i):
        
        
        
        offspring = tools.selBest(self.population, self.nElistists) + self.toolbox.select(self.population, (self.population_size-self.nElistists), (self.population_size-self.nElistists))
        
        offspring = list(map(self.toolbox.clone, offspring))
        
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            
            if random.random() < self.crossProb:
                
               self.toolbox.mate(child1,child2) 
               
               del child1.fitness.values
               del child2.fitness.values
               
        for mutant in offspring:
            
            if random.random() < self.mutateProb:
                
                self.toolbox.mutate(mutant)
                
                del mutant.fitness.values
                
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        
        fitnesses = map(self.toolbox.evaluate, invalid_ind)
        
        for ind, fit in zip(invalid_ind, fitnesses):
            
            ind.fitness.values = fit
            
        self.population[:] = offspring
        
        
        if i%10 == 0:
            fits = [ind.fitness.values[0] for ind in self.population]

            length = len(self.population)
            mean = sum(fits) / length
            sum2 = sum(x*x for x in fits)
            std = abs(sum2 / length - mean**2)**0.5

            print("  Min %s" % min(fits))
            print("  Max %s" % max(fits))
            print("  Avg %s" % mean)
            print("  Std %s" % std)

    
    def optimize_NN(self, images, labels):
        
        offspring = tools.selBest(self.population, self.nElistists) + self.toolbox.select(self.population, (self.population_size-self.nElistists), (self.population_size-self.nElistists))
        
        offspring = list(map(self.toolbox.clone, offspring))
        
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            
            if random.random() < self.crossProb:
                
               self.toolbox.mate(child1,child2) 
               
               del child1.fitness.values
               del child2.fitness.values
               
        for mutant in offspring:
            
            if random.random() < self.mutateProb:
                
                self.toolbox.mutate(mutant)
                
                del mutant.fitness.values
                
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        
        print(len(invalid_ind))
        fitnesses = []
        for individual in (invalid_ind):
            
            weight = self.separatevariables(individual)
            
            self.weight_assign(weight)
            
            fitness = self.toolbox.evaluate_nn(images, labels)
                
            fitnesses.append(fitness)
         
        
        for ind, fit in zip(invalid_ind, fitnesses):
            
            ind.fitness.values = fit
            
        self.population[:] = offspring
        
        best_individual= tools.selBest(self.population,1)[0]
        best_weight = self.separatevariables(best_individual)
        self.weight_assign(best_weight)
        loss = best_individual.fitness.values[0]
        
        
        return loss
        
# ga = GA(objective=None,population_size=200,dimension=2, numOfBits=10)
# ga.initInd()


# for i in range(100):
    
#     print("Generattion {}".format(i))
#     ga.optimize(i)
    
# best = tools.selBest(ga.population,1)[0]

# print("Best individal is {} which is {} and {}".format(best, ga.separatevariables(best), best.fitness.values))
# #print("Decoded x1, x2,  is {}  and {} ".format(ga.separatevariables(best)[0], ga.separatevariables(best)[1]))
    
    
    
    
