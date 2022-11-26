
import random
import numpy as np
import array
from sympy.combinatorics.graycode import GrayCode
from sympy.combinatorics.graycode import gray_to_bin
from deap import creator 
from deap import base 
from deap import tools
from deap import algorithms
from deap import benchmarks
import torch
from tqdm import tqdm

# softmax activation function
softmax = torch.nn.Softmax(dim=1)

# Ecncoding type

ENCODING = ['binary','real']

"""
Genetic algorithms
"""
class GeneticAlgorithms():
    
    
    def __init__ (self, objective, population_size,
                  dimension,
                  numOfBits = 10,
                  nElitists=1,
                  crossPoint=3, 
                  crossProb=0.6, 
                  flipProb =1, 
                  mutateProb=0.1,
                  lower_bound = -1,
                  upper_bound = 1,
                  encoding = "binary"
                  ):
        
        if encoding not in ENCODING:
            
            raise ValueError("The type of encoding should be in this list {}".format(ENCODING))
             
        
        
    
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
        self.encoding = encoding
        self.population = None 
        self.model = None
        self.device =None
        self.objective = objective

    
        
        def uniform(low,up,size=None):
            try:
                return [random.uniform(a,b) for a,b in zip(low,up)]
            except TypeError:
                
                return  [random.uniform(a,b) for a, b in zip([low] * size, [up] *size)]

       

        """
        Objective funtion or Loss funciton
        """
        def objective_nn(data, labels):
                 
            self.model.train()
            pred = self.model(data)
            loss = self.objective(softmax(pred),labels).item()
            
            proba = softmax(pred).cpu().detach().numpy()
            
            pred_labels = [np.argmax(i) for i in proba]
            
            pred_labels = np.array(pred_labels)
            
            # calculate accuracy 
            
            correct = 0
            accuracy = 0
            
            gt_labels = labels.cpu().detach().numpy()
            
            for p, g in zip(pred_labels, gt_labels):
                
                if  p==g : 
                    
                    correct+=1
                    
            accuracy = 100 * (correct/len(gt_labels))
            
                
            return (loss,) , accuracy
        
        
        """
        Deap Libraries section 
        """
        
        # Deap Creator
        self.creator = creator
        self.creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        if self.encoding == "binary":
       
            self.creator.create("Individual", list, fitness=creator.FitnessMin, acc=list)
            
        else: 
            
            self.creator.create("Individual",array.array, typecode='d', fitness=creator.FitnessMin)
        
            
    
        
        # Deap Toolbox
        self.toolbox = base.Toolbox()
        if self.encoding == "binary":
            self.toolbox.register("attr_bool", random.randint, 0, 1)
            self.toolbox.register("individual", tools.initRepeat, self.creator.Individual, self.toolbox.attr_bool, self.numOfBits*self.dimension)
            self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
            self.toolbox.register("mate", tools.cxTwoPoint)
            self.toolbox.register("mutate",tools.mutFlipBit, indpb= self.flipProb)
            
        else:
            self.toolbox.register("attr_float", uniform,self.lower_bound, self.upper_bound, self.dimension)
            self.toolbox.register("individual", tools.initIterate, self.creator.Individual, self.toolbox.attr_float)
            self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
            self.toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=self.lower_bound, up=self.upper_bound, eta=20.0)
            self.toolbox.register("mutate",tools.mutPolynomialBounded, low=self.lower_bound, up=self.upper_bound, eta=20.0,  indpb= 1/self.dimension)
            
            
            
        
        
        self.toolbox.register("select",tools.selRoulette, fit_attr='fitness')
        self.toolbox.register("evaluate_nn",objective_nn)
   
    
        
     
    """
    Initialize  a  population of genetic algorithms using for optimzing a weight for neural network
    """     
    def initPop(self, model, device, data):
        
        #  number of population
        self.population = self.toolbox.population(n=self.population_size)
        
        self.model = model
        
        self.device = device
        
        fitnesses = [] 
        accuracy  = []
        
        print("Calculating fitness of individual")
        
        # iterate to each individual in a population
        for individual in tqdm(self.population):
            
            if self.encoding == "binary":
           
                # convert a binary representation of a value from individual to a  value represent weight 
                weight = self.separatevariables(individual)
                
            else:
                
                weight = individual
            
            
            #  assign weight to a model 
            self.weight_assign(weight)
           
            # calculate a fitness  for individual
            for images, labels in data:
                
                images = images.to(device)
                labels = labels.to(device)
                fitness, acc = self.toolbox.evaluate_nn(images, labels)
                
                fitnesses.append(fitness)
                accuracy.append(acc)
        
        # assign a fitness to each individual 
        for individual , fitness , acc in zip(self.population,fitnesses, accuracy):
            
            individual.fitness.values = fitness
            individual.acc = acc
            
    """
    Seperate decison variable 
    
    This is the same function in a constructor
    """
    def separatevariables(self, v):
            
        variable = []
        num_bits = self.numOfBits
            
        bit_counter = 0 
        for i in range(self.dimension):
                 
            variable.append(self.chrom2real(v[bit_counter:num_bits*(1+i)]))
            bit_counter+=num_bits
             
        return variable
            

    """
    Convert chromosome to real number
    
    This is the same funciton in the constructor
    """ 
    def chrom2real(self, c):
        
            indasstring = ''.join(map(str,c))
            degray = gray_to_bin(indasstring)
            numasint = int(degray, 2)
            numinrange= self.lower_bound + ( (self.upper_bound-self.lower_bound)*(numasint/(self.maxnum-1)))
            
            return numinrange
        
        
    """
    Assign a weight to a model 
    
    """     
    def weight_assign(self, individual):
        
        # convert a list of weight from individual into numpy array      
        ind = np.array(individual)
        
        # count the number of parameters  of a model
        params_no = sum( params.numel() for params in self.model.parameters())
        
        # The lenght of the paramters should have the same size to the dimension of decision variable (or the length of its value) 
        assert params_no == len(ind)
        
        # count the number of weight that has been assign 
        params_count = 0 
        
        # iterate to every layer of model 
        for layer in self.model.parameters():
            
            # count the number of weight in this layer 
            # select  a same size of the value of decsion variable to  a weight of a model in current layer
            # reshape an the array to shape shape as the weight of model
            weight = ind[params_count: params_count+layer.numel()].reshape(layer.data.shape)
        
            # Assign the weight to current layer
            layer.data = torch.nn.parameter.Parameter(torch.FloatTensor(weight).to(self.device))
            
            # incremen the parameter count 
            params_count += layer.numel()
        
        
             
  
    
    
    """
    Optimizing a neural network
    """
    def search(self, images, labels):
        
        # select an offspring for reproduction 
        offspring = tools.selBest(self.population, self.nElistists) + self.toolbox.select(self.population, (self.population_size-self.nElistists))
        # clone offspring
        offspring = list(map(self.toolbox.clone, offspring))
        
        # Crossover process
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            
            
            # if the random number is less than probability perform crossover 
            if random.random() < self.crossProb:
               
               # perform crossover 
               self.toolbox.mate(child1,child2) 
                
               # delete a fitness of a child generated form crossover 
               del child1.fitness.values
               del child2.fitness.values
        
        
        
        # Muatation process 
        for mutant in offspring:
           
            # if random number genreate is less that the probabilty then perform mutation 
            if random.random() < self.mutateProb:
               
                # perform mutation 
                self.toolbox.mutate(mutant)
                
                # delete the fitness of a mutant child 
                del mutant.fitness.values
        
        # Only select the individual that just perform crossover and mutation
        # to recalculate its fitness after performing reproduction 
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        
        fitnesses = []
        accuracy = []
        
        # Calculate a fitness for a new offspring
        for individual in (invalid_ind):
            
            
            if self.encoding == "binary":
                weight = self.separatevariables(individual)
            
            else:
                
                weight = individual
            
            self.weight_assign(weight)
            
            fitness, acc= self.toolbox.evaluate_nn(images, labels)
                
            fitnesses.append(fitness)
            accuracy.append(acc)
         
        # Assign a new fitness of a new offspring 
        for ind, fit, acc in zip(invalid_ind, fitnesses, accuracy):
            
            ind.fitness.values = fit
            ind.acc =acc
        
        # Survival Selection
        self.population[:] =  tools.selBest(self.population + offspring, self.population_size)
         
        
        # select the best individual
        best_individual= tools.selBest(self.population,1)[0]
        best_acc = best_individual.acc
        # for i in self.population:
        #     print(i.fitness.values[0])
        
        
        if self.encoding == "binary":
            best_weight = self.separatevariables(best_individual)
        else :
            best_weight = best_individual 
            
        self.weight_assign(best_weight)
        loss = best_individual.fitness.values[0]
        
        
        return loss , best_acc
    
    

