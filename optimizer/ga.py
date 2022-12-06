
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

# Survial Selection type
SELECTION = ['steady','gen']


"""
Genetic algorithms
"""
class GeneticAlgorithms():
    
    def __init__ (self, 
                  objective, 
                  population_size,
                  model,
                  device,
                  data,
                  numOfBits = 10,
                  nElitists=1,
                  crossProb=0.6, 
                  flipProb =1, 
                  mutateProb=0.1,
                  lower_bound = -1,
                  upper_bound = 1,
                  encoding = "binary",
                  selection="steady"
                  ):
        
        
        if encoding not in ENCODING:
            
            raise ValueError("The type of encoding should be in this list {}".format(ENCODING))

        if selection not in SELECTION:

            raise ValueError("The type of  selection should be in this list {}".format(SELECTION))
             
        # neural network model        
        self.model = model
        # device 
        self.device = device
        # size of population 
        self.population_size = population_size
        # dimension of Decision varivale 
        self.dimension = sum([ params.numel() for params in self.model.parameters()])
        # Number of bit represent values in each dimension
        self.numOfBits = numOfBits 
        # Number of Elistism
        self.nElistists = nElitists 
        # Probabilities for crossover 
        self.crossProb = crossProb 
        # Probabilities of flipping a bit in mutation
        self.flipProb = flipProb/(self.dimension * numOfBits) 
        # Probabilities for Mutation
        self.mutateProb = mutateProb 
        # Lower bound of a decision variable values
        self.lower_bound = lower_bound 
        # Upper bound of a decision variable values
        self.upper_bound = upper_bound 
        # Max number
        self.maxnum = 2**self.numOfBits 
        # Type of encoding
        self.encoding = encoding
        # Objective function  
        self.objective = objective
        # Type of survival selection
        self.selection = selection

    
        """
        Function to create a individual for real coded GA 
        """
        def uniform(low,up,size=None):
            try:
                return [random.uniform(a,b) for a,b in zip(low,up)]
            except TypeError:
                
                return  [random.uniform(a,b) for a, b in zip([low] * size, [up] *size)]

       

        """
        *******
        fitness : Objective funtion or Loss funciton for calculating a fitness
        *******
        
        ****** 
        inputs : 
        ******
        
            x : inputs features
            
            y : ground truth
        
        ******* 
        outputs :
        *******
        
            (loss,) : Tuple containing inverse fitness values(loss) of a model for minimizng  and null values
            
            accuracy : accuracy of a model
            
        """
        def fitness(x, y):
                 
            self.model.train()
            pred = self.model(x)
            loss = self.objective(softmax(pred),y).item()
            
            proba = softmax(pred).cpu().detach().numpy()
            
            pred_labels = [np.argmax(i) for i in proba]
            
            pred_labels = np.array(pred_labels)
            
            # calculate accuracy 
            
            correct = 0
            accuracy = 0
            
            gt_labels = y.cpu().detach().numpy()
            
            for p, g in zip(pred_labels, gt_labels):
                
                if  p==g : 
                    
                    correct+=1
                    
            accuracy = 100 * (correct/len(gt_labels))
            
                
            return (1/loss,) , accuracy
        
        
        """
        Deap Libraries section 
        """
        
        # Deap Creator
        self.creator = creator
        # The optmization is set to maximization at the begining (1.0,) using deap libray
        # since Roulette Wheel selection in deap doesnt not support minimization
        # To solve this we maximizing the inverse values of the fitness to make it minimization
        self.creator.create("FitnessMin", base.Fitness, weights=(1.0,)) 
        if self.encoding == "binary":
       
            self.creator.create("Individual", list, fitness=creator.FitnessMin, acc=list)
            
        else: 
            
            self.creator.create("Individual",array.array, typecode='d', fitness=creator.FitnessMin)
        
             
        # Deap Toolbox
        self.toolbox = base.Toolbox()
        
        if self.encoding == "binary":
        # Binary encoding(Gray) GA
            self.toolbox.register("attr_bool", random.randint, 0, 1)
            self.toolbox.register("individual", tools.initRepeat, self.creator.Individual, self.toolbox.attr_bool, self.numOfBits*self.dimension)
            self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
            self.toolbox.register("mate", tools.cxTwoPoint)
            self.toolbox.register("mutate",tools.mutFlipBit, indpb= self.flipProb)
            
        else:
        # Real coded GA
            self.toolbox.register("attr_float", uniform,self.lower_bound, self.upper_bound, self.dimension)
            self.toolbox.register("individual", tools.initIterate, self.creator.Individual, self.toolbox.attr_float)
            self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
            self.toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=self.lower_bound, up=self.upper_bound, eta=20.0)
            self.toolbox.register("mutate",tools.mutPolynomialBounded, low=self.lower_bound, up=self.upper_bound, eta=20.0,  indpb= 1/self.dimension)
            
        # Set Roulette Wheel selection  for enviromental selection
        self.toolbox.register("select",tools.selRoulette, fit_attr='fitness')
        self.toolbox.register("evaluate_nn",fitness)
        
        # Create Population 
        self.population = self.toolbox.population(n=self.population_size)
        
        
         
        """
        *******
        initPop : Initialize  a  population of genetic algorithms using for optimzing a weight for neural network and calculate its fitness values
        *******
        
        ******
        inputs :
        ******
        
            data : Pytorch's dataloader containg dataset for evaluating fitness
        
        *******
        outputs :
        *******
        
            None
        """   
        def initPop(data):
            
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
                    
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    fitness, acc = self.toolbox.evaluate_nn(images, labels)
                    
                    fitnesses.append(fitness)
                    accuracy.append(acc)
            
            # assign a fitness to each individual 
            for individual , fitness , acc in zip(self.population,fitnesses, accuracy):
                
                individual.fitness.values = fitness
                individual.acc = acc
        
        # Calculate fitness of all individual 
        initPop(data) 
   
    """
    *****************
    spearatevariables :  Separate a chromosome of individuals and convert chromosome into real values
    *****************
    
    ****** 
    inputs :
    ******
    
        individual :  Individual containig list of chromosome
    
    *******
    outputs :
    *******
    
        variable : list of contain a weight or values of decision variables
    
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
    **********
    chrom2real: Convert chromosome into real numbers
    **********
   
    ******
    inputs:
    ******
    
        c : chromosome  representing a weight 
    
    ******* 
    outputs:
    *******
    
        numinrange : real values of chromosome 
    """ 
    def chrom2real(self, c):
    
        
            indasstring = ''.join(map(str,c))
            degray = gray_to_bin(indasstring)
            numasint = int(degray, 2)
            numinrange= self.lower_bound + ((self.upper_bound-self.lower_bound)*(numasint/(self.maxnum-1)))
            
            return numinrange
        
    """
    **********
    weight_assing :  Assgin a weight to neuralnetwork
    **********
    
    ****** 
    inputs :
    ******
        individual : individual  cotainig a weight in (binary(or gray) or real coding) 
    *******
    outputs :
    *******
    
        None 
    
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
    ******
    search : perform a Genetic Algorithms optimization 
    ******
    
    ******
    inputs :
    ******
    
        x : inputs features
        
        y :  ground truth of a images
    
    *******
    outputs : 
    *******
    
        loss :  loss of a best individual
        
        best_acc :  accuracy of a best individual
    
    """
    def search(self, x, y):
 
        # selection
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
            
            fitness, acc= self.toolbox.evaluate_nn(x, y)
                
            fitnesses.append(fitness)
            accuracy.append(acc)
         
        # Assign a new fitness of a new offspring 
        for ind, fit, acc in zip(invalid_ind, fitnesses, accuracy):
            
            ind.fitness.values = fit
            ind.acc =acc
        
        # selection

        if self.selection == 'steady':

            # Steady state
            self.population[:] =  tools.selBest(self.population + offspring, self.population_size) 
        else :

            # Generational 
            self.population[:] = offspring
         
        
        # select the best individual
        best_individual= tools.selBest(self.population,1)[0]
        # get accuracy of best individual 
        best_acc = best_individual.acc
        
        # get the weight from the best individual  
        if self.encoding == "binary":
            best_weight = self.separatevariables(best_individual)
        else :
            best_weight = best_individual 
        
        # Asign the weight back to the model 
        self.weight_assign(best_weight)
        
        # Get the loss 
        loss = best_individual.fitness.values[0]
         
        return 1/loss , best_acc
    
    

