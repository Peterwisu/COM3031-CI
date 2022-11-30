import random
import numpy as np
import array
from sympy.combinatorics.graycode import GrayCode
from sympy.combinatorics.graycode import gray_to_bin
from sympy.combinatorics.graycode import bin_to_gray
from deap import creator 
from deap import base 
from deap import tools
from deap import algorithms
from deap import benchmarks
import torch
from torch import optim
from tqdm import tqdm

# softmax activation function
softmax = torch.nn.Softmax(dim=1)

"""
****************************
Lamarckian Memetic algorithm
****************************
"""

class memeticAlgorithms():
    
    
     
    def __init__ (self,
                  objective, 
                  population_size,
                  model,
                  data,
                  device,
                  numOfBits = 10,
                  nElitists=1,
                  crossProb=0.6, 
                  flipProb =1, 
                  mutateProb=0.1,
                  lower_bound = -1,
                  upper_bound = 1,
                  encoding = "binary",
                  ls_iter = 5,
                  ls_lr=0.001,
                  ):

        # neural network model
        self.model = model
        # device
        self.device =device
        
        # size of pupulation  
        self.population_size = population_size 
        # dimension of decision variable
        self.dimension = sum([params.numel() for params in self.model.parameters()])
        
        # Number of bits ( only required in binary or gray coding)
        self.numOfBits = numOfBits 
        # Number of Elistis 
        self.nElistists = nElitists 
        
        # Cross over probabilties
        self.crossProb = crossProb 
        
        # objective function
        
        # Bit flip probabilites 
        self.flipProb = flipProb/(self.dimension * numOfBits) 
        # Mutation probabilities
        self.mutateProb = mutateProb 
        
        
        
        
        # Max number
        self.maxnum = 2**self.numOfBits
        # Type of encoding
        self.encoding = encoding
        
        # Objective functions 
        self.objective = objective
        # Lower bound
        self.lower_bound = lower_bound
        # Upper bound
        self.upper_bound =upper_bound
        
        # local search iteration 
        self.ls_iter = ls_iter
        # local search learning rate
        self.ls_lr = ls_lr
        # local search optimizer
        self.optimizer = optim.Rprop( self.model.parameters() , lr = self.ls_lr)
        
        
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
        self.creator.create("Individual", list, fitness=creator.FitnessMin, acc=list)
        
        # Deap Toolbox
        self.toolbox = base.Toolbox()
        self.toolbox.register("attr_bool", random.randint, 0, 1)
        self.toolbox.register("individual", tools.initRepeat, self.creator.Individual, self.toolbox.attr_bool, self.numOfBits*self.dimension)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate",tools.mutFlipBit, indpb= self.flipProb)
        self.toolbox.register("select",tools.selRoulette, fit_attr='fitness')
        self.toolbox.register("evaluate_nn",objective_nn)
        
        
        
        # Create population
        self.population = self.toolbox.population(n=self.population_size)
        
        def initPop(data): 
            fitnesses = [] 
            accuracy  = []
            
            print("Calculating fitness of individual")
            
            # iterate to each individual in a population
            for individual in tqdm(self.population):
                weight = self.separatevariables(individual)
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
        initPop(data)
        
    """
    **********
    spearatevariables :  Separate a chromosome of individuals and convert chromosome into real values
    **********
    
    ****** 
    inputs :
    ******
    
        individual :  Individual containig list of chromosome
    
    *******
    outputs :
    *******
    
        variable : list of contain a weight or values of decision variables
    
    """ 
    def separatevariables(self, individual):
            
        variable = []
        num_bits = self.numOfBits
            
        bit_counter = 0 
        for i in range(self.dimension):
                 
            variable.append(self.chrom2real(individual[bit_counter:num_bits*(1+i)]))
            bit_counter+=num_bits
             
        return variable
            
        
    
            
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
    *******************
    weightsOutOfNetwork
    *******************
    
    ****** 
    inputs
    ******
    
        None
    
    ******* 
    returns
    *******

        weights : weight inside a network
    """
    def weightsOutOfNetwork(self):
        weights = []
        #Collecting data from every layer and flattening it into one single array
        for layer, param in self.model.state_dict().items():
        
            flattened = (param.flatten().detach().clone().cpu().numpy()).tolist()
            weights += flattened
        
        return weights 
    
    """
    **********
    rprop : 
    **********
    
    ****** 
    inputs :
    ******
    
    *******
    outputs :
    *******
    """ 
    def rprop(self,trainingSet, trainingResults):
        
    
        for i in range(self.ls_iter):
            self.optimizer.zero_grad()   # zero the gradient buffers
            output = self.model(trainingSet)
            loss = self.objective(softmax(output), trainingResults)
            
            loss.backward()
            self.optimizer.step()    # Does the update
       
        
    """
    **********
    real2chrom : Convert a weight(real number) into chromosome(gray coding)
    **********
    
    ****** 
    inputs:  
    ****** 
    
        weights : weight in form of real number in range of lower and upper bound
    
    ******* 
    returns:
    *******
    
        grey : values of chromosome in gray code
    
    """ 
    def real2chrom(self, weights):
       
        numasint = int(((weights - self.lower_bound)/(self.upper_bound - self.lower_bound))*(self.maxnum-1))
        
        if numasint == self.maxnum:
            numasint -=1
        chromosome = str(bin(numasint))[2:]
       
        if (len(chromosome) < self.numOfBits):
            chromosome = ("0" * (self.numOfBits - len(chromosome))) + chromosome
    
        binary = ''.join(map(str, chromosome))

        grey = bin_to_gray(binary)

        return grey
    
    
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
        numinrange= self.lower_bound + ( (self.upper_bound-self.lower_bound)*(numasint/(self.maxnum-1)))
        
        return numinrange
    
    
    """
    ************* 
    weights2chrom : Retrive a list of chromosome in individual from a weights
    *************
    
    ******
    inputs: 
    ******
    
        weight: weight of a networks
    
    *******
    outputs:
    *******
    
        individual:  individual containing list of chromosome

    """ 
    def weights2chrom(self,weights): # Insert weights into a single chromosome
   
        individual = []
        for weight in weights:
            if weight > self.upper_bound :
                weight = self.upper_bound
            elif weight< self.lower_bound:
                weight = self.lower_bound
            chromosome = self.real2chrom(weight)
            chromList = list(map(int, list(chromosome)))
            for x in chromList:
                individual.append(x)
        
        return individual
    
   
    
    """
    **********
    lamarckian : 
    **********
    
    ****** 
    inputs :
    ******
    
    *******
    outputs
    """ 
    def lamarckian(self,individual, images, labels):
        # Inserting individual into network and calculating the loss
        sorted_ind = self.separatevariables(individual)
        self.weight_assign(sorted_ind)
        self.rprop(images, labels)
   
        # Creating a new individual based on the weights received from the network
        indi = self.weights2chrom(self.weightsOutOfNetwork())
        return indi
    
    
    """
    ******
    search : perform a optimization 
    ******
    
    ******
    inputs :
    ******
    
        images : inputs images data
        
        labels :  ground truth of a images
    
    *******
    outputs : 
    *******
    
        loss :  loss of a best individual
        
        best_acc :  accuracy of a best individual
    
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
        for child in offspring:
          # Remove offspring and replace them with a completely new one
          offspring.remove(child)
          lamarck_ind = self.lamarckian(child, images, labels)
          offspring.append(creator.Individual(lamarck_ind))

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
        self.population[:] = offspring
        
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
