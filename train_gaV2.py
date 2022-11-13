import torch
from model import Classifier

if __name__ == "__main__":
    

    savename ="CIFAR-10_SGD"

    #  Setup tensorboard
    writer = SummaryWriter("../CI_logs/{}".format(savename))

    device = "cuda" if torch.cuda.is_available else "cpu"

    batch_size = 32

    nepochs = 20

    print("Using  **{}** as a device ".format(device))
    print("Batch Size : {}".format(batch_size))
    print("Iteration : {} epochs".format(nepochs))
    

    print("Loading dataset ....")
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
    
    # Prepare Dataset
    training_set = torchvision.datasets.CIFAR10(root='./../data', train=True, download=True, transform=transform)

    testing_set  = torchvision.datasets.CIFAR10(root='./../data', train=False, download=True, transform=transform)


    training_data, validation_data = random_split(training_set, [40000, 10000])
    
    

    train_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True, num_workers=2)
    validation_loader = DataLoader(validation_data, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(testing_set, batch_size=batch_size, shuffle=True, num_workers=2)

    print("Training Dataset: {}".format(len(training_data)))
    print("Validation Dataset: {}".format(len(validation_data)))
    print("Testing Dataset: {}".format(len(testing_set)))
    
    # labels of dataset
    classes = ['plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    print("Classes in dataset  : {} ".format(classes))

    
    # Classifier Models
    model = Classifier().to(device)

    # Loss function Objective function 
    CrossEntropy = nn.CrossEntropyLoss()

    # Optimizer 
    optimizer = optim.SGD([params  for params in model.parameters() if params.requires_grad], lr=0.001)
    
    
    # close tensorboard writer
    writer.close()




