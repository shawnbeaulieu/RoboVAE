import glob
import numpy as np
from VariationalBayes import VAE

blueprint = [1000, 750, 375, 4]
convolutions = [(4, 4, 4, 1, 8), (4, 4, 4, 8, 16)]
#convolutions = 0

HYPERPARAMETERS = {

        "batch_size": 32,
        "regularizer": 1E-6,
        "learning_rate": 3E-4,
        "dropout": True,
        "dropout_rate": 0.50,
        "num_classes": 0

}

meta_graph = glob.glob("*.meta")

if len(meta_graph) == 0:
    meta_graph = None
    new_graph = True
else:
    meta_graph = meta_graph[0]
    new_graph = False

model = VAE(blueprint, HYPERPARAMETERS, convolutions, meta_graph=meta_graph, new_graph=new_graph)
synthetic_data = np.random.randn(10,10,10)

cost = model.Fit(synthetic_data.reshape(1,-1))
print(cost)


"""

For 10 classes

>> num_samples = 1
>> labels = np.array([0,1,0,0,0,0,0,0,0,0]).reshape(1,-1)

>> sample = model.Generate(num_samples, labels=labels).reshape(28,28)

Reshape into dimensions of original input

"""

