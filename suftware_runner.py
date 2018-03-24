import suftware
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
print(os.getcwd())


# print API main doc_string
#print(stuft.__doc__)


# print class doc_string
#print(stuft.Deft1D.__doc__)


# Use cases and tests

# load data different ways
#data = np.loadtxt('./data/old_faithful_eruption_times.dat').astype(np.float)
#data = np.loadtxt('./data/old_faithful_eruption_times.dat')
data = np.genfromtxt('./data/old_faithful_eruption_times.dat')

posterior_samples = 200

# initialize deft object
deft = suftware.Deft1D(data,num_posterior_samples=posterior_samples,bounding_box=(-10,10),posterior_sampling_method='Lap')

Qstar = deft.get_Q_star()
xs = deft.get_grid()
plt.plot(xs,Qstar.evaluate(xs))

Q_samples = deft.get_Q_samples()


# make a plot using the grid
xs = deft.get_grid()
for s in range(posterior_samples):
    plt.plot(xs,Q_samples[s].evaluate(xs))
#plt.plot(xs,Q_samples.evaluate(xs),'o')
plt.show()


# run fit
#deft.fit(data)


# bad key: # the following will not work:
#deft.get_params('G')

# get parameter based on valid key
deft.get_params('num_grid_points')
deft.get_params('alpha')

# get all parameters
deft.get_params()

# return entire class dictionary
#print(deft.__dict__)

# test methods
#print('G: ',deft.get_num_grid_points())
#print('h: ',deft.get_h())
deft.get_bounding_box()
grid = deft.get_grid()



#Qstar = deft.get_Q_star()
#Qstar.evaluate(0.2)
deft.get_results()

#print(deft.get_results()['phi_weights'])


print(type(deft.get_results()['Q_star']))


#print(deft.get_results()['phi_samples'])


# make a plot using the grid
#xs = deft.get_grid()
#plt.plot(xs,Qstar.evaluate(xs),'o')
#plt.show()

# get results
deft.get_results()

# get deft result by key
deft.get_results('phi_star')

# access particular results pythonically
deft.get_results()['phi_star']

# setting parameter by invalid key value
#deft.set_params('G',10)

# setting parameter by invalid key value
deft.set_params('num_grid_points',11)

#set parameters via dictionary

#print(deft.get_params())

#print(deft.get_params())

# set incorrect params
#d = {"alpha":2,"G":2}
#deft.set_params(**d)

print(deft.get_params())


# parameters correctly
d = {"num_grid_points":10,"alpha":2}
deft.set_params(**d)

#print(deft.get_params())

