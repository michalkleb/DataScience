import numpy as np

class bird:
   def __init__(self, dimension):
        """
        Arguments:
        dimension - dimension of the position array

        """
        self.dimension=dimension
        self.velocity=np.repeat(0,dimension)
        self.xy = np.random.uniform(-2,2,size=(2,))
      

def update_position_bare_bones(bird, best_bird,previous_best):
    """" Updates position function - bare bones variant """
    sigma = np.array(abs(bird.xy-best_bird.xy))
    test = np.random.normal((bird.xy+best_bird.xy)/2,sigma)
    
    return test

def update_position_canonical(bird, best_bird,previous_best):
    """" Updates position function - canonical variant """

    v=np.array([velocity(previous_best.xy[n],best_bird.xy[n],bird.xy[n],bird.velocity[n]) for n in range(bird.dimension)])
    new_pos=np.array(bird.xy+v)
  
    return  new_pos 


def velocity(previous_best,best_bird,bird,v):
    """" Returns the veolocity of the particle"""

    ind_tend = np.random.uniform(0,2.05)*(previous_best - bird)
    soc_tend = np.random.uniform(0,2.05)*(best_bird - bird)

    chi=0.7298
    velocity=chi*(v+ind_tend+soc_tend)

    return velocity 

def PSO(num_of_birds, dimension, function_to_optimize, update_function):
    """ 
    Particle Swarm Optimization
    Arguments:
    - num_of_birds - number of the particles
    - function_to_optimize - function which will be optimized
    - update_fucntion - name of method to optimize
    """

    #Choosing the variant of the update algorithm
    update_function={
            "bare_bones": update_position_bare_bones,
            "canonical": update_position_canonical
    }[update_function]
    
    #Initalization of arrays: particles and best positions
    birds = np.array([bird(dimension) for i in range(num_of_birds)])
    result = np.array([ function_to_optimize(i.xy) for i in birds])

    #Verification which particle has the best position
    best_bird = np.argmin(result)
    previous_best=best_bird

    #Add to array the best solution
    best_evo=np.array([birds[best_bird].xy])
    
    for idx in range(600):
        for i in range(num_of_birds):

            #Calculating the new position of each particle
            new_position=update_function(birds[i],birds[best_bird],birds[previous_best])
            new_value=function_to_optimize(new_position)
            
            #If new position is better than previous one, updating the position
            if new_value < result[i]:
                birds[i].xy = new_position
                result[i] = new_value
                
            
        #Verification which particle now is the best one
        previous_best=best_bird
        best_bird = np.argmin(result)

        #Append to the array new best solution
        best_evo=np.append(best_evo,[birds[best_bird].xy],axis=0)
        
        #If all the particles are nearly in the same position: stop
        if np.std(result) < 0.001:
            break

    
    return (idx ,best_evo)




