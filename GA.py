import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


dna_size = 30
pop_size = 2000
c_rate = 0.8 #crossover_rate
m_rate = 0.001 #mutation_rate
n_generate = 100
x1_bound = [-1, 1]
x2_bound = [-1, 1]
x = np.linspace(0, n_generate, 100)



def F(x, y):
    return x**2+y**2-0.3*np.cos(3*np.pi*x)-0.4*np.cos(4*np.pi*y)+0.7


def plot_3d(ax):

   X = np.linspace(*x1_bound, 100)
   Y = np.linspace(*x2_bound, 100)
   X, Y = np.meshgrid(X, Y)
   Z = F(X, Y)

   ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm)
   #ax.set_zlim(-10, 10)
   ax.set_xlabel('x1')
   ax.set_ylabel('x2')
   ax.set_zlabel('z')
   plt.pause(3)
   plt.show()




def translateDNA(pop):

    x_pop = pop[:, 1::2]#奇数列表示x
    y_pop = pop[:, ::2]#偶数列表示y
    x = x_pop.dot(2**np.arange(dna_size)[::-1])/float(2**dna_size-1)*(x1_bound[1]-x1_bound[0])+x1_bound[0]
    y = y_pop.dot(2**np.arange(dna_size)[::-1])/float(2**dna_size-1)*(x2_bound[1]-x2_bound[0])+x2_bound[0]
    return x, y


def get_fitness(pop):

    x1, x2 = translateDNA(pop)
    pred = F(x1, x2)
    return -(pred-np.max(pred))+1e-3


def crossover(pop, crossover_rate):

    new_pop = []
    for father in pop:
        child = father
        if np.random.rand() < crossover_rate:
            mother = pop[np.random.randint(pop_size)]
            cross_points = np.random.randint(low=0, high=dna_size*2)
            child[cross_points:] = mother[cross_points:]
        mutation(child, m_rate)
        new_pop.append(child)

    return new_pop


def mutation(child, mutation_rate):

    if np.random.rand() < mutation_rate:
        mutate_point = np.random.randint(0, dna_size*2)
        child[mutate_point] = child[mutate_point] ^ 1


def select(pop, fitness):
    idx = np.random.choice(np.arange(pop_size), size=pop_size, replace=True, p=fitness/(fitness.sum()))
    return pop[idx]


def print_info(pop):
    fitness = get_fitness(pop)
    max_fitness_index = np.argmax(fitness)
    print("max_fitness:", fitness[max_fitness_index])
    x1, x2 = translateDNA(pop)
    print("最优基因型:", pop[max_fitness_index])
    print("(x1,x2):", (x1[max_fitness_index], x2[max_fitness_index]))
    print("f(x1,x2)=", F(x1[max_fitness_index], x2[max_fitness_index]))


if __name__ == "__main__":

    fig = plt.figure()
    ax = Axes3D(fig)
    fig.add_axes(ax)
    plt.ion()
    plot_3d(ax)
    mean_pop_fitness = []
    best_fitness = []
    pop = np.random.randint(2, size=(pop_size, dna_size*2))
    for _ in range(n_generate):
        x1, x2 = translateDNA(pop)
        if 'sca' in globals():
            sca.remove()

        sca = ax.scatter(x1, x2, F(x1, x2), c='red', marker='*')
        #plt.gca().view_init(10, 5)
        plt.show()
        plt.pause(0.1)
        pop = np.array(crossover(pop, c_rate))
        fitness = get_fitness(pop)
        pop = select(pop, fitness)
        mean_pop_fitness.append(np.mean(F(x1, x2)))
        best_fitness_index = np.argmax(fitness)
        best_fitness.append(F(x1[best_fitness_index], x2[best_fitness_index]))

    print_info(pop)
    plt.ioff()
    plot_3d(ax)
    #plt.legend()
    #plt.figure()
    plt.plot(x, mean_pop_fitness, label='mean fitness')
    plt.title('mean fitness')
    plt.legend()
    plt.figure()
    plt.plot(x, best_fitness, label='best person fitness')
    plt.title('best person fitness')
    plt.legend()
    plt.show()






