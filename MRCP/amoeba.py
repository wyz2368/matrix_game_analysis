from math import sqrt
import numpy as np
from functools import partial
from utils import regret_of_variable
from utils import upper_bouned_regret_of_variable
from utils import project_onto_unit_simplex
from utils import Cache
from utils import find_all_deviation_payoffs

# Amoeba uses the simplex method of Nelder and Mead to maximize a
# function of 1 or more variables, constraints are put into place
# according to Patrick Jordan's Thesis "practical strategic reaso
# ning with applications in market games" section 7.2
#   Copyright (C) 2020  Gary Qiurui Ma, Yongzhao Wang, Strategic Reasing Group

# Ameoba Parameters
alpha = 1 # reflection
gamma = 2 # expansion
rho = 0.5 # contraction
sigma = 0.5 # shrinking param.

def check_within_probability_simplex(var):
    '''
    check variable is within nprobability simplex
    Here we only check the scope of each single variable since mathematically
    initialization guarantees the sum of each variable equal to 1 in amoeba.
    '''
    return np.all(var >= 0) and np.all(var <= 1)

def variable_projection(variables, sections):
    """
    Project decision variables onto simplex respectively.
    :param variables: ameoba decision variables
    :param sections: a list of the number of variables per player.
    :return:
    """
    pointer = 0
    for ele in np.cumsum(sections):
        variables[pointer:ele] = project_onto_unit_simplex(variables[pointer:ele])
        pointer = ele
    return variables

def infeasibility_handling(var, sections, base, step_size, minus, infeasibility = "proj",):
    """
    Handling situation where ameoba variables exceeds the probability simplex.
    :param var: ameoba decision variables
    :param sections:
    :param base:
    :param minus:
    :param step_size:
    :param infeasibility: method to handle infeasibility. Options: "proj": projection onto simplex, otherwise, shrink step size.
    :return:
    """
    if not check_within_probability_simplex(var):
        if infeasibility == "proj":
            var = variable_projection(var, sections)
        else:
            while not check_within_probability_simplex(var):
                step_size /= 2
                var = base + step_size * (base - minus)
    return var



def shrink_simplex(sorted_simplex, sorted_value, func):
    """
    Please beware that simplex and function value supplied must be sorted
    based on the function value, the smallest value at the front
    Input:
        sorted_simplex: the simplex of length nvar+1. The first the best
        sorted_value  : the fvalue of simplexes
        func          : function to evaluate the points
        sigma         : hyperparamter to shrink the simplex
    """
    assert len(sorted_simplex) == len(sorted_value)
    simplex, fvalue = [sorted_simplex[0]], [sorted_value[0]]
    for i in range(1,len(sorted_simplex)):
        simplex.append(simplex[0] + sigma * (sorted_simplex[i]-simplex[0]))
        fvalue.append(func(simplex[i]))
    return simplex, fvalue 

def amoeba_mrcp(empirical_game,
                full_game,
                approximation = False,
                var='uni',
                max_iter=5000,
                ftolerance=1.e-4,
                xtolerance=1.e-4):
    """
    Note each varibale in the amoeba variable is two times the length of the strategies
    Input:
        empirical_game : each player's strategy set
        full_game      : the full meta game to compute mrcp on
        approximation  : whether to approximate the regret of mixed strategy using deviation payoff of pure profile.
        var            : initial guessing for the solution. defaulted to uniform
        max_iter       : maximum iteration of amoeba to automatically end
        ftolerance     : smallest difference of best and worst vertex to converge
        xtolerance     : smallest difference in average point and worst point of simplex
    """
    def normalize(sections, variables):
        """
        A variable made of len(sections) parts, each of the parts is
        in a probability simplex
        Input:
            variables: the varible that amoeba is searching through
            sections : a list containing number of element for each section.
                       Typically it is the list of number of strategies
        Output:
            A normalized version of the varibales by sections
        """
        pointer = 0
        for ele in np.cumsum(sections):
            variables[pointer:ele] /= sum(variables[pointer:ele])
            pointer = ele
        return variables

    # construct function for query
    if approximation:
        # Calculate the upper-bounded regret of mixed strategy profile.
        caches = [Cache(), Cache]
        caches = find_all_deviation_payoffs(empirical_games=empirical_game,
                                            meta_game=full_game,
                                            caches=caches)
        func = partial(upper_bouned_regret_of_variable,
                       empirical_games=empirical_game,
                       meta_game=full_game,
                       caches=caches)
    else:
        # Calculate the exact regret of mixed strategy profile.
        func = partial(regret_of_variable,
                    empirical_games=empirical_game,
                    meta_game=full_game)

    # TODO: check if repeated action is allowed in emprical game.
    sections = [len(ele) for ele in empirical_game]    # num strategies for players
    normalize = partial(normalize, sections=sections)  # force into simplex
    if var == 'uni':
        var = np.ones(sum(sections))      # the initial point of search from uniform
    elif var == 'rand': # random initial points
        var = np.random.rand(sum(sections))
    else:
        assert len(var) == sum(sections), 'initial points incorrect shape'

    var = normalize(variables=var)       

    nvar = sum(sections)                  # total number of variables to minimize over
    nsimplex = nvar + 1                   # number of points in the simplex

    # Set up the simplex. The first point is the guess. All sides of simplex
    # have length |c|. Please tweak this value should constraints be violated
    # assume if vertexes on simplex is normalized, then reflection, expansion
    # shrink will be on the probability simplex
    c = 1
    val_b = c/nvar/sqrt(2)*(sqrt(nvar+1)-1)
    val_a = val_b + c/sqrt(2)

    simplex = [0] * nsimplex
    simplex[0] = var[:]
    
    for i in range(nvar):
        addition_vector = np.ones(sum(sections))*val_b
        addition_vector[i] = val_a
        simplex[i+1] = normalize(variables=simplex[0]+addition_vector)

    fvalue = []
    for i in range(nsimplex):  # set the function values for the simplex
        fvalue.append(func(simplex[i]))

    # Start of the Ameoba Method.
    iteration = 0
    while iteration < max_iter:

        # sort the simplex and the fvalue the last one is the worst
        sort_index = np.argsort(fvalue)
        fvalue = [fvalue[ele] for ele in sort_index]
        simplex = [simplex[ele] for ele in sort_index]

        # get the average of the the n points except from the worst
        x_a = np.average(np.array(simplex[:-1]), axis=0)
        assert check_within_probability_simplex(x_a), 'centroid not in probability simplex'

        # determine the termination criteria
        # 1. distance between average and worst
        simscale = np.sum(np.absolute(x_a-simplex[-1]))/nvar
        # 2. distance between best and worst function values
        fscale = (abs(fvalue[0])+abs(fvalue[-1]))/2.0
        if fscale != 0.0:
            frange = abs(fvalue[0]-fvalue[-1])/fscale
        else:
            frange = 0.0  # all the fvalues are zero in this case

        # Convergence Checking
        if (ftolerance <= 0.0 or frange < ftolerance) \
                and (xtolerance <= 0.0 or simscale < xtolerance):
            return np.split(simplex[0],sections[:-1]),fvalue[0],iteration

        # perform reflection to acquire x_r,evaluate f_r
        alpha = 1
        x_r = x_a + alpha*(x_a-simplex[-1])
        x_r = infeasibility_handling(var=x_r,
                                     sections=sections,
                                     base=x_a,
                                     step_size=alpha,
                                     minus=simplex[-1])
        f_r = func(x_r)

        # expansion if the reflection is better
        if f_r < fvalue[0]:    # expansion if the reflection is better
            gamma = 1
            x_e = x_r + gamma*(x_r-x_a)
            x_e = infeasibility_handling(var=x_e,
                                         sections=sections,
                                         base=x_r,
                                         step_size=gamma,
                                         minus=x_a)

            f_e = func(x_e)
            if f_e < f_r: # accept expansion and replace the worst point
                simplex[-1] = x_e
                fvalue[-1] = f_e
            else:               # refuse expansion and accept reflection
                simplex[-1] = x_r
                fvalue[-1] = f_r
        elif f_r < fvalue[-2]:  # accept reflection when better than lousy
            simplex[-1] = x_r
            fvalue[-1] = f_r
        else:
            if f_r > fvalue[-1]: # inside contract if reflection is worst than worst
                x_c = x_a - 0.5 * (x_a-simplex[-1]) # 0.5 being a hyperparameter
                f_c = func(x_c)
                if f_c < fvalue[-1]: # accept inside contraction
                    simplex[-1] = x_c
                    fvalue[-1] = f_c
                else:
                    simplex, fvalue = shrink_simplex(simplex, fvalue, func)
            else:                # outside contract if reflection better than worse
                x_c = x_a + alpha*0.5*(x_a-simplex[-1]) # 0.5 being a hyperparameter
                f_c = func(x_c)
                if f_c < f_r:    # accept contraction
                    simplex[-1] = x_c
                    fvalue[-1] = f_c
                else:
                    simplex, fvalue = shrink_simplex(simplex,fvalue,func)
        iteration += 1
    sort_index = np.argsort(fvalue)
    fvalue = [fvalue[ele] for ele in sort_index]
    simplex = [simplex[ele] for ele in sort_index]
    return np.split(simplex[0],sections[:-1]), fvalue[0], iteration
