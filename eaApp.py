import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import optimizer
import benchmarks
import pandas as pd

# Set page title and favicon.
st.set_page_config(
    page_title="Evolutionary Algorithm Racing", page_icon='🐎',
)

"""
# Evolutionary Algorithm Racing 🐎
[![GitHubLicense](https://img.shields.io/github/license/VicZH/EaRacing)](https://github.com/VicZH/EaRacing)
[![GitHubIssues](https://img.shields.io/github/issues/VicZH/EaRacing)](https://github.com/VicZH/EaRacing/issues)
"""

st.markdown("<br>", unsafe_allow_html=True)

"""
Let's find out which evolutionary algorithm will win the race:

1. Choice one or multiple evolutionary algorithms below
2. Pick a problem to test
3. Using the slider to give the dimension of the test porblem, number of candidate solutions for a evolutionary algorithm and maximum number of function calls
4. Click the run button to see results
---
"""

# with layout_col1:
ea_options = st.multiselect(
    'Which evolutionary algorithms you want to compare',
    ['DE','SSA', 'PSO', 'GA', 'BAT','FFA','GWO','WOA','MVO','MFO','CS','HHO','SCA','JAYA'],
    ['DE'])

test_fun = st.selectbox('Which function do you want to test', ['F%d'%s for s in range(1,13)])
optfname = benchmarks.TestFunctionDetails[test_fun]
st.latex(optfname[0])
st.latex(r'z_i=x_i-rand(%g,%g)'%(optfname[1],optfname[2]))
st.latex(r'x_i\in[%g,%g]'%(optfname[1],optfname[2]))

num_dim = int(st.slider('How many unknowns (dimensions)', 1, 200, 30))

num_pop = int(st.slider('How many candidate solutions', 5, 100, 30))

num_evl = int(st.slider('How many function calls', 10000, 100000, 10000, step=10000))

if_start = 0
if_stop = 0
m = st.markdown("""
<style>
div.stButton > button:first-child {
    display: block;
    width:30%;
    margin-left: auto;
    margin-right: auto;
}
</style>""", unsafe_allow_html=True)
if len(ea_options):
    if_start = st.button('run')

ea_options.sort()
allEAs = []
allEA_best = []
if if_start:
    sol_shift = np.random.random(num_dim)*(optfname[2]-optfname[1])+optfname[1]
    max_iter = int(num_evl/num_dim)
    progress_bar = st.progress(0)
    res_plt = st.container()
    st.text('Expected objective value is: 0.')
    res_text = st.empty()
    num_ea = len(ea_options)
    for opt_id in ea_options:
        allEAs += [optimizer.initalisation(opt_id, test_fun, sol_shift, optfname[1],optfname[2], num_dim, num_pop, num_evl)]
        allEA_best += [allEAs[-1].best]
    for run_id in range(max_iter):
        # if if_stop: st.text("stop")
        for ea_id,opt_id in enumerate(ea_options):
            ea = allEAs[ea_id]
            ea.update(run_id+1)
            allEA_best[ea_id] = ea.best
        # for plot
        if run_id == 0:
            __res_val = np.zeros((1,len(allEAs)))
            __res_columns = []
            for ea_id,ea in enumerate(allEAs):
                __res_val[0,ea_id] = np.log(ea.best)/np.log(10)
                __res_columns += [ea.optimizer]
            __res = pd.DataFrame(__res_val,columns=__res_columns)
            res_plot = res_plt.line_chart(__res)
        else:
            __res_val = np.zeros((1,len(allEAs)))
            __res_columns = []
            for ea_id,ea in enumerate(allEAs):
                __res_val[0,ea_id] = np.log(ea.best)/np.log(10)
                __res_columns += [ea.optimizer]
            __res = pd.DataFrame(__res_val,columns=__res_columns)
            res_plot.add_rows(__res)
        res_best = 'Current best objective values are:\n'
        res_best += '-'*70+'\n'
        res_rank = np.argsort(allEA_best)
        for ea_id,ea in enumerate(allEAs):
            current_ea_rank = np.where(res_rank==ea_id)[0][0]+1
            res_best += '||%3.3f'%(ea.best) +' '*(50//(num_ea+1))*current_ea_rank 
            res_best += '%s🏇'%ea.optimizer + ' ' *(50//(num_ea+1))*(num_ea-current_ea_rank+1) 
            res_best += '\n'
            res_best += '-'*70+'\n'
        res_text.text(res_best)
        progress_bar.progress((run_id/max_iter))
    # show solutions
    sols = sol_shift
    allSolTitle = ['Solutions']
    for ea in allEAs:
        sols = np.vstack((sols,np.array(ea.bestIndividual)))
        allSolTitle += [ea.optimizer]
    df = pd.DataFrame(sols.T,columns=allSolTitle)
    st.dataframe(df)        


st.markdown("<br>", unsafe_allow_html=True)

st.header('So, What is an Evolutionary Algorithm?')

st.write("""
In short, evolutionary algorithm is just a way 
to solve an optimisation problem. Evolutionary 
algorithms tackle the optimisation problem via 
a population-based metaheuristic approach which 
does not need to compute a gradient to 
update the candidate solutions.  
The procedures for an example of evolutionary 
algorithm are summaried as follow:

1. first, randomly generate a set of candidate solutions
2. then, calculate the objective value given by each candidate solution
3. sometime, we need to rank or sort all candidate solutions and find the best one, for minimisation problem we will need to find the one gives the smallest objective value and for the maximisation problem we will need to find the one gives the largest objective value
4. next, update the candidate solutions by comparing current candidate solutions
5. last, comparing the objective value given by each new candidate solution and select some good candidate solutions pass to the next iteration loop
6. above step will contitune until we reach to the maximum number of function call
""")

st.header('Good, so what is the advantages and disadvantages of Evolutionary Algorithms?')
st.write("""
Comparing to gradient-based optimisation method such gradient descent method, 
people claimed that evolutionary algorithm has the 
benefit of

1. do not need to compute the gradient of the objective (test or loss) function
2. so evolutionary algorithm can easily handle objective functions
that are flat, dynamic, noisy and discontinue.
3. it can solve optimisation problem subjected to many constrains
4. additionally, evolutionary algorithm has less 
chance to be trapped in the local miminums
5. last but not least, evolutionary algorithm can find multiple global
optimal solutions at the same time

However, the drawbacks of evolutionary algorithm are

1. evolutionary algorithm finds the local miminum much slower than the 
gradient-based optimisation method
2. evolutionary algorithm often hard to handle high 
dimensional optimisation problem
3. evolutionary algorithm does not have a clear 
stopping criterion 

""")

st.header('Cool, can I repurpose or test my own evolutionary algorithm?')
st.write("""
Definitely, please feel free to use the open sourced codes on [GitHub](https://github.com/VicZH/EaRacing).
If you have any suggestions or questions, please [let me know](https://github.com/VicZH/EaRacing/issues).
""")


st.header('What algorithm abbreviation stands for?')
st.write("""

1. **BAT: Bat Algorithm** 
    
    Yang, Xin-She. "A new metaheuristic bat-inspired algorithm." Nature inspired cooperative strategies for optimization (NICSO 2010). Springer Berlin Heidelberg, 2010. 65-74. DOI: http://dx.doi.org/10.1007/978-3-642-12538-6_6
2. **CS: Cuckoo Search**

    Yang, Xin-She, and Suash Deb. "Cuckoo search via Lévy flights." Nature & Biologically Inspired Computing, 2009. NaBIC 2009. World Congress on. IEEE, 2009. DOI: http://dx.doi.org/10.1109/NABIC.2009.5393690
3. **DE: Differential Evolution**

    Storn, Rainer. "On the usage of differential evolution for function optimization." Proceedings of North American Fuzzy Information Processing. IEEE, 1996.
4. **FFA: Firfly Algorithm**

    Yang, Xin-She. "Firefly algorithm, stochastic test functions and design optimisation." International Journal of Bio-Inspired Computation 2.2 (2010): 78-84. http://dx.doi.org/10.1504/IJBIC.2010.032124
5. **GA: Genetic algorithm**

    Holland, J. "Genetic algorithms". Scientific American. 1992
6. **GWO: Grey Wolf Optimizer**

    S. Mirjalili, S. M. Mirjalili, A. Lewis, Grey Wolf Optimizer, accepted in Advances in Engineering Software , vol. 69, pp. 46-61, 2014, DOI:http://dx.doi.org/10.1016/j.advengsoft.2013.12.007
7. **HHO: Harris hawks optimization**

    Ali Asghar Heidari, Seyedali Mirjalili, Hossam Faris, Ibrahim Aljarah, Majdi Mafarja, Huiling Chen, "Harris hawks optimization: Algorithm and applications. ", Future Generation Computer Systems, DOI: https://doi.org/10.1016/j.future.2019.02.028
8. **JAYA: **

    Rao, R. "Jaya: A simple and new optimization algorithm for solving constrained and unconstrained optimization problems." International Journal of Industrial Engineering Computations 7.1 (2016): 19-34.
9. **MFO: Moth-flame optimization**

    S. Mirjalili, Moth-flame optimization algorithm: A novel nature-inspired heuristic paradigm, Knowledge-based Systems, Volume 89, Pages 229-249, DOI: http://dx.doi.org/10.1016/j.knosys.2015.07.006
10. **MVO: Multi-Verse Optimizer**

    S. M. Mirjalili, A. Hatamlou, " Multi-Verse Optimizer: a nature-inspired algorithm for global optimization " ,Neural Computing and Applications, in press, 2015, DOI: http://dx.doi.org/10.1007/s00521-015-1870-7
11. **PSO: Particle Swarm Optimization**

    Kennedy, J. and Eberhart, R. "Particle swarm optimization. In Neural Networks, 1995. Proceedings.", IEEE International Conference on, volume 4, pages 1942–1948 vol.4. 1995
12. **SCA: **

    Mirjalili, Seyedali. "SCA: a sine cosine algorithm for solving optimization problems." Knowledge-based systems 96 (2016): 120-133.
13. **SSA: **

    Mirjalili, Seyedali, et al. "Salp Swarm Algorithm: A bio-inspired optimizer for engineering design problems." Advances in Engineering Software 114 (2017): 163-191.
14. **WOA: Whale Optimization Algorithm**

    S. Mirjalili, A. Lewis, The Whale Optimization Algorithm, Advances in Engineering Software , in press, 2016, DOI: http://dx.doi.org/10.1016/j.advengsoft.2016.01.008
""")


st.markdown("<br>", unsafe_allow_html=True)

st.info("""
        Credit: 

        This small app was created by [YP Zhang] (https://www.linkedin.com/in/yunpeng-zhang-34434865/) 
        and the source code can be found [here](https://github.com/VicZH/EaRacing).
        This app was inspired by many great works like [evolutionary play ground](https://evo-algorithm-playground.herokuapp.com/) 
        created by Alexandre Mundim, [Code Generator](https://traingenerator.jrieke.com/) created by Johannes Rieke
        and [tensorflow playground](https://playground.tensorflow.org/).
        
        The evolutionary algorithm tested here are based on the python scripts from 
        [EvoloPy](https://github.com/7ossam81/EvoloPy) project. Most of the 
        original python files are modified with some bugs fixed.

        Please feel free to reach out and let me know if you found any issue.
    """)

