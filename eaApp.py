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
        res_rank = np.argsort(allEA_best)
        for ea_id,ea in enumerate(allEAs):
            current_ea_rank = np.where(res_rank==ea_id)[0][0]+1
            res_best += '||' +' '*(50//(num_ea+1))*current_ea_rank 
            res_best += '🏇' + '=' *(50//(num_ea+1))*(num_ea-current_ea_rank+1) 
            res_best += ' %s: %f\n'%(ea.optimizer,ea.best)
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

