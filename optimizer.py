import optimizers.PSO as pso
import optimizers.MVO as mvo
import optimizers.GWO as gwo
import optimizers.MFO as mfo
import optimizers.CS as cs
import optimizers.BAT as bat
import optimizers.WOA as woa
import optimizers.FFA as ffa
import optimizers.SSA as ssa
import optimizers.GA as ga
import optimizers.HHO as hho
import optimizers.SCA as sca
import optimizers.JAYA as jaya
import optimizers.DE as de
import benchmarks

def initalisation(algo, function_name, sol_shift, lb, ub, dim, popSize, numEvl):

    x = 0
    if algo == "DE":
        x = de.DE(getattr(benchmarks, function_name), sol_shift, lb, ub, dim, popSize, numEvl)
    elif algo == "SSA":
        x = ssa.SSA(getattr(benchmarks, function_name), sol_shift, lb, ub, dim, popSize, numEvl)
    elif algo == "PSO":
        x = pso.PSO(getattr(benchmarks, function_name), sol_shift, lb, ub, dim, popSize, numEvl)
    elif algo == "GA":
        x = ga.GA(getattr(benchmarks, function_name), sol_shift, lb, ub, dim, popSize, numEvl)
    elif algo == "BAT":
        x = bat.BAT(getattr(benchmarks, function_name), sol_shift, lb, ub, dim, popSize, numEvl)
    elif algo == "FFA":
        x = ffa.FFA(getattr(benchmarks, function_name), sol_shift, lb, ub, dim, popSize, numEvl)
    elif algo == "GWO":
        x = gwo.GWO(getattr(benchmarks, function_name), sol_shift, lb, ub, dim, popSize, numEvl)
    elif algo == "WOA":
        x = woa.WOA(getattr(benchmarks, function_name), sol_shift, lb, ub, dim, popSize, numEvl)
    elif algo == "MVO":
        x = mvo.MVO(getattr(benchmarks, function_name), sol_shift, lb, ub, dim, popSize, numEvl)
    elif algo == "MFO":
        x = mfo.MFO(getattr(benchmarks, function_name), sol_shift, lb, ub, dim, popSize, numEvl)
    elif algo == "CS":
        x = cs.CS(getattr(benchmarks, function_name), sol_shift, lb, ub, dim, popSize, numEvl)
    elif algo == "HHO":
        x = hho.HHO(getattr(benchmarks, function_name), sol_shift, lb, ub, dim, popSize, numEvl)
    elif algo == "SCA":
        x = sca.SCA(getattr(benchmarks, function_name), sol_shift, lb, ub, dim, popSize, numEvl)
    elif algo == "JAYA":
        x = jaya.JAYA(getattr(benchmarks, function_name), sol_shift, lb, ub, dim, popSize, numEvl)
    return x


