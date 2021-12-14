
import numpy
import math

# define the function blocks
def prod(it):
    p = 1
    for n in it:
        p *= n
    return p


def Ufun(x, a, k, m):
    y = k * ((x - a) ** m) * (x > a) + k * ((-x - a) ** m) * (x < (-a))
    return y

def F1(x):
    x = x - numpy.random.random()
    s = numpy.sum((x) ** 2)
    return s


def F2(x):
    o = sum(abs((x))) + prod(abs((x)))
    return o


def F3(x):
    dim = len(x) + 1
    o = 0
    for i in range(1, dim):
        o = o + (numpy.sum(x[0:i])) ** 2
    return o


def F4(x):
    o = max(abs(x))
    return o


def F5(x):
    dim = len(x)
    o = numpy.sum(
        100 * (x[1:dim] - (x[0 : dim - 1] ** 2)) ** 2 + (x[0 : dim - 1] - 1) ** 2
    )
    return o


def F6(x):
    o = numpy.sum(abs((x + 0.5)) ** 2)
    return o


def F7(x):
    dim = len(x)

    w = [i for i in range(len(x))]
    for i in range(0, dim):
        w[i] = i + 1
    o = numpy.sum(w * (x ** 4)) + numpy.random.uniform(0, 1)
    return o


def F8(x):
    o = sum(-x * (numpy.sin(numpy.sqrt(abs(x)))))
    return o


def F9(x):
    dim = len(x)
    o = numpy.sum(x ** 2 - 10 * numpy.cos(2 * math.pi * x)) + 10 * dim
    return o


def F10(x):
    dim = len(x)
    o = (
        -20 * numpy.exp(-0.2 * numpy.sqrt(numpy.sum(x ** 2) / dim))
        - numpy.exp(numpy.sum(numpy.cos(2 * math.pi * x)) / dim)
        + 20
        + numpy.exp(1)
    )
    return o


def F11(x):
    dim = len(x)
    w = [i for i in range(len(x))]
    w = [i + 1 for i in w]
    o = numpy.sum(x ** 2) / 4000 - prod(numpy.cos(x / numpy.sqrt(w))) + 1
    return o


def F12(x):
    dim = len(x)
    o = (math.pi / dim) * (
        10 * ((numpy.sin(math.pi * (1 + (x[0] + 1) / 4))) ** 2)
        + numpy.sum(
            (((x[: dim - 1] + 1) / 4) ** 2)
            * (1 + 10 * ((numpy.sin(math.pi * (1 + (x[1 :] + 1) / 4)))) ** 2)
        )
        + ((x[dim - 1] + 1) / 4) ** 2
    ) + numpy.sum(Ufun(x, 10, 100, 4))
    return o


def F13(x):
    if x.ndim==1:
        x = x.reshape(1,-1)

    o = 0.1 * (
        (numpy.sin(3 * numpy.pi * x[:,0])) ** 2
        + numpy.sum(
            (x[:,:-1] - 1) ** 2
            * (1 + (numpy.sin(3 * numpy.pi * x[:,1:])) ** 2), axis=1
        )
        + ((x[:,-1] - 1) ** 2) * (1 + (numpy.sin(2 * numpy.pi * x[:,-1])) ** 2)
    ) + numpy.sum(Ufun(x, 5, 100, 4))
    return o


TestFunctionDetails = {
    "F1": [
        r'F_1(\textit{\textbf{x}}) = \sum\limits^D_{i=1} z^2_i', 
        -100, 100],
    "F2": [
        r'F_2(\textit{\textbf{x}}) = \sum\limits^D_{i=1} |z_i| + \prod\limits^D_{i=1}|z_i|', 
        -10, 10],
    "F3": [
        r'F_3(\textit{\textbf{x}}) = \sum\limits^D_{i=1} \left(\sum\limits^i_{j=1}z_j\right)^2', 
        -100, 100],
    "F4": [
        r'F_4(\textit{\textbf{x}}) = \max\left(\textit{\textbf{z}}\right)', 
        -100, 100],
    "F5": [
        r'F_5(\textit{\textbf{x}}) = \sum\limits^{D-1}_{i=1}\left[100\left(z_{i+1}-z_i^2\right)^2+\left(z_i - 1\right)^2\right]', 
        -30, 30],
    "F6": [
        r'F_6(\textit{\textbf{x}}) = \sum\limits^D_{i=1} \left(z_i+0.5\right)^2', 
        -100, 100],
    "F7": [
        r'F_7(\textit{\textbf{x}}) = \sum\limits^D_{i=1} i\cdot z_i^4 + \text{rand}[0,1)', 
        -1.28, 1.28],
    "F8": [
        r'F_8(\textit{\textbf{x}}) = \sum\limits^D_{i=1} -x_i \sin\sqrt{|x_i|}+D\cdot 418.98288727243369', 
        -500, 500],
    "F9": [
        r'F_9(\textit{\textbf{x}}) = \sum\limits^D_{i=1} \left[z^2_i-10\cos \left(2\pi z_i\right) +10 \right]', 
        -5.12, 5.12],
    "F10": [
        r'F_{10}(\textit{\textbf{x}}) = -20\exp\left(-0.2\sqrt{\frac{1}{D}\sum\limits^{D}_{i=1}z_i^2}\right)-\exp\left(\frac{1}{D}\sum\limits^D_{i=1}\cos\left(2\pi z_i\right)\right)+20+\exp(1)', 
        -32, 32],
    "F11": [
        r'F_{11}(\textit{\textbf{x}}) = \frac{1}{4000}\sum\limits^D_{i=1}z^2_i-\prod\limits^D_{i=1}\cos\left(\frac{z_i}{\sqrt{i}}\right)+1', 
        -600, 600],
    "F12": [
        r'F_{12}(\textit{\textbf{x}}) = \frac{\pi}{D}\left\{10\sin^2\left(\pi \omega_1\right)+\sum\limits^{D-1}_{i=1}\left(\omega_i-1\right)^2\left[1+10\sin^2\left(\pi \omega_{i+1}\right)\right]+\left(\omega_D - 1\right)^2 \right\}', 
        -50, 50],
}
