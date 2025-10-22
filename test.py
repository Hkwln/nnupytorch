import numpy
def postionalencoding(seq_length, d, n:int = 10000) -> numpy.tensor:
    p = numpy.zeros(seq_lenght)
    for k in range(seq_length):
        for i in numpy.arange(int (d/2)):
            denominator= np.power(n, 2*i/d)
            p[k, 2*i] =np.sin(k/denominator)
            p[k, 2*+1] = np.cos(k/denominator)

    return p

    
