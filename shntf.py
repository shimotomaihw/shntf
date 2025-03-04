
import numpy
import scipy
import matplotlib.pyplot as plt
import pandas

def main():
    n = 4
    ai = 3
    bi = 13
    ci = 103
    uo = numpy.random.randint(0, 100, ai*n).reshape(ai, n)
    vo = numpy.random.randint(0, 100, bi*n).reshape(bi, n)
    wo = numpy.random.randint(0, 100, ci*n).reshape(ci, n)
    print(uo)
    print(vo)
    print(wo)
    uovo = numpy.dot(uo, vo.T)
    print(uovo)
    uovowo = numpy.einsum('il,jl,kl->ijk', uo, vo, wo)
    print(uovowo)
    r1,r2,r3, info = ntf3(uovowo, 3)
    print('r1\n',r1)
    print('r2\n',r2)
    print('r3\n',r3)
    #print('loglikelihood', loglikelihood(uovowo, r1, r2, r3))
    print('loglikelihoodGamma', loglikelihoodGamma(uovowo, r1, r2, r3))
    print('aic', aic(uovowo, r1, r2, r3))
    error2 = []
    samplen = 10
    maxrank = 12
    nvec = numpy.arange(1,maxrank+1)
    dfer = pandas.DataFrame(index=numpy.arange(samplen*maxrank), columns=['n','trial', 'MSE'])
    ind = 0
    for i in nvec:
        for j in range(samplen):
            uo = numpy.random.randint(0, 10, ai*n).reshape(ai, n)
            vo = numpy.random.randint(0, 10, bi*n).reshape(bi, n)
            wo = numpy.random.randint(0, 10, ci*n).reshape(ci, n)
            #uovowo = numpy.einsum('il,jl,kl->ijk', uo, vo, wo)
            r1,r2,r3, info = ntf3(uovowo, i)
            dfer.loc[ind, ['n','trial','MSE']] = [i, j, info['error2'][-1]]
            ind += 1
        error2 += [info['error2'][-1]]
    dfer.to_csv('ntf3_rank_error2.csv', index=False)
    print(dfer)
    dfer.boxplot(by='n', column='MSE')
    plt.savefig('ntf3_rank_error2_boxplot.png')
    plt.savefig('ntf3_rank_error2_boxplot.svg')
    plt.figure()
    plt.plot(error2)
    plt.grid()
    plt.yscale('log')
    plt.savefig('ntf3_rank_error2.png')
    plt.savefig('ntf3_rank_error2.svg')
    return

def loglikelihood(m, r1, r2, r3):
    r123 = numpy.einsum('il,jl,kl->ijk', r1, r2, r3)
    return numpy.sum(m*numpy.log(r123)-r123)

def loglikelihoodGamma(m, r1, r2, r3):
    """ Gamma Distribution Log Likelihood
    m: numpy.ndarray
    r1: numpy.ndarray
    r2: numpy.ndarray
    r3: numpy.ndarray
    """
    r123 = numpy.einsum('il,jl,kl->ijk', r1, r2, r3)
    #lp = numpy.sum(numpy.log(r123)*m-(r123))
    k = 5
    mask = (m!=0) * (r123!=0)
    m0 = m[mask]
    r0 = r123[mask]
    lp = numpy.sum((k-1)*numpy.log(m0)-m0/r0 - numpy.log(scipy.special.gamma(m0)) - k*numpy.log(r0))
    return lp


def aic(m, r1, r2, r3):
    return -2*loglikelihood(m, r1, r2, r3)+2*(r1.size+r2.size+r3.size)

def ntf3(m, n, iter=4) -> tuple:
    r1 = numpy.zeros((m.shape[0],n))
    r2 = numpy.zeros((m.shape[1],n))
    r3 = numpy.zeros((m.shape[2],n))
    r1 = numpy.random.rand(m.shape[0],n)
    r2 = numpy.random.rand(m.shape[1],n)
    r3 = numpy.random.rand(m.shape[2],n)
    #r1[:,0] = m.mean(axis=(1,2))
    #r2[:,0] = m.mean(axis=(0,2))
    #r3[:,0] = m.mean(axis=(0,1))
    error1 = []
    error2 = []
    for i in range(iter):
        r123 = numpy.einsum('il,jl,kl->ijk', r1, r2, r3)
        error1 += [numpy.mean(numpy.abs(m-r123))]
        error2 += [numpy.mean(r123*r123)]
        print(f'ntf3:error1={error1[-1]} error2={error2[-1]}') 
        lower = numpy.einsum('sk,tk,rst->rk',r2,r3,r123)
        upper = numpy.einsum('rst,sk,tk->rk',m,r2,r3)
        r1 = r1*(upper/lower)
        print('ntf3:r1\n',r1)
        r123 = numpy.einsum('il,jl,kl->ijk', r1, r2, r3)
        lower = numpy.einsum('rk,tk,rst->sk',r1,r3,r123)
        upper = numpy.einsum('rst,rk,tk->sk',m,r1,r3)
        r2 = r2*(upper/lower)
        print('ntf3:r2\n',r2)
        r123 = numpy.einsum('il,jl,kl->ijk', r1, r2, r3)
        lower = numpy.einsum('rk,sk,rst->tk',r1,r2,r123)
        upper = numpy.einsum('rst,rk,sk->tk',m,r1,r2)
        r3 = r3*(upper/lower)
        print('ntf3:r3\n',r3)
    r123 = numpy.einsum('il,jl,kl->ijk', r1, r2, r3)
    plt.figure()
    plt.plot(error1)
    plt.plot(error2)
    plt.grid()
    plt.yscale('log')
    plt.savefig('ntf3_error.png')
    plt.savefig('ntf3_error.svg')
    plt.ylim(1e-1, 1e3)
    plt.savefig('ntf3_error1e3.png')
    plt.savefig('ntf3_error1e3.svg')
    info = {'error1':error1, 'error2':error2, 'r123':r123}
    return r1,r2,r3, info

# u_{r,k}=u_{r, k}\displaystyle\frac{\sum_s\sum_tx_{r,s,t}v_{s,k}w_{t,k}}{\sum_s\sum_tv_{s,k}w_{t,k}\sum_{k^{'}}u_{r,k'}v_{s,k'}w_{t,k'}}
# v_{s,k}=v_{s, k}\displaystyle\frac{\sum_s\sum_tx_{r,s,t}u_{r,k}w_{t,k}}{\sum_r\sum_tu_{r,k}w_{t,k}\sum_{k^{'}}u_{r,k'}v_{s,k'}w_{t,k'}}
# w_{t,k}=w_{t, k}\displaystyle\frac{\sum_s\sum_tx_{r,s,t}v_{s,k}u_{r,k}}{\sum_s\sum_tv_{s,k}u_{r,k}\sum_{k^{'}}u_{r,k'}v_{s,k'}w_{t,k'}}

if __name__ == '__main__':
    main()
