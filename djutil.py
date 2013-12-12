def clist():return[2+2j,3+2j,1.75+1j,2.5+1j,3+1j,3.25+1j]

#def blist():return
from random import *
def ranbin(n):
   
    return [randint(0,1) for b in range(1,n+1)]

def iseq(bl):
    el=[]
    
    for x in range(len(bl)-1):
        if bl[x] == bl[x+1]:
            el.append(bl[x+1])
        else:el=[]
            

    return el        

def image():
    import image
    im=image.file2image("dance.png")
    return im

def imrep():return[[(255,255,255),(155,155,100)],[(100,100,100),(90,90,90)]]

def ir0(imfile):return imfile[0]

def irx(ir0):return [x for (x,y,z) in ir0]

def iry(ir0):return [y for (x,y,z) in ir0]

def irz(ir0):return [z for (x,y,z) in ir0]
                     
def dbl(ir):return [2*r for r in ir]

def test(ir0):return[2*x for x in ir0]

#inputs a list of numbers and a divisor outputs all numvers that are not multiples 
def filter(num,div):return[x for x in num if div % x !=0]

def mlist(lst):return[list(range(x+1)) for x in lst]

def ds():return [{1:'a',2:'b',3:'c',4:'d',5:'e'},{'a':'apple','b':'banana','c':'coco', 'd':'dates', 'e':'eggs'}]

def dkeys(d):return {k for k in d.keys()}

def dvals(d):return {v for v in d.values()}

def fog(d1,d2):
    import dictutil
    d1k=list(d1.keys())
    d1v=dictutil.dict2list(d1,d1.keys())
    d2v=dictutil.dict2list(d2,d1v)
    return dictutil.list2dict(d2v,d1k)

def nlist(): return [1,2,3,4,5,6,7,8,9]

def power(lst,ex):return[x**ex for x in lst]

def loop(m, fx,fy,fz):
    lst=m
    for i in range(len(m)):
        for p in range(len(m[i])):
           x=lst[i][p][0]
           y=lst[i][p][1]
           z=lst[i][p][2]
           lst[i][p]=(fx(x),fy(y),fz(z))
    return lst

def modsq(v, m=256):return ((v*v)%m)

def modcu(v):return((v**3)%256)

def modplus(v):return((v+20)%256)

def polar(n):
    l=list(range(1,n+1))
    import math
    return[math.e**((200*math.pi*1j)/x) for x in l]
   
#http://stackoverflow.com/questions/477486/python-decimal-range-step-value
def drange(start, stop, step):
    r = start
    while r < stop:
       yield r
       r += step
       
def randlist(dim,le):
    import random
    import timeit
    return [random.randint(dim[0],dim[1]) for r in range(le)]

def randset(dim,le):
    import random
    import timeit
    return {random.randint(dim[0],dim[1]) for r in range(le)}

def oneslist(le): return[1 for r in range(le)]

def ilist(mi,ma):
    i0=range(mi, ma, 1)
    #return(["%c" % x for x in i0])
    return i0
def dlist():
   i0=drange(2, 3, 0.1)
   return(["%g" % x for x in i0])

def gtvec():
    v=tvec()
    for d in v.D:
        if d in v.f:
            print(v.f[d])
            

def tvec():return Vec({'a','b','c','d'}, {'a':1})

def zero_vec(D):return Vec(D,{})

def setitem(v,d,val): v.f[d] = val

def getitem(v,d): return v.f[d] if d in v.f else 0

def scalar_mul(v, alpha, s=0):
    if s==0:
       return Vec(v.D, {d:alpha*value for d, value in v.f.items()})
    if s==1:
       return Vec(v.D, {d:alpha*getitem(v,d) for d in v.D})

def add(u,v):
    return Vec(u.D,{d:getitem(u,d)+getitem(v,d) for d in u.D})

def neg(v): return scalar_mul(v,-1)
   
class Vec:
    def __init__(self, labels, function):
        self.D = labels
        self.f = function
        
 #list u.v to dot product float      
def list_dot(u,v): return sum(u[i]*v[i] for i in range(len(u)))

def dot_product_list(needle, haystack):
    s=len(needle)
    return [list_dot(needle, haystack[i:i+s]) for i in range(len(haystack)-(s-1))]

def average(l):return sum(l)/len(l)

 
#returns standard deviation from a list of floats
def standev(l):
    import math
    ave= average(l)
    sd1=sum([(x-ave)**2 for x in l])
    var=sd1/len(l)
    return math.sqrt(var)


def tvserv():
    
   news =  input(" News current affairs ")
   sports = input(" Sports ")
   doco = input(" doccumentary ")
   drama = input(" drama ")
   comedy = input(" comedy ")
   action = input(" action ")
   panel = input(" Panel / game ")
   datain({'news':news , 'sports': sports, 'dodo':doco, 'drama':drama,'comedy':comedy,'action':action,'panel':panel})
       
def datain(d):
    import pickle
    data=dataout()
    data.append(d)
    pickle.dump(data, open( "save.p", "wb" ))

def dataout(file):
    import pickle
    data=pickle.load( open( file, "rb" ) )
    return data
       
def datainit(file):
     import pickle
     data=[]
     pickle.dump(data, open( file, "wb" ))

def transform(a,ai,b,bi,c,ci,d,L):
    tr=[a*x**ai +b*x**bi + c*x**ci + d for x in L]
    return tr

def quad(a,ai,b,c,L):
    tr=[a*x**ai +b*x**b + c for x in L]
    return tr

    
def djplot(lsta,lstb,lstc):
    import matplotlib.pyplot as plt
    plt.plot(lsta,'r',lstb,'b',lstc,'g')
    plt.show()
def djplot(lsta):
    import matplotlib.pyplot as plt
    plt.plot(lsta,'r')
    plt.show()

def argand(a):
    import matplotlib.pyplot as plt
    import numpy as np
    for x in range(len(a)):
        plt.plot([0,a[x].real],[0,a[x].imag],'ro-',label='python')
    limit=np.max(np.ceil(np.absolute(a))) # set limits for axis
    plt.xlim((-limit,limit))
    plt.ylim((-limit,limit))
    plt.ylabel('Imaginary')
    plt.xlabel('Real')
    plt.show()


def djco():
    import numpy as np
    from matplotlib import pyplot as plt

    # generate data
    x = np.r_[0:100:30j]
    y = np.r_[0:1:20j]
    X, Y = np.meshgrid(x, y)
    #Z = X*np.exp(1j*Y) # some arbitrary complex data
    Z=X + 2j*Y

    plotit(x,y,Z, 'real')
    plotit(x,y,Z.real, 'explicit real')
    plotit(x,y,Z.imag, 'imagenary')

    plt.show()

def plotit(X,Y,z, title):
    import numpy as np
    from matplotlib import pyplot as plt
    plt.figure()
    cs = plt.contour(X,Y,z) # contour() accepts complex values
    plt.clabel(cs, inline=1, fontsize=10) # add labels to contours
    plt.title(title)
    plt.savefig(title+'.png')

def rvec():
    import vec
    D={'radio','sensor','memory','cpu'}
    v0=vec.Vec(D,{'radio':1,'cpu':3})
    v1=vec.Vec(D,{'sensor':2,'cpu':4})
    v2=vec.Vec(D,{'memory':3, 'cpu':1})
    VT=[v0,v1,v2]
    return VT
           
