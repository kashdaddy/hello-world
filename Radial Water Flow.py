# Radial Water Flow.py
"""
Created on 20 July 2015

A simulation created to analyse the effects of groundwater flow on BHE system 
performance. Uses 2D radial coordinates. 

Author: James Kasherman
"""

from math import pi, log
from numpy import zeros, linspace, ones, transpose
import matplotlib.pyplot as plt

# Define parameters
T_i = 273.15 + 21.4 # Temperature, in K
T_in = 273.15 + 60 # well temperature at top, in K
T_out = 273.15 + 30 # well temperature at bottom, in K

r_borehole = 0.03 # borehole radius

#Define boundary conditions for the problem (2 needed as 2nd order in space)
r_min = r_borehole # minimum value for r, = 0 
r_max = 16 + r_borehole # maximum value for r, = 1000

z_min = 0.0 # minimum value for y, = 0 
z_max = 100.0 # maximum value for y, = 1000

def initial(r,z): return T_i
    
# Spatial discretization
class nodes:
    def __init__(self,noodles,ks,k2,caps,cap2):
        self.noodles = noodles
        self.Nr = self.noodles + 1 + 2 # number of nodes in space dimension
        self.dr = (r_max-r_min)/(self.Nr-3) # distance between each node
        self.r = linspace (r_min, r_max, self.Nr-2) # create vector of nodes in r
        
        self.Nz = self.noodles + 1 + 2 # number of nodes in space dimension
        self.dz = (z_max-z_min)/(self.Nz-3) # distance between each node
        self.z = linspace (z_min, z_max, self.Nz-2) # create vector of nodes in y
        
        self.ks = ks # soil 1
        self.k2 = k2 # soil 2 or water
        self.caps = caps
        self.cap2 = cap2
    
    def watertable(self,porosity=0.12):
        self.porosity = porosity
        self.k1 = self.ks/(1-self.porosity)
        self.cap1 = self.caps/(1-self.porosity)
        k_w = self.k2*self.porosity + self.k1*(1-self.porosity)
        cap_w = self.cap2*self.porosity + self.cap1*(1-self.porosity)
        return k_w, cap_w
        
    def k_(self,depth1,por):
        self.k_w, self.cap_w = self.watertable(por)
        k = ones((self.Nr,self.Nz))*self.k_w
        for i in range(0,int(depth1/z_max*self.noodles)):
            k[i,:] = self.ks
        return k

    def alpha(self,depth1,por):
        a = ones((self.Nr,1))*self.k_w/(self.cap_w)
        for i in range(0,int(depth1/z_max*self.noodles)):
            a[i] = self.k1/(self.cap1)
        return a
        
    def q(self,T,Tn,k): # Returns heat transfer rate
        q = 2*pi*self.dz*k*(T-Tn)/log((r_min+self.dr)/r_min)
        return q

#Time steps
dt = 10800 # Time steps


def T_borehole(Nr,i,T): # Defines borehole temperature
    p = - (T_in - T_out) * i/(Nr-1)*10/8 + T_in
    if p >= T_out:
#        print(p)
        return p
    else: return T

# Integrate in time
def iterate(Nodes,years,ks,por1,kw,caps,cap2,w_lev,v):
    Q = []
    time = []
    t_final = years*31471200
    N = nodes(Nodes,ks,kw,caps,cap2)
    k_ = N.k_(w_lev,por1)    
    dif = N.alpha(w_lev,por1)
    T_np1 = zeros((N.Nr,N.Nz), float)
    t = 0.0
    T_n = ones((N.Nr,N.Nz)) * initial(N.r,N.z)
    flow = v/31471200
    # Apply Initial Borehole Condition
    for i in range(0,N.Nz): 
        T_n[i,0] = T_borehole(N.Nz,i,T_i)
    while t < t_final:
        Q_ = []
        time.append(t/86400)
        t += dt
        r_ = r_min
        for k in range(1,N.Nz-1):
            r_ += N.dr
            for j in range(1,N.Nr-1):
                T_np1[j,k] = T_n[j,k] + dt * ( dif[j] * ( (T_n[j,k+1] + T_n[j,k-1] - 2*T_n[j,k]) / N.dr**2 + (T_n[j,k+1] - T_n[j,k-1]) / (2*N.dr*r_) + (T_n[j+1,k] + T_n[j-1,k] - 2*T_n[j,k])/ N.dz**2) - por1*cap2*flow/caps*(T_n[j,k+1] - T_n[j,k-1]) / (2*N.dr*r_)) 
        T_np1[0,:] = T_np1[1,:]
        T_np1[N.Nr-1,:] = T_np1[N.Nr-2,:]
        T_np1[:,0] = T_np1[:,1]
        T_np1[:,N.Nz-1] = T_np1[:,N.Nz-2]
        for i in range(0,N.Nr): 
            T_np1[i,0] = T_borehole(N.Nr,i,T_np1[i,0])
            Q_.append(N.q(T_n[i,0],T_n[i,1],k_[i,0]))
        Q.append(sum(Q_))
        T_n = T_np1.copy() # ready for the next step
        
    return N.r, N.z, T_n, time, Q

if __name__ == "__main__": 
    # Soil Properties
    k_sandstone = 4.4
    k_clay = 0.56
    k_silt = 1.04
    k_shale = 2.5
    k_water = 0.6
    por_sandstone = 0.18
    por_clay = 0.47
    por_silt = 0.475
    por_shale = 0.0525
    cap_sandstone = 3560000
    cap_clay = 3300000
    cap_silt = 2850000
    cap_shale = 3940000
    cap_water = 4180000
    
    # Iterate
    F = iterate(20,1,k_clay,por_clay,k_water,cap_clay,cap_water,10,0)
    G = iterate(20,1,k_clay,por_clay,k_water,cap_clay,cap_water,30,0)
    H = iterate(20,1,k_clay,por_clay,k_water,cap_clay,cap_water,60,0)
    I = iterate(20,1,k_clay,por_clay,k_water,cap_clay,cap_water,100,0)
    
    # Plot Temperature Contour Plot
    plt.figure()
    plt.title("Temperature Distribution (K), 60m Groundwater")
    plt.contour(transpose(H[0]),-H[1],H[2][1:-1,1:-1],16)
    plt.colorbar()
    plt.xlabel("Radius (m)")
    plt.ylabel("Depth (m)")
    
    # Plot Heat Transfer Rates
    plt.figure()
    plt.title("Heat Transfer Rate for Different Groundwater Depths")
    plt.hold(True)
    plt.plot(F[3],F[4],'r',label='10m Depth')
    plt.plot(G[3],G[4],'b',label='30m Depth')
    plt.plot(H[3],H[4],'g',label='60m Depth')
    plt.plot(I[3],I[4],'m',label='No Groundwater')
    plt.xlabel("Time (days)")
    plt.ylabel("Heat transfer rate (W)")
    plt.ylim([0,3500])
    plt.legend(bbox_to_anchor=(1, 1), fontsize='small')
    
    # Plot Final Value Comparison
    plt.figure()
    plt.title("Heat Transfer Rate After 1 Year")
    plt.plot([10,30,60,100],[F[4][-1],G[4][-1],H[4][-1],I[4][-1]],'ro')
    plt.xlabel("Groundwater Depth")
    plt.ylabel("Heat Transfer Rate")
    
    plt.show()