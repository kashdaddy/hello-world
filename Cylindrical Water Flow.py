# Cylindrical Water Flow.py
"""
Created on 20 September 2015

A simulation created to analyse the effects of groundwater flow on BHE system 
performance. Uses 3D cylindrical coordinates. 

Author: James Kasherman
"""

from math import pi, log, cos
from numpy import zeros, linspace, ones, transpose, append, vstack
import matplotlib.pyplot as plt
import time as tt

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

def initial(): return T_i
    
# Spatial discretization
class nodes:
    def __init__(self,nod_r,nod_th,nod_z,ks,k2,caps,cap2):
        self.Nr = nod_r + 1 + 2 # number of nodes in radial space dimension
        self.dr = (r_max-r_min)/(self.Nr-3) # distance between each node
        self.r = linspace (r_min, r_max, self.Nr-2) # create vector of nodes in r
        
        self.Nz = nod_z + 1 + 2 # number of nodes in vertical space dimension
        self.dz = (z_max-z_min)/(self.Nz-3) # distance between each node
        self.z = linspace (z_min, z_max, self.Nz-2) # create vector of nodes in z
        
        self.Nth = nod_th # number of nodes in angular space dimension
        self.dth = 2*pi/self.Nth # angle between each node
        self.th = linspace (0, (self.Nth-1)/self.Nth*2*pi, self.Nth) # create vector of nodes in th
        
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
        k = ones((self.Nz,1))*self.k_w
        for i in range(0,int(depth1/z_max*self.Nz)):
            k[i] = self.ks
        return k

    def alpha(self,depth1,por):
        a = ones((self.Nz,1))*self.k_w/(self.cap_w)
        for i in range(0,int(depth1/z_max*self.Nz)):
            a[i] = self.k1/(self.cap1)
        return a
        
    def q(self,T,Tn,k): # Returns heat transfer rate
        q = 2*pi*self.dz*k*(T-Tn)/log((r_min+self.dr)/r_min)/self.Nth
        return q
    
#Time steps
dt = 21600 # Time steps


def T_borehole(Nr,i,T): # Defines borehole temperature
    p = - (T_in - T_out) * i/(Nr-1)*10/8 + T_in
    if p >= T_out:
#        print(p)
        return p
    else: return T

# Integrate in time
def iterate(Nr,Nth,Nz,years,ks,por1,kw,caps,cap2,w_lev,vel):
    start = tt.clock()
    Q = []
    time = []
    r_ = []
    t_final = years*31471200
    N = nodes(Nr,Nth,Nz,ks,kw,caps,cap2)
    k_ = N.k_(w_lev,por1)    
    dif = N.alpha(w_lev,por1)
    t = 0.0
    T_n = ones((N.Nz,N.Nth,N.Nr)) * initial()
    V = vel/31471200
    for j in range(0,N.Nz): 
        for k in range(0,N.Nth):
            T_n[j,k,0] = T_borehole(N.Nz,j,T_i)
    T_np1 = T_n.copy()
    r_iterate = r_min
    for l in range(0,N.Nr):
        r_.append(r_iterate)
        r_iterate += N.dr

    while t < t_final:
        Q_2 = []
        time.append(t/86400)
        t += dt
        for j in range(1,N.Nz-1):
            for l in range(1,N.Nr-1):
                for k in range(0,N.Nth): 
                    if w_lev < z_max*(j)/N.Nz: 
                        if k+1 >= N.Nth:
                            T_np1[j,k,l] = T_n[j,k,l] + dt * ( dif[j] * ( (T_n[j,k,l+1] + T_n[j,k,l-1] - 2*T_n[j,k,l]) / N.dr**2 + (T_n[j,k,l+1] - T_n[j,k,l-1]) / (2*N.dr*r_[l]) + (T_n[j+1,k,l] + T_n[j-1,k,l] - 2*T_n[j,k,l])/ N.dz**2 + (T_n[j,0,l] + T_n[j,k-1,l] - 2*T_n[j,k,l]) / (N.dth**2 * r_[l]**2)) + por1*cap2*V*cos(N.th[k])/caps*(T_n[j,k,l+1] - T_n[j,k,l-1]) / (2*N.dr*r_[l]))
                        else: 
                            T_np1[j,k,l] = T_n[j,k,l] + dt * ( dif[j] * ( (T_n[j,k,l+1] + T_n[j,k,l-1] - 2*T_n[j,k,l]) / N.dr**2 + (T_n[j,k,l+1] - T_n[j,k,l-1]) / (2*N.dr*r_[l]) + (T_n[j+1,k,l] + T_n[j-1,k,l] - 2*T_n[j,k,l])/ N.dz**2 + (T_n[j,k+1,l] + T_n[j,k-1,l] - 2*T_n[j,k,l]) / (N.dth**2 * r_[l]**2)) + por1*cap2*V*cos(N.th[k])/caps*(T_n[j,k,l+1] - T_n[j,k,l-1]) / (2*N.dr*r_[l]))
                    else:
                        if k+1 >= N.Nth:
                            T_np1[j,k,l] = T_n[j,k,l] + dt * ( dif[j] * ( (T_n[j,k,l+1] + T_n[j,k,l-1] - 2*T_n[j,k,l]) / N.dr**2 + (T_n[j,k,l+1] - T_n[j,k,l-1]) / (2*N.dr*r_[l]) + (T_n[j+1,k,l] + T_n[j-1,k,l] - 2*T_n[j,k,l])/ N.dz**2 + (T_n[j,0,l] + T_n[j,k-1,l] - 2*T_n[j,k,l]) / (N.dth**2 * r_[l]**2)))
                        else: 
                            T_np1[j,k,l] = T_n[j,k,l] + dt * ( dif[j] * ( (T_n[j,k,l+1] + T_n[j,k,l-1] - 2*T_n[j,k,l]) / N.dr**2 + (T_n[j,k,l+1] - T_n[j,k,l-1]) / (2*N.dr*r_[l]) + (T_n[j+1,k,l] + T_n[j-1,k,l] - 2*T_n[j,k,l])/ N.dz**2 + (T_n[j,k+1,l] + T_n[j,k-1,l] - 2*T_n[j,k,l]) / (N.dth**2 * r_[l]**2)))
        T_np1[0,:,:] = T_np1[1,:,:]
        T_np1[N.Nz-1,:,:] = T_np1[N.Nz-2,:,:]
        T_np1[:,:,0] = T_np1[:,:,1]
        T_np1[:,:,N.Nr-1] = T_np1[:,:,N.Nr-2]
        for i in range(0,N.Nz): 
            Q_1 = []
            for j in range(0,N.Nth):
                T_np1[i,j,0] = T_borehole(N.Nz,i,T_np1[i,j,0])
                Q_1.append(N.q(T_n[i,j,0],T_n[i,j,1],k_[i]))
            Q_2.append(sum(Q_1))
        Q.append(sum(Q_2)[0])
    end = tt.clock()
    
    print('time =', end - start, 'seconds. ')
        
    return N.r.tolist(), N.z.tolist(), N.th.tolist(), T_n, time, Q

if __name__ == "__main__": 
    # Soil Properties
    k_sandstone = 4.4
    k_clay = 0.56
    k_silt = 1.04
    k_water = 0.6
    por_sandstone = 0.18
    por_clay = 0.47
    por_silt = 0.475
    cap_sandstone = 3560000
    cap_clay = 3300000
    cap_silt = 2850000
    cap_water = 4180000
    
    # Iterate    
    F = iterate(10,16,10,1,k_clay,por_clay,k_water,cap_clay,cap_water,10,0)
    G = iterate(10,16,10,1,k_clay,por_clay,k_water,cap_clay,cap_water,10,20)
    H = iterate(10,16,10,1,k_clay,por_clay,k_water,cap_clay,cap_water,10,100)
    
    # Plot Heat Transfer Rate
    plt.figure(4)
    plt.title("Heat Transfer Rate for Different Nodal Resolutions")
    plt.hold(True)
    plt.plot(F[4],F[5],'r',label='No Groundwater Flow')
    plt.plot(G[4],G[5],'c',label='20 m/year')
    plt.plot(H[4],H[5],'k',label='100 m/year')
    plt.xlabel("Time (days)")
    plt.ylabel("Heat transfer rate (W)")
    plt.yscale('log')
    plt.legend(bbox_to_anchor=(1, 1), fontsize='small')
    
    # Cross-sectional Temperature Contour Plots
    plt.figure()
    af = append(F[2],2*pi)
    pf = vstack((F[3][2,:,1:-1],F[3][2,0,1:-1]))
    figf, axf = plt.subplots(subplot_kw=dict(projection='polar'))
    caxf = axf.contourf(af,F[0],transpose(pf),30)
    cb = figf.colorbar(caxf)
    plt.title('30m Depth, No Groundwater Flow')
    
    plt.figure()
    ag = append(G[2],2*pi)
    pg = vstack((G[3][2,:,1:-1],G[3][2,0,1:-1]))
    figg, axg = plt.subplots(subplot_kw=dict(projection='polar'))
    caxg = axg.contourf(ag,G[0],transpose(pg),30)
    cb = figg.colorbar(caxg)
    plt.title('30m Depth, 20 m/year')
    
    plt.figure()
    ah = append(H[2],2*pi)
    ph = vstack((H[3][2,:,1:-1],H[3][2,0,1:-1]))
    figh, axh = plt.subplots(subplot_kw=dict(projection='polar'))
    caxh = axh.contourf(ah,H[0],transpose(ph),30)
    cb = figh.colorbar(caxh)
    plt.title('30m Depth, 100 m/year')
    
    # Plot Comparison
    plt.figure()
    plt.title("Heat Transfer Rate After 1 Year")
    plt.plot([0,20,100],[F[5][-1],G[5][-1],H[5][-1]],'ro-')
    plt.xlabel("Groundwater Flow Rate")
    plt.ylabel("Heat Transfer Rate")
    
    plt.show()