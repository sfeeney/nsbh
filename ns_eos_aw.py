import numpy as np
from scipy.interpolate import interp1d
import sympy as spy
import lal
import lalsimulation as lalsim
from glue.ligolw import ligolw, lsctables, table, utils, array

def choose_func(eos):
    if eos.lower()=="dd2":
        radii, masses = np.loadtxt("data/{}_mr.dat".format(eos.lower()), unpack=True)
        interp_func = interp1d(masses, radii, kind="quadratic")
        return dd2_lambda_from_mass, interp_func, max(masses), min(masses)
    else:
        raise ValueError("EOS not implemented!")


def dd2_lambda_from_mass(m):
    coeffs = {-5: -23020.6,
              -4: 194720.,
              -3: -658596.,
              -2: 1.33938e6,
              -1: -1.78004e6,
              0: 1.60491e6,
              1: -992989.,
              2: 416080.,
              3: -112946.,
              4: 17928.5,
              5: -1263.34}
    return sum([coeffs[n] * m**n for n in coeffs])


def load_table(sim_file):
    
    xml_doc = utils.load_filename(sim_file,
            contenthandler=lsctables.use_in(ligolw.LIGOLWContentHandler))
    
    sim_inspiral_table = table.get_table(xml_doc,
            lsctables.SimInspiralTable.tableName)
    
    return sim_inspiral_table


def load_samples(samples_file):
    return np.genfromtxt(samples_file, names=True)
    

def get_tidal_coupling_constant(s):
    """
    As in Dietrich, Bernuzzi & Tichy - doi:10.1103/PhysRevD.96.121501; arXiv:1706.02969
    """
    return 2. / 13 * (1 + 12 * s.m_bh / s.m_ns) * (s.m_ns / (s.m_ns + s.m_bh) / s.compactness)**5 * s.love


def get_chi_eff(s):
    """
    As in Dietrich, Bernuzzi & Tichy - doi:10.1103/PhysRevD.96.121501; arXiv:1706.02969
    """
    X_bh = s.m_bh / (s.m_bh + s.m_ns)
    X_ns = s.m_ns / (s.m_ns + s.m_bh)
    x_bh = X_bh * s.s_bh_z
    x_ns = X_ns * s.s_ns_z
    return x_bh + x_ns - (38./113 * X_bh * X_ns * (s.s_bh_z + s.s_ns_z))


def get_tidal_phase(s, omegas):
    """
    As in Dietrich, Bernuzzi & Tichy - doi:10.1103/PhysRevD.96.121501; arXiv:1706.02969
    """
    X_bh = s.m_bh / (s.m_bh + s.m_ns)
    X_ns = s.m_ns / (s.m_ns + s.m_bh)
    c_newt = 39./16
    n_1 = -17.428
    n_32 = 31.867
    n_2 = -26.414
    n_52 = 62.362
    d_1 = n_1 - (3115./1248)
    d_32 = 36.089
    x = (0.5 * (s.m_bh + s.m_ns) * omegas)**(2./3.)
    z1 = 1 + n_1 * x + n_32 * x**(3./2.) + n_2 * x**2 + n_52 * x**(5./2.)
    z2 = 1 + d_1 * x + d_32 * x**(3./2.)
    return -1 * get_tidal_coupling_constant(s) * c_newt / X_bh / X_ns * x**(5./2.) * z1 / z2

            
class Foucart:
    """
    Generates a class to calculate the remnant mass of an NSBH merger following
    Foucart et al. (2018) - doi:10.1103/PhysRevD.98.081501; arXiv:1807.00011
    """
    def __init__(self, sim, eos="WFF1"):
        # Load if from sim inspiral table
        if isinstance(sim, lsctables.SimInspiral):
            self.sim = sim
            
            # Set NS
            self.m_ns = sim.mass2
            self.s_ns_x = sim.spin2x
            self.s_ns_y = sim.spin2y
            self.s_ns_z = sim.spin2z

            # Set BH
            self.m_bh = sim.mass1
            self.s_bh_x = sim.spin1x
            self.s_bh_y = sim.spin1y
            self.s_bh_z = sim.spin1z
            
            self.s_bh = np.sqrt(self.s_bh_x**2 + self.s_bh_y**2 + self.s_bh_z**2)
            self.s_bh_tilt = np.arccos(self.s_bh_z/self.s_bh)
            
        # Load if from LALInference posterior samples
        elif isinstance(sim, np.void):
            # Set NS
            self.m_ns = sim["m2"]

            # Set BH
            self.m_bh = sim["m1"]
            self.s_bh = sim["a1"]
            self.s_bh_tilt = sim["tilt1"]
            self.s_bh_z = self.s_bh * np.cos(self.s_bh_tilt)

        elif isinstance(sim, dict):
            
            # Set NS
            self.m_ns = sim['mass2']
            self.s_ns_x = sim['spin2x']
            self.s_ns_y = sim['spin2y']
            self.s_ns_z = sim['spin2z']

            # Set BH
            self.m_bh = sim['mass1']
            self.s_bh_x = sim['spin1x']
            self.s_bh_y = sim['spin1y']
            self.s_bh_z = sim['spin1z']
            
            self.s_bh = np.sqrt(self.s_bh_x**2 + self.s_bh_y**2 + self.s_bh_z**2)
            self.s_bh_tilt = np.arccos(self.s_bh_z/self.s_bh)
        
        else:
            raise NotImplementedError("Input of type {} not yet supported!".format(type(sim)))
            
        #print("BH = {:.2f}, NS = {:.2f}".format(self.m_bh, self.m_ns))
        
        if eos in list(lalsim.SimNeutronStarEOSNames):
            eos_obj = lalsim.SimNeutronStarEOSByName(eos)
            self.eos = lalsim.CreateSimNeutronStarFamily(eos_obj)

            # Get limiting NS masses and ensure valid input
            m_max = lalsim.SimNeutronStarMaximumMass(self.eos)
            self.m_max = m_max / lal.MSUN_SI
            assert(self.m_ns < self.m_max)
            m_min = lalsim.SimNeutronStarFamMinimumMass(self.eos)
            self.m_min = m_min / lal.MSUN_SI
            assert(self.m_ns > self.m_min)

            # Get NS radius
            self.r_ns = lalsim.SimNeutronStarRadius(self.m_ns * lal.MSUN_SI, self.eos)

            # NS tidal deformability
            self.compactness = self.get_compactness()
            self.love = lalsim.SimNeutronStarLoveNumberK2(self.m_ns * lal.MSUN_SI, self.eos)
            self.lamb = 2. / 3. * self.love * self.compactness**-5

        else:
            eos_func, mr_func, self.m_max, self.m_min = choose_func(eos)
            assert(self.m_ns < self.m_max)
            assert(self.m_ns > self.m_min)
            self.lamb = eos_func(self.m_ns)
            self.r_ns = mr_func(self.m_ns) * 1e3
            self.compactness = self.get_compactness()
        
        #self.cutoff_frequency = self.get_cutoff_frequency()
        #self.disruption_q = self.get_disruption_q()
        
    def get_cutoff_frequency(self):
        """
        Calculates the cut-off frequency following
        Pannarale et al. (2015) - doi:10.1103/PhysRevD.92.081504; arXiv:1509.06209
        Pannarale et al. (2015) - doi:10.1103/PhysRevD.92.084050; arXiv:1509.00512
        """
        f = np.zeros((4,4,4))
        f[0,0,0] = 1.38051e-1
        f[1,0,0] = -2.36698
        f[0,1,0] = -3.07791e-2
        f[0,0,1] = 3.06474e-2
        f[2,0,0] = 1.19668e1
        f[0,2,0] = 1.81262e-3
        f[0,0,2] = 4.31813e-2
        f[1,1,0] = 2.89424e-1
        f[1,0,1] = -1.61434e-1
        f[0,1,1] = 9.30676e-4
        f[3,0,0] = -1.46271e1
        f[0,3,0] = -6.89872e-5
        f[0,0,3] = -2.29830e-3
        f[2,1,0] = 2.73922e-1
        f[1,2,0] = -4.69093e-3
        f[2,0,1] = 1.75728e-1
        f[1,0,2] = -2.04964e-1
        f[0,2,1] = 5.52098e-4
        f[0,1,2] = -5.79629e-3
        f[1,1,1] = -9.09280e-2
        
        # Coefficients given in units where total mass = 1, need conversion factor
        f *= (self.m_bh + self.m_ns) * lal.MSUN_SI * lal.G_SI / lal.C_SI**2
        
        f_cutoff = 0
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    if i + j + k <= 3:
                        f_cutoff += f[i,j,k] * self.compactness**i * (self.m_bh / self.m_ns)**j * self.s_bh**k
        
        return f_cutoff
        
    def get_disruption_q(self):
        """
        Calculates the disruption limit mass ratio following
        Pannarale et al. (2015) - doi:10.1103/PhysRevD.92.081504; arXiv:1509.06209
        Pannarale et al. (2015) - doi:10.1103/PhysRevD.92.084050; arXiv:1509.00512
        """
        a = np.zeros((4,4))
        a[0,0] = 4.59676e1
        a[1,0] = -6.68812e2
        a[0,1] = 2.78668e1
        a[2,0] = 3.56791e3
        a[1,1] = -2.79252e2
        a[0,2] = 1.07053e1
        a[3,0] = -6.69647e3
        a[2,1] = 7.55858e2
        a[1,2] = -5.51855e1
        a[0,3] = 4.01679e-1
        
        qd = 0
        for i in range(4):
            for j in range(4):
                if i + j <= 3:
                    qd += a[i,j] * self.compactness**i * self.s_bh**j
        
        return qd
    
    def get_compactness(self):
        return lal.G_SI * self.m_ns * lal.MSUN_SI / (self.r_ns * lal.C_SI**2)
    
    def risco(self):
        z1 = 1 + (1 - self.s_bh**2)**(1./3.) * ((1 + self.s_bh)**(1./3.) + (1 - self.s_bh)**(1./3.))
        z2 = np.sqrt(3 * self.s_bh**2 + z1**2)
        
        return 3 + z2 - np.sign(self.s_bh) * np.sqrt((3 - z1) * (3 + z1 + 2 * z2))
    
    def risco_root(self, tol=1e-5):
        """
        Find R_ISCO by solving to find the roots of:
        (r * (r - 6))**2 - chi**2 * (2 * r * (3 * r + 14) - 9 * chi**2)
        which expands to:
        r**4
            - 12 * r**3
                + 6 * (6 - chi**2) * r**2
                    - 28 * chi**2 * r
                        + 9 * chi**4
        """
        print("WARNING: This function does not yet discern between "
              "roots corresponding to posititve and negative spins.")
        a = 6
        b = 1 + np.sqrt(3) + np.sqrt(3 + 2 * np.sqrt(3))
        
        p = np.poly1d([1, -12, 6*(6-self.s_bh**2), -28*self.s_bh**2, 9*self.s_bh**4])
        
        R = [r.real for r in np.roots(p) if r.imag < tol]
        
        return R
    
    def risco_root_sanity(self):
        r, chi = spy.symbols("r, chi")
        p = (r * (r - 6))**2 - self.s_bh**2 * (2 * r * (3 * r + 14) - 9 * self.s_bh**2)
        p2 = (r * (r - 6))**2 - chi**2 * (2 * r * (3 * r + 14) - 9 * chi**2)
        
        return p2.as_poly(), spy.nroots(p)
    
    def risso_polar(self, tol=1e-5):
        """
        Find the polar R_ISSO by solving to find the roots of:
        r**3 * (r**2 * (r - 6) + chi**2 * (3 * r + 4)) 
           + chi**4 * (3 * r * (r - 2) + chi**2)
        which expands to:
        r**6
            - 6 * r**5
                + 3 * chi**2 * r**4
                    + 4 * chi**2 * r**3
                        + 3 * chi**4 * r**2
                            - 6 * chi**4 * r
                                + chi**6
        The value must lie between 6 and 1 + sqrt(3) + sqrt(3 + 2 * sqrt(3))
        """
        a = 6
        b = 1 + np.sqrt(3) + np.sqrt(3 + 2 * np.sqrt(3))
        
        p = np.poly1d([1, -6, 3*self.s_bh**2, 4*self.s_bh**2, 3*self.s_bh**4, -6*self.s_bh**4, self.s_bh**6])
        
        R = [r.real for r in np.roots(p) if r.real - tol <= a and r.real + tol >= b and r.imag < tol]
        assert len(R) == 1
        
        return R[0]
    
    def risso_polar_sanity(self):
        r, chi = spy.symbols("r, chi")
        p2 = r**3 * (r**2 * (r - 6) + chi**2 * (3 * r + 4)) + chi**4 * (3 * r * (r - 2) + chi**2)
        print(p2.as_poly())
        
        p = r**3 * (r**2 * (r - 6) + self.s_bh**2 * (3 * r + 4)) + self.s_bh**4 * (3 * r * (r - 2) + self.s_bh**2)
        
        return spy.nroots(p)
        
    def risso(self, tol=1e-5):
        r = spy.symbols("r")
        
        c = np.cos(self.s_bh_tilt)
        
        x1 = self.s_bh**2 * (3 * self.s_bh**2 + 4 * r *(2 * r - 3)) + r**2 * (15 * r * (r - 4) + 28)
        x2 = 6 * r**4 * (r**2 - 4)
        x = self.s_bh**2 * x1 - x2
        
        y1 = self.s_bh**4 * (self.s_bh**4 + r**2 * (7 * r * (3 * r - 4) + 36))
        y2 = 6 * r * (r - 2) * (self.s_bh**6 + 2 * r**3 * (self.s_bh**2 * (3 * r + 2) + 3 * r**2 * (r - 2)))
        y = y1 + y2
        
        z = (r * (r - 6))**2 - self.s_bh**2 * (2 * r * (3 * r + 14) - 9 * self.s_bh**2)
        
        p = r**8 * z + self.s_bh**2 * (1 - c**2) * (self.s_bh**2 * (1 - c**2) * y - 2 * r**4 * x)
        roots = np.array(spy.nroots(p, maxsteps=int(1e4)), dtype="complex128")
        
        a = self.risso_polar()
        b = self.risco()
        
        R = [r.real for r in roots if r.real - tol <= max(a, b) and r.real + tol >= min(a, b) and r.imag < tol]
        try:
            assert len(R) == 1
        except AssertionError:
            print(R)
        
        return R[0]
    
    def remnant_mass(self, a=0.406, b=0.139, c=0.255, d=1.761):
        if self.s_bh_tilt == 0:
            R = self.risco()
        elif self.s_bh_tilt == np.pi/2 or self.s_bh_tilt == -np.pi/2:
            R = self.risso_polar()
        else:
            R = self.risso()

        eta = (self.m_bh / self.m_ns) / (1 + (self.m_bh / self.m_ns))**2
        p1 = a * (1 - 2 * self.compactness) / eta**(1./3.)
        p2 = b * R * self.compactness / eta
        
        return self.m_ns * (np.maximum(p1 - p2 + c, 0))**d
    
    def remnant_mass_lambda(self, a=0.308, b=0.124, c=0.283, d=1.536):
        if self.s_bh_tilt == 0:
            R = self.risco()
        elif self.s_bh_tilt == np.pi/2 or self.s_bh_tilt == -np.pi/2:
            R = self.risso_polar()
        else:
            R = self.risso()

        rho = (15 * self.lamb)**-0.2
        eta = (self.m_bh / self.m_ns) / (1 + (self.m_bh / self.m_ns))**2
        p1 = a * (1 - 2 * rho) / eta**(1./3.)
        p2 = b * R * rho / eta
        
        return self.m_ns * (np.maximum(p1 - p2 + c, 0))**d