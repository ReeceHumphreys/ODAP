import numpy as np
import scipy
from enum import IntEnum
import lognorm as lognorm

debris_category = IntEnum('Category', 'rb sc soc')


""" ----------------- Mean ----------------- """
def make_mean_AM(debris_type):

    def RB_mean_AM(lambda_c):

        mean_am_1 = np.empty_like(lambda_c)
        mean_am_2 = np.empty_like(lambda_c)

        mean_am_1[lambda_c<=-0.5] = -0.45
        I = (lambda_c>-0.5) & (lambda_c<0)
        mean_am_1[I] = -0.45 - (0.9*(lambda_c[I] +0.5))
        mean_am_1[lambda_c>=0] = -0.9

        mean_am_2.fill(-0.9)

        return np.array([mean_am_1,mean_am_2])

    def SC_mean_AM(lambda_c):
        mean_am_1 = np.empty_like(lambda_c)
        mean_am_2 = np.empty_like(lambda_c)

        mean_am_1[lambda_c<=-1.1] = -0.6
        I = (lambda_c>-1.1) & (lambda_c<0)
        mean_am_1[I] = -0.6 - (0.318*(lambda_c[I] +1.1))
        mean_am_1[lambda_c>=0] = -0.95

        mean_am_2[lambda_c<=-0.7] = -1.2
        I = (lambda_c>-0.7) & (lambda_c<-0.1)
        mean_am_2[I] = -1.2 - (1.333*(lambda_c[I] + 0.7))
        mean_am_2[lambda_c>=-0.1] = -2.0

        return np.array([mean_am_1,mean_am_2])

    def SOC_mean_AM(lambda_c):

        mean_am_1 = np.empty_like(lambda_c)
        mean_am_2 = np.empty_like(lambda_c)

        mean_am_1[lambda_c<=-1.75] = -0.3
        I = (lambda_c>-1.75) & (lambda_c<-1.25)
        mean_am_1[I] = -0.3 - (1.4*(lambda_c[I] +1.75))
        mean_am_1[lambda_c>=-1.25] = -1.0

        mean_am_2.fill(0)
        return np.array([mean_am_1,mean_am_2])

    if debris_type == debris_category.rb:
        return RB_mean_AM
    elif debris_type == debris_category.sc:
        return SC_mean_AM
    else:
        return SOC_mean_AM

""" ----------------- Standard Deviation ----------------- """

def make_standard_dev_AM(debris_type):

    def RB_std_dev_AM(lambda_c):

        std_dev_1 = np.empty_like(lambda_c)
        std_dev_2 = np.empty_like(lambda_c)

        std_dev_1.fill(0.55)

        std_dev_2[lambda_c<=-1.0] = 0.28
        I = (lambda_c>-1.0) & (lambda_c<0.1)
        std_dev_2[I] = 0.29 - (0.1636*(lambda_c[I] +1))
        std_dev_2[lambda_c>=0.1] = 0.1

        return np.array([std_dev_1,std_dev_1])


    def SC_std_dev_AM(lambda_c):

        std_dev_1 = np.empty_like(lambda_c)
        std_dev_2 = np.empty_like(lambda_c)

        std_dev_1[lambda_c<=-1.3] = 0.1
        I = (lambda_c>-1.3) & (lambda_c<-0.3)
        std_dev_1[I] = 0.1 + (0.2*(lambda_c[I] +1.3))
        std_dev_1[lambda_c>=-0.3] = 0.3

        std_dev_2[lambda_c<=-0.5] = 0.5
        I = (lambda_c>-0.5) & (lambda_c<-0.3)
        std_dev_2[I] = 0.5 - ((lambda_c[I] + 0.5))
        std_dev_2[lambda_c>=-0.3] = 0.3

        return np.array([std_dev_1,std_dev_1])


    def SOC_std_dev_AM(lambda_c):
        std_dev_1 = np.empty_like(lambda_c)
        std_dev_2 = np.empty_like(lambda_c)

        std_dev_1[lambda_c<=-3.5] = 0.2
        I = (lambda_c>-3.5)
        std_dev_1[I] = 0.2 + (0.1333*(lambda_c[I] +3.5))

        std_dev_2.fill(0)

        return np.array([std_dev_1,std_dev_1])

    if debris_type == debris_category.rb:
        return RB_std_dev_AM
    elif debris_type == debris_category.sc:
        return SC_std_dev_AM
    else:
        return SOC_std_dev_AM

""" ----------------- Alpha ----------------- """
def alpha_AM(lambda_c, debris_type):
    def RB_alpha_AM(lambda_c):
        alpha = 1
        # dev1 rule
        if lambda_c <= -1.4:
            alpha = 1
        elif (lambda_c > -1.4 and lambda_c < 0):
            alpha = 1 - (0.3571*(lambda_c + 1.4))
        else:
            alpha = 0.5
        return alpha

    def SC_alpha_AM(lambda_c):
        alpha = 1
        # dev1 rule
        if lambda_c <= -1.95:
            alpha = 0
        elif (lambda_c > -1.95 and lambda_c < 0.55):
            alpha = 0.3 + (0.4*(lambda_c + 1.2))
        else:
            alpha = 1
        return alpha

    def SOC_alpha_AM(lambda_c):
        # Is not used by SOC, for saftey returning 1
        alpha = 1
        return alpha

    if debris_type == debris_category.rb:
        return RB_alpha_AM(lambda_c)
    elif debris_type == debris_category.sc:
        return SC_alpha_AM(lambda_c)
    else:
        return SOC_alpha_AM(lambda_c)

alpha_AM = np.vectorize(alpha_AM)

""" ----------------- Distribution A/M ----------------- """
def distribution_AM(lambda_c, debris_type):

    N = len(lambda_c)
    lambda_c = np.array(lambda_c)

    mean_factory = make_mean_AM(debris_type)
    std_dev_factor = make_standard_dev_AM(debris_type)

    mean_preSwitch = np.array(mean_factory(lambda_c))
    std_dev_preSwitch = np.array(std_dev_factor(lambda_c))

    alpha = np.array(alpha_AM(lambda_c, debris_category.rb)) # This takes a long time
    switch = np.random.uniform(0,1, N)

    if debris_type == debris_category.rb or debris_type == debris_category.sc:

        means = np.empty(N)
        I,J = switch<alpha, switch>=alpha
        means[I] = mean_preSwitch[0, I]
        means[J] = mean_preSwitch[1, J]

        devs = np.empty(N)
        devs[I] = std_dev_preSwitch[0, I]
        devs[J] = std_dev_preSwitch[1, J]

        return np.random.normal(means, devs, N)

    else:
        means = mean_preSwitch[0]
        devs = std_dev_preSwitch[0]

        return np.random.normal(means, devs, N)

""" ----------------- Area ----------------- """
def avg_area(L_c):

    A = np.copy(L_c)
    I = A < 0.00167 #(m)
    A[I] = 0.540424 * A[I]**2
    I = A >= 0.00167 #(m)
    A[I] = 0.556945 * A[I]**2.0047077

    return A



""" ----------------- Mean ----------------- """
def mean_deltaV(kai, explosion):
    if explosion == True:
        return (0.2 * kai) + 1.85
    else:
        # Is a collision
        return (0.9 * kai) + 2.9

mean_deltaV = np.vectorize(mean_deltaV)

""" ----------------- Standard Deviation ----------------- """
def std_dev_deltaV():
    return 0.4
std_dev_deltaV = np.vectorize(std_dev_deltaV)

""" ----------------- Distribution delta V ----------------- """
def distribution_deltaV(chi, v_c, explosion=False):
    N = len(chi)
    mean = mean_deltaV(chi, True)
    dev  = std_dev_deltaV()
# print(np.mean(mean),dev)
    max_itr = 5000
    i = 0
    base = 10
    centered = np.random.normal(0, dev, N)
    I = np.nonzero(base**(mean+centered)>1.3*v_c)[0]
    n = len(I)
    while n != 0 and i<max_itr:
        centered[I] = np.random.normal(0, dev, n)
        #I = np.nonzero(base**(mean+centered)>1.3*v_c)[0]
        J = np.nonzero(base**(mean[I] + centered[I])>1.3*v_c)[0]
        I = I[J]
        n = len(I)
        i+=1
        print(n)
    centered[I] = np.log10(1.3*v_c)
    result = base**centered
    return result

""" ----------------- Unit vector delta V ----------------- """
def unit_vector(N):
    vectors = np.random.normal(0, 1, np.array([N, 3]))
    vectors /= np.sqrt((vectors**2).sum(axis=1))[:, None]
    return vectors

def velocity_vectors(N, target_velocity, velocities):
    unit_vectors = unit_vector(N)
    velocity_vectors = velocities[:, None] * unit_vectors
    return target_velocity + velocity_vectors

from numpy.random import uniform

""" ----------------- Num. Fragments & Char. Length----------------- """
def number_fragments(l_characteristic, m_target, m_projectile, v_impact, is_catastrophic, debris_type, explosion):
    # Defining reference Mass
    if explosion == True: # Can be multiplied by scaling factor S
        return 6*(l_characteristic)**(-1.6)
    else:
        m_ref = 0
        if is_catastrophic: m_ref = m_target + m_projectile
        else: m_ref = m_projectile * (v_impact)**2
        return 0.1 * (m_ref)**0.75 * l_characteristic**-1.71


def characteristic_lengths(m_target, m_projectile, v_impact, is_catastrophic, debris_type, explosion):
    bins = np.geomspace(0.001, 1, 100)
    N_fragments = number_fragments(bins, m_target, m_projectile, v_impact, is_catastrophic, debris_type, explosion)
    N_per_bin = np.array(N_fragments[:-1] - N_fragments[1:]).astype(int)
    L_c = np.concatenate([uniform(bins[i], bins[i+1], size=N_per_bin[i]) for i in range(len(bins) - 1)])

    return L_c

def fragmentation(m_target, m_projectile, v_impact, is_catastrophic, debris_type, explosion):

    prelim_L_c = characteristic_lengths(m_target, m_projectile, v_impact, is_catastrophic, debris_type, explosion)
    prelim_lambda_c = np.log10(prelim_L_c)
    prelim_areas = avg_area(prelim_L_c)
    prelim_AM = np.array(distribution_AM(prelim_lambda_c, debris_type))
    prelim_masses = prelim_areas / 10**prelim_AM

    if explosion == True:

        unaccounted_mass = m_target - np.sum(prelim_masses)

        n_large_deb = np.random.randint(2, 8) # Pick 2-8 pieces of deb > 1m to spread out the rest of the mass

       # Using 10**-4 to enure endpoints are not included
        mass_range = np.linspace(10**-4, (unaccounted_mass - 10**-4), 10**4) # Create mass range, will use `n_large_deb` to split into sections
        ranges = np.sort(np.random.choice(mass_range, n_large_deb - 1, replace=False))
        ranges = np.concatenate([[0],ranges,[unaccounted_mass]])

        # Note adding zero for subtraction to work (correct dims) then dropping it afterward
        mass_per_deb = np.concatenate((ranges[1:],np.zeros(1))) - ranges
        mass_per_deb = np.resize(mass_per_deb, mass_per_deb.size - 1)


        # For L_c > 1, A/M Distribution is basically deterministic, therefore will just use avg value, can get using np.inf
        assumed_AM_factory = make_mean_AM(debris_type)
        assumed_len = np.ones(mass_per_deb.shape)
        assumed_AM = assumed_AM_factory(assumed_len)

        # Each mean has two possible values, randomly pick one of them for each piece of deb
        AM_choices = np.random.choice([0,1], len(mass_per_deb), replace=True)
        assumed_AM = 10**np.array([assumed_AM[AM_choices[i], i] for i in range(assumed_AM.shape[1])])

        # mass * AM = A(L_c), therefore can reverse Area function for L_c
        area = mass_per_deb * assumed_AM
        found_L_c = np.sort((area / 0.556945)**(1/2.0047077)) # Inversing the Area function defined above
        found_lambda_c = np.log10(found_L_c)
        found_areas = avg_area(found_L_c)

        found_AM = np.array(distribution_AM(found_lambda_c, debris_type))
        found_masses = found_areas / assumed_AM # Using assumed A/M since A/M is a distribution and could get diff values.

        L_c = np.concatenate([prelim_L_c, found_L_c])
        areas = np.concatenate([prelim_areas, found_areas])
        masses = np.concatenate([prelim_masses, found_masses])
        AM = np.concatenate([prelim_AM, assumed_AM])

        return L_c, areas, masses, AM
    else:
        # Is a collision
        if is_catastrophic == True:

            unaccounted_mass = (m_target + m_projectile) - np.sum(prelim_masses)
            # Put the rest of the mass in many fragments in last bin
            deposit_bin = (np.geomspace(0.001, 1, 100)[-1] + np.geomspace(0.001, 1, 100)[-2])/2


            n_large_deb = np.random.randint(15, 50) # Pick 2-8 pieces of deb > 1m to spread out the rest of the mass

           # Using 10**-4 to enure endpoints are not included
            mass_range = np.linspace(10**-4, (unaccounted_mass - 10**-4), 10**4) # Create mass range, will use `n_large_deb` to split into sections
            ranges = np.sort(np.random.choice(mass_range, n_large_deb - 1, replace=False))
            ranges = np.concatenate([[0],ranges,[unaccounted_mass]])

            # Note adding zero for subtraction to work (correct dims) then dropping it afterward
            found_masses = np.concatenate((ranges[1:],np.zeros(1))) - ranges
            found_masses = np.resize(found_masses, found_masses.size - 1)

            found_L_c = np.ones_like(found_masses) * deposit_bin
            found_areas = avg_area(found_L_c)
            found_AM = found_areas / found_masses

            L_c = np.concatenate([prelim_L_c, found_L_c])
            areas = np.concatenate([prelim_areas, found_areas])
            masses = np.concatenate([prelim_masses, found_masses])
            AM = np.concatenate([prelim_AM, found_AM])

            return L_c, areas, masses, AM

        else:
            # Is a non catastrophic collision, Deposit remaining mass in 1 large piece of deb
            unaccounted_mass = np.array([(m_target + m_projectile) - np.sum(prelim_masses)])

            # For L_c > 1, A/M Distribution is basically deterministic, therefore will just use avg value, can get using np.inf
            assumed_AM_factory = make_mean_AM(debris_type)
            assumed_len = np.ones(unaccounted_mass.shape)
            assumed_AM = assumed_AM_factory(assumed_len)

            # Each mean has two possible values, randomly pick one of them for each piece of deb
            AM_choices = np.random.choice([0,1], len(unaccounted_mass), replace=True)
            assumed_AM = 10**np.array([assumed_AM[AM_choices[i], i] for i in range(assumed_AM.shape[1])])

             # mass * AM = A(L_c), therefore can reverse Area function for L_c
            area = unaccounted_mass * assumed_AM
            found_L_c = np.sort((area / 0.556945)**(1/2.0047077)) # Inversing the Area function defined above
            found_lambda_c = np.log10(found_L_c)
            found_areas = avg_area(found_L_c)

            L_c = np.concatenate([prelim_L_c, found_L_c])
            areas = np.concatenate([prelim_areas, found_areas])
            masses = np.concatenate([prelim_masses, unaccounted_mass])
            AM = np.concatenate([prelim_AM, assumed_AM])

            return L_c, areas, masses, AM
