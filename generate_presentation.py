"""
Molecular Dynamics Simulation of Lennard-Jones Argon Fluid
Course: Computational Chemistry
Modules: PES (Module 4) -> Molecular Dynamics (Module 5) -> Thermodynamics (Module 3)
"""

import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------------------
# 1. Physical Constants and Argon Parameters
# -------------------------------------------------------------
SIGMA = 3.4e-10        # m (3.4 Angstroms)
EPSILON_KB = 119.8     # K (epsilon / k_B)
K_B = 1.380649e-23     # J/K
EPSILON = EPSILON_KB * K_B  # Joules
MASS_AR = 39.948 * 1.660539e-27  # kg (~6.63e-26 kg)
TIME_UNIT = np.sqrt(MASS_AR * (SIGMA**2) / EPSILON)  # ~2.15 ps

# -------------------------------------------------------------
# 2. Simulation Setup (Reduced Lennard-Jones Units)
# -------------------------------------------------------------
class LJSimulation:
    def __init__(self, n_particles=108, density=0.8442, target_temp=0.728, dt=0.004, r_cut=2.5):
        """
        Default parameters represent liquid Argon near triple point:
        density (rho*) = 0.8442 (~1.4 g/cm3)
        temperature (T*) = 0.728 (~87.2 K)
        """
        self.N = n_particles
        self.rho = density
        self.T_target = target_temp
        self.dt = dt
        self.r_cut = r_cut
        self.r_cut_sq = r_cut ** 2
        
        # Simulation Box Length
        self.L = (self.N / self.rho) ** (1.0 / 3.0)
        self.half_L = self.L / 2.0
        self.V = self.L ** 3
        
        # Initialize positions, velocities, and forces
        self.pos = self._init_fcc_lattice()
        self.vel = self._init_velocities()
        self.forces = np.zeros_like(self.pos)
        self.pot_energy = 0.0
        self.virial = 0.0
        
        # Calculate initial forces
        self._compute_forces()
        
    def _init_fcc_lattice(self):
        """Place particles on a Face-Centered Cubic (FCC) lattice to avoid steric clashes."""
        num_cells = int(np.ceil((self.N / 4.0) ** (1.0 / 3.0)))
        a = self.L / num_cells
        basis = np.array([
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.5, 0.0, 0.5],
            [0.0, 0.5, 0.5]
        ]) * a
        
        positions = []
        for x in range(num_cells):
            for y in range(num_cells):
                for z in range(num_cells):
                    origin = np.array([x, y, z]) * a
                    for b in basis:
                        if len(positions) < self.N:
                            positions.append(origin + b)
        return np.array(positions)

    def _init_velocities(self):
        """Sample velocities from Maxwell-Boltzmann distribution and zero net momentum."""
        vel = np.random.normal(0.0, np.sqrt(self.T_target), (self.N, 3))
        # Remove center-of-mass drift
        vel -= np.mean(vel, axis=0)
        # Rescale to target temperature
        current_temp = np.sum(vel**2) / (3.0 * self.N)
        vel *= np.sqrt(self.T_target / current_temp)
        return vel

    def _compute_forces(self):
        """Calculate pairwise Lennard-Jones forces with Minimum Image Convention."""
        self.forces.fill(0.0)
        self.pot_energy = 0.0
        self.virial = 0.0
        
        for i in range(self.N - 1):
            # Displacement vectors
            dr = self.pos[i] - self.pos[i + 1:]
            
            # Minimum Image Convention (Periodic Boundary Condition)
            dr = dr - self.L * np.round(dr / self.L)
            
            r_sq = np.sum(dr**2, axis=1)
            mask = r_sq < self.r_cut_sq
            
            r_sq_filtered = r_sq[mask]
            dr_filtered = dr[mask]
            
            inv_r2 = 1.0 / r_sq_filtered
            inv_r6 = inv_r2 ** 3
            inv_r12 = inv_r6 ** 2
            
            # Force magnitude scalar: F(r)/r = 48 * (inv_r14 - 0.5 * inv_r8)
            f_over_r = 48.0 * inv_r2 * (inv_r12 - 0.5 * inv_r6)
            
            # Vector forces
            f_vectors = dr_filtered * f_over_r[:, np.newaxis]
            
            self.forces[i] += np.sum(f_vectors, axis=0)
            self.forces[i + 1:][mask] -= f_vectors
            
            # Potential energy: 4 * (1/r^12 - 1/r^6) - shift at cutoff
            e_cut = 4.0 * ((1.0 / self.r_cut**12) - (1.0 / self.r_cut**6))
            pair_pot = 4.0 * (inv_r12 - inv_r6) - e_cut
            self.pot_energy += np.sum(pair_pot)
            
            # Virial sum: sum(r_ij . F_ij)
            self.virial += np.sum(f_over_r * r_sq_filtered)

    def step_velocity_verlet(self):
        """Velocity-Verlet symplectic time integration."""
        # 1. Half-step velocity update + full step position update
        self.pos += self.vel * self.dt + 0.5 * self.forces * (self.dt ** 2)
        # Apply Periodic Boundary Conditions (wrap particles inside box)
        self.pos = np.mod(self.pos, self.L)
        
        # Store old forces
        old_forces = np.copy(self.forces)
        
        # 2. Compute new forces at new positions
        self._compute_forces()
        
        # 3. Complete velocity update
        self.vel += 0.5 * (old_forces + self.forces) * self.dt

    def compute_rdf(self, n_bins=100, max_r=None):
        """Compute the Radial Distribution Function g(r)."""
        if max_r is None:
            max_r = self.half_L
        edges = np.linspace(0.0, max_r, n_bins + 1)
        dr = edges[1] - edges[0]
        hist = np.zeros(n_bins)
        
        for i in range(self.N - 1):
            d_pos = self.pos[i] - self.pos[i + 1:]
            d_pos = d_pos - self.L * np.round(d_pos / self.L)
            distances = np.linalg.norm(d_pos, axis=1)
            hist += np.histogram(distances, bins=edges)[0]
            
        r_vals = 0.5 * (edges[:-1] + edges[1:])
        shell_volumes = 4.0 * np.pi * (r_vals ** 2) * dr
        ideal_counts = self.rho * shell_volumes * (self.N / 2.0)
        gr = hist / ideal_counts
        return r_vals, gr

    def run(self, steps=1000, sample_interval=5):
        """Run the simulation and log energy, temperature, and pressure."""
        times = []
        e_kin_list = []
        e_pot_list = []
        e_tot_list = []
        temp_list = []
        press_list = []
        
        print(f"Starting MD Simulation of {self.N} particles...")
        print(f"Density rho* = {self.rho:.4f}, Target T* = {self.T_target:.3f}, Timestep dt* = {self.dt}")
        
        for step in range(steps):
            self.step_velocity_verlet()
            
            # Simple Berendsen-style velocity scaling during initial equilibration (first 200 steps)
            if step < 200 and step % 10 == 0:
                cur_t = np.sum(self.vel**2) / (3.0 * self.N)
                self.vel *= np.sqrt(self.T_target / cur_t)
                
            if step % sample_interval == 0:
                kin_energy = 0.5 * np.sum(self.vel**2)
                tot_energy = kin_energy + self.pot_energy
                temp = (2.0 * kin_energy) / (3.0 * self.N)
                
                # Thermodynamic pressure from Virial theorem: P = rho*T + W/(3V)
                pressure = (self.rho * temp) + (self.virial / (3.0 * self.V))
                
                times.append(step * self.dt)
                e_kin_list.append(kin_energy / self.N)
                e_pot_list.append(self.pot_energy / self.N)
                e_tot_list.append(tot_energy / self.N)
                temp_list.append(temp)
                press_list.append(pressure)
                
            if step % 200 == 0:
                print(f"Step {step:4d}/{steps} | T* = {temp_list[-1]:.3f} | E_tot* = {e_tot_list[-1]:.4f}")
                
        return {
            "time": np.array(times),
            "e_kin": np.array(e_kin_list),
            "e_pot": np.array(e_pot_list),
            "e_tot": np.array(e_tot_list),
            "temp": np.array(temp_list),
            "pressure": np.array(press_list)
        }

# -------------------------------------------------------------
# 3. Execution and Plot Generation
# -------------------------------------------------------------
if __name__ == "__main__":
    np.random.seed(42)
    sim = LJSimulation(n_particles=108, density=0.8442, target_temp=0.728, dt=0.004)
    data = sim.run(steps=1000, sample_interval=2)
    
    # Compute Radial Distribution Function g(r)
    r_vals, gr = sim.compute_rdf(n_bins=80)
    
    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Lennard-Jones Fluid (Argon) Molecular Dynamics Simulation", fontsize=15, fontweight="bold")
    
    # 1. Energy Conservation Plot
    t_ps = data["time"] * TIME_UNIT * 1e12  # Picoseconds
    axes[0, 0].plot(t_ps, data["e_tot"], 'k-', label="Total Energy (E*)", linewidth=1.8)
    axes[0, 0].plot(t_ps, data["e_pot"], 'b--', label="Potential Energy (U*)", alpha=0.7)
    axes[0, 0].plot(t_ps, data["e_kin"], 'r--', label="Kinetic Energy (K*)", alpha=0.7)
    axes[0, 0].set_title("Energy Evolution & Conservation (Microcanonical Test)")
    axes[0, 0].set_xlabel("Time (ps)")
    axes[0, 0].set_ylabel("Energy per Particle (ε)")
    axes[0, 0].grid(True, linestyle="--", alpha=0.5)
    axes[0, 0].legend()
    
    # 2. Temperature Fluctuations
    t_kelvin = data["temp"] * EPSILON_KB
    axes[0, 1].plot(t_ps, t_kelvin, 'darkorange', label="Instantaneous Temp")
    axes[0, 1].axhline(y=0.728 * EPSILON_KB, color='red', linestyle=':', label="Target T (~87.2 K)")
    axes[0, 1].set_title("Equilibration and Temperature Fluctuations")
    axes[0, 1].set_xlabel("Time (ps)")
    axes[0, 1].set_ylabel("Temperature (K)")
    axes[0, 1].grid(True, linestyle="--", alpha=0.5)
    axes[0, 1].legend()
    
    # 3. Pressure Fluctuations
    p_bar = data["pressure"] * (EPSILON / (SIGMA**3)) * 1e-5  # Convert to bar
    axes[1, 0].plot(t_ps, p_bar, 'purple')
    axes[1, 0].set_title("Virial Equation of State: Pressure Fluctuations")
    axes[1, 0].set_xlabel("Time (ps)")
    axes[1, 0].set_ylabel("Pressure (bar)")
    axes[1, 0].grid(True, linestyle="--", alpha=0.5)
    
    # 4. Radial Distribution Function g(r)
    r_angstrom = r_vals * SIGMA * 1e10
    axes[1, 1].plot(r_angstrom, gr, 'teal', linewidth=2)
    axes[1, 1].axhline(y=1.0, color='gray', linestyle='--')
    axes[1, 1].set_title("Radial Distribution Function g(r) [Liquid Structure]")
    axes[1, 1].set_xlabel("r (Å)")
    axes[1, 1].set_ylabel("g(r)")
    axes[1, 1].grid(True, linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    plt.savefig("simulation_results.png", dpi=300)
    print("Simulation complete! Results saved to 'simulation_results.png'.")