"""Simulation of our solar system"""

import numpy as np
from matplotlib import pyplot as plt
import matplotlib
from copy import deepcopy
import matplotlib.animation as animation


class Planet:
    """
    Units:
    Mass in earth mass,
    Distance in AU,
    Velocity in AU/day.
    """
    G = 8.8874e-10  # Gravitational constant in AU^3 / (ME * days^2)

    # G = 1.1840e-4  # Gravitational constant in AU^3 / (ME * years^2)

    def __init__(self, name, mass, x, y, vx, vy, color, in_or_out):
        self.name = name
        self.mass = mass
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.color = color
        self.prev_x = []
        self.prev_y = []
        self.pos = in_or_out

    def r(self, planet):
        """Calculate distance to other planet."""
        r2 = (self.x - planet.x) ** 2 + (self.y - planet.y) ** 2
        return np.sqrt(r2)

    def pot_energy(self, planets):
        """Calculate potential energy of self."""
        U = 0
        for p in planets:
            U -= self.G * self.mass * p.mass / self.r(p)
        return U

    def kin_energy(self):
        """Calculate kinetic energy of self."""
        return 0.5 * self.mass * (self.vx ** 2 + self.vy ** 2)

    def energy(self, planets):
        """Calculate total energy"""
        return self.pot_energy(planets) + self.kin_energy()

    def _force(self, planet):
        """Help class, calculate gravitational force."""
        return self.G * self.mass * planet.mass / (self.r(planet) ** 2)

    def force(self, planets):
        """Calculate the gravitational force from other planets. Returns x-component, y-component as list."""
        fx = 0
        fy = 0
        for p in planets:
            if self.r(p) == 0:
                pass
            else:
                F = self._force(p)  # Force
                dx = p.x - self.x
                dy = p.y - self.y
                theta = np.arctan2(dy, dx)
                fx += F * np.cos(theta)
                fy += F * np.sin(theta)
        return [fx, fy]


def unit_conv(m_s=None, AU_d=None, mass_kg=None):
    AU = 149597870700
    day = 3600 * 24
    M_earth = 5.972e24
    if m_s is not None:
        return m_s * day / AU
    if AU_d is not None:
        return AU_d * AU / day
    if mass_kg is not None:
        return mass_kg / M_earth


def planets_positions_colors(planets):
    """Returns two lists (x and y) of the planets positions."""
    x = []
    y = []
    colors = []
    for p in planets:
        x.append(p.x)
        y.append(p.y)
        colors.append(p.color)
    return x, y, colors


def planets_prev_positions_colors(planets):
    x = []
    y = []
    colors = []
    for p in planets:
        x.append(p.prev_x)
        y.append(p.prev_y)
        colors.append(p.color)
    return x, y, colors


def plot_planets(x, y, colors):
    for i in range(len(x)):
        if colors[i] == "yellow":
            plt.plot(x[i], y[i], '.', color=colors[i], markersize=20)
        elif colors[i] == "silver":
            plt.plot(x[i], y[i], '.', color=colors[i], markersize=10)
        else:
            plt.plot(x[i], y[i], '.', color=colors[i], markersize=20)


def plot_orbits(x, y, colors):
    for i in range(len(x)):
        if colors[i] == "yellow":
            pass
        else:
            plt.plot(x[i], y[i], color=colors[i], linewidth=0.8)


def verlet(planets, dt):
    """Velocity Verlet method with time step dt."""
    updated_planets = []
    for p in planets:
        p.prev_x.append(p.x)
        p.prev_y.append(p.y)
        upd = deepcopy(p)
        fx, fy = p.force(planets)
        upd.x += upd.vx * dt + 0.5 * fx * dt * dt / p.mass
        upd.y += upd.vy * dt + 0.5 * fy * dt * dt / p.mass
        updated_planets.append(upd)
    i = 0
    updated_planets2 = []
    for upd in updated_planets:
        upd2 = deepcopy(upd)
        fx, fy = planets[i].force(planets)
        fx_new, fy_new = upd.force(updated_planets)
        upd2.vx += 0.5 * (fx + fx_new) * dt / upd.mass
        upd2.vy += 0.5 * (fy + fy_new) * dt / upd.mass
        updated_planets2.append(upd2)
        i += 1
    return updated_planets2


def symplectic_euler(planets, dt):
    """Symplectic Euler method (also known as Euler-Cromer) with time step dt."""
    updated_planets = []
    for p in planets:
        p.prev_x.append(p.x)
        p.prev_y.append(p.y)
        fx, fy = p.force(planets)
        upd = deepcopy(p)
        upd.vx += dt * fx / p.mass
        upd.vy += dt * fy / p.mass
        upd.x += upd.vx * dt
        upd.y += upd.vy * dt
        updated_planets.append(upd)
    return updated_planets


def euler(planets, dt):
    updated_planets = []
    for p in planets:
        p.prev_x.append(p.x)
        p.prev_y.append(p.y)
        fx, fy = p.force(planets)
        upd = deepcopy(p)
        upd.x += upd.vx * dt
        upd.y += upd.vy * dt
        upd.vx += fx * dt / p.mass
        upd.vy += fy * dt / p.mass
        updated_planets.append(upd)
    return updated_planets


"""
Simulation of the solar system. Initial values retrieved 2019-01-09.
Masses (in earth masses): https://en.wikipedia.org/wiki/Earth_mass
Velocities (in AU/day): https://ssd.jpl.nasa.gov/horizons.cgi
"""

sun = Planet("sun", 332946, 0, 0, 0, 0, "yellow", "in")
mercury = Planet("mercury", 0.0553, -1.734e-1, -4.312e-1, 2.045e-2, -9.111e-3, "grey", "in")
venus = Planet("venus", 0.815, -6.396e-1, 3.254e-1, -9.257e-3, -1.812e-2, "orange", "in")
earth = Planet("earth", 1, -3.064e-1, 9.344e-1, -1.663e-2, -5.431e-3, "b", "in")
mars = Planet("mars", 0.107, 1.017e0, 1.052e0, -9.528e-3, 1.092e-2, "r", "in")
jupiter = Planet("jupiter", 317.8, -2.079e0, -4.926e0, 6.867e-3, -2.582e-3, "goldenrod", "out")
saturn = Planet("saturn", 95.2, 2.001e0, -9.858e0, 5.168e-3, 1.090e-3, "khaki", "out")
uranus = Planet("uranus", 14.5, 1.700e1, 1.027e1, -2.056e-3, 3.181e-3, "skyblue", "out")
neptune = Planet("neptune", 17.1, 2.899e1, -7.464e0, 7.687e-4, 3.057e-3, "b", "out")

planets = [sun, mercury, venus, earth, mars, jupiter, saturn, uranus, neptune]

# Axis limits for inner and outer solar system
lim_in = 2
lim_out = 32

t = 0  # time
dt = 1  # In days

fig = plt.figure(figsize=(7.2, 6))
matplotlib.rcParams.update({'font.size': 15})
nsteps = 1


def animate(framenr, in_or_out):
    global planets, t
    for i in range(nsteps):
        # planets = symplectic_euler(planets, dt)
        planets = verlet(planets, dt)
        # planets = euler(planets, dt)
    fig.clear()
    if in_or_out == "i":
        plan = [p for p in planets if p.pos == "in"]
        lim = lim_in
    else:
        plan = [planets[0]] + [p for p in planets if p.pos == "out"]
        lim = lim_out
    x, y, colors = planets_positions_colors(plan)
    prev_x, prev_y, colors = planets_prev_positions_colors(plan)
    plt.axis([-lim, lim, -lim, lim])
    plt.xlabel("x [AU]")
    plt.ylabel("y [AU]")
    plot_planets(x, y, colors)
    plot_orbits(prev_x, prev_y, colors)
    plt.text(25, 25, str(t), color="white", fontsize=16)
    ax = plt.gca()
    ax.set_facecolor("black")

    t += 1


# Animate inner solar system
anim = animation.FuncAnimation(fig, animate, fargs=("i"), interval=100)

# Animate outer solar system
# anim = animation.FuncAnimation(fig, animate, fargs=("o"), interval=100)

plt.show()
