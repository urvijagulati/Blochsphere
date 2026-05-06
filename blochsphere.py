
import numpy as np
from vispy import scene
from vispy.app import Timer

class Qubit:
    def __init__(self):
        # |ψ⟩ = alpha|0⟩ + beta|1⟩
        # stored as a 2-element complex numpy array
        self.state = np.array([1+0j, 0+0j])

    def bloch_vector(self):
        a, b = self.state
        x = 2 * np.real(np.conj(a) * b)
        y = 2 * np.imag(np.conj(a) * b)
        z = (np.abs(a)**2 - np.abs(b)**2)
        return np.array([x, y, z], dtype=float)

    def apply(self, gate: np.ndarray):
        self.state = gate @ self.state
        # renormalize to fight float drift
        self.state /= np.linalg.norm(self.state)

    def reset(self):
        self.state = np.array([1+0j, 0+0j])

    def set_from_angles(self, theta, phi):
        """Set state from Bloch sphere angles θ (polar), φ (azimuthal)."""
        self.state = np.array([
            np.cos(theta / 2),
            np.exp(1j * phi) * np.sin(theta / 2)
        ])

    @property
    def prob_zero(self):
        return float(np.abs(self.state[0])**2)

    @property
    def prob_one(self):
        return float(np.abs(self.state[1])**2)


def add_sphere(view):
    
    # Latitude rings (horizontal)
    for lat in np.linspace(-75, 75, 7):
        r = np.cos(np.radians(lat))
        z = np.sin(np.radians(lat))
        t = np.linspace(0, 2*np.pi, 80)
        pts = np.column_stack([
            r * np.cos(t),
            r * np.sin(t),
            np.full_like(t, z)
        ])
        scene.visuals.Line(
            pos=pts, color=(0.6, 0.7, 0.8, 0.4),
            connect="strip", parent=view.scene
        )

    # Longitude lines (vertical)
    for lon in np.linspace(0, 150, 6):
        t = np.linspace(0, 2*np.pi, 80)
        pts = np.column_stack([
            np.cos(np.radians(lon)) * np.sin(t),
            np.sin(np.radians(lon)) * np.sin(t),
            np.cos(t)
        ])
        scene.visuals.Line(
            pos=pts, color=(0.6, 0.7, 0.8, 0.4),
            connect="strip", parent=view.scene
        )

def add_axes(view):
    
    axes = [
        ([ 1.4,0,0], [-1.4,0,0]),  # X
        ([0, 1.4,0], [0,-1.4,0]),  # Y
        ([0,0, 1.4], [0,0,-1.4]),  # Z
    ]
    for a, b in axes:
        pts = np.array([a, b], dtype=float)
        scene.visuals.Line(
            pos=pts, color=(0.3, 0.3, 0.3, 0.6),
            connect="strip", width=2,
            parent=view.scene
        )

def add_labels(view):
    
    labels = [
        ([0, 0,  1.6], "|0⟩"),
        ([0, 0, -1.6], "|1⟩"),
        ([ 1.6, 0, 0], "|+⟩"),
        ([-1.6, 0, 0], "|−⟩"),
        ([0,  1.6, 0], "|i⟩"),
        ([0, -1.6, 0], "|−i⟩"),
    ]
    for pos, text in labels:
        t = scene.visuals.Text(
            text=text,
            pos=pos,
            color=(0.1, 0.1, 0.1, 1.0),
            font_size=28,          # ← bigger
            bold=True,
            face='OpenSans',
            parent=view.scene,
        )
        t.order = 10 
        
        
        
class BlochArrow:

    def __init__(self, view):
        self.view = view

        self.line = scene.visuals.Line(
            pos=np.array([[0,0,0],[0,0,1]], dtype=float),
            color=(0.85, 0.15, 0.15, 1.0),
            width=3,
            connect="strip",
            parent=view.scene,
        )

        self.tip = scene.visuals.Markers(
            parent=view.scene
        )
        self.tip.set_data(
            pos=np.array([[0,0,1]], dtype=float),
            face_color=(0.85, 0.15, 0.15, 1.0),
            size=12,
        )

        self.proj = scene.visuals.Line(
            pos=np.array([[0,0,0],[0,0,0]], dtype=float),
            color=(0.85, 0.15, 0.15, 0.25),
            width=1,
            connect="strip",
            parent=view.scene,
        )

    def update(self, bvec):
        x, y, z = bvec
        self.line.set_data(
            pos=np.array([[0,0,0],[x,y,z]], dtype=float)
        )
        self.tip.set_data(
            pos=np.array([[x,y,z]], dtype=float),
            face_color=(0.85, 0.15, 0.15, 1.0),
            size=12,
        )
        # Projection drops straight down to z=0 plane
        self.proj.set_data(
            pos=np.array([[x,y,z],[x,y,0]], dtype=float)
        )
        
        
class Trail:

    def __init__(self, view, max_points=200):
        self.max_points = max_points
        self.points = []

        self.line = scene.visuals.Line(
            pos=np.array([[0,0,0],[0,0,1]], dtype=float),
            color=(1.0, 0.5, 0.0, 0.7),
            width=2,
            connect="strip",
            parent=view.scene,
        )
        self.line.visible = False

    def add(self, bvec):
        self.points.append(np.array(bvec, dtype=float))
        if len(self.points) > self.max_points:
            self.points.pop(0)
        self._redraw()

    def _redraw(self):
        if len(self.points) < 2:
            return
        self.line.visible = True
        self.line.set_data(pos=np.array(self.points))

    def clear(self):
        self.points = []
        self.line.visible = False
        



def slerp(v1, v2, t):
    v1n = v1 / (np.linalg.norm(v1) + 1e-9)
    v2n = v2 / (np.linalg.norm(v2) + 1e-9)
    dot = np.clip(np.dot(v1n, v2n), -1.0, 1.0)
    omega = np.arccos(dot)
    if abs(omega) < 1e-6:
        return v2
    return (np.sin((1-t)*omega)/np.sin(omega))*v1 + \
           (np.sin(t*omega)/np.sin(omega))*v2

class GateAnimator:

    N_STEPS = 40        # frames per gate animation
    INTERVAL = 0.02     # seconds per frame (~50fps)

    def __init__(self, arrow, trail, canvas):
        self.arrow = arrow
        self.trail = trail
        self.canvas = canvas

        self._start = None
        self._end   = None
        self._step  = 0
        self._busy  = False
        self._queue = []

        self._timer = Timer(
            interval=self.INTERVAL,
            connect=self._tick,
            start=False,
        )

    def animate(self, start_vec, end_vec):
        self._queue.append((np.array(start_vec), np.array(end_vec)))
        if not self._busy:
            self._next()

    def _next(self):
        if not self._queue:
            self._busy = False
            return
        self._start, self._end = self._queue.pop(0)
        self._step = 0
        self._busy = True
        self._timer.start()

    def _tick(self, event):
        t = self._step / self.N_STEPS
        vec = slerp(self._start, self._end, t)
        self.arrow.update(vec)
        self.trail.add(vec)
        self.canvas.update()
        self._step += 1
        if self._step > self.N_STEPS:
            self._timer.stop()
            self.arrow.update(self._end)
            self._next()

    @property
    def busy(self):
        return self._busy
    
    
    

s = 1 / np.sqrt(2)

X = np.array([[0, 1],   [1,  0]],  dtype=complex)
Y = np.array([[0,-1j],  [1j, 0]],  dtype=complex)
Z = np.array([[1, 0],   [0, -1]],  dtype=complex)
H = np.array([[s,  s],  [s, -s]],  dtype=complex)
S = np.array([[1, 0],   [0,  1j]], dtype=complex)
T = np.array([[1, 0],   [0,  np.exp(1j*np.pi/4)]], dtype=complex)

def Rx(angle):
    c, s2 = np.cos(angle/2), np.sin(angle/2)
    return np.array([[c, -1j*s2], [-1j*s2, c]], dtype=complex)

def Ry(angle):
    c, s2 = np.cos(angle/2), np.sin(angle/2)
    return np.array([[c, -s2], [s2, c]], dtype=complex)

def Rz(angle):
    return np.array([
        [np.exp(-1j*angle/2), 0],
        [0, np.exp( 1j*angle/2)]
    ], dtype=complex)


BINDINGS = {
    'x': X,
    'y': Y,
    'z': Z,
    'h': H,
    's': S,
    't': T,
}

def handle_key(event, qubit, arrow, trail, animator):
    key = event.key.name.lower()
    applied = False

    before = qubit.bloch_vector().copy()

    if key in BINDINGS:
        qubit.apply(BINDINGS[key])
        applied = True
    elif key == 'r':
        qubit.reset()
        trail.clear()
        arrow.update(qubit.bloch_vector())
        print("  Reset to |0⟩")
        return
    elif key == '1':
        qubit.apply(Rx(np.radians(45)));  applied = True
    elif key == '2':
        qubit.apply(Rx(np.radians(90)));  applied = True
    elif key == '3':
        qubit.apply(Ry(np.radians(45)));  applied = True
    elif key == '4':
        qubit.apply(Ry(np.radians(90)));  applied = True
    elif key == '5':
        qubit.apply(Rz(np.radians(45)));  applied = True
    elif key == '6':
        qubit.apply(Rz(np.radians(90)));  applied = True
    elif key == 'c':
        trail.clear()
        print("  Trail cleared")
        return

    if applied:
        after = qubit.bloch_vector().copy()
        animator.animate(before, after)
        print_state(qubit)

def print_state(qubit):
    a, b = qubit.state
    bv = qubit.bloch_vector()
    print(f"  α={a.real:+.3f}{a.imag:+.3f}i  "
          f"β={b.real:+.3f}{b.imag:+.3f}i  "
          f"P0={qubit.prob_zero*100:.0f}%  "
          f"Bloch=({bv[0]:.2f},{bv[1]:.2f},{bv[2]:.2f})")