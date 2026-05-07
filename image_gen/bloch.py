"""
Bloch Sphere Trajectory Visualiser — Interactive CLI
Requirements: numpy, matplotlib
    pip install numpy matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
import os

import signal
signal.signal(signal.SIGINT, signal.SIG_DFL)

import time
OUTPUT_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    f'bloch_{time.strftime("%Y%m%d_%H%M%S")}.png'
)
# ── Quantum gates ──────────────────────────────────────────────────────────────

I = np.eye(2, dtype=complex)
X = np.array([[0, 1],   [1, 0]],   dtype=complex)
Y = np.array([[0, -1j], [1j, 0]],  dtype=complex)
Z = np.array([[1, 0],   [0, -1]],  dtype=complex)
H = np.array([[1, 1],   [1, -1]],  dtype=complex) / np.sqrt(2)
S = np.array([[1, 0],   [0, 1j]],  dtype=complex)
T = np.array([[1, 0],   [0, np.exp(1j*np.pi/4)]], dtype=complex)

def Rx(a): c,s=np.cos(a/2),np.sin(a/2); return np.array([[c,-1j*s],[-1j*s,c]],dtype=complex)
def Ry(a): c,s=np.cos(a/2),np.sin(a/2); return np.array([[c,-s],[s,c]],dtype=complex)
def Rz(a): return np.array([[np.exp(-1j*a/2),0],[0,np.exp(1j*a/2)]],dtype=complex)

GATE_DESCRIPTIONS = {
    'X':  'Pauli-X  — NOT gate, flips |0>↔|1>, 180° around X-axis',
    'Y':  'Pauli-Y  — flip + phase, 180° around Y-axis',
    'Z':  'Pauli-Z  — flips sign of |1>, 180° around Z-axis',
    'H':  'Hadamard — puts |0> into superposition |+>',
    'S':  'Phase    — 90° rotation around Z-axis',
    'T':  'T gate   — 45° rotation around Z-axis',
    'RX': 'Rx(θ)   — custom angle rotation around X-axis',
    'RY': 'Ry(θ)   — custom angle rotation around Y-axis',
    'RZ': 'Rz(θ)   — custom angle rotation around Z-axis',
}

GATE_OPTIONS = list(GATE_DESCRIPTIONS.keys())

def get_gate_matrix(name):
    simple = {'X': X, 'Y': Y, 'Z': Z, 'H': H, 'S': S, 'T': T}
    if name in simple:
        return name, simple[name]
    if name in ('RX', 'RY', 'RZ'):
        while True:
            try:
                raw = input(f"  Angle for {name} in degrees: ").strip()
                angle = float(raw) * np.pi / 180
                deg = int(round(float(raw)))
                break
            except ValueError:
                print("  ✗ Please enter a number.")
        return f'{name}({deg})', {'RX': Rx, 'RY': Ry, 'RZ': Rz}[name](angle)
    return None, None

# ── Bloch sphere maths ─────────────────────────────────────────────────────────

def state_to_bloch(psi):
    psi = psi / np.linalg.norm(psi)
    rho = np.outer(psi, psi.conj())
    return np.array([2*np.real(rho[0,1]), 2*np.imag(rho[1,0]),
                     np.real(rho[0,0]-rho[1,1])])

def apply_gate(gate, psi):
    psi_new = gate @ psi
    return psi_new / np.linalg.norm(psi_new)

def interpolate_states(psi_start, psi_end, steps=50):
    pts = []
    for t in np.linspace(0, 1, steps):
        psi_t = (1-t)*psi_start + t*psi_end
        psi_t /= np.linalg.norm(psi_t)
        pts.append(state_to_bloch(psi_t))
    return pts

# ── Drawing ────────────────────────────────────────────────────────────────────

TRAJ_COLORS = ['#e63946','#f4a261','#2a9d8f','#6a4c93','#457b9d',
               '#e76f51','#43aa8b','#f3722c','#577590','#c77dff']

def draw_sphere(ax):
    u = np.linspace(0, 2*np.pi, 40)
    v = np.linspace(0, np.pi, 20)
    ax.plot_wireframe(np.outer(np.cos(u),np.sin(v)),
                      np.outer(np.sin(u),np.sin(v)),
                      np.outer(np.ones_like(u),np.cos(v)),
                      color='steelblue', alpha=0.1, linewidth=0.8)
    t = np.linspace(0, 2*np.pi, 200)
    ax.plot(np.cos(t), np.sin(t), np.zeros_like(t), 'k-',  alpha=0.25, linewidth=1.2)
    ax.plot(np.cos(t), np.zeros_like(t), np.sin(t), 'k--', alpha=0.15, linewidth=0.9)
    ax.plot(np.zeros_like(t), np.cos(t), np.sin(t), 'k--', alpha=0.15, linewidth=0.9)

def draw_axes(ax):
    for d in [(1,0,0),(0,1,0),(0,0,1)]:
        ax.quiver(0,0,0,*[v*1.35 for v in d], color='#888', linewidth=0.9,
                  arrow_length_ratio=0.07)
    for (x,y,z,lbl) in [(1.52,0,0,'|+>'),(- 1.52,0,0,'|->'),(0,1.52,0,'|i>'),
                         (0,-1.52,0,'|-i>'),(0,0,1.52,'|0>'),(0,0,-1.52,'|1>')]:
        ax.text(x,y,z,lbl,fontsize=10,ha='center',va='center',
                color='#333',fontweight='bold')

def draw_trajectory(ax, segments_by_gate):
    for gi, pts_seg in enumerate(segments_by_gate):
        color = TRAJ_COLORS[gi % len(TRAJ_COLORS)]
        pts = np.array(pts_seg)
        alphas = np.linspace(0.5, 1.0, max(len(pts)-1, 1))
        for i, alpha in enumerate(alphas):
            ax.plot(pts[i:i+2,0], pts[i:i+2,1], pts[i:i+2,2],
                    color=color, linewidth=2.2, alpha=alpha)

def draw_state_arrow(ax, vec, color, alpha=1.0):
    ax.quiver(0,0,0,*vec, color=color, linewidth=2.2,
              arrow_length_ratio=0.12, alpha=alpha)

def label_point(ax, vec, name, color):
    ax.scatter(*vec, color=color, s=40, zorder=5)
    ax.text(vec[0]+0.12, vec[1]+0.12, vec[2]+0.12,
            name, fontsize=8.5, color=color, fontweight='bold')

# OUTPUT_FILE = 'bloch_trajectory.png'

def render(gate_sequence):
    psi = np.array([1, 0], dtype=complex)
    states    = [psi.copy()]
    waypoints = [state_to_bloch(psi)]
    for _, gate in gate_sequence:
        psi = apply_gate(gate, psi)
        states.append(psi.copy())
        waypoints.append(state_to_bloch(psi))

    segments = [interpolate_states(states[i], states[i+1])
                for i in range(len(states)-1)]

    # if os.path.exists(OUTPUT_FILE):
    #     os.remove(OUTPUT_FILE)

    fig = plt.figure(figsize=(9, 7))
    ax  = fig.add_subplot(111, projection='3d')
    draw_sphere(ax)
    draw_axes(ax)
    draw_trajectory(ax, segments)

    gate_names = ['|0>'] + [n for n,_ in gate_sequence]
    wp_colors  = ['#1a6bb5'] + [TRAJ_COLORS[i % len(TRAJ_COLORS)]
                                 for i in range(len(gate_sequence))]

    for i, (pt, name, col) in enumerate(zip(waypoints, gate_names, wp_colors)):
        label_point(ax, pt, name, col)
        if i == 0:
            draw_state_arrow(ax, pt, '#1a6bb5', alpha=0.5)
    draw_state_arrow(ax, waypoints[-1], wp_colors[-1])

    from matplotlib.lines import Line2D
    handles = [
        Line2D([0],[0], color=TRAJ_COLORS[i % len(TRAJ_COLORS)], linewidth=2.5,
               label=f'{gate_names[i]} -> {gate_names[i+1]}')
        for i in range(len(gate_sequence))
    ]
    ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(-0.08,1.0),
              fontsize=8.5, framealpha=0.7)

    ax.set_title('Trajectory: ' + ' -> '.join(n for n,_ in gate_sequence),
                 fontsize=13, pad=12)
    ax.set_xlim([-1.6,1.6]); ax.set_ylim([-1.6,1.6]); ax.set_zlim([-1.6,1.6])
    ax.set_box_aspect([1,1,1]); ax.axis('off')
    plt.tight_layout()
    filename = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f'bloch_{time.strftime("%Y%m%d_%H%M%S")}.png'
    )

    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.show()

    print(f"\n  Saved -> {filename}\n")

# ── CLI ────────────────────────────────────────────────────────────────────────

def print_menu():
    print("\n  Available gates:")
    for name, desc in GATE_DESCRIPTIONS.items():
        print(f"    {name:<4}  {desc}")
    print("    Q     Done — render the plot\n")

def main():
    print("=" * 55)
    print("  Bloch Sphere Trajectory Visualiser")
    print("=" * 55)
    print_menu()

    gate_sequence = []

    while True:
        prompt = "  Enter first gate" if not gate_sequence else \
                 f"  Gate {len(gate_sequence)+1} (or Q to render)"
        raw = input(f"{prompt}: ").strip().upper()

        if raw == 'Q':
            if not gate_sequence:
                print("  Add at least one gate first.\n")
                continue
            break

        if raw not in GATE_OPTIONS:
            valid = ', '.join(GATE_OPTIONS)
            print(f"  Unknown gate '{raw}'. Options: {valid}\n")
            continue

        display_name, matrix = get_gate_matrix(raw)
        gate_sequence.append((display_name, matrix))
        seq = ' -> '.join(n for n,_ in gate_sequence)
        print(f"  Added {display_name}  |  Sequence: {seq}\n")

    print(f"\n  Rendering: {' -> '.join(n for n,_ in gate_sequence)} ...")
    render(gate_sequence)

if __name__ == '__main__':
    main()