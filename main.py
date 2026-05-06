
from vispy import app, scene
from blochsphere import *

canvas = scene.SceneCanvas(
    title="Bloch Sphere  |  H X Y Z S T  |  1-6: Rotations  |  R: Reset  |  C: Clear trail",
    size=(800, 700),
    show=True,
    keys="interactive",
)
canvas.bgcolor = "#f0f2f5"

view = canvas.central_widget.add_view()
view.camera = scene.cameras.TurntableCamera(
    fov=45, distance=4.5, elevation=25, azimuth=30,
)

add_sphere(view)
add_axes(view)
add_labels(view)

qubit    = Qubit()
arrow    = BlochArrow(view)
trail    = Trail(view)
animator = GateAnimator(arrow, trail, canvas)

arrow.update(qubit.bloch_vector())
trail.add(qubit.bloch_vector())

print("\n  Bloch Sphere ready!")
print("  H X Y Z S T — gates")
print("  1/2: Rx45/90   3/4: Ry45/90   5/6: Rz45/90")
print("  R — reset    C — clear trail\n")
print("  Try: H → Z → H  (should equal X)\n")

@canvas.events.key_press.connect
def on_key(event):
    handle_key(event, qubit, arrow, trail, animator)

app.run()