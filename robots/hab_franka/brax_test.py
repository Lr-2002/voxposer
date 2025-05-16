import brax
from brax.tools.urdf import UrdfConverter
from brax.io.image import render_array
from PIL import Image

if __name__ == "__main__":
    with open("panda_arm.urdf") as f:
        xml_data = f.read()
    brax_urdf_obj = UrdfConverter(xml_data, add_collision_pairs=False)

    _CONFIG = brax_urdf_obj.config

    _CONFIG.bodies[0].frozen.all = True # fixed base
    _CONFIG.gravity.z = -9.8
    _CONFIG.collide_include.add() # ignore collisions

    _CONFIG.dt = 0.1
    _CONFIG.substeps: 10

    sys = brax.System(_CONFIG)
    qp = sys.default_qp() # THIS PRODUCES NaNs

    Image.fromarray(render_array(sys, qp, 600, 400)).show()
    breakpoint()

    # qp, _ = sys.step(qp, [])