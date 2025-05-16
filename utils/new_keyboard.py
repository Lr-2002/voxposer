from PIL import Image
from habitat_sim.utils.common import d3_40_colors_rgb
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import time


FORWARD_KEY="w"
LEFT_KEY="a"
RIGHT_KEY="d"
LOOK_UP_KEY = '1'
LOOK_DOWN_KEY = '2'
FINISH="f"

class KeyController:
    def __init__(self, env):
        self.env = env
        self.fig, _= plt.subplots(figsize=(20, 8))
        self.fig.canvas.mpl_disconnect(self.fig.canvas.manager.key_press_handler_id)
        self.cid = self.fig.canvas.mpl_connect('key_press_event', self)
        plt.title("Use 'w', 'a', 'd', '1' and '2' to control, press 'f' to finish.")
        plt.show()

    def display_sample(self, rgb_obs, semantic_obs, depth_obs):
        rgb_img = Image.fromarray(rgb_obs, mode="RGBA")

        semantic_img = Image.new("P", (semantic_obs.shape[1], semantic_obs.shape[0]))
        semantic_img.putpalette(d3_40_colors_rgb.flatten())
        semantic_img.putdata((semantic_obs.flatten() % 40).astype(np.uint8))
        semantic_img = semantic_img.convert("RGBA")

        depth_img = Image.fromarray((depth_obs / 10 * 255).astype(np.uint8), mode="L")

        arr = [rgb_img, semantic_img, depth_img]

        titles = ['RGB', 'Semantic', 'Depth']
        plt.clf()
        for i, data in enumerate(arr):
            ax = plt.subplot(1, 3, i+1)
            ax.axis('off')
            ax.set_title(titles[i])
            ax.imshow(data)
        self.fig.canvas.draw()

    
    def __call__(self, event):
        keystroke = event.key
        action = None
        if keystroke == FORWARD_KEY:
            print("action: FORWARD")
            action = 'move_forward'
        elif keystroke == LEFT_KEY:
            print("action: LEFT")
            action = 'turn_left'
        elif keystroke == RIGHT_KEY:
            print("action: RIGHT")
            action = 'turn_right'
        elif keystroke == LOOK_UP_KEY:
            print("action: LOOK UP")
            action = 'look_up'
        elif keystroke == LOOK_DOWN_KEY:
            print("action: LOOK DOWN")
            action = 'look_down'
        elif keystroke == FINISH:
            print("action: FINISH")
            event.canvas.mpl_disconnect(self.cid)
            return
        else:
            print("INVALID KEY")
        if action != None:
            a = time.time()
            obs = self.env.step(action)
            b = time.time()
            print("sim_time: ", b-a)
            a = time.time()
            self.display_sample(obs['color_1st'], obs['semantic_1st'], obs['depth_1st'])
            b = time.time()
            print("plot_time: ", b-a)