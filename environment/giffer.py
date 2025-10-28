"""A class to store a gif of the simulation that was just run

  Example
from giffer import SumoGif

sim = SumoInterface()
gif = SumoGif(sim, draw_cars=True)
for i in range(100):
    sim.step()
    gif.update()
gif.save("Result.gif")
"""
# PIP INSTALL PILLOW
from PIL import Image, ImageDraw
from math import floor

from sumo_interface import proj, perp

SCALE = 4
T_PER_FRAME = 0.4
MAX_TIME = 60
LW = floor(3.2 / SCALE )# Lane Width
LR = 2 / SCALE   # Light Radius
CR = floor(1.5 / SCALE )  # Car Radius
MT = MAX_TIME / 16

def _scale(pt, pt1=None):
    x, y = pt if pt1 is None else (pt, pt1)
    return (x - 100) * SCALE, (y - 100) * SCALE

class SumoGif:
    def __init__(self, sim, draw_cars=False):
        self.sim       = sim
        self.draw_cars = draw_cars
        
        # Sensor positions
        self.sp = self.sim.get_sensor_positions()
        # Light positions
        self.lp = [(l[2], l[3]) for l in self.sim.get_lane_midpoints('in')]
        # Text (queue length) positions
        self.tp = [perp(proj(l[2:], l[:2], 60), l[:2], 20) for l in self.sim.get_lane_midpoints('in')]

        self.frames = []

        self._gen_background()

    def save(self, name):
        duration = sim.step_count() / T_PER_FRAME
        if not self.frames:
            return
        self.frames[0].save(name, save_all=True, append_images=self.frames[1:], optimize=False, duration=duration)

    def _gen_background(self):
        # "SETTINGS"
        self.w, self.h = 200*SCALE, 200*SCALE
        
        self.bg = Image.new("RGBA", (self.w, self.h), 'white')
        self.bgdraw = ImageDraw.Draw(self.bg)
        
        # 1) Draw intersection
        intersection = [_scale(x, y) for x, y in self.sim.get_intersection_shape()]
        self.bgdraw.polygon(intersection, fill="#ddddddff", outline="#dddddd")
        
        # 2) Draw lanes
        lanes = [(_scale(x0, y0), _scale(x1, y1)) for x0, y0, x1, y1 in self.sim.get_lane_midpoints('in')]
        for l in lanes:
            self.bgdraw.line(l, fill="#ddddddff", width=0)
        lanes = [(_scale(x0, y0), _scale(x1, y1)) for x0, y0, x1, y1 in self.sim.get_lane_midpoints('out')]
        for l in lanes:
            self.bgdraw.line(l, fill="#ddddddff", width=0)

        # 3) Draw Sensors
        # self.sensors = Image.new("RGBA", self.bg.size, (255, 255, 255, 0))
        # self.sdraw = ImageDraw.Draw(self.sensors)

        # 4) Draw Sensors
        for s in self.sp:
            for x, y in s:
                pts = [_scale(x-LW, y-LW), _scale(x+LW, y+LW)]
                # self.sdraw.ellipse(pts, fill="#dddddd88", outline="#ddddddFF")
                self.bgdraw.ellipse(pts, fill="#FFFFFFFF", outline="#ddddddFF")

    def car_frame(self):
        frame = Image.new("RGBA", self.bg.size, (255, 255, 255, 0))
        fdraw = ImageDraw.Draw(frame)
        for x, y, a, l, w in self.sim.get_car_midpoints():
            pts = [_scale(x-w, y-w), _scale(x+w, y+w)]
            fdraw.ellipse(pts, fill="#00000066", outline="#000000FF")
        return frame

    def sensor_frame(self):
        frame = Image.new("RGBA", self.bg.size, (255, 255, 255, 0))
        fdraw = ImageDraw.Draw(frame)
        for lp, lt in zip(self.sp, self.sim.get_occupied_time()):
            for (x, y), t in zip(lp, lt):
                if t > 0:
                    border = "#8844DDEE"
                else:
                    continue
                if t > MAX_TIME:
                    fill = "#002255FF"
                else:
                    v = int(t // MT)
                    v = v if v <= 15 else 15
                    fill = "#002255" + hex(v).replace("0x", "") * 2
                pts = [_scale(x-LW, y-LW), _scale(x+LW, y+LW)]
                fdraw.ellipse(pts, fill=fill, outline=border)
        return frame
    
    def light_frame(self):
        frame = Image.new("RGBA", self.bg.size, (255, 255, 255, 0))
        fdraw = ImageDraw.Draw(frame)
        for (x, y), g in zip(self.lp, self.sim.get_lights()):
            if g:
                fill = "#00FF00FF"
            else:
                fill = "#FF0000FF"
            pts = [_scale(x-LR, y-LR), _scale(x+LR, y+LR)]
            fdraw.ellipse(pts, fill=fill, outline=fill)
        return frame
    
    def queue_len_frame(self):
        frame = Image.new("RGBA", self.bg.size, (255, 255, 255, 0))
        fdraw = ImageDraw.Draw(frame)
        for (x, y), q in zip(self.tp, self.sim.get_queue_length()):
            fdraw.text(_scale(x, y), str(q), fill="#00BBBBFF")
            # fdraw.ellipse(pts, str(q), fill="#00BBBBFF")
        return frame

    def time_frame(self):
        frame = Image.new("RGBA", self.bg.size, (255, 255, 255, 0))
        fdraw = ImageDraw.Draw(frame)
        fdraw.text((self.bg.size[0] * 0.9, self.bg.size[1] * 0.1),
                    "Time: " + str(self.sim.get_time()),
                    fill="#000000FF")
        return frame

    def compose(self):
        frame = self.bg.copy()
        if self.draw_cars:
            frame = Image.alpha_composite(frame, self.car_frame())
        frame = Image.alpha_composite(frame, self.sensor_frame())
        frame = Image.alpha_composite(frame, self.light_frame())
        frame = Image.alpha_composite(frame, self.queue_len_frame())
        frame = Image.alpha_composite(frame, self.time_frame())
        return frame

    def update_buffer(self):
        self.frames.append(self.compose())

if __name__ == "__main__":
    from sumo_interface import SumoInterface
    from time import time, sleep

    sim = SumoInterface("map_1")
    sim.set_car_prob([0.2] * 12)
    gif = SumoGif(sim, True)
    for i in range(100):
        sim.step()
        gif.update_buffer()
    gif.save("temp.gif")    

    sleep(0.2)