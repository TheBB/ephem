from colour import Color
import importlib
from itertools import chain
from math import cos, pi, radians, sin, sqrt, tan
from functools import partial
import numpy as np
from vispy import app, gloo, visuals, scene
from vispy.scene.cameras.perspective import PerspectiveCamera
from vispy.scene.cameras.turntable import TurntableCamera


class SkyBoxVisual(visuals.Visual):

    VERTEX = """
    void main()
    {
        gl_Position = $transform(vec4($position, 1));
    }
    """

    FRAGMENT = """
    void main()
    {
        gl_FragColor = $color;
    }
    """

    def __init__(self, color=(1, 1, 1, 1), res=100, steps=12, radius=1, width=2):
        visuals.Visual.__init__(self, self.VERTEX, self.FRAGMENT)

        sp_azims = np.linspace(0, 2*pi, 2*steps, endpoint=False)
        fl_azims = np.linspace(0, 2*pi, 2*res)
        sp_decls = np.linspace(-pi/2, pi/2, steps + 1)[1:-1]
        fl_decls = np.linspace(sp_decls[0], sp_decls[-1], res)

        m_vertices = np.vstack(
            (np.array([(radius * cos(decl) * cos(azim),
                        radius * cos(decl) * sin(azim),
                        radius * sin(decl))
                       for azim in sp_azims
                       for decl in fl_decls], dtype=np.float32),
             np.array([(radius * cos(decl) * cos(azim),
                        radius * cos(decl) * sin(azim),
                        radius * sin(decl))
                       for decl in sp_decls
                       for azim in fl_azims], dtype=np.float32)))

        m_indices = np.vstack(
            np.array([(a, a+1) for a in range(k*res, (k+1)*res-1)], dtype=np.uint32)
            for k in range(0, 2*steps),
        )
        p_indices = np.vstack(
            np.array([(a, a+1) for a in range(k*2*res, (k+1)*2*res-1)], dtype=np.uint32)
            for k in range(0, steps-1),
        )
        indices = np.vstack((m_indices, 2*steps*res + p_indices))
        self.full_indices = gloo.IndexBuffer(np.reshape(indices, np.prod(indices.shape)))

        ms = (steps - 1) // 2
        indices = 2*steps*res + np.array([
            (a, a+1) for a in range(ms*2*res, (ms+1)*2*res-1)
        ], dtype=np.uint32)
        self.sparse_indices = gloo.IndexBuffer(np.reshape(indices, np.prod(indices.shape)))

        self.vbo = gloo.VertexBuffer(m_vertices)
        self.shared_program.vert['position'] = self.vbo
        self._draw_mode = 'lines'

        self.full = True
        self.width = float(width)
        self.set_color(color)

    @property
    def full(self):
        return self._index_buffer is self.full_indices

    @full.setter
    def full(self, value):
        self._index_buffer = {
            True: self.full_indices,
            False: self.sparse_indices,
        }[value]

    def set_color(self, color):
        self.shared_program.frag['color'] = color

    def _prepare_transforms(self, view):
        view.view_program.vert['transform'] = view.get_transform()

    def _prepare_draw(self, view):
        gl = None
        try:
            gl = importlib.import_module('OpenGL.GL')
        except Exception:
            pass

        if gl:
            gl.glEnable(gl.GL_LINE_SMOOTH)
            px_scale = self.transforms.pixel_scale
            gl.glLineWidth(px_scale * self.width)

SkyBox = scene.visuals.create_visual_node(SkyBoxVisual)


class OutwardCamera(PerspectiveCamera):

    _state_props = PerspectiveCamera._state_props + ('elevation', 'azimuth')

    def __init__(self, elevation=30.0, azimuth=30.0, **kwargs):
        PerspectiveCamera.__init__(self, **kwargs)
        self._event_value = None
        self._actual_distance = 0.0
        self._distance = None

        self.elevation = elevation
        self.azimuth = azimuth

    @property
    def elevation(self):
        return self._elevation

    @elevation.setter
    def elevation(self, elev):
        elev = float(elev)
        self._elevation = min(90, max(-90, elev))
        self.view_changed()

    @property
    def azimuth(self):
        return self._azimuth

    @azimuth.setter
    def azimuth(self, azim):
        azim = float(azim)
        while azim < -180:
            azim += 360
        while azim > 180:
            azim -= 360
        self._azimuth = azim
        self.view_changed()

    def viewbox_mouse_event(self, event):
        if event.handled or not self.interactive:
            return

        if event.type == 'mouse_wheel':
            s = 1.02 ** -event.delta[1]
            self.fov = min(max(s * self.fov, 1.0), 90.0)
            self.view_changed()
        elif event.type == 'mouse_release':
            self._event_value = None
        elif event.type == 'mouse_press':
            event.handled = True
        elif event.type == 'mouse_move':
            if event.press_event is None:
                return
            if 1 in event.buttons:
                a = event.mouse_event.press_event.pos
                b = event.mouse_event.pos
                if self._event_value is None:
                    self._event_value = self.azimuth, self.elevation

                _, h = self._viewbox.size

                d_azim = (b - a)[0] / h * self.fov
                self.azimuth = self._event_value[0] + d_azim

                d_elev = (b - a)[1] / h * self.fov
                self.elevation = self._event_value[1] + d_elev

    def _update_transform(self, event=None):
        if self._viewbox is None:
            return

        fx = fy = self._scale_factor
        w, h = self._viewbox.size
        if w / h > 1:
            fx *= w / h
        else:
            fy *= h / w

        self._update_projection_transform(fx, fy)

        unit = [[-1, 1], [1, -1]]
        vrect = [[0, 0], self._viewbox.size]
        self._viewbox_tr.set_mapping(unit, vrect)
        transforms = [n.transform for n in
                      self._viewbox.scene.node_path_to_child(self)[1:]]
        camera_tr = self._transform_cache.get(transforms).inverse
        full_tr = self._transform_cache.get([self._viewbox_tr,
                                             self._projection,
                                             camera_tr])
        self._transform_cache.roll()
        self._set_scene_transform(full_tr)

    def _update_projection_transform(self, fx, fy):
        d = self.depth_value
        dist = fy / (2 * tan(radians(self._fov)/2))
        val = sqrt(d)
        self._projection.set_perspective(self._fov, fx/fy, dist/val, dist*val)
        self._update_camera_pos()

    def _update_camera_pos(self):
        ch_em = self.events.transform_change
        with ch_em.blocker(self._update_transform):
            tr = self.transform
            tr.reset()

            pp1 = np.array([(0, 0, 0), (0, 0, -1), (1, 0, 0), (0, 1, 0)])
            pp2 = np.array([(0, 0, 0), (0, 1, 0), (1, 0, 0), (0, 0, 1)])
            tr.set_mapping(pp1, pp2)

            tr.rotate(self.elevation, (1, 0, 0))
            tr.rotate(self.azimuth + 180, (0, 0, 1))


class GUI:

    def __init__(self):
        canvas = scene.SceneCanvas(keys='interactive', show=True)
        view = canvas.central_widget.add_view()
        view.bgcolor = '#000000'
        self.canvas = canvas
        self.view = view

        self.box_local = SkyBox(parent=view.scene)
        self.box_equatorial = SkyBox(parent=view.scene)

        self.directions = []
        for text, direction in [('N', (0,1,0)), ('E', (1,0,0)),
                                ('S', (0,-1,0)), ('W', (-1,0,0))]:
            self.directions.append(scene.visuals.Text(
                text, parent=view.scene, color='orange', pos=direction,
                bold=True, font_size=20,
            ))

        self.cam_out = OutwardCamera(fov=45, azimuth=0, elevation=0)
        self.cam_in = TurntableCamera(fov=45, distance=3, center=(0,0,0), azimuth=0, elevation=0)
        view.camera = self.cam_out

        self.set_system('local')

        self.handlers = {
            'C': self.camera_switch,
            'Q': self.camera_coords,
            '1': partial(self.set_system, 'local'),
            '2': partial(self.set_system, 'equatorial'),
            # '3': partial(self.set_system, 'ecliptic'),
        }

        @canvas.connect
        def on_key_press(event):
            self.on_key_press(event)

    def on_key_press(self, event):
        if event.handled:
            return
        try:
            self.handlers[event.key.name]()
            event.handled = True
        except KeyError:
            pass

    def camera_switch(self):
        old_cam = self.view.camera
        new_cam = next(c for c in {self.cam_out, self.cam_in} if old_cam is not c)
        new_cam.azimuth = old_cam.azimuth
        new_cam.elevation = old_cam.elevation
        self.view.camera = new_cam
        self.canvas.update()

    def camera_coords(self):
        print('{:.2f} {:.2f}'.format(self.view.camera.azimuth, self.view.camera.elevation))

    def set_system(self, system):
        tr = visuals.transforms.MatrixTransform()
        if system == 'equatorial':
            tr.rotate(30, (1, 0, 0))
        self.box_local.transform = tr
        self.box_local.full = system == 'local'
        cl = Color(hsl=(0, 0, .5))
        if system == 'local':
            cl.luminance *= 0.3
        self.box_local.set_color(tuple((*cl.rgb, 1)))

        tr = visuals.transforms.MatrixTransform()
        if system == 'local':
            tr.rotate(-30, (1, 0, 0))
        self.box_equatorial.transform = tr
        self.box_equatorial.full = system == 'equatorial'
        cl = Color(hsl=(0.6, 0.63, .5))
        if system == 'equatorial':
            cl.luminance *= 0.3
        self.box_equatorial.set_color(tuple((*cl.rgb, 1)))

        for d in self.directions:
            d.visible = system == 'local'

        self.canvas.update()


def run():
    GUI()
    app.run()
