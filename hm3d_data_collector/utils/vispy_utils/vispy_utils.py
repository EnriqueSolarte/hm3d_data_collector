import numpy as np
import vispy
from functools import partial
from vispy.scene import visuals
from vispy.visuals.transforms import STTransform
from vispy.scene.visuals import Text
from vispy import app, scene, io
import sys
from matplotlib.colors import hsv_to_rgb
import vispy.io as vispy_file
import os
from imageio import imwrite
from vispy import app, gloo


def get_vispy_plot(list_xyz, caption=""):
    from geometry_perception_utils.image_utils import add_caption_to_image

    try:
        img = plot_list_pcl(
            [] + list_xyz, return_canvas=True, shape=(512, 512))
        img = add_caption_to_image(img, f"{caption}", position=(20, 20))
    except:
        img = np.zeros((512, 512, 3), dtype=np.uint8)
        img = add_caption_to_image(img, f"{caption}", position=(20, 20))
        img = add_caption_to_image(
            img, f"PLOTTING FAILED !!!", position=(50, 20))
    return img


def plot_list_pcl(
        list_pcl, size=1, colors=None, scale_factor=None, up="-y",
        return_canvas=False, shape=(1024, 1024),
        elevation=90, azimuth=180, roll=0):
    if colors is not None:
        colors.__len__() == list_pcl.__len__()
        colors = np.vstack(colors).T/255

    else:
        colors = get_color_list(number_of_colors=list_pcl.__len__())
    pcl_colors = []
    for pcl, c in zip(list_pcl, colors.T):
        pcl_colors.append(np.ones_like(pcl)*c.reshape(3, 1))
    if scale_factor is None or scale_factor < 0:
        scale_factor = compute_scale_factor(np.hstack(list_pcl).T)
    return plot_color_plc(np.hstack(list_pcl).T, color=np.hstack(pcl_colors).T, size=size, scale_factor=scale_factor, return_canvas=return_canvas, shape=shape,
                          elevation=elevation, azimuth=azimuth, roll=roll, up=up)


def get_color_list(array_colors=None, fr=0.1, return_list=False, number_of_colors=None):
    """
    Returns a different color RGB for every element in the array_color
    """
    if array_colors is not None:
        number_of_colors = len(array_colors)

    h = np.linspace(0.1, 0.8, number_of_colors)
    # np.random.shuffle(h)
    # values = np.linspace(0, np.pi, number_of_colors)
    colors = np.ones((3, number_of_colors))

    colors[0, :] = h

    return hsv_to_rgb(colors.T).T


def setting_viewer(main_axis=True, bgcolor="black", caption="", shape=(512, 512)):
    canvas = vispy.scene.SceneCanvas(show=True, bgcolor=bgcolor)
    # size_win = shape[0]
    canvas.size = shape

    t1 = Text(caption, parent=canvas.scene, color="white")
    t1.font_size = 24
    t1.pos = canvas.size[0] // 2, canvas.size[1] // 10

    view = canvas.central_widget.add_view()
    view.camera = "arcball"  # turntable / arcball / fly / perspective

    if main_axis:
        visuals.XYZAxis(parent=view.scene)

    return view, canvas


def setting_pcl(view, size=5, edge_width=2, antialias=0):
    scatter = visuals.Markers()
    scatter.set_gl_state(
        'additive',
        blend=False,
        blend_equation='func_add',
        blend_func=('src_alpha', 'zero'),
        cull_face=True,
        depth_test=True
    )
    # scatter.set_gl_state(depth_test=True)
    scatter.antialias = 0
    view.add(scatter)
    return partial(scatter.set_data, size=size, edge_width=edge_width)


def compute_scale_factor(points, factor=3):
    # distances = np.linalg.norm(points, axis=1)
    distances = np.max(abs(points), axis=0)

    # max_size = np.quantile(distances, 0.8)
    max_size = np.max(distances)

    # if max_size > 50:
    #     return 50
    return factor * max_size


def plot_color_plc(
    points,
    color=(1, 1, 1, 1),
    size=1,
    plot_main_axis=True,
    background="black",
    scale_factor=None,
    caption="",
    return_canvas=False,
    elevation=90,
    azimuth=0,
    up="-y",
    roll=0,
    shape=(2000, 2000)
):

    view, canvas = setting_viewer(
        main_axis=plot_main_axis, bgcolor=background, caption=caption, shape=shape)
    view.camera = vispy.scene.TurntableCamera(
        elevation=elevation, azimuth=azimuth, roll=roll, fov=0, up=up
    )
    # view.camera = vispy.scene.TurntableCamera(elevation=90,
    #                                           azimuth=0,
    #                                           roll=0,
    #                                           fov=0,
    #  up='-y')
    if scale_factor is None or scale_factor < 0:
        scale = compute_scale_factor(points)
    else:
        scale = scale_factor
    view.camera.scale_factor = scale
    draw_pcl = setting_pcl(view=view)
    draw_pcl(points, edge_color=color, size=size)

    if return_canvas:
        return canvas
        # img = canvas.render()[:, :, :3]
        # view.canvas.app.quit()
        # view.canvas.close()
        # del canvas
        # del view
        # vispy.app.quit()
        # gc.collect()

        return img

    vispy.app.run()
