import plotly.graph_objects as go
import numpy as np

def draw_occ(occ):
    nx, ny, nz = occ.shape
    x, y, z = np.linspace(0, nx, nx), np.linspace(0, ny, ny), np.linspace(0, nz, nz)
    x, y, z = np.meshgrid(x, y, z, indexing='ij')
    x, y, z = x.flatten(), y.flatten(), z.flatten()
    occ_mark = occ.flatten()
    color_map = np.array([
        [0, 0, 0],          # others               black
        [255, 120, 50],     # barrier              orange
        [255, 192, 203],    # bicycle              pink         √
        [255, 255, 0],      # bus                  yellow       √
        [0, 150, 245],      # car                  blue         √
        [0, 255, 255],      # construction_vehicle cyan         √
        [255, 127, 0],      # motorcycle           dark orange  √
        [255, 0, 0],        # pedestrian           red          √
        [255, 240, 150],    # traffic_cone         light yellow
        [135, 60, 0],       # trailer              brown        √
        [160, 32, 240],     # truck                purple       √
        [255, 0, 255],      # driveable_surface    dark pink
        [139, 137, 137],    # other_flat           dark red
        [75, 0, 75],        # sidewalk             dard purple
        [150, 240, 80],     # terrain              light green
        [230, 230, 250],    # manmade              white
        [0, 175, 0],        # vegetation           green
        [255, 255, 255],    # Free                 White
    ])

    occ_color = color_map[occ_mark]
    mask = occ_mark != 17

    fig = go.Figure(data=[go.Scatter3d(
        x=x[mask],
        y=y[mask],
        z=z[mask],
        mode='markers',
        marker=dict(
            size=5,
            color=occ_color[mask]/255.0,                # set color to an array/list of desired values
            opacity=0.8,
            symbol='square'
        )
    )])

    # tight layout
    # fig.update_layout(scene = dict(aspectmode='data'), margin=dict(l=0, r=0, b=0, t=0))

    fig.update_layout(scene=dict(
        aspectmode='data',
        camera=dict(
            eye=dict(x=.0, y=.0, z=3.0),  # 调整观察者位置
            up=dict(x=0, y=0, z=1)           # 定义上方向
        )
        ), margin=dict(l=0, r=0, b=0, t=0))
    return fig

