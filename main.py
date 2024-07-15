import pygame as pg
import torch
import threading
import concurrent.futures
from math import sin, cos, tan, radians, prod, log10
import matplotlib.pyplot as plt

# ?????
# TODO: reduce allocations by reusing buffers
# TODO: triangle culling when all points are hidden behind one triangle
# TODO: potentially add a way to do fragment shaders (using either cuda or pytorch)
# TODO: proper gpu support meaning decoupling the rendering from pygame and using pytorch to generate the triangle mask

# !!!!!
# TODO: textures

pg.init()

SCREEN_SIZE = (1000, 1000)
MAX_RENDER_SIZE = 2 * prod(SCREEN_SIZE)
BASE_BRIGHTNESS = 0.3
MULTI_THREAD_INSIDE_MODEL = False
TORCH_USE_CUDA = False
DBG_STATS = {
    "widths": [[]],
    "heights": [[]],
    "zstd": [[]]  # z.std()
}
DBG_PLOT_SETTINGS = {
    "transform": {  # whether log10 is applied
        "widths": lambda x: log10(x + 1),
        "heights": lambda x: log10(x + 1),
        "zstd": lambda x: x,
    }
}


# TODO fix
def triangle_mask(triangle, topleft, bottomright, w, h):
    """
    Casts a ray upwards from the point (x, y) and checks if it intersects the triangles sides an odd number of times
      | yes -> inside
      | no -> outside
    :param bottomright: tuple of bottom right coordinates
    :param topleft: tuple of top left coordinates
    :param triangle: 3x2 torch.Tensor
    :param w: width of the triangle mask
    :param h: height of the triangle mask
    :return: w x h boolean torch.Tensor - mask of the triangle
    """
    x, y = triangle[:, 0], triangle[:, 1]
    assert triangle.shape == (3, 2)
    # f(x) = kx + d
    kab = (y[1] - y[0]) / (x[1] - x[0])
    dab = y[0] - x[0] * kab
    kac = (y[2] - y[0]) / (x[2] - x[0])
    dac = y[0] - x[0] * kab
    kbc = (y[2] - y[1]) / (x[2] - x[1])
    dbc = y[1] - x[1] * kab
    W = torch.tensor([kab, kac, kbc], device=triangle.device).view((1, 1, 3))
    b = torch.tensor([dab, dac, dbc], device=triangle.device).view((1, 1, 3))
    pf_idx = torch.tensor([0, 0, 1])  # first point idx
    ps_idx = torch.tensor([1, 2, 1])  # second point idx
    grid_x = torch.linspace(topleft[0], bottomright[0], w).view((w, 1, 1))
    grid_y = torch.linspace(topleft[1], bottomright[1], h).view((1, h, 1))
    is_in_range = ((x[pf_idx] <= grid_x) & (grid_x <= x[ps_idx])) | ((x[ps_idx] <= grid_x) & (grid_x <= x[pf_idx]))
    intersects_line = (W * grid_x + b >= grid_y)
    line_masks = intersects_line & is_in_range
    mask = line_masks.to(dtype=torch.uint8).sum(-1) % 2 == 1  # check if intersects odd num triangle sides
    return mask


def rotation_matrix(rx, ry, rz, device=torch.device('cpu')):
    """
    :param device: torch.device - device to put the tensor on
    :param rx: float - x angle in radians
    :param ry: float - y angle in radians
    :param rz: float - z angle in radians
    :return: torch.Tensor (3, 3) - 3d rotation matrix
    """
    return torch.tensor([
        [
            cos(rz) * cos(ry),
            cos(rz) * sin(ry) * sin(rx) - sin(rz) * cos(rx),
            cos(rz) * sin(ry) * cos(rx) + sin(rz) * sin(rx)],
        [
            sin(rz) * cos(ry),
            sin(rz) * sin(ry) * sin(rx) + cos(rz) * cos(rx),
            sin(rz) * sin(ry) * cos(rx) - cos(rz) * sin(rx),
        ],
        [
            -sin(ry),
            cos(ry) * sin(rx),
            cos(ry) * cos(rx),
        ]
    ], device=device)


class Camera:
    def __init__(
            self,
            pos: (float, float, float) = (0, 0, 0),
            rot: (float, float, float) = (0, 0, 0),
            fov: float = 60,
            device=torch.device('cpu')
    ):
        self.pos = torch.tensor(pos, device=device)
        self.x_rot, self.y_rot, self.z_rot = rot
        self.fov = fov
        self.device = device
        self.focal_width = tan(radians(fov)) / 2

    def fov_scaled_rotation_matrix(self):
        return rotation_matrix(self.x_rot, self.y_rot, self.z_rot, self.device) / self.focal_width


class ObjectModel:
    def __init__(self, model_filepath: str, pos: (float, float, float),
                 rot: (float, float, float) = (0, 0, 0), device=torch.device('cpu')):
        """
        :param filepath: path to load object model from
        :param pos: absolute position in space (triangles read from file are relative to this)
        :param rot: rotation of object in radians (roll pitch yaw)
        """
        self.model_filepath = model_filepath
        self.pos = torch.tensor(pos, dtype=torch.float32, device=device)
        self.rot = torch.tensor(rot, dtype=torch.float32, device=device)
        triangles, textures, texture_orients, texture_data = self.read_from_fp()
        self.triangles = triangles.to(device)
        self.texture_data = texture_data.to(device)
        self.textures = textures.to(device)
        self.texture_orientations = texture_orients.to(device)

    def read_from_fp(self):
        result, textures, orients, texture_data = [], [], [], []
        with open(self.model_filepath, 'r') as csv_file:
            for row in csv_file.read().splitlines():
                if row.strip() == '#':
                    break
                point_strings = []
                inside_point = False
                coord_end = 0
                while ')' in row[coord_end + 1:]:
                    coord_end = row[coord_end + 1:].index(')') + coord_end + 1
                for char in row[:coord_end + 1]:
                    if char == ')':
                        point_strings[-1] += ')'
                        inside_point = False
                    elif inside_point:
                        point_strings[-1] += char
                    elif char == '(':
                        point_strings.append('(')
                        inside_point = True
                    elif char == ',' or char == ' ':
                        pass
                    else:
                        raise Exception(f"Invalid character encountered while loading {self.model_filepath}")
                points = tuple(eval(point_str) for point_str in point_strings)
                texture_path, *orientation = row[coord_end + 1:].strip(" ,").split(',')
                orientation = [int(o) for o in orientation]
                texture = pg.surfarray.pixels3d(pg.image.load(texture_path))
                result.append(points)
                for i in range(len(texture_data)):
                    if (texture == texture_data[i]).all():
                        break
                else:
                    texture_data.append(texture)
                    i = len(texture_data) - 1
                textures.append(i)
                orients.append(orientation)
        return torch.tensor(result, dtype=torch.float32), torch.tensor(textures, dtype=torch.int32), torch.tensor(
            orients, dtype=torch.int32), torch.tensor(texture_data, dtype=torch.uint8)

    # sub-function of project
    def _render_triangle(
            self, tri_idx, widths, heights, projected_triangles, depth_buffer, screen_buffer, tri_idx_buffer, device,
            z_a, norm_ax_inv, z_ax_gradient, rotated_triangles3, buffer_lock=None
    ):
        X = torch.round(projected_triangles[tri_idx, :, 0]).to(dtype=torch.int32) + SCREEN_SIZE[0] // 2
        Y = torch.round(projected_triangles[tri_idx, :, 1]).to(dtype=torch.int32) + SCREEN_SIZE[1] // 2
        tri = projected_triangles[tri_idx].round().to(dtype=torch.int32) + torch.tensor(
            (SCREEN_SIZE[0] // 2, SCREEN_SIZE[1] // 2), device=device)
        x_min, y_min = X.min(), Y.min()
        width, height = widths[tri_idx].round().item(), heights[tri_idx].round().item()
        width, height = int(width), int(height)
        DBG_STATS["widths"][-1].append(width)
        DBG_STATS["heights"][-1].append(height)
        if width * height > MAX_RENDER_SIZE:
            DBG_STATS["zstd"][-1].append(0)
            return

        p = torch.stack(
            [torch.arange(x_min, x_min + width, device=device).reshape((width, 1)).repeat_interleave(height, dim=1),
             torch.arange(y_min, y_min + height, device=device).reshape((1, height)).repeat_interleave(width, dim=0)],
            dim=-1)

        start_horizontal = max(0, min(SCREEN_SIZE[0], p[0, 0, 0]))
        end_horizontal = max(0, min(SCREEN_SIZE[0], p[-1, 0, 0]))  # + 1 TODO?
        start_vertical = max(0, min(SCREEN_SIZE[1], p[0, 0, 1]))
        end_vertical = max(0, min(SCREEN_SIZE[1], p[0, -1, 1]))  # + 1 TODO?

        topleft = torch.tensor([start_horizontal, start_vertical], device=device)
        bottomright = torch.tensor([end_horizontal, end_vertical], device=device)
        p_rel_topleft = p - topleft

        start_horizontal_rel = max(0, min(min(width, SCREEN_SIZE[0] - start_horizontal), p_rel_topleft[0, 0, 0]))
        end_horizontal_rel = max(0, min(min(width, SCREEN_SIZE[0] - start_horizontal),
                                        p_rel_topleft[-1, 0, 0]))  # + 1 TODO?
        start_vertical_rel = max(0, min(min(height, SCREEN_SIZE[1] - start_vertical), p_rel_topleft[0, 0, 1]))
        end_vertical_rel = max(0, min(min(height, SCREEN_SIZE[1] - start_vertical),
                                      p_rel_topleft[0, -1, 1]))  # + 1 TODO?

        A_rel_topleft = torch.tensor((X[0], Y[0]), device=device) - topleft
        # NOTE: this linspace is centered around A = X[0], Y[0]
        # TODO off by one error probably (slicing should be avoidable but isnt)
        tri_fragments = torch.stack(
            torch.meshgrid(
                torch.linspace(-A_rel_topleft[0] / width, 1 - A_rel_topleft[0] / width, width, device=device),
                torch.linspace(-A_rel_topleft[1] / height, 1 - A_rel_topleft[1] / height, height, device=device),
                indexing='ij'),
            dim=-1)[start_horizontal_rel:end_horizontal_rel, start_vertical_rel:end_vertical_rel]
        tri_fragments_ax = tri_fragments @ norm_ax_inv[tri_idx]  # change of basis here
        z = z_a[tri_idx] + tri_fragments_ax @ z_ax_gradient[tri_idx]  # interpolated z buffer
        buffer_surf = pg.Surface(
            (end_horizontal - start_horizontal, end_vertical - start_vertical))  # todo: explore depth=1
        # if width * height > prod(SCREEN_SIZE):
        #     pass  # breakpoint slap
        pg.draw.polygon(buffer_surf, 1,
                        ((X[0] - x_min, Y[0] - y_min), (X[1] - x_min, Y[1] - y_min), (X[2] - x_min, Y[2] - y_min)))
        buffer_surfarray = pg.surfarray.array2d(buffer_surf)
        is_triangle = torch.tensor(
            buffer_surfarray,  # [start_horizontal_rel:end_horizontal_rel, start_vertical_rel:end_vertical_rel],
            dtype=torch.bool, device=device)
        # is_triangle = triangle_mask(tri, topleft, bottomright, width, height)[start_horizontal_rel:end_horizontal_rel, start_vertical_rel:end_vertical_rel]

        DBG_STATS["zstd"][-1].append(z.std())

        inf = float('inf')
        z[~is_triangle] = inf
        z_truncated = z  # truncated before depth interpolation
        # z_truncated = z[
        #               start_horizontal_rel:end_horizontal_rel,
        #               start_vertical_rel:end_vertical_rel
        #               ]
        if prod(z_truncated.shape) == 0:
            return

        tri3 = rotated_triangles3[tri_idx]
        surface_normal = (tri3[1] - tri3[0]).cross(tri3[2] - tri3[0])
        surface_normal /= (surface_normal ** 2).sum().sqrt()
        brightness = surface_normal[2].abs()  # normal @ camera = normal.z
        brightness = BASE_BRIGHTNESS + (1 - BASE_BRIGHTNESS) * brightness
        texture = self.texture_data[self.textures[tri_idx]]
        texture_orient = self.texture_orientations[tri_idx]
        shape_tensor = torch.tensor((texture.shape[0] - 1, texture.shape[1] - 1), device=device)
        texture_idx = tri_fragments_ax * shape_tensor
        torch.round_(texture_idx)
        torch.clamp_(texture_idx, torch.zeros((2,)), shape_tensor)
        texture_idx = texture_orient * texture_idx.to(dtype=torch.int32) - torch.where(texture_orient == 1, 0, 1)
        proj_texture = texture[texture_idx[:, :, 0], texture_idx[:, :, 1]]
        color = (proj_texture.to(dtype=torch.float32) * brightness).to(torch.uint8)

        # this function will be parallelized, so we need to lock the buffer to avoid race conditions
        if buffer_lock is not None:
            buffer_lock.acquire()
        is_closer = z_truncated < depth_buffer[start_horizontal:end_horizontal, start_vertical:end_vertical]
        depth_buffer[start_horizontal:end_horizontal, start_vertical:end_vertical][is_closer] = z_truncated[
            is_closer]
        if screen_buffer is not None:  # could be none since tri_idx_buffer exists
            screen_buffer[start_horizontal:end_horizontal, start_vertical:end_vertical][is_closer] = color[is_closer]
        tri_idx_buffer[start_horizontal:end_horizontal, start_vertical:end_vertical][is_closer] = tri_idx
        if buffer_lock is not None:
            buffer_lock.release()

    def project(self, camera: Camera, filter_non_rendered=True, depth_buffer=None, screen_buffer=None,
                tri_idx_buffer=None, render_without_depth_buffer=False) -> (torch.Tensor, torch.Tensor):
        # I sincerely apologize to anyone reading this for the sins I have committed in this function's code
        """Returns 2 torch tensors, which have the following meanings respectively:
        - depth buffer
        - screen buffer
        """
        device = self.triangles.device
        DBG_STATS["widths"].append([])
        DBG_STATS["heights"].append([])
        DBG_STATS["zstd"].append([])
        # Do projection
        triangles_rot = self.triangles @ rotation_matrix(*self.rot.tolist(), device)
        triangles = (self.pos + triangles_rot).to(dtype=torch.float32)
        relative_triangles = triangles - camera.pos
        scaled_rotation_matrix = camera.fov_scaled_rotation_matrix()  # with this scaled_rotation_matrix, projection_matrix = identity
        projected_triangles3 = relative_triangles @ scaled_rotation_matrix
        z_distances = projected_triangles3[:, :, 2]
        all_z_negative = torch.all(z_distances <= 0, dim=1)  # all z pos <= 0 (if 0, dont render because divide by 0)
        should_render = ~all_z_negative

        if filter_non_rendered:  # optionally, filter triangles where all z positions are negative (-> behind camera)
            projected_triangles3 = projected_triangles3[should_render]
        z_distances = projected_triangles3[:, :, 2]  # if relative_triangles changed because of above

        # always have logging for all triangles, also non-rendered ones
        n_non_rendered = triangles.shape[0] - projected_triangles3.shape[0]
        DBG_STATS["widths"][-1].extend([0] * n_non_rendered)
        DBG_STATS["heights"][-1].extend([0] * n_non_rendered)
        DBG_STATS["zstd"][-1].extend([0] * n_non_rendered)

        num_processed_triangles = len(
            projected_triangles3)  # number of triangles actually being processed (-> not out of screen)

        z_distances_unsqueezed = z_distances.reshape(z_distances.shape + (1,))  # unsqueeze(-1)
        # TODO maybe add focal length in denominator?
        # TODO remove this * 1000 scaling by figuring out focal lengths
        # scale and remove z component
        projected_triangles = (projected_triangles3 / z_distances_unsqueezed.abs())[:, :, :2] * SCREEN_SIZE[1]

        if render_without_depth_buffer:
            surf = pg.surfarray.make_surface(screen_buffer.cpu().detach().numpy())
            for tri_idx in range(num_processed_triangles):
                tri = projected_triangles[tri_idx]
                X, Y = tri[:, 0] + SCREEN_SIZE[0] // 2, tri[:, 1] + SCREEN_SIZE[1] // 2
                pg.draw.polygon(surf, self.colors[tri_idx].tolist(), (
                    (X[0].item(), Y[0].item()), (X[1].item(), Y[1].item()),
                    (X[2].item(), Y[2].item())))
            screen_buffer[:] = torch.tensor(pg.surfarray.array3d(surf), dtype=screen_buffer.dtype, device=device)
            return depth_buffer, screen_buffer

        # compute depth buffers
        if depth_buffer is None:
            depth_buffer = torch.full(SCREEN_SIZE, float('inf'), dtype=torch.float32, device=device)
        if screen_buffer is None:
            screen_buffer = torch.zeros(SCREEN_SIZE + (3,), dtype=torch.uint8, device=device)
        if tri_idx_buffer is None:
            tri_idx_buffer = torch.full(SCREEN_SIZE + (3,), -1, dtype=torch.int32, device=device)

        a, b, c = projected_triangles[:, 0, :], projected_triangles[:, 1, :], projected_triangles[:, 2,
                                                                              :]  # unpack triangles into points
        dims = projected_triangles.max(1)[0] - projected_triangles.min(1)[0] + 1
        widths, heights = dims[:, 0], dims[:, 1]
        ab = b - a
        ac = c - a
        norm_ab = ab / dims
        norm_ac = ac / dims
        z_a, z_b, z_c = z_distances[:, 0], z_distances[:, 1], z_distances[:, 2]
        z_ab_gradient = (z_b - z_a)  # unnormalized, interpolation values are x, y E [0, 1]
        z_ac_gradient = (z_c - z_a)
        norm_ax = torch.stack([norm_ab, norm_ac], dim=1)
        norm_ax_inv = torch.linalg.inv(norm_ax)
        z_ax_gradient = torch.stack([z_ab_gradient, z_ac_gradient], dim=1)

        # do a change of bases to express a linspace with norm_ax as basis vectors
        buffer_lock = threading.Lock()
        if MULTI_THREAD_INSIDE_MODEL:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                executor.map(
                    lambda tri_idx: self._render_triangle(
                        tri_idx, widths, heights, projected_triangles, depth_buffer,
                        screen_buffer, tri_idx_buffer, device, z_a, norm_ax_inv, z_ax_gradient, projected_triangles3,
                        buffer_lock),
                    range(num_processed_triangles)
                )
        else:
            for tri_idx in range(num_processed_triangles):
                self._render_triangle(
                    tri_idx, widths, heights, projected_triangles, depth_buffer,
                    screen_buffer, tri_idx_buffer, device, z_a, norm_ax_inv, z_ax_gradient, projected_triangles3,
                )
        return depth_buffer, screen_buffer


class PygameDisplayManager:
    def reset_buffers(self):
        self.depth_buffer = torch.full(SCREEN_SIZE, float('inf'), dtype=torch.float32, device=self.device)
        self.screen_buffer = torch.zeros(SCREEN_SIZE + (3,), dtype=torch.uint8, device=self.device)
        self.tri_idx_buffer = torch.full(SCREEN_SIZE + (3,), -1, dtype=torch.int32, device=self.device)

    def __init__(self, device=torch.device('cpu')):
        self.device = device
        self.reset_buffers()
        self.screen_size = SCREEN_SIZE
        self.screen_width, self.screen_height = SCREEN_SIZE
        self.win = pg.display.set_mode(SCREEN_SIZE)

    def render(self, model: ObjectModel, camera: Camera, render_without_depth_buffer=False):
        # w, h = SCREEN_SIZE
        # centered_triangles = model.triangles + torch.tensor([w, h]) / 2
        # for tri in centered_triangles:
        #     pg.draw.polygon(self.win, (255, 255, 255), tri.tolist(), 5)
        depth_buffer, screen_buffer = model.project(
            camera, True, self.depth_buffer,
            self.screen_buffer, self.tri_idx_buffer, render_without_depth_buffer
        )
        # NOTE: below not needed as render function checks that already
        # TODO probably needed again as I parallelize rendering (for fusing multiple depth / screen buffers)
        # is_closer = depth_buffer < self.depth_buffer
        # self.depth_buffer[is_closer] = depth_buffer[is_closer]
        # self.screen_buffer[is_closer] = screen_buffer[is_closer]

    def draw(self, render_depth_buffer=False):
        self.win.fill((0, 0, 0))
        if render_depth_buffer:
            depth_buffer = (self.depth_buffer * 10).clamp(0, 255).to(torch.uint8)
            depth_buffer = torch.where(self.depth_buffer < float('inf'), 255 - depth_buffer, 0).unsqueeze(
                -1).repeat_interleave(3, dim=-1).cpu().detach().numpy()
            # depth_buffer = 256 - (torch.clamp(self.depth_buffer, 0, 32) * 1000).floor().cpu().detach().numpy()
            self.win.blit(pg.surfarray.make_surface(depth_buffer), (0, 0))
        else:
            self.win.blit(pg.surfarray.make_surface(self.screen_buffer.cpu().detach().numpy()), (0, 0))
        self.reset_buffers()


def main():
    MODEL_PATHS = ["assets/block.csv"]
    running = True
    device = torch.device('cuda') if torch.cuda.is_available() and TORCH_USE_CUDA else torch.device('cpu')
    if device == 'cuda':
        print("CUDA available, using GPU")
    if device == 'cpu' and TORCH_USE_CUDA:
        print("WARNING: CUDA not available, using CPU")
    pdm = PygameDisplayManager(device)
    N_OBJ = 2
    models = [ObjectModel(MODEL_PATH, (cos(6.28 * t / N_OBJ), sin(6.28 * t / N_OBJ), 7), (0, 1, 1), device) for t in range(N_OBJ) for MODEL_PATH in MODEL_PATHS]
    n_triangles = sum(model.triangles.shape[0] for model in models)
    camera = Camera((0, 0, 0), (0, 0, 0), 0.1, device)
    clock = pg.time.Clock()
    font = pg.font.Font(None, 30)
    n_frames = 0
    spinning_paused = False
    while running:
        clock.tick(9999)  # wont get that many fps anyway
        fps = clock.get_fps()
        for model in models:
            pdm.render(model, camera)
        pdm.draw()
        fps_text = font.render(f"FPS: {fps:.2f}", True, (255, 255, 255))
        pdm.win.blit(fps_text, (10, 10))
        pg.display.update()
        camera.y_rot = sin(n_frames / 20) * 0.18
        camera.x_rot = cos(n_frames / 20) * 0.12
        # print("frame")
        for e in pg.event.get():
            if e.type == pg.QUIT:
                running = False
            elif e.type == pg.MOUSEBUTTONDOWN:
                spinning_paused = True
            elif e.type == pg.MOUSEBUTTONUP:
                spinning_paused = False
        if not spinning_paused:
            camera.y_rot += 0.02 * 0
            for i, model in enumerate(models):
                model.rot[0] += 0.03 * (i + 1)
                model.rot[1] -= 0.01 * (i + 1)
            # models[1].pos[1] += 0.01
        n_frames += 1
    return n_frames, n_triangles


if __name__ == '__main__':
    n_frames, n_triangles = main()
    x = range(n_frames * n_triangles)  # 12 = n_triangles
    fig, axs = plt.subplots(3)
    for i, (name, data) in enumerate(DBG_STATS.items()):
        transform = DBG_PLOT_SETTINGS["transform"][name]
        data = [transform(x) for f in data for x in f]
        axs[i].plot(x, data)
        axs[i].set_title(name)

    # x, y = torch.meshgrid(torch.arange(n_frames), torch.arange(12))
    # fig = plt.figure()
    # ax3d = fig.add_subplot(111, projection='3d')
    # ax3d.plot(x, y, flattened_widths)
    # ax3d.plot(x, y, flattened_heights)

    plt.show()
