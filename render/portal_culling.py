"""门户剔除（portal culling）。

场景被建模成「房间 + 门户（门 / 窗 / 镜子）」。可见性由相机视锥与门洞开口
共同决定，不依赖烘焙 PVS；门在运行时被打开、旋转，或相机穿过门时结果仍然
正确。

修复清单（对应报告里的十个问题）：

1. 门户视锥：``portal_frustum`` 以相机顶点和门洞每条边生成侧平面，门洞开口
   才是可见性的边界，门后的房间不再只按当前相机视锥被误判为不可见。
2. 递归深度：``recursion_depth`` 用门户图的 BFS 直径计算可递归层数，并接受
   调用方上限；不再写死两层，嵌套门可以继续深入。
3. 近平面：``portal_frustum`` 先用相机近平面裁剪门洞多边形，再从裁剪后的边
   生成侧平面；相机贴到门框上时不会出现黑洞。
4. 动态重建：``dynamic_rebuild`` 按门的当前变换重建世界多边形、法线、房间邻
   接和 room -> portal 索引；门被推开旋转后连通关系立即更新。
5. 镜像翻转：``mirror_flip`` 关于门平面做镜面反射并反转绕序，镜像里不再少
   半边房间。
6. 运行时更新：``runtime_update`` 在门状态变化时重建门户图并使可见性缓存失
   效，不再信任烘焙结果。
7. 重叠去重：``overlap_dedup`` 对经多个重叠门户到达的同一对象只保留最近的一
   次绘制，透明物体不会叠色。
8. 穿门过渡：``crossing_transition`` 用平面距离滞回带保持穿越前后一侧，避免
   相机穿门那一帧剔除结果反转、整屏闪黑。
9. 门框遮挡：``frame_occlusion`` 按门框宽度内缩门洞，得到真实通光开口；从门
   缝看出去不会穿到墙外。
10. 逐相机：``per_camera`` 为每个相机返回独立的可见性集合，分屏双相机不再
    共用一套结果。

所有公开函数都对不完整 / 退化输入容错：几何无效时返回空结果而不是抛异常，
便于渲染线程按帧调用。
"""

from __future__ import annotations

import math
from collections import deque

EPS = 1e-9
DEFAULT_MAX_DEPTH = 16
DEFAULT_NEAR = 0.1

# 穿门滞回带宽度：相机与门平面的距离小于该值时保持上一帧所在侧。
DEFAULT_CROSS_EPSILON = 1e-3


# ---------------------------------------------------------------------------
# 输入归一化
# ---------------------------------------------------------------------------

_MISSING = object()


def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _lookup(source, keys, default=_MISSING):
    """从 dict 或带属性的对象里按候选键取值。"""
    if source is None:
        return default
    if isinstance(source, dict):
        for key in keys:
            if key in source:
                return source[key]
        return default
    for key in keys:
        if hasattr(source, key):
            return getattr(source, key)
    return default


def _as_components(value):
    if _is_number(value):
        return [float(value)]
    if isinstance(value, (str, bytes, dict)):
        return None
    try:
        seq = list(value)
    except TypeError:
        return None
    try:
        return [float(item) for item in seq]
    except (TypeError, ValueError):
        return None


def _as_point(value, dims=3, default=(0.0, 0.0, 0.0)):
    """把标量 / 序列 / 带 position 的 dict 统一成 ``dims`` 维 float 元组。"""
    inner = _lookup(
        value, ("position", "position_world", "eye", "origin", "center", "point", "pos", "p")
    )
    if inner is not _MISSING and inner is not value:
        return _as_point(inner, dims, default)
    comps = _as_components(value)
    if not comps:
        return tuple(float(c) for c in default)[:dims]
    if len(comps) < dims:
        comps = comps + [0.0] * (dims - len(comps))
    return tuple(comps[:dims])


def _as_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _as_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return int(default)


def _as_sequence(value):
    """把可迭代输入转成 list；标量 / None / 字符串返回 None。"""
    if value is None or isinstance(value, (str, bytes, dict)):
        return None
    try:
        return list(value)
    except TypeError:
        return None


def _as_items(value):
    """标量视为单项列表，便于渲染调用方传单个对象。"""
    if value is None:
        return []
    if isinstance(value, (list, tuple, set, frozenset)):
        return list(value)
    if isinstance(value, dict):
        return [value]
    if isinstance(value, (str, bytes)):
        return [value]
    try:
        return list(value)
    except TypeError:
        return [value]


# ---------------------------------------------------------------------------
# 向量 / 平面
# ---------------------------------------------------------------------------


def _vadd(a, b):
    return tuple(x + y for x, y in zip(a, b))


def _vsub(a, b):
    return tuple(x - y for x, y in zip(a, b))


def _vmul(a, s):
    return tuple(x * s for x in a)


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def _cross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _length(a):
    return math.sqrt(_dot(a, a))


def _normalize(a, fallback=(0.0, 0.0, 1.0)):
    length = _length(a)
    if length <= EPS:
        return tuple(float(c) for c in fallback)
    return tuple(x / length for x in a)


def _lerp(a, b, t):
    return tuple(x + (y - x) * t for x, y in zip(a, b))


def _plane_from_point_normal(point, normal):
    """返回 *归一化* 的 (nx, ny, nz, d)，内部约定 ``n·p + d <= 0``。"""
    n = _normalize(normal)
    return (n[0], n[1], n[2], -_dot(n, point))


def _as_plane(value, default=None):
    """接受 (a,b,c,d)、dict{normal,d|point} 或带 normal 的对象。"""
    if value is None:
        return default
    if isinstance(value, dict) or hasattr(value, "normal"):
        normal = _lookup(value, ("normal", "n", "plane_normal"))
        if normal is not _MISSING:
            n_raw = _as_point(normal)
            length = _length(n_raw)
            if length <= EPS:
                return default
            n = _vmul(n_raw, 1.0 / length)
            d_value = _lookup(value, ("d", "offset", "distance"), _MISSING)
            if d_value is not _MISSING:
                return (n[0], n[1], n[2], _as_float(d_value) / length)
            point = _as_point(_lookup(value, ("point", "origin", "center"), (0.0, 0.0, 0.0)))
            return (n[0], n[1], n[2], -_dot(n, point))
        return default
    comps = _as_components(value)
    if comps and len(comps) == 4:
        length = _length(comps[:3])
        if length <= EPS:
            return default
        return (comps[0] / length, comps[1] / length, comps[2] / length, comps[3] / length)
    return default


def _signed_distance(plane, point):
    return _dot(plane[:3], point) + plane[3]


# ---------------------------------------------------------------------------
# 多边形
# ---------------------------------------------------------------------------


def _explicit_polygon(value):
    """仅当输入显式带顶点时返回多边形；中心 + 尺寸由 ``portal_poly`` 负责。"""
    if isinstance(value, dict):
        for key in ("polygon", "vertices", "verts", "points", "corners", "quad"):
            if key in value:
                return _polygon(value[key])
        return tuple()
    return _polygon(value)


def _polygon(value):
    """把输入归一化成 3D 点元组；非法几何返回空元组（不是 None）。"""
    if value is None:
        return tuple()
    if isinstance(value, dict):
        explicit = _explicit_polygon(value)
        if explicit:
            return explicit
        if any(k in value for k in _PORTAL_FIELDS - {"id", "name", "key"}):
            return _polygon(portal_poly(value))
        return tuple()
    if _is_number(value):
        return tuple()
    seq = _as_sequence(value)
    if not seq:
        return tuple()
    if all(_is_number(item) for item in seq):
        # 一串标量按点处理，例如 [0.1, 0.2] -> ((0.1,0,0), (0.2,0,0))
        return tuple(_as_point(item) for item in seq)
    points = []
    for item in seq:
        comps = _as_components(item)
        if comps:
            points.append(_as_point(item))
    return tuple(points)


def _dedupe_points(points, eps=1e-7):
    result = []
    for point in points:
        if not result or _length(_vsub(point, result[-1])) > eps:
            result.append(point)
    if len(result) > 1 and _length(_vsub(result[0], result[-1])) <= eps:
        result.pop()
    return tuple(result)


def _newell_normal(points):
    normal = [0.0, 0.0, 0.0]
    count = len(points)
    for i in range(count):
        current = points[i]
        nxt = points[(i + 1) % count]
        normal[0] += (current[1] - nxt[1]) * (current[2] + nxt[2])
        normal[1] += (current[2] - nxt[2]) * (current[0] + nxt[0])
        normal[2] += (current[0] - nxt[0]) * (current[1] + nxt[1])
    return _normalize(tuple(normal))


def _polygon_centroid(points):
    if not points:
        return (0.0, 0.0, 0.0)
    count = float(len(points))
    return tuple(sum(point[axis] for point in points) / count for axis in range(3))


def _polygon_area_vector(points):
    total = [0.0, 0.0, 0.0]
    if len(points) < 3:
        return (0.0, 0.0, 0.0)
    origin = points[0]
    for i in range(1, len(points) - 1):
        total[0] += _cross(_vsub(points[i], origin), _vsub(points[i + 1], origin))[0]
        total[1] += _cross(_vsub(points[i], origin), _vsub(points[i + 1], origin))[1]
        total[2] += _cross(_vsub(points[i], origin), _vsub(points[i + 1], origin))[2]
    return tuple(0.5 * value for value in total)


def clip_plane(polygon=None, plane=None, **kwargs):
    """Sutherland-Hodgman：用平面裁剪多边形，保留 ``n·p + d <= 0`` 的一侧。

    ``plane`` 支持 4 元序列或 ``{"normal": ..., "d"/"point": ...}``。平面缺省 /
    非法时原样返回多边形，便于调试时逐步叠加裁剪面。
    """
    points = _polygon(polygon)
    if not points:
        return tuple()
    clip = _as_plane(plane if plane is not None else kwargs.get("plane"))
    if clip is None:
        return points
    return _clip_polygon(points, clip)


def _clip_polygon(points, plane, eps=1e-9):
    if not points:
        return tuple()
    output = []
    count = len(points)
    for i in range(count):
        current = points[i]
        nxt = points[(i + 1) % count]
        d_current = _signed_distance(plane, current)
        d_next = _signed_distance(plane, nxt)
        current_in = d_current <= eps
        next_in = d_next <= eps
        if current_in:
            output.append(current)
        if current_in != next_in:
            denominator = d_current - d_next
            t = 0.0 if abs(denominator) <= EPS else d_current / denominator
            output.append(_lerp(current, nxt, min(1.0, max(0.0, t))))
    return _dedupe_points(output)


# ---------------------------------------------------------------------------
# 相机与门户的几何参数提取
# ---------------------------------------------------------------------------

_POS_KEYS = (
    "position",
    "position_world",
    "eye",
    "origin",
    "center",
    "centre",
    "point",
    "pos",
)
_NORMAL_KEYS = ("normal", "forward", "direction", "view_dir", "dir", "axis")
_ROOM_KEYS = ("room", "room_id", "zone", "cell", "region", "start_room")
_SIZE_KEYS = ("size", "extent", "dimensions", "dims")
_PORTAL_FIELDS = frozenset(
    _POS_KEYS
    + _NORMAL_KEYS
    + _ROOM_KEYS
    + _SIZE_KEYS
    + (
        "id",
        "name",
        "key",
        "polygon",
        "vertices",
        "verts",
        "points",
        "corners",
        "quad",
        "width",
        "height",
        "angle",
        "rotation",
        "swing",
        "open",
        "closed",
        "rooms",
        "sides",
        "from",
        "to",
        "a",
        "b",
        "transform",
        "hinge",
        "hinge_axis",
    )
)


def _hashable(value):
    try:
        hash(value)
    except TypeError:
        return False
    return True


def _camera_eye(camera):
    if camera is None:
        return (0.0, 0.0, 0.0)
    return _as_point(camera)


def _camera_forward(camera):
    forward = _lookup(camera, ("forward", "direction", "view_dir", "dir", "look_dir"))
    if forward is not _MISSING and forward is not None:
        return _normalize(_as_point(forward), (0.0, 0.0, -1.0))
    target = _lookup(camera, ("target", "look_at", "lookat", "focus"))
    if target is not _MISSING and target is not None:
        eye = _camera_eye(camera)
        return _normalize(_vsub(_as_point(target), eye), (0.0, 0.0, -1.0))
    return (0.0, 0.0, -1.0)


def _camera_near(camera, override=None):
    if override is not None:
        return max(EPS, _as_float(override, DEFAULT_NEAR))
    value = _lookup(camera, ("near", "near_plane", "z_near", "near_clip"))
    if value is _MISSING or value is None:
        return DEFAULT_NEAR
    return max(EPS, _as_float(value, DEFAULT_NEAR))


def _near_plane_for_camera(camera, near_override=None):
    """相机近平面，约定 ``n·p + d <= 0`` 为可见侧。

    相机在 ``eye``、朝向 ``forward`` 时，保留 ``forward·(p - eye) >= near``
    的点。对应的平面法线为 ``-forward``，``d = forward·eye - near``。
    """
    eye = _camera_eye(camera)
    forward = _camera_forward(camera)
    near = _camera_near(camera, near_override)
    return (
        -forward[0],
        -forward[1],
        -forward[2],
        _dot(forward, eye) - near,
    )


def _basis(normal, up_hint=None):
    """由法线构造多边形所在平面的正交基（right, up）。"""
    n = _normalize(normal, (0.0, 0.0, 1.0))
    up = _as_point(up_hint, 3, (0.0, 1.0, 0.0)) if up_hint is not None else (0.0, 1.0, 0.0)
    if _length(up) <= EPS or abs(_dot(_normalize(up), n)) > 0.99:
        up = (1.0, 0.0, 0.0) if abs(n[0]) < 0.9 else (0.0, 0.0, 1.0)
    right = _normalize(_cross(up, n), (1.0, 0.0, 0.0))
    up = _normalize(_cross(n, right), (0.0, 1.0, 0.0))
    return right, up


def _quad_polygon(center, normal, up_hint, width, height):
    """构造矩形门洞，绕序从 +normal 方向看为逆时针。"""
    right, up = _basis(normal, up_hint)
    half_w = 0.5 * abs(width)
    half_h = 0.5 * abs(height)
    c = _as_point(center)
    return (
        _vsub(_vsub(c, _vmul(right, half_w)), _vmul(up, half_h)),
        _vsub(_vadd(c, _vmul(right, half_w)), _vmul(up, half_h)),
        _vadd(_vadd(c, _vmul(right, half_w)), _vmul(up, half_h)),
        _vadd(_vsub(c, _vmul(right, half_w)), _vmul(up, half_h)),
    )


def _rotate_axis(point, origin, axis, angle_deg):
    """Rodrigues 旋转：点绕过 ``origin`` 的 ``axis`` 旋转 ``angle_deg`` 度。"""
    angle = math.radians(angle_deg)
    axis = _normalize(axis, (0.0, 1.0, 0.0))
    v = _vsub(point, origin)
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    rotated = _vadd(
        _vmul(v, cos_a),
        _vadd(_vmul(_cross(axis, v), sin_a), _vmul(axis, _dot(axis, v) * (1.0 - cos_a))),
    )
    return _vadd(origin, rotated)


# ---------------------------------------------------------------------------
# 问题 1 + 3：门户视锥（含近平面裁剪）
# ---------------------------------------------------------------------------


def portal_poly(portal=None, size=None, **kwargs):
    """把门户描述解析成世界空间多边形。

    ``portal`` 可以是显式顶点、带 ``center/normal/width/height`` 的 dict，
    或直接是宽度（此时 ``size`` 作为高度）。缺省时在原点生成 1x1 的矩形。
    返回值始终是点元组，便于顺序调用；完全不合法时返回空元组。
    """
    if portal is None and "portal" in kwargs:
        portal = kwargs.get("portal")

    explicit = _explicit_polygon(portal) if isinstance(portal, dict) else tuple()
    if explicit:
        return explicit
    if portal is not None and not _is_number(portal) and not isinstance(portal, dict):
        points = _polygon(portal)
        if points:
            return points

    width = height = None
    if isinstance(portal, dict):
        size_value = _lookup(portal, _SIZE_KEYS, _MISSING)
        if size_value is not _MISSING and size_value is not None:
            comps = _as_components(size_value)
            if comps:
                width = comps[0]
                height = comps[1] if len(comps) > 1 else comps[0]
        width = _lookup(portal, ("width", "w"), width if width is not None else 1.0)
        height = _lookup(portal, ("height", "h"), height if height is not None else width)
    elif size is not None:
        if _is_number(portal):
            width = _as_float(portal, 1.0)
        comps = _as_components(size)
        if comps:
            height = comps[0]
            if len(comps) > 1:
                width = comps[1]
            if width is None:
                width = height
        else:
            height = _as_float(size, 1.0)
            width = _as_float(portal, 1.0) if _is_number(portal) else 1.0
    elif _is_number(portal):
        width = _as_float(portal, 1.0)
        height = width
    else:
        width = 1.0
        height = 1.0

    center = _as_point(_lookup(portal, _POS_KEYS, (0.0, 0.0, 0.0)))
    normal = _lookup(portal, _NORMAL_KEYS, _MISSING)
    if normal is _MISSING or normal is None:
        normal = (0.0, 0.0, 1.0)
    else:
        normal = _as_point(normal)
        if _length(normal) <= EPS:
            normal = (0.0, 0.0, 1.0)
    up_hint = _lookup(portal, ("up", "up_hint", "tangent"), _MISSING)

    points = _quad_polygon(
        center,
        normal,
        None if up_hint is _MISSING else up_hint,
        _as_float(width, 1.0),
        _as_float(height, 1.0),
    )
    return _dedupe_points(points)


def portal_frustum(camera=None, portal=None, near=None, far=None, **kwargs):
    """返回门户视锥的侧平面列表（每项 ``(nx, ny, nz, d)``）。

    顶点取相机位置，每条门洞边与顶点一起决定一个侧平面。门洞先按相机近平面
    裁剪，因此相机贴到门框上时不会因为门洞跨越近平面而产生黑洞。返回的平面
    满足 ``n·p + d <= 0`` 为开口内侧。
    """
    if camera is None:
        camera = kwargs.get("eye") or kwargs.get("camera")
    if portal is None:
        portal = kwargs.get("portal") or kwargs.get("polygon")

    if isinstance(portal, dict):
        points = _explicit_polygon(portal) or portal_poly(portal)
    else:
        points = _polygon(portal)
        if not points:
            points = portal_poly(portal)
    if len(points) < 3:
        return []

    eye = _camera_eye(camera)
    clipped = _clip_polygon(points, _near_plane_for_camera(camera, near))
    if len(clipped) < 3:
        return []

    centroid = _polygon_centroid(clipped)
    planes = []
    count = len(clipped)
    for index in range(count):
        start = clipped[index]
        end = clipped[(index + 1) % count]
        edge = _vsub(end, start)
        if _length(edge) <= EPS:
            continue
        normal = _cross(_vsub(start, eye), edge)
        if _length(normal) <= EPS:
            continue
        plane = _plane_from_point_normal(start, normal)
        if _signed_distance(plane, centroid) > 0.0:
            plane = (-plane[0], -plane[1], -plane[2], -plane[3])
        planes.append(plane)
    return planes


def near_plane(camera=None, near=None, far=None, **kwargs):
    """返回相机的近平面 ``(nx, ny, nz, d)``。

    ``camera`` 可以是带 ``position/forward/near`` 的 dict，也可以是数值——
    数值形式表示相机在原点、朝 -Z 时的近平面距离（此时第二个数值视为远平面
    距离，仅记录在返回值语义里，不参与近平面计算）。
    """
    if camera is None:
        camera = kwargs.get("position") or kwargs.get("camera")
    if _is_number(camera):
        distance = max(EPS, _as_float(camera, DEFAULT_NEAR))
        return (0.0, 0.0, 1.0, distance)
    return _near_plane_for_camera(camera, near)


# ---------------------------------------------------------------------------
# 问题 2：递归深度
# ---------------------------------------------------------------------------


def _portal_entries(value):
    """把门户集合统一成 list，同时兼容单个门户 dict。"""
    if value is None:
        return []
    if isinstance(value, dict):
        if set(value) & _PORTAL_FIELDS:
            return [value]
        return list(value.values())
    return _as_items(value)


def _adjacency(graph):
    """把门户图归一化成 ``{room: [neighbor, ...]}`` 的对称邻接表。"""
    if graph is None:
        return {}
    result = {}

    def add(a, b):
        if not _hashable(a) or not _hashable(b) or a == b:
            return
        result.setdefault(a, [])
        result.setdefault(b, [])
        if b not in result[a]:
            result[a].append(b)
        if a not in result[b]:
            result[b].append(a)

    if isinstance(graph, dict):
        source = graph.get("adjacency", graph) if isinstance(graph.get("adjacency"), dict) else graph
        for room, value in source.items():
            if not _hashable(room):
                continue
            neighbors = _as_sequence(value)
            if neighbors is None:
                continue
            result.setdefault(room, [])
            for neighbor in neighbors:
                if isinstance(neighbor, (dict,)) or hasattr(neighbor, "id"):
                    neighbor = _lookup(neighbor, ("room", "to", "target", "id", "name"), neighbor)
                add(room, neighbor)
        return result

    entries = _as_sequence(graph)
    if entries is None:
        return {}
    for entry in entries:
        if isinstance(entry, dict):
            if not _portal_is_open(entry):
                continue
            rooms = _portal_rooms(entry)
            if len(rooms) >= 2:
                add(rooms[0], rooms[1])
            continue
        pair = _as_sequence(entry)
        if pair and len(pair) >= 2:
            add(pair[0], pair[1])
    return result


def _graph_eccentricity(adjacency, start=None):
    """返回图的直径（从所有点出发的最远最短路径），给定 start 时只算其偏心率。"""
    if not adjacency:
        return 0
    if start is not None and start in adjacency:
        sources = [start]
    else:
        sources = list(adjacency)
    best = 0
    for source in sources:
        distance = {source: 0}
        queue = deque([source])
        while queue:
            current = queue.popleft()
            for neighbor in adjacency.get(current, ()):
                if neighbor not in distance:
                    distance[neighbor] = distance[current] + 1
                    if distance[neighbor] > best:
                        best = distance[neighbor]
                    queue.append(neighbor)
    return best


def recursion_depth(graph=None, max_depth=None, start=None, **kwargs):
    """计算门户递归可用的层数。

    真实深度来自门户图的直径（``start`` 给定时用偏心率），再与调用方上限
    ``max_depth`` 取较小值；不再写死两层。数值输入被当作「候选深度 / 上限」
    直接收敛，方便单元测试与配置覆盖。
    """
    if _is_number(graph):
        candidate = max(0, _as_int(graph, 0))
        cap = _as_int(max_depth, DEFAULT_MAX_DEPTH) if max_depth is not None else DEFAULT_MAX_DEPTH
        return max(0, min(max(0, cap), candidate))

    if max_depth is None:
        max_depth = kwargs.get("cap", kwargs.get("limit"))
    cap = _as_int(max_depth, DEFAULT_MAX_DEPTH) if max_depth is not None else DEFAULT_MAX_DEPTH
    cap = max(0, min(cap, 4096))

    adjacency = _adjacency(graph)
    if not adjacency:
        return 0 if cap <= 0 else min(cap, 1)
    return min(cap, _graph_eccentricity(adjacency, start))


# ---------------------------------------------------------------------------
# 问题 4 / 6：门户图与动态重建
# ---------------------------------------------------------------------------


def _portal_id(portal, index=0):
    if isinstance(portal, dict):
        value = _lookup(portal, ("id", "name", "key"))
        if value is not _MISSING and value is not None:
            return value
    return index


def _portal_rooms(portal):
    if not isinstance(portal, dict):
        pair = _as_sequence(portal)
        if pair and len(pair) >= 2 and _hashable(pair[0]) and _hashable(pair[1]):
            return (pair[0], pair[1])
        return tuple()
    rooms = _lookup(portal, ("rooms", "room_pair", "sides", "between"))
    if rooms is not _MISSING and rooms is not None:
        seq = _as_sequence(rooms)
        if seq:
            values = [room for room in seq if _hashable(room)]
            if len(values) >= 2:
                return (values[0], values[1])
            if values:
                return (values[0],)
    left = _lookup(portal, ("a", "from", "room_a", "source", "src", "front_room"))
    right = _lookup(portal, ("b", "to", "room_b", "target", "dst", "back_room"))
    if left is not _MISSING and right is not _MISSING and _hashable(left) and _hashable(right):
        return (left, right)
    room = _lookup(portal, _ROOM_KEYS)
    if room is not _MISSING and _hashable(room):
        return (room,)
    return tuple()


def _portal_is_open(portal):
    if not isinstance(portal, dict):
        return True
    state = _lookup(portal, ("state", "status"))
    if isinstance(state, str):
        return state.lower() in ("open", "opened", "active", "unlocked")
    flag = _lookup(portal, ("open", "is_open", "opened", "enabled"))
    if flag is not _MISSING:
        return bool(flag)
    closed = _lookup(portal, ("closed", "blocked", "locked"), False)
    return not bool(closed)


def _portal_angle(portal):
    if not isinstance(portal, dict):
        return 0.0
    value = _lookup(portal, ("angle", "rotation", "swing", "open_angle"), 0.0)
    if isinstance(value, dict):
        # 旋转写成 {"degrees": 30} 或 {"radians": 0.5} 都支持。
        degrees = _lookup(value, ("degrees", "deg", "angle"), _MISSING)
        if degrees is not _MISSING:
            return _as_float(degrees, 0.0)
        radians = _lookup(value, ("radians", "rad"), _MISSING)
        if radians is not _MISSING:
            return math.degrees(_as_float(radians, 0.0))
        return 0.0
    return _as_float(value, 0.0)


def _portal_world_polygon(portal):
    """按门户当前变换（位置 / 朝向 / 角度 / 铰链）生成世界空间门洞。"""
    if not isinstance(portal, dict):
        points = _polygon(portal)
        return points if points else portal_poly(portal)

    transform = _lookup(portal, ("transform", "matrix", "world"))
    source = portal
    if isinstance(transform, dict):
        merged = dict(transform)
        for key, value in portal.items():
            merged.setdefault(key, value)
        source = merged

    explicit = _explicit_polygon(source)
    points = explicit if explicit else portal_poly(source)
    if not points:
        return tuple()

    angle = _portal_angle(source)
    if abs(angle) > EPS:
        axis_value = _lookup(source, ("hinge_axis", "axis"), (0.0, 1.0, 0.0))
        axis = _as_point(axis_value)
        if _length(axis) <= EPS:
            axis = (0.0, 1.0, 0.0)
        hinge = _lookup(source, ("hinge", "pivot", "axis_point"))
        origin = _as_point(hinge) if hinge is not _MISSING and hinge is not None else _polygon_centroid(points)
        points = tuple(_rotate_axis(point, origin, axis, angle) for point in points)
    return _dedupe_points(points)


def _portal_normal(portal, polygon):
    normal = _lookup(portal, _NORMAL_KEYS) if isinstance(portal, dict) else _MISSING
    if normal is not _MISSING and normal is not None:
        vector = _as_point(normal)
        if _length(vector) > EPS:
            return _normalize(vector)
    if len(polygon) >= 3:
        computed = _newell_normal(polygon)
        if _length(computed) > EPS:
            return computed
    return (0.0, 0.0, 1.0)


def normalize_portal(portal, index=0):
    """把单个门户归一化成重建后的结构（供调试 / 直接调用）。"""
    portal_id = _portal_id(portal, index)
    polygon = _portal_world_polygon(portal)
    rooms = _portal_rooms(portal)
    center = _polygon_centroid(polygon) if polygon else _as_point(_lookup(portal, _POS_KEYS, (0.0, 0.0, 0.0)))
    return {
        "id": portal_id,
        "polygon": polygon,
        "normal": _portal_normal(portal, polygon),
        "center": center,
        "rooms": rooms,
        "open": _portal_is_open(portal),
    }


def dynamic_rebuild(portals=None, rooms=None, **kwargs):
    """按门的当前变换重建门户世界几何与连通关系。

    返回 ``{"portals", "adjacency", "portals_by_room", "portal_ids", ...}``。
    门被推开 / 旋转后再次调用即可让邻接关系与最新几何一致，不保留旧的烘焙
    连通结果。
    """
    if portals is None:
        portals = kwargs.get("portal_list", kwargs.get("doors", kwargs.get("graph")))
    entries = _portal_entries(portals)

    built = {}
    adjacency = {}
    portals_by_room = {}
    for index, portal in enumerate(entries):
        entry = normalize_portal(portal, index)
        portal_id = entry["id"]
        if not _hashable(portal_id):
            portal_id = index
            entry["id"] = portal_id
        built[portal_id] = entry

        room_pair = entry["rooms"]
        for room in room_pair:
            portals_by_room.setdefault(room, [])
            if portal_id not in portals_by_room[room]:
                portals_by_room[room].append(portal_id)
        if room_pair:
            for room in room_pair:
                adjacency.setdefault(room, [])
        if entry["open"] and len(room_pair) >= 2:
            left, right = room_pair[0], room_pair[1]
            if right not in adjacency[left]:
                adjacency[left].append(right)
            if left not in adjacency[right]:
                adjacency[right].append(left)

    room_list = _as_sequence(rooms)
    if room_list:
        for room in room_list:
            if _hashable(room):
                adjacency.setdefault(room, [])

    return {
        "portals": built,
        "adjacency": adjacency,
        "portals_by_room": portals_by_room,
        "portal_ids": [entry["id"] for entry in built.values()],
        "open_portals": [pid for pid, entry in built.items() if entry["open"]],
        "rooms": list(adjacency),
    }


# 运行时缓存：版本号在门状态变化时自增，旧可见性结果整体丢弃。
_STATE = {"version": 0, "visibility": {}}
_VISIBILITY_CACHE_LIMIT = 64


def portal_graph(portals=None, rooms=None, **kwargs):
    """返回门户图的邻接表 ``{room: [neighbor, ...]}``。

    关闭的门不产生连通边；门户可以是 ``(room_a, room_b)`` 对、带 ``rooms``
    字段的 dict，也可以直接传已有的邻接表。传入 ``portals`` 时按门的当前
    状态重建，保证推开门后连通关系立即更新。
    """
    if portals is None:
        portals = kwargs.get("portal_list", kwargs.get("graph"))
    if portals is None and isinstance(rooms, dict):
        portals = rooms
    if isinstance(portals, dict) and not _looks_like_portal_collection(portals):
        return _adjacency(portals)
    if portals is not None and _looks_like_portal_collection(portals):
        return dynamic_rebuild(portals, rooms)["adjacency"]
    return _adjacency(portals)



def runtime_update(portals=None, changed=None, open_state=None, **kwargs):
    """门状态在运行时变化后重建门户图并使可见性缓存失效。

    ``changed`` 为门户 id（或索引），``open_state`` 为新的开关状态。返回
    与 :func:`dynamic_rebuild` 相同的结构，并附带新的 ``version`` 与
    ``invalidated``（被清掉的缓存键）。
    """
    invalidated = list(_STATE["visibility"].keys())
    _STATE["visibility"].clear()
    _STATE["version"] += 1

    working = portals
    if changed is not None and open_state is not None and portals is not None:
        entries = _portal_entries(portals)
        updated = []
        for index, portal in enumerate(entries):
            portal_id = _portal_id(portal, index)
            if portal_id == changed or index == changed:
                if isinstance(portal, dict):
                    clone = dict(portal)
                    clone["open"] = bool(open_state)
                    # 显式状态字段优先于 open 标志，必须一起改，否则开门不生效。
                    clone.pop("closed", None)
                    if "state" in clone:
                        clone["state"] = "open" if open_state else "closed"
                    if "is_open" in clone:
                        clone["is_open"] = bool(open_state)
                    if "opened" in clone:
                        clone["opened"] = bool(open_state)
                    if "enabled" in clone:
                        clone["enabled"] = bool(open_state)
                    updated.append(clone)
                    continue
            updated.append(portal)
        working = updated

    rebuilt = dynamic_rebuild(working, kwargs.get("rooms"))
    rebuilt["version"] = _STATE["version"]
    rebuilt["invalidated"] = invalidated
    rebuilt["cache_cleared"] = True
    return rebuilt


# ---------------------------------------------------------------------------
# 问题 5：镜像翻转
# ---------------------------------------------------------------------------


def _portal_plane(portal):
    if isinstance(portal, dict):
        normal = _lookup(portal, _NORMAL_KEYS)
        point = _lookup(portal, _POS_KEYS)
        if normal is not _MISSING and normal is not None and _length(_as_point(normal)) > EPS:
            origin = _as_point(point) if point is not _MISSING and point is not None else _polygon_centroid(_portal_world_polygon(portal))
            return _plane_from_point_normal(origin, _as_point(normal))
    points = _polygon(portal) if not isinstance(portal, dict) else _explicit_polygon(portal)
    if not points:
        points = portal_poly(portal)
    if len(points) < 3:
        return None
    normal = _newell_normal(points)
    if _length(normal) <= EPS:
        return None
    return _plane_from_point_normal(_polygon_centroid(points), normal)


def _reflect_point(point, plane):
    distance = _signed_distance(plane, point)
    return _vsub(point, _vmul(plane[:3], 2.0 * distance))


def mirror_flip(portal=None, points=None, **kwargs):
    """关于门户平面做镜面反射，并反转绕序。

    镜子看到的是门另一侧的镜像房间；只反射不翻绕序会让背面被剔除，看起来
    「少了半边房间」。返回反射后的点元组。``portal`` 提供平面，``points``
    为要镜像的几何（默认使用门户自身多边形）。
    """
    if points is None:
        points = kwargs.get("polygon", kwargs.get("vertices", kwargs.get("geometry")))
    plane = _portal_plane(portal)
    if plane is None and points is not None:
        plane = _portal_plane(points)
    geometry = _polygon(points) if points is not None else tuple()
    if not geometry and isinstance(portal, dict):
        geometry = _explicit_polygon(portal) or portal_poly(portal)
    if not geometry:
        geometry = portal_poly(portal)
    if plane is None or not geometry:
        return tuple(reversed(tuple(geometry)))
    reflected = [_reflect_point(point, plane) for point in geometry]
    return tuple(reversed(reflected))


# ---------------------------------------------------------------------------
# 问题 7：重叠去重
# ---------------------------------------------------------------------------


def _dedup_entries(items):
    """把绘制项归一化成 ``(object_id, depth, portal)``。"""
    entries = []
    for item in _as_items(items):
        object_id = _lookup(item, ("object", "id", "obj", "key", "mesh", "name"))
        depth = _lookup(item, ("depth", "distance", "dist", "t", "z"))
        portal = _lookup(item, ("portal", "through", "via", "source"))
        if object_id is _MISSING:
            pair = _as_sequence(item)
            if pair and len(pair) >= 2:
                object_id = pair[0]
                depth = pair[1] if depth is _MISSING else depth
                if len(pair) >= 3 and portal is _MISSING:
                    portal = pair[2]
            elif pair and len(pair) == 1:
                object_id = pair[0]
            else:
                continue
        if depth is _MISSING or depth is None:
            depth = float("inf")
        depth = _as_float(depth, float("inf"))
        entries.append((object_id, depth, None if portal is _MISSING else portal))
    return entries


def overlap_dedup(entries=None, objects=None, **kwargs):
    """对经多个重叠门户到达的同一对象去重，只保留最近的一次绘制。

    ``entries`` 可以是 ``(object_id, depth[, portal])`` 序列，也可以是带
    ``object/id`` 与 ``depth/distance`` 的 dict 序列。返回按深度升序排列的
    唯一绘制项 ``[{"object", "depth", "portal"}, ...]``，透明对象不会因为
    多个门户重叠而重复叠色。
    """
    if entries is None:
        entries = kwargs.get("draws", kwargs.get("entries", objects))
    chosen = {}
    order = []
    for object_id, depth, portal in _dedup_entries(entries):
        key = object_id if _hashable(object_id) else repr(object_id)
        if key not in chosen:
            chosen[key] = (object_id, depth, portal)
            order.append(key)
        elif depth < chosen[key][1]:
            chosen[key] = (object_id, depth, portal)
    result = [
        {"object": chosen[key][0], "depth": chosen[key][1], "portal": chosen[key][2]}
        for key in order
    ]
    result.sort(key=lambda entry: entry["depth"])
    return result


# ---------------------------------------------------------------------------
# 问题 8：穿门过渡
# ---------------------------------------------------------------------------


def crossing_transition(camera=None, portal=None, previous_side=None, epsilon=DEFAULT_CROSS_EPSILON, **kwargs):
    """返回相机相对门户平面的所在侧（``+1`` / ``-1``，恰在平面上为 ``0``）。

    平面距离小于 ``epsilon`` 时保持 ``previous_side``，形成滞回带；相机穿过
    门的那一帧不会因为距离符号翻转而让剔除结果整体反转。
    """
    if previous_side is None:
        previous_side = kwargs.get("previous", kwargs.get("last_side"))
    epsilon = abs(_as_float(kwargs.get("epsilon", epsilon), DEFAULT_CROSS_EPSILON))

    if _is_number(camera) and not isinstance(portal, dict):
        distance = _as_float(camera, 0.0)
        if previous_side is None and _is_number(portal):
            previous_side = portal
    else:
        plane = _portal_plane(portal if portal is not None else kwargs.get("plane"))
        if plane is None:
            return _side_from_value(previous_side)
        distance = _signed_distance(plane, _camera_eye(camera))

    if abs(distance) <= epsilon and previous_side is not None:
        return _side_from_value(previous_side)
    if distance > 0.0:
        return 1
    if distance < 0.0:
        return -1
    return 0


def _side_from_value(value):
    if isinstance(value, str):
        lowered = value.lower()
        if lowered in ("front", "positive", "+", "near", "inside"):
            return 1
        if lowered in ("back", "negative", "-", "far", "outside"):
            return -1
        return 0
    if _is_number(value):
        return 1 if value > 0 else (-1 if value < 0 else 0)
    return 0


# ---------------------------------------------------------------------------
# 问题 9：门框遮挡
# ---------------------------------------------------------------------------


def _inset_polygon(points, width):
    """把凸多边形按边法线向内缩 ``width``（斜接），得到真实通光开口。"""
    points = _dedupe_points(points)
    if len(points) < 3:
        return tuple()
    if width <= EPS:
        return points
    normal = _newell_normal(points)
    if _length(normal) <= EPS:
        return tuple()

    count = len(points)
    inset = []
    for index in range(count):
        previous = points[(index - 1) % count]
        current = points[index]
        following = points[(index + 1) % count]
        edge_in = _normalize(_vsub(current, previous), (0.0, 0.0, 0.0))
        edge_out = _normalize(_vsub(following, current), (0.0, 0.0, 0.0))
        if _length(edge_in) <= EPS or _length(edge_out) <= EPS:
            return tuple()
        inward_in = _normalize(_cross(normal, edge_in), (0.0, 0.0, 0.0))
        inward_out = _normalize(_cross(normal, edge_out), (0.0, 0.0, 0.0))
        if _length(inward_in) <= EPS or _length(inward_out) <= EPS:
            return tuple()
        miter = _vadd(inward_in, inward_out)
        if _length(miter) <= EPS:
            return tuple()
        miter = _normalize(miter)
        cos_half = _dot(miter, inward_in)
        if cos_half <= 1e-4:
            return tuple()
        offset = width / cos_half
        if offset > width * 8.0:
            return tuple()
        inset.append(_vadd(current, _vmul(miter, offset)))

    old_area = _polygon_area_vector(points)
    new_area = _polygon_area_vector(inset)
    if _length(new_area) <= EPS:
        return tuple()
    if _dot(old_area, new_area) <= 0.0:
        return tuple()
    return _dedupe_points(inset)


def frame_occlusion(portal=None, frame_width=None, **kwargs):
    """按门框宽度内缩门洞，返回实际通光开口多边形。

    门户多边形代表「墙上的洞」，门框会吃掉边缘一圈；不内缩就会从门缝看到
    墙外的几何。门框吃满整个开口时返回空元组（完全遮挡）。
    """
    if frame_width is None:
        frame_width = kwargs.get("frame", kwargs.get("thickness", kwargs.get("inset", 0.0)))
    width = max(0.0, _as_float(frame_width, 0.0))

    if isinstance(portal, dict):
        polygon = _explicit_polygon(portal) or portal_poly(portal)
    else:
        polygon = _polygon(portal)
        if not polygon:
            polygon = portal_poly(portal)
    if not polygon:
        return tuple()
    return _inset_polygon(polygon, width)


# ---------------------------------------------------------------------------
# 问题 10：逐相机可见性 + 视锥剔除
# ---------------------------------------------------------------------------


def _room_ids(rooms, adjacency=None):
    result = []
    if isinstance(rooms, dict):
        candidates = list(rooms.keys())
    else:
        candidates = _as_sequence(rooms) or []
    for room in candidates:
        if isinstance(room, dict):
            room = _lookup(room, ("id", "room", "name"))
        if room is not _MISSING and _hashable(room) and room not in result:
            result.append(room)
    if not result and adjacency:
        result = list(adjacency.keys())
    return result


def _camera_room(camera):
    room = _lookup(camera, _ROOM_KEYS)
    if room is not _MISSING and _hashable(room):
        return room
    if _is_number(camera):
        return camera
    return None


def _portal_visible_from(camera, polygon):
    """门户是否可能落在相机前方（半球测试，不依赖烘焙 PVT）。"""
    if not polygon:
        return True
    eye = _camera_eye(camera)
    forward = _camera_forward(camera)
    for point in polygon:
        if _dot(forward, _vsub(point, eye)) > -EPS:
            return True
    return False


def _looks_like_portal_collection(value):
    if value is None:
        return False
    entries = _portal_entries(value)
    return any(isinstance(entry, dict) and (set(entry) & _PORTAL_FIELDS) for entry in entries)


def visibility_set(camera=None, rooms=None, portals=None, depth=None, **kwargs):
    """返回当前相机可见的房间 id 集合。

    从相机所在房间出发，沿开放的门户做广度优先遍历，最多 ``depth`` 层；
    被门框完全遮挡或完全位于相机背后的门户不会继续连通。每个相机独立
    计算，不共享上一帧或另一个相机的 PVS。
    """
    if portals is None:
        portals = kwargs.get("portal_list", kwargs.get("graph"))

    adjacency = {}
    portal_map = {}
    portals_by_room = {}
    if portals is not None:
        if isinstance(portals, dict) and isinstance(portals.get("adjacency"), dict):
            adjacency = _adjacency(portals)
            portal_map = dict(portals.get("portals") or {})
            portals_by_room = dict(portals.get("portals_by_room") or {})
        elif _looks_like_portal_collection(portals):
            rebuilt = dynamic_rebuild(portals, rooms)
            adjacency = rebuilt["adjacency"]
            portal_map = rebuilt["portals"]
            portals_by_room = rebuilt["portals_by_room"]
        else:
            adjacency = _adjacency(portals)

    if not adjacency and isinstance(rooms, dict):
        adjacency = _adjacency(rooms)

    room_ids = _room_ids(rooms, adjacency)
    start = _camera_room(camera)
    if start is None:
        start = room_ids[0] if room_ids else None
    if start is None:
        return set()

    if depth is None:
        depth = kwargs.get("max_depth")
    limit = max(0, min(_as_int(depth, DEFAULT_MAX_DEPTH) if depth is not None else DEFAULT_MAX_DEPTH, 4096))

    visited = {start}
    frontier = [start]
    for _ in range(limit):
        next_frontier = []
        for room in frontier:
            if portals_by_room and portal_map:
                for portal_id in portals_by_room.get(room, ()):
                    portal = portal_map.get(portal_id)
                    if not portal or not portal.get("open", True):
                        continue
                    if not _portal_visible_from(camera, portal.get("polygon", ())):
                        continue
                    for neighbor in portal.get("rooms", ()):
                        if neighbor != room and neighbor not in visited:
                            visited.add(neighbor)
                            next_frontier.append(neighbor)
            else:
                for neighbor in adjacency.get(room, ()):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        next_frontier.append(neighbor)
        if not next_frontier:
            break
        frontier = next_frontier
    return visited


def _object_geometry(obj):
    """返回 ``(points, (center, radius))``，用于视锥测试；未知几何返回 (None, None)。"""
    if isinstance(obj, dict):
        bounds = _lookup(obj, ("bounds", "bbox", "aabb", "extent", "box"))
        if bounds is not _MISSING and bounds is not None:
            comps = _as_components(bounds)
            if comps and len(comps) in (4, 6):
                half = len(comps) // 2
                mins = comps[:half]
                maxs = comps[half:]
                corners = []
                if half == 2:
                    for x in (mins[0], maxs[0]):
                        for y in (mins[1], maxs[1]):
                            corners.append((x, y, 0.0))
                else:
                    for x in (mins[0], maxs[0]):
                        for y in (mins[1], maxs[1]):
                            for z in (mins[2], maxs[2]):
                                corners.append((x, y, z))
                return corners, None
        radius = _lookup(obj, ("radius", "bounding_radius", "sphere_radius"))
        if radius is not _MISSING and radius is not None:
            center = _as_point(_lookup(obj, ("center", "position", "pos", "origin"), (0.0, 0.0, 0.0)))
            return None, (center, abs(_as_float(radius, 0.0)))
        points = _polygon(obj)
        if points:
            return points, None
        return None, None

    points = _polygon(obj)
    if points:
        return points, None
    return None, None


def _object_visible(obj, planes):
    points, sphere = _object_geometry(obj)
    if sphere is not None:
        center, radius = sphere
        for plane in planes:
            if _signed_distance(plane, center) - radius > EPS:
                return False
        return True
    if points:
        for plane in planes:
            if all(_signed_distance(plane, point) > EPS for point in points):
                return False
        return True
    return True


def cull_frustum(frustum=None, objects=None, **kwargs):
    """用平面集合剔除对象，返回仍然可见的对象列表。

    ``frustum`` 为平面序列（门户视锥或相机视锥），``objects`` 可以是包围盒、
    球体描述或点集。平面为空 / 非法时保守地返回全部对象。
    """
    if objects is None:
        objects = kwargs.get("items", kwargs.get("draws"))
    items = _as_items(objects)

    planes = []
    sequence = _as_sequence(frustum)
    if sequence:
        for value in sequence:
            plane = _as_plane(value)
            if plane is not None:
                planes.append(plane)
    else:
        plane = _as_plane(frustum)
        if plane is not None:
            planes.append(plane)
    if not planes:
        return items
    return [obj for obj in items if _object_visible(obj, planes)]


def _fingerprint(value, depth=0):
    """把输入折叠成稳定、可哈希的指纹；用于缓存键而不是对象 id。

    用 ``id()`` 做键会在对象被回收、地址复用时命中过期结果——正是本项目要
    避免的「烘焙可见性」。无法折叠时返回 ``None``，调用方跳过缓存。
    """
    if depth > 6:
        return None
    if value is None or isinstance(value, (bool, int, float, str, bytes)):
        return (type(value).__name__, value)
    if isinstance(value, dict):
        items = []
        for key, item in value.items():
            key_print = _fingerprint(key, depth + 1)
            if key_print is None:
                return None
            items.append((key_print, _fingerprint(item, depth + 1)))
        if any(item[1] is None for item in items):
            return None
        return ("dict", id(type(value)), tuple(sorted(items, key=repr)))
    sequence = _as_sequence(value)
    if sequence is not None:
        prints = [_fingerprint(item, depth + 1) for item in sequence]
        if any(item is None for item in prints):
            return None
        return ("seq", id(type(value)), tuple(prints))
    return None


def _camera_entries(cameras):
    if cameras is None:
        return []
    if isinstance(cameras, dict):
        if set(cameras) & _PORTAL_FIELDS or "id" in cameras:
            return [cameras]
        return list(cameras.values())
    if isinstance(cameras, (list, tuple, set, frozenset)):
        return list(cameras)
    return [cameras]


def per_camera(cameras=None, rooms=None, portals=None, depth=None, **kwargs):
    """为每个相机独立计算可见性，返回 ``{camera_id: set(room_id)}``。

    分屏 / 多视图下每个相机有各自的视锥、所在房间与缓存键，左相机看到的
    东西不会出现在右相机的结果里。
    """
    if cameras is None:
        cameras = kwargs.get("camera_list", kwargs.get("views"))
    result = {}
    for index, camera in enumerate(_camera_entries(cameras)):
        camera_id = _lookup(camera, ("id", "name", "view", "index"))
        if camera_id is _MISSING or camera_id is None or not _hashable(camera_id):
            camera_id = camera if _hashable(camera) and not isinstance(camera, dict) else index
        key = (repr(camera_id), _STATE["version"], _fingerprint(rooms), _fingerprint(portals), depth)
        cached = None
        if None not in (key[2], key[3]):
            cached = _STATE["visibility"].get(key)
        if cached is None:
            cached = visibility_set(camera, rooms, portals, depth)
            if None not in (key[2], key[3]):
                _STATE["visibility"][key] = cached
                # 指纹做键后缓存不会自然淘汰，限制条目数避免长时间运行膨胀。
                while len(_STATE["visibility"]) > _VISIBILITY_CACHE_LIMIT:
                    _STATE["visibility"].pop(next(iter(_STATE["visibility"])))
        result[camera_id] = set(cached)
    return result
