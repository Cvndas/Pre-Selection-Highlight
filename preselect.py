# SPDX-License-Identifier: GPL-3.0-or-later
"""Maya-style pre-selection highlight in mesh edit mode.

Derived from ``src/allset/operators/preselect.py`` of Maya Workflow Kit for Blender
(https://github.com/5hyuns/maya-workflow-kit-for-blender, Copyright 2026 ohnsk, GPL-3.0-or-later),
made standalone: kit preferences, pivot editing, the kit's manipulator and its extrude handles are
replaced by this add-on's own preferences and a simplified manipulator-hover guard.

The component under the mouse is highlighted before you click, following the select mode:
a vertex = square point, an edge = line, a face = outline + translucent fill.

Blender's ``GIZMO_GT_mesh_preselect_elem_3d`` picks the nearest vertex in every select mode, so this is a
Python gizmo with its own picking:

- ``pick_*`` (numpy, no bpy): nearest vertex / edge within the pick radius on screen that the ``visible``
  callback accepts; a face = the one under the cursor (BVH ray cast).
- Occlusion: a per-object ``BVHTree`` of the visible faces, cached until the next depsgraph update;
  X-ray turns it off.
- The gizmo has no invoke / modal, so a click goes on to the active select tool.
"""

import math

import bpy
import gpu
import numpy as np
from bpy_extras import view3d_utils
from gpu_extras.batch import batch_for_shader
from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree

MODES = {'EDIT_MESH'}
SELECT_TOOL_PREFIX = "builtin.select"      # select, select_box, select_circle, select_lasso
TRANSFORM_TOOLS = {"builtin.move": 'TRANSLATE', "builtin.scale": 'RESIZE', "builtin.rotate": 'ROTATE',
                   "builtin.transform": 'TRANSFORM'}

HANDLE_PX = 12.0        # keep the hover off a manipulator this close
FACE_DOT_PX = 75.0      # X-ray: farthest a face centre dot may be from the cursor
ROTATE_RING = 1.25      # rotate gizmo: disc of gizmo_size / 2 x this

_last = []    # the live gizmo group
_cache = {}   # object pointer -> MeshCache


def prefs():
    try:
        return bpy.context.preferences.addons[__package__].preferences
    except KeyError:
        return None


# --- pure helpers -------------------------------------------------------------------------------------

def project(co, persp, width, height):
    """Screen xy (N x 2) and an in-front mask of world points ``co`` (N x 3) under ``persp`` (4 x 4)."""
    co = np.asarray(co, dtype=np.float64).reshape(-1, 3)
    m = np.asarray(persp, dtype=np.float64)
    h = co @ m[:, :3].T + m[:, 3]
    w = h[:, 3]
    front = w > 1e-9
    w = np.where(front, w, 1.0)
    xy = np.empty((len(co), 2))
    xy[:, 0] = (h[:, 0] / w + 1.0) * 0.5 * width
    xy[:, 1] = (h[:, 1] / w + 1.0) * 0.5 * height
    return xy, front


def segment_distance(p, a, b):
    """Distances from point ``p`` to segments a[i]-b[i] (N x 2 each) and the segment parameters t."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    d = b - a
    ll = (d * d).sum(axis=1)
    t = np.where(ll > 1e-12, ((np.asarray(p) - a) * d).sum(axis=1) / np.where(ll > 1e-12, ll, 1.0), 0.0)
    t = np.clip(t, 0.0, 1.0)
    q = a + d * t[:, None]
    return np.hypot(q[:, 0] - p[0], q[:, 1] - p[1]), t


def pick_vert(xy, ok, mouse, radius, visible):
    """Index of the nearest ``ok`` point within ``radius`` that ``visible(i)`` accepts, or None.
    Only points inside the square around the cursor get the distance maths and the sort."""
    if len(xy) == 0:
        return None
    dx = xy[:, 0] - mouse[0]
    dy = xy[:, 1] - mouse[1]
    cand = np.flatnonzero(ok & (np.abs(dx) <= radius) & (np.abs(dy) <= radius))
    if len(cand) == 0:
        return None
    dist = np.hypot(dx[cand], dy[cand])
    for k in np.argsort(dist):
        if dist[k] > radius:
            break
        i = int(cand[k])
        if visible(i):
            return i
    return None


def pick_edge(xy, edges, ok, mouse, radius, visible, fallback=(), front=None):
    """Index of the nearest ``ok`` edge within ``radius`` (screen) that ``visible(i, t)`` accepts; else the
    nearest of ``fallback`` (the edges of the face under the cursor); else None.

    ``front`` (optional, per vertex): edges with an endpoint behind the camera are skipped.
    Only edges whose screen bounding box reaches the cursor circle get the segment-distance maths and the
    sort, so the cost per mouse move is a few vectorised comparisons instead of work on every edge."""
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    if len(edges) == 0:
        return None
    a = xy[edges[:, 0]]
    b = xy[edges[:, 1]]
    mx, my = mouse
    near = ok & (np.minimum(a[:, 0], b[:, 0]) <= mx + radius) & (np.maximum(a[:, 0], b[:, 0]) >= mx - radius) \
              & (np.minimum(a[:, 1], b[:, 1]) <= my + radius) & (np.maximum(a[:, 1], b[:, 1]) >= my - radius)
    cand = np.flatnonzero(near)
    if front is not None and len(cand):
        cand = cand[front[edges[cand, 0]] & front[edges[cand, 1]]]
    if len(cand):
        dist, t = segment_distance(mouse, a[cand], b[cand])
        for k in np.argsort(dist):
            if dist[k] > radius:
                break
            i = int(cand[k])
            if visible(i, float(t[k])):
                return i
    fb = [i for i in fallback if ok[i] and (front is None or (front[edges[i, 0]] and front[edges[i, 1]]))]
    if fb:
        d, _t = segment_distance(mouse, a[fb], b[fb])
        return int(fb[int(np.argmin(d))])
    return None


# --- per-object cache ---------------------------------------------------------------------------------

class MeshCache:
    """World vertex positions, edges, visible-face BVH and (lazily) selection of one mesh.
    Rebuilt after geometry / transform updates only (selection updates keep it)."""

    def __init__(self, ob):
        import bmesh
        self.ob_ptr = ob.as_pointer()
        self.name = ob.name
        bm = bmesh.from_edit_mesh(ob.data)      # local only: never stored, it can be freed under us
        bm.verts.ensure_lookup_table()
        bm.edges.ensure_lookup_table()
        bm.faces.ensure_lookup_table()
        self.mw = ob.matrix_world.copy()
        self.inv = self.mw.inverted_safe()
        m = np.array(self.mw, dtype=np.float64)
        local = np.array([v.co[:] for v in bm.verts], dtype=np.float64).reshape(-1, 3)
        self.co = local @ m[:3, :3].T + m[:3, 3]
        self.vok = np.array([not v.hide for v in bm.verts], dtype=bool)
        self.edges = np.array([(e.verts[0].index, e.verts[1].index) for e in bm.edges],
                              dtype=np.int64).reshape(-1, 2)
        self.eok = np.array([not e.hide for e in bm.edges], dtype=bool)
        hidden = [f.index for f in bm.faces if f.hide]
        if not hidden:
            self.bvh_map = None                         # BVH index = face index
            self.bvh = BVHTree.FromBMesh(bm) if len(bm.faces) else None
        else:
            vis = [f.index for f in bm.faces if not f.hide]
            self.bvh_map = vis
            self.bvh = BVHTree.FromPolygons([v.co.copy() for v in bm.verts],
                                            [tuple(v.index for v in bm.faces[i].verts) for i in vis]) if vis else None
        self._sel = None
        self._dots = None

    def _bm(self):
        """A fresh reference to the live edit mesh of this object (ReferenceError when it is gone)."""
        import bmesh
        ob = bpy.data.objects.get(self.name)
        if ob is None or ob.mode != 'EDIT' or ob.as_pointer() != self.ob_ptr:
            raise ReferenceError("edit mesh is gone")
        bm = bmesh.from_edit_mesh(ob.data)
        bm.faces.ensure_lookup_table()
        return bm

    def face_verts(self, i):
        return tuple(v.index for v in self._bm().faces[i].verts)

    def face_edges(self, i):
        return tuple(e.index for e in self._bm().faces[i].edges)

    def selection(self):
        """(world points of the selected vertices, active element centre or None), cached."""
        if self._sel is None:
            import bmesh
            bm, mw = self._bm(), self.mw
            pts = [mw @ v.co for v in bm.verts if v.select and not v.hide]
            el, active = bm.select_history.active, None
            if isinstance(el, bmesh.types.BMVert):
                active = mw @ el.co
            elif isinstance(el, bmesh.types.BMEdge):
                active = mw @ ((el.verts[0].co + el.verts[1].co) / 2)
            elif isinstance(el, bmesh.types.BMFace):
                active = mw @ el.calc_center_median()
            self._sel = (pts, active)
        return self._sel

    def face_dots(self):
        """(face indices, world centres) of the visible faces, built on first use (X-ray face picking)."""
        if self._dots is None:
            idx, flat, counts = [], [], []
            for f in self._bm().faces:
                if f.hide:
                    continue
                vs = [v.index for v in f.verts]
                idx.append(f.index)
                flat.extend(vs)
                counts.append(len(vs))
            if idx:
                starts = np.cumsum([0] + counts[:-1])
                sums = np.add.reduceat(self.co[np.array(flat, dtype=np.int64)], starts, axis=0)
                cen = sums / np.array(counts, dtype=np.float64)[:, None]
            else:
                cen = np.zeros((0, 3))
            self._dots = (np.array(idx, dtype=np.int64), cen)
        return self._dots

    def ray(self, origin, direction):
        """(world distance, face index) of the first visible face along a world ray, or None."""
        if self.bvh is None:
            return None
        lo = self.inv @ origin
        ld = (self.inv.to_3x3() @ direction).normalized()
        loc, _n, idx, _d = self.bvh.ray_cast(lo, ld)
        if loc is None:
            return None
        return (self.mw @ loc - origin).length, (idx if self.bvh_map is None else self.bvh_map[idx])


def caches(context):
    out = []
    for ob in context.objects_in_mode_unique_data or ():
        if ob is None or ob.type != 'MESH':
            continue
        c = _cache.get(ob.as_pointer())
        if c is None or c.name != ob.name:
            c = _cache[ob.as_pointer()] = MeshCache(ob)
        out.append((ob, c))
    return out


@bpy.app.handlers.persistent
def invalidate(_scene=None, depsgraph=None):
    """depsgraph_update_post: drop the caches on a geometry / transform change; a selection-only change
    keeps them and only forgets the selection."""
    if depsgraph is not None:
        for u in depsgraph.updates:
            if isinstance(u.id, (bpy.types.Mesh, bpy.types.Object)) and (u.is_updated_geometry or u.is_updated_transform):
                break
        else:
            for c in _cache.values():
                c._sel = None
            return
    _cache.clear()


@bpy.app.handlers.persistent
def _drop_all(*_args):
    """load_post / undo_post / redo_post: edit-mesh data may be gone, rebuild lazily."""
    _cache.clear()


# --- policy -------------------------------------------------------------------------------------------

def tool_id(context):
    try:
        tool = context.workspace.tools.from_space_view3d_mode(context.mode, create=False)
    except (AttributeError, TypeError):
        return None
    return tool.idname if tool is not None else None


def overlays_on(context):
    """False when the viewport's Overlays toggle is off."""
    sd = context.space_data
    try:
        return sd is not None and sd.type == 'VIEW_3D' and sd.overlay.show_overlays
    except AttributeError:
        return False


def _enabled(context):
    p = prefs()
    if p is None or not p.enabled or context.mode not in MODES or not overlays_on(context):
        return False
    tid = tool_id(context)
    if tid is None:
        return False
    return tid.startswith(SELECT_TOOL_PREFIX) or (p.with_transform_tools and tid in TRANSFORM_TOOLS)


def view_stamp(rv3d):
    """The view as a comparable tuple: changes whenever the camera moves (orbit, pan, zoom, numpad views,
    trackpad gestures). Comparing it is how navigation is detected; no other operator is inspected."""
    return tuple(v for row in rv3d.perspective_matrix for v in row)


def _xray(space):
    shading = space.shading
    return shading.show_xray_wireframe if shading.type == 'WIREFRAME' else shading.show_xray


def _handles_block(context, region, rv3d, mouse, cs):
    """Simplified: the mouse is over the disc around the pivot where the transform tool's gizmo lives,
    so the gizmo (not the hovered component) gets the click."""
    kind = TRANSFORM_TOOLS.get(tool_id(context))
    if kind is None:
        return False
    sels = [c.selection() for _ob, c in cs]
    pts = [p for s_ in sels for p in s_[0]]
    if not pts:
        return False
    ts = context.scene.tool_settings
    pivot = ts.transform_pivot_point
    if pivot == 'CURSOR':
        center = context.scene.cursor.location.copy()
    elif pivot == 'ACTIVE_ELEMENT' and any(s_[1] is not None for s_ in sels):
        center = next(s_[1] for s_ in sels if s_[1] is not None)
    elif pivot == 'BOUNDING_BOX_CENTER':
        arr = np.array([p[:] for p in pts])
        center = Vector((arr.min(axis=0) + arr.max(axis=0)) / 2.0)
    else:
        center = sum(pts, Vector()) / len(pts)
    c2 = view3d_utils.location_3d_to_region_2d(region, rv3d, center)
    if c2 is None:
        return False
    scale = context.preferences.system.ui_scale or 1.0
    size = context.preferences.view.gizmo_size
    if kind in ('ROTATE', 'TRANSFORM'):
        reach = size / 2.0 * scale * ROTATE_RING
    else:
        reach = size * scale * 0.9          # arrows / scale cubes extend roughly one gizmo size
    return math.hypot(mouse[0] - c2.x, mouse[1] - c2.y) <= reach + HANDLE_PX * scale


def pick(context, region, rv3d, mouse):
    """(object pointer, 'VERT' / 'EDGE' / 'FACE', index) under ``mouse`` (region pixels), or None."""
    cs = caches(context)
    if not cs or _handles_block(context, region, rv3d, mouse, cs):
        return None
    p = prefs()
    ts = context.tool_settings
    vmode, emode, fmode = tuple(ts.mesh_select_mode)
    xray = _xray(context.space_data)
    scale = context.preferences.system.ui_scale or 1.0
    radius = (p.radius if p else 18.0) * scale
    origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, mouse)
    direction = view3d_utils.region_2d_to_vector_3d(region, rv3d, mouse)
    hit = None   # (distance, cache, face index)
    if xray:
        # X-ray: like Blender, the face is the one whose centre dot is nearest the cursor on screen,
        # whatever lies in front of it.
        if fmode or emode:
            persp_x = rv3d.perspective_matrix
            limit = FACE_DOT_PX * scale
            for _ob, c in cs:
                fidx, cen = c.face_dots()
                if len(fidx) == 0:
                    continue
                dxy, dfront = project(cen, persp_x, region.width, region.height)
                dd = np.hypot(dxy[:, 0] - mouse[0], dxy[:, 1] - mouse[1])
                dd = np.where(dfront, dd, np.inf)
                k = int(np.argmin(dd))
                if dd[k] <= limit and (hit is None or dd[k] < hit[0]):
                    hit = (float(dd[k]), c, int(fidx[k]))
    elif fmode or emode:
        for _ob, c in cs:
            h = c.ray(origin, direction)
            if h is not None and (hit is None or h[0] < hit[0]):
                hit = (h[0], c, h[1])

    def occluded(pt):
        if xray:
            return False
        p2 = view3d_utils.location_3d_to_region_2d(region, rv3d, pt)
        if p2 is None:
            return True
        o = view3d_utils.region_2d_to_origin_3d(region, rv3d, p2)
        d = pt - o
        dist = d.length
        if dist < 1e-12:
            return False
        d /= dist
        for _ob, c in cs:
            h = c.ray(o, d)
            if h is not None and h[0] < dist * (1.0 - 1e-4) - 1e-6:
                return True
        return False

    if fmode and not (vmode or emode):
        return None if hit is None else (hit[1].ob_ptr, 'FACE', hit[2])
    best = None   # (screen distance, result)
    persp = rv3d.perspective_matrix
    for _ob, c in cs:
        xy, front = project(c.co, persp, region.width, region.height)
        if vmode:
            i = pick_vert(xy, c.vok & front, mouse, radius, lambda i, c=c: not occluded(Vector(c.co[i])))
            if i is not None:
                d = math.hypot(xy[i, 0] - mouse[0], xy[i, 1] - mouse[1])
                if best is None or best[1][1] != 'VERT' or d < best[0]:
                    best = (d, (c.ob_ptr, 'VERT', i))
        if emode and not vmode and (best is None or best[1][1] != 'VERT'):
            fallback = c.face_edges(hit[2]) if hit is not None and hit[1] is c else ()

            def vis(i, t, c=c):
                a, b = c.co[c.edges[i, 0]], c.co[c.edges[i, 1]]
                return not occluded(Vector(a + (b - a) * t))
            i = pick_edge(xy, c.edges, c.eok, mouse, radius, vis, fallback, front)
            if i is not None:
                d, _t = segment_distance(mouse, xy[c.edges[[i], 0]], xy[c.edges[[i], 1]])
                d = float(d[0]) if i not in fallback else radius + 1.0
                if best is None or d < best[0]:
                    best = (d, (c.ob_ptr, 'EDGE', i))
    return best[1] if best is not None else None


# --- the gizmo ----------------------------------------------------------------------------------------

class PRESELECT_GT_highlight(bpy.types.Gizmo):
    bl_idname = "PRESELECT_GT_highlight"

    def setup(self):
        self.target = None          # (object pointer, kind, index)
        self.stamp = None           # view the target was picked in

    def test_select(self, context, location):
        region, rv3d = context.region, context.region_data
        stamp = view_stamp(rv3d)
        moving = self.stamp is not None and stamp != self.stamp    # the view changed since the last move
        self.stamp = stamp
        if moving or not overlays_on(context):    # camera moving, or Overlays off: no highlight, no picking
            target = None
        else:
            try:
                target = pick(context, region, rv3d, (float(location[0]), float(location[1])))
            except (ReferenceError, IndexError, RuntimeError):   # the edit mesh changed under the cache
                invalidate()
                target = None
        if target != self.target:
            self.target = target
            region.tag_redraw()
        return -1 if target is None else 0

    def draw(self, context):
        if not self.is_highlight or self.target is None:
            return
        if not overlays_on(context) or view_stamp(context.region_data) != self.stamp:
            return                  # the camera moved since the pick: hide until the next mouse move
        c = _cache.get(self.target[0])
        p = prefs()
        if c is None or p is None:
            return
        kind, i = self.target[1], self.target[2]
        scale = context.preferences.system.ui_scale or 1.0
        region = context.region
        rgb = tuple(p.color)
        try:
            gpu.state.depth_test_set('NONE')
            gpu.state.blend_set('ALPHA')
            if kind == 'VERT':
                # point_size_set is ignored by several GPU backends, so draw an exact pixel-sized square
                # in screen space instead.
                p2 = view3d_utils.location_3d_to_region_2d(region, context.region_data, Vector(c.co[i]))
                if p2 is None:
                    return
                half = p.vert_size * scale * 0.5
                x0, y0, x1, y1 = p2.x - half, p2.y - half, p2.x + half, p2.y + half
                w, h = float(region.width), float(region.height)
                gpu.matrix.push()
                gpu.matrix.push_projection()
                try:
                    gpu.matrix.load_identity()
                    gpu.matrix.load_projection_matrix(Matrix(((2.0 / w, 0, 0, -1.0), (0, 2.0 / h, 0, -1.0),
                                                              (0, 0, -1.0, 0), (0, 0, 0, 1.0))))
                    shader = gpu.shader.from_builtin('UNIFORM_COLOR')
                    shader.uniform_float("color", (*rgb, 1.0))
                    batch_for_shader(shader, 'TRIS', {"pos": [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]},
                                     indices=[(0, 1, 2), (0, 2, 3)]).draw(shader)
                finally:
                    gpu.matrix.pop_projection()
                    gpu.matrix.pop()
            else:
                if kind == 'EDGE':
                    lines = [tuple(c.co[c.edges[i, 0]]), tuple(c.co[c.edges[i, 1]])]
                else:
                    verts = [tuple(c.co[v]) for v in c.face_verts(i)]
                    fill = gpu.shader.from_builtin('UNIFORM_COLOR')
                    tris = [(0, k, k + 1) for k in range(1, len(verts) - 1)]
                    fill.uniform_float("color", (*rgb, p.face_alpha))
                    batch_for_shader(fill, 'TRIS', {"pos": verts}, indices=tris).draw(fill)
                    lines = []
                    for k in range(len(verts)):
                        lines += [verts[k], verts[(k + 1) % len(verts)]]
                shader = gpu.shader.from_builtin('POLYLINE_UNIFORM_COLOR')
                shader.uniform_float("viewportSize", (region.width, region.height))
                shader.uniform_float("lineWidth", p.line_width * scale)
                shader.uniform_float("color", (*rgb, 1.0))
                batch_for_shader(shader, 'LINES', {"pos": lines}).draw(shader)
        except ReferenceError:
            pass
        finally:
            gpu.state.blend_set('NONE')


class PRESELECT_GGT_highlight(bpy.types.GizmoGroup):
    bl_idname = "PRESELECT_GGT_highlight"
    bl_label = "Pre-selection Highlight"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'WINDOW'
    bl_options = {'3D', 'PERSISTENT', 'SELECT'}

    @classmethod
    def poll(cls, context):
        return _enabled(context)

    @classmethod
    def setup_keymap(cls, keyconfig):
        # Blender only lets a gizmo be hovered when the group's keymap has an item matching the modifiers
        # currently held. The default keymap has no Alt item, so the highlight vanished while Option / Alt
        # was held. Every combination of Alt / Shift / Ctrl gets an item here, INCLUDING the unmodified
        # one (an earlier attempt without it disabled the highlight altogether). The gizmo sets
        # hide_keymap, so these items never start a tweak: they only keep the highlight hoverable.
        try:
            km = keyconfig.keymaps.new(name="Pre-selection Highlight", space_type='VIEW_3D',
                                       region_type='WINDOW')
            if len(km.keymap_items) == 0:
                for alt in (False, True):
                    for shift in (False, True):
                        for ctrl in (False, True):
                            km.keymap_items.new("gizmogroup.gizmo_tweak", 'LEFTMOUSE', 'PRESS',
                                                alt=alt, shift=shift, ctrl=ctrl)
            return km
        except Exception as ex:   # never break the highlight: fall back to Blender's default keymap
            print("Pre-selection Highlight: keymap setup failed:", ex)
            return None

    def setup(self, context):
        self.gz = self.gizmos.new(PRESELECT_GT_highlight.bl_idname)
        self.gz.hide_keymap = True   # a click goes to the select tool, not to a gizmo tweak
        _last[:] = [self]


classes = (PRESELECT_GT_highlight, PRESELECT_GGT_highlight)


def register_handlers():
    h = bpy.app.handlers
    if invalidate not in h.depsgraph_update_post:
        h.depsgraph_update_post.append(invalidate)
    for lst in (h.load_post, h.undo_post, h.redo_post):
        if _drop_all not in lst:
            lst.append(_drop_all)


def clear():
    _last.clear()
    _cache.clear()
    h = bpy.app.handlers
    if invalidate in h.depsgraph_update_post:
        h.depsgraph_update_post.remove(invalidate)
    for lst in (h.load_post, h.undo_post, h.redo_post):
        if _drop_all in lst:
            lst.remove(_drop_all)
