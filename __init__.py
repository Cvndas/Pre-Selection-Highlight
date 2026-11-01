# SPDX-License-Identifier: GPL-3.0-or-later
"""Pre-selection Highlight: see the vertex / edge / face under the cursor before you click it.

Standalone add-on derived from Maya Workflow Kit for Blender (GPL-3.0-or-later,
https://github.com/5hyuns/maya-workflow-kit-for-blender, Copyright 2026 ohnsk).
"""

import bpy
from bpy.props import BoolProperty, FloatProperty, FloatVectorProperty

from . import preselect


def _redraw(self, context):
    for win in context.window_manager.windows:
        for area in win.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()


class PreselectPreferences(bpy.types.AddonPreferences):
    bl_idname = __package__

    enabled: BoolProperty(name="Pre-selection Highlight", default=True, update=_redraw,
                          description="Highlight the component under the cursor in Edit Mode before clicking")
    with_transform_tools: BoolProperty(
        name="With Move / Rotate / Scale tools", default=True, update=_redraw,
        description="Also highlight while a transform tool is active (stays off over its gizmo)")
    color: FloatVectorProperty(name="Colour", subtype='COLOR_GAMMA', size=3, min=0.0, max=1.0,
                               default=(0.8, 239 / 255, 1.0), update=_redraw)   # #CCEFFF
    face_alpha: FloatProperty(name="Face Fill Opacity", min=0.0, max=1.0, default=0.35, update=_redraw)
    vert_size: FloatProperty(name="Vertex Size", min=1.0, max=30.0, default=3.0, subtype='PIXEL', update=_redraw)
    line_width: FloatProperty(name="Line Width", min=1.0, max=10.0, default=2.0, subtype='PIXEL', update=_redraw)
    radius: FloatProperty(name="Pick Radius", min=2.0, max=100.0, default=18.0, subtype='PIXEL',
                          description="Cursor distance (pixels) within which a vertex or edge is picked")

    def draw(self, context):
        col = self.layout.column()
        col.prop(self, "enabled")
        col.prop(self, "with_transform_tools")
        col.separator()
        col.prop(self, "color")
        col.prop(self, "face_alpha")
        col.prop(self, "vert_size")
        col.prop(self, "line_width")
        col.prop(self, "radius")


def _overlay_draw(self, context):
    p = preselect.prefs()
    if p is not None and context.mode == 'EDIT_MESH':
        self.layout.prop(p, "enabled", text="Pre-selection Highlight")



_classes = (PreselectPreferences, *preselect.classes)


def register():
    for cls in _classes:
        bpy.utils.register_class(cls)
    preselect.register_handlers()
    bpy.types.VIEW3D_PT_overlay_edit_mesh.append(_overlay_draw)


def unregister():
    bpy.types.VIEW3D_PT_overlay_edit_mesh.remove(_overlay_draw)
    preselect.clear()
    for cls in reversed(_classes):
        bpy.utils.unregister_class(cls)
