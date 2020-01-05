import bpy
import numpy as np
import png
import pyexr
from math import pi

dir = "/tmp/blender_renders/"

cam = bpy.data.objects['Camera']
scene = bpy.context.scene

bpy.data.scenes['Scene'].frame_start = 1
bpy.data.scenes['Scene'].frame_end = 1

bpy.context.scene.render.use_compositing = True
bpy.context.scene.use_nodes = True
tree = bpy.context.scene.node_tree
links = tree.links

# Clean nodes
for n in tree.nodes:
    tree.nodes.remove(n)

# Setup views for different outputs
renderlayer = tree.nodes.new('CompositorNodeRLayers')
rl_out = {
    "Image" : None,
    "Alpha" : None,
    "Depth" : None,
    "Normal" : None, 
    "DiffDir" : None,
    "DiffInd" : None,
    "DiffCol" : None    
}

for output in renderlayer.outputs:
    if output.name in rl_out.keys():
        rl_out[output.name] = output

width = 1280
height = 960

bpy.context.scene.render.resolution_x = width
bpy.context.scene.render.resolution_y = height

bpy.context.scene.render.resolution_percentage = 100 #make sure scene height and width are ok (edit)

# Only one composite can be active at a time, so we have to render multiple times
composite = tree.nodes.new('CompositorNodeComposite')
links.new(rl_out["Image"], composite.inputs[0])

# Connect image to color
vcolor = tree.nodes.new('CompositorNodeViewer')
vcolor.name = "Color Output"
links.new(rl_out["Image"], vcolor.inputs[0]) # image -> image
links.new(rl_out["Alpha"], vcolor.inputs[1]) # alpha -> alpha

print("Rendering color")
bpy.ops.render.render()

img_color = np.array(bpy.data.images['Viewer Node'].pixels)
img_color = np.reshape(img_color * 255, (height, width, 4))
img_color = img_color[:, :, :-1]

tree.nodes.remove(vcolor)

# Connect depth to image channel
vdepth = tree.nodes.new('CompositorNodeViewer')
vdepth.name = "Depth Output"
vdepth.use_alpha = True
links.new(rl_out["Depth"], vdepth.inputs[0])

print("Rendering depth")
bpy.ops.render.render()

img_depth = np.array(bpy.data.images['Viewer Node'].pixels)
img_depth = np.reshape(img_depth * 1000, (height, width, 4))
cam_min_mm = 100
cam_max_mm = 2000
img_depth[img_depth < cam_min_mm] = 0
img_depth[img_depth > cam_max_mm] = 0
img_depth = img_depth[:, :, :-1]

img_mask = np.zeros_like(img_depth)
img_mask[img_depth > 0] = 1
img_mask = np.array(img_mask, dtype=np.bool)

tree.nodes.remove(vdepth)

# Connect normals to normals
vnormals = tree.nodes.new('CompositorNodeViewer')
vnormals.name = "Normals Output"
vnormals.use_alpha = False

links.new(rl_out["Normal"], vnormals.inputs[0])

print("Rendering normals")
bpy.ops.render.render()

img_normals = np.array(bpy.data.images['Viewer Node'].pixels)
img_normals = np.reshape(img_normals, (height, width, 4))
img_normals = img_normals[:, :, :-1]

tree.nodes.remove(vnormals)

# Albedo (diffuse color + glossy color from cycle lighting pass)
# No glossy color in our passes
valbedo = tree.nodes.new('CompositorNodeViewer')
valbedo.name = "Albedo Output"
valbedo.use_alpha = False

# DiffCol
links.new(rl_out["DiffCol"], valbedo.inputs[0])

print("Rendering albedo")
bpy.ops.render.render()

img_albedos = np.array(bpy.data.images['Viewer Node'].pixels)
img_albedos = np.reshape(img_albedos * 255, (height, width, 4))
img_albedos = img_albedos[:, :, :-1]

tree.nodes.remove(valbedo)

# Shading (diffuse direct + diffuse indirect from cycle lighting pass)
vshading = tree.nodes.new('CompositorNodeViewer')
vshading.name = "Shading Output"
vshading.use_alpha = False

vadd = tree.nodes.new('CompositorNodeMath') # default: Add

links.new(rl_out["DiffDir"], vadd.inputs[0])
links.new(rl_out["DiffInd"], vadd.inputs[1])
links.new(vadd.outputs[0], vshading.inputs[0])

print("Rendering shading")
bpy.ops.render.render()

img_shading = np.array(bpy.data.images['Viewer Node'].pixels)
img_shading = np.reshape(img_shading * 255, (height, width, 4))
img_shading = img_shading[:, :, :-1]

tree.nodes.remove(vadd)
tree.nodes.remove(vshading)

# Process images

print(img_color.shape, np.min(img_color), np.max(img_color))
print(img_depth.shape, np.min(img_depth), np.max(img_depth))
print(img_mask.shape, np.min(img_mask), np.max(img_mask))
print(img_normals.shape, np.min(img_normals), np.max(img_normals))
print(img_albedos.shape, np.min(img_albedos), np.max(img_albedos))
print(img_shading.shape, np.min(img_shading), np.max(img_shading))

img_color[img_mask < 1] = 0
img_color = np.reshape(img_color, (-1, width * 3))
img_color = np.flip(img_color)
img_color = np.array(img_color, dtype=np.uint8)

img_depth = np.reshape(img_depth[:, :, 0], (-1, width))
img_depth = np.flip(img_depth)

img_depth_png = np.array(img_depth, dtype=np.uint16) 

img_normals = np.array(img_normals, dtype=np.float) 
img_normals = (img_normals + 1) * 255/2 # range [-1, 1] to [0, 255/4095]
img_normals[img_mask < 1] = 0

img_normals = np.reshape(img_normals, (height, width, 3))
img_normals = np.flip(img_normals)

img_normals_png = np.reshape(img_normals, (-1, width * 3))
img_normals_png = np.array(img_normals_png, dtype=np.uint8)

img_mask = np.reshape(img_mask[:, :, 0], (-1, width))
img_mask = np.flip(img_mask)

img_albedos = np.reshape(img_albedos, (-1, width * 3))
img_albedos = np.flip(img_albedos)
img_albedos = np.array(img_albedos, dtype=np.uint8)

img_shading = np.reshape(img_shading, (height, width, 3))
img_shading = np.flip(img_shading)

img_shading_png = np.reshape(img_shading, (-1, width * 3))
img_shading_png = np.array(img_shading_png, dtype=np.uint8)

print("Saving color to", dir + "color.png")
with open(dir + "color.png", "wb") as f_color:
    w = png.Writer(width, height)
    w.write(f_color, img_color)

print("Saving depth to", dir + "depth.png")
with open(dir + "depth.png", "wb") as f_depth:
    w = png.Writer(width, height, greyscale=True, bitdepth=16)
    w.write(f_depth, img_depth_png)

print("Saving normals to", dir + "normals.png")
with open(dir + "normals.png", "wb") as f_normals:
    w = png.Writer(width, height)
    w.write(f_normals, img_normals_png)

print("Saving mask to", dir + "mask.png")
with open(dir + "mask.png", "wb") as f_mask:
    w = png.Writer(width, height, greyscale=True, bitdepth=1)
    w.write(f_mask, img_mask)

print("Saving albedo to", dir + "albedo.png")
with open(dir + "albedo.png", "wb") as f_albedo:
    w = png.Writer(width, height)
    w.write(f_albedo, img_albedos)

print("Saving shading to", dir + "shading.png")
with open(dir + "shading.png", "wb") as f_shading:
    w = png.Writer(width, height)
    w.write(f_shading, img_shading_png)

print("Saving depth to", dir + "depth.exr")
pyexr.write(dir + "depth.exr", img_depth.astype(np.float32))

print("Saving shading", img_shading.shape, "to", dir + "shading.exr")
pyexr.write(dir + "shading.exr", img_shading.astype(np.float32), 
            channel_names=['R','G','B'])

print("Saving normals", img_shading.shape, "to", dir + "normals.exr")
pyexr.write(dir + "normals.exr", img_normals.astype(np.float32), 
            channel_names=['R','G','B'])

print("Done!")
