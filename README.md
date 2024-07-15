# torchrender3d
A triangle-projection-based barebones 3D renderer with the most basic steps of the rendering process built from scratch using pytorch. **Purely educational**

# What it's not
- A game engine
- fast
- usable
- a serious project

# What it is
- educational (primarily for me)
- barely working 3D rendering from scratch

# Explanation

I derived some of the algorithms and formulas myself:
- vertex projection
- depth interpolation
- very primitive lighting
- texture mapping
... and I'm going to explain what I did here

## Vertex projection
1. apply rotation to vertices of one object's triangles according to object rotation
    - I implemented this using rotation matrices
2. rotate vertices of object according to camera orientation so camera is looking towards (0, 0, 1)
    - Again, rotation matrices ftw
3. divide every rotated point's coordinates by its z position to get (x/z, y/z, z/z)
    - Note: z = (vertex.pos - camera.pos) @ camera.orient, where @ denotes dot product
4. remove the z component as it is now z/z = 1

And now you've got the projected 2D positions where on the screen your vertices will land... **almost**.
If one or two of the vertices of a triangle are behind the camera and the rest is in front of the camera, this approach will not work.
So what do you do to fix this?
Well, in my hacky renderer, I just skip the triangle XD.

## Depth buffering/interpolation
Depth buffering is a rendering technique where you have a buffer that stores the z distance of whatever is rendered
on every pixel from the camera. Then, when rendering all your triangles, you choose what to render to the 
screen at some pixel based on how far away all the triangles that cover the pixel are from the camera.
Sounds easy enough, right?
There's just a slight problem... we don't actually know how far away every point on a triangles surface is from the
camera, just how far away the vertices are.
This is where depth interpolation comes into play.

### Depth interpolation
1. using the points A, B, C of the 2D projected triangle (result of the vertex projection step) to compute vectors AB and AC
    - AB, AC are the vectors pointing from A to B and C respectively
2. doing a [change of basis](https://www.youtube.com/watch?v=P2LTAUO1TdA&list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab&index=13)
to express every point (x, y) within the triangle as some combination of AB and AC, so 
(x, y) -> (b, c) such that (x, y) = b * AB + c * AC.
    - Note that this is equivalent to saying (x, y) = ((AB_0, AC_0), (AB_1, AC_1)) @ (b, c) where @ denotes matrix-vector multiplication.
    - Therefore, (b, c) = ((AB_0, AC_0), (AB_1, AC_1))^(-1) @ (x, y) which is a change of basis
3. then, the depth (distance from camera in z direction) of point (x, y) is
z = z_A + b * (z_B - z_A) + c * (z_C - z_A)
    - Note this is just a linear interpolation

### Depth buffering
Now, we got the z distance of every point within the triangle from the camera using the algorithm described in the _Depth interpolation_ section.

We can now use this to render our triangles.

```pseudocode
init depth buffer to all float('inf')
init screen buffer to all 0s  # screen buffer is what is rendered
for every triangle
    for every pixel in triangle
        compute z using depth interpolation
        if entry in depth buffer at that pixel is bigger than computed z value
            update the entry in the depth buffer to computed z value
            compute color of that pixel on current triangle (lighting and textures)
            update screen buffer content at pixel to new computed color
draw screen buffer (or depth buffer :D)
```

## Lighting

For very primitive lighting, compute the dot product of the surface normal of the triangle and the lighting direction, 
both normalized vectors. Then clamp the result in range [-1, 1] between 0 and 1, since you can't have negative lighting.

The surface normal is computed by taking the cross product of AB and AC, which, intuitively, means finding the vector
that is normal to both of those vectors (and thus the triangle surface) and obeys the right-hand-rule.
However, since you probably don't want the choice of A, B, and C (which is arbitrary) to decide the direction of your
surface normal (either 1 * cross_prod or -1 * cross_prod), you have to choose based on some criteria.
I think what makes the most sense is choosing the one that is facing the camera, at least in simple cases, 
though one could also somehow attach some information about which surface normal to choose to the triangle.

## Texture mapping
In simple terms, I used the change of basis from the _Depth interpolation_ section to map the 2D coordinates with the alternative bases to the texture coordinates.
This was done by multiplying these coordinates (within the triangle, each coordinate is between 0 and 1) by the texture size, rounding and clamping them to the texture size.
Then, I just used the texture coordinates to get the color of the pixel on the texture that corresponds to the 2D coordinates and wrote that to the screen buffer.

```pseudocode
texture_coords = clamp(round(uv_triangle_coords * (texture_size - 1)), 0, texture_size - 1)
```