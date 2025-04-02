#PBR class definition
#from curses import OK
import moderngl
from PIL import Image
import numpy as np
import os
from pyrr import Matrix44, Vector3
tile_maps_keys = ['basecolor','diffuse','displacement','height','metallic','normal','opacity','roughness','specular','blend_mask']

"""
    PBR class:
        A wrapper class for tile maps and rendered image.
        The class has a renderer function, to_render(), but scene parameters like the matricies we see in __init__() we couldn't find right one to use.
        Vertex and Fragment Shaders are credit to and based as written in the comments below.
"""
class PBR:
    def __init__(self, tile_maps = None):
        self.render = None
        self.tile_maps = {}
        if isinstance(tile_maps, dict):
            #self.tile_maps = {tile: tile_maps.get(tile, None) for tile in tile_maps_keys}
            for tile in tile_maps_keys:
                if tile in tile_maps:
                    self.tile_maps[tile] = tile_maps[tile]
                else:
                    self.tile_maps[tile] = None
        # Define object transformations (position, rotation, scale)
        self.model_matrix = Matrix44.from_translation(Vector3([0.0, 0.0, 0.0]))  # Model matrix (Identity by default)

        # Define camera view matrix (look at the object)
        eye = Vector3([0.0, 0.0, 3.0])        # Camera position
        target = Vector3([0.0, 0.0, 0.0])     # Look-at target
        up = Vector3([0.0, 1.0, 0.0])         # Up direction
        self.view_matrix = Matrix44.look_at(eye, target, up)

        # Normal matrix (inverse-transpose of the Model-View matrix)
        self.model_view = self.view_matrix @ self.model_matrix
        self.normal_matrix = np.linalg.inv(self.model_view[:3, :3]).T  # Extract the 3x3 upper-left part
        aspect_ratio = 1
        self.projection_matrix = Matrix44.perspective_projection(45.0, aspect_ratio, 0.1, 100.0)

        # Texture rotation matrix (optional)
        """angle = np.radians(45)  # Rotate textures 45 degrees
        texture_rotation = np.array([
            [np.cos(angle), -np.sin(angle), 0.0, 0.0],
            [np.sin(angle),  np.cos(angle), 0.0, 0.0],
            [0.0,            0.0,           1.0, 0.0],
            [0.0,            0.0,           0.0, 1.0]
        ], dtype='f4')"""
        self.texture_rotation = texture_rotation = np.identity(4, dtype='f4') # No rotation (0 degrees)
        self.texture_repeat = np.array([1, 1], dtype='f4')
        self.camera_position = np.array([0.0, 2.0, 5.0], dtype=np.float32)  # Default camera position - slightly above and in front
    
    def save(self, dir):
        if not os.path.isdir(dir):
            os.makedirs(dir, exist_ok=True)
        for tile in self.tile_maps:
            if self.tile_maps[tile] is not None:
                self.tile_maps[tile].save(f"{dir}/{tile}.png")
        if self.render is not None:
            self.render.save(f"{dir}/render.png")



    def set_matrices(self, model, view, projection):
        self.model_matrix = model
        self.view_matrix = view
        self.projection_matrix = projection
    def set_camera_position(self, cp):
        self.camera_position = cp
    def is_tile_maps_empty(self):
        # Returns True when tile_maps is an empty dict, or when all values are None/False.
        return self.tile_maps == {} or (not any(self.tile_maps.values()))
    def to_render(self):
        # Returns A PBR rendered material fit for Material Segmentation
        if not self.render is None:
            return self.render
        if self.is_tile_maps_empty():
            return None
        
        """# Initialize ModernGL context
        ctx = moderngl.create_context(
            standalone=True,
            backend='egl',
            # These are OPTIONAL if you want to load a specific version
            libgl='libGL.so.1',
            libegl='libEGL.so.1',
        )"""
        # Headless ModernGL context
        ctx = moderngl.create_standalone_context()

        # Load textures
        basecolor_img = self.tile_maps["basecolor"]  # RGB
        normal_img = self.tile_maps["normal"]  # RGBA
        roughness_img = self.tile_maps["roughness"].convert('L')  # Grayscale
        metallic_img = self.tile_maps["metallic"].convert('L')  # Grayscale
        height_img = self.tile_maps["height"].convert('L')  # Grayscale

        basecolor = ctx.texture(basecolor_img.size, 3, basecolor_img.tobytes())
        normalmap = ctx.texture(normal_img.size, 4, normal_img.tobytes())
        roughnessmap = ctx.texture(roughness_img.size, 1, roughness_img.tobytes())  # Single channel
        metallicmap = ctx.texture(metallic_img.size, 1, metallic_img.tobytes())  # Single channel
        #heightmap = ctx.texture(height_img.size, 1, height_img.tobytes(), dtype = 'f2')  # Single channel - for 16bit map
        heightmap = ctx.texture(height_img.size, 1, height_img.tobytes())  # Testing is done with a bigger than 16bit.

        basecolor.use(location=0)
        normalmap.use(location=1)
        roughnessmap.use(location=2)
        metallicmap.use(location=3)
        heightmap.use(location=4)
        #Shader Program         #Need to learn more on shader making.
        #Vertex and Fragment shader - @Credit Jon Macey
        # Based on PBR lectures from https://nccastaff.bournemouth.ac.uk/jmacey/
        vertex_shader = '''
        #version 300 es
        // Based on https://learnopengl.com/PBR/Theory
        // textures from https://freepbr.com/
        precision highp float;
        uniform mat4 modelMatrix;
        uniform mat4 modelViewMatrix;
        uniform mat4 projectionMatrix;
        uniform mat3 normalMatrix;
        uniform mat4 textureRotation;
        uniform vec2 textureRepeat;
        in vec3 position;
        in vec3 normal;
        in vec2 uv;

        out vec2 TexCoords;
        out vec3 WorldPos;
        out vec3 Normal;

        void main()
        {
          WorldPos = vec3(modelMatrix * vec4(position, 1.0f));
          Normal=normalMatrix*normal;
          TexCoords=mat2(textureRotation)*(uv*textureRepeat);  
          gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );
        }

        '''
        fragment_shader = '''
        #version 300 es
        precision highp float;

        in vec2 TexCoords;
        in vec3 WorldPos;
        in vec3 Normal;

        layout (location = 0) out vec4 fragColour;

        // Material parameters
        uniform sampler2D albedoMap;
        uniform sampler2D normalMap;
        uniform sampler2D metallicMap;
        uniform sampler2D roughnessMap;
        uniform sampler2D heightMap;  // New height map
        uniform float heightScale;    // Height map scaling factor
        uniform float roughnessScale;

        // Lights
        uniform vec3 lightPositions[4];
        uniform vec3 lightColors[4];
        uniform float exposure;
        uniform vec3 cameraPosition;

        const float PI = 3.14159265359;

        // Parallax Mapping
        vec2 parallaxMapping(vec2 texCoords, vec3 viewDir) {
            // Parallax mapping with linear interpolation for smoother results
            const int numLayers = 10; // Reduce iterations for performance
            float layerDepth = 1.0 / float(numLayers);
            float currentLayerDepth = 0.0;

            vec2 deltaTexCoords = normalize(viewDir.xy) * heightScale / float(numLayers);
            vec2 currentTexCoords = texCoords;

            float currentHeight = texture(heightMap, currentTexCoords).r;

            for (int i = 0; i < numLayers; i++) {
                currentLayerDepth += layerDepth;
                currentTexCoords -= deltaTexCoords;
                float sampledHeight = texture(heightMap, currentTexCoords).r;

                // If we've passed the height, interpolate between layers
                if (currentLayerDepth >= sampledHeight) {
                    vec2 prevTexCoords = currentTexCoords + deltaTexCoords;
                    float prevHeight = texture(heightMap, prevTexCoords).r;

                    float weight = (currentLayerDepth - sampledHeight) / (currentLayerDepth - prevHeight);
                    return mix(currentTexCoords, prevTexCoords, weight);
                }
            }

            return texCoords; // Default to base coordinates if no match
        }


        // Normal Mapping
        vec3 getNormalFromMap() {
            vec3 tangentNormal = texture(normalMap, TexCoords).xyz * 2.0 - 1.0;

            vec3 Q1  = dFdx(WorldPos);
            vec3 Q2  = dFdy(WorldPos);
            vec2 st1 = dFdx(TexCoords);
            vec2 st2 = dFdy(TexCoords);

            vec3 N   = normalize(Normal);
            vec3 T  = normalize(Q1 * st2.t - Q2 * st1.t);
            vec3 B  = -normalize(cross(N, T));
            mat3 TBN = mat3(T, B, N);

            return normalize(TBN * tangentNormal);
        }

        float DistributionGGX(vec3 N, vec3 H, float roughness) {
            float a = roughness * roughness;
            float a2 = a * a;
            float NdotH = max(dot(N, H), 0.0);
            float NdotH2 = NdotH * NdotH;

            float nom   = a2;
            float denom = (NdotH2 * (a2 - 1.0) + 1.0);
            denom = PI * denom * denom;

            return nom / denom;
        }

        float GeometrySchlickGGX(float NdotV, float roughness) {
            float r = (roughness + 1.0);
            float k = (r * r) / 8.0;

            float nom   = NdotV;
            float denom = NdotV * (1.0 - k) + k;

            return nom / denom;
        }

        float GeometrySmith(vec3 N, vec3 V, vec3 L, float roughness) {
            float NdotV = max(dot(N, V), 0.0);
            float NdotL = max(dot(N, L), 0.0);
            float ggx2 = GeometrySchlickGGX(NdotV, roughness);
            float ggx1 = GeometrySchlickGGX(NdotL, roughness);

            return ggx1 * ggx2;
        }

        vec3 fresnelSchlick(float cosTheta, vec3 F0) {
            return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
        }

        void main() {
            vec3 viewDir = normalize(cameraPosition - WorldPos);
    

            // Adjusted texture coordinates with parallax mapping
            vec2 parallaxTexCoords = parallaxMapping(TexCoords, viewDir);
            //vec2 parallaxTexCoords = TexCoords;

            // Material properties
            vec3 albedo = pow(texture(albedoMap, parallaxTexCoords).rgb, vec3(2.2));
            //vec3 albedo = texture(albedoMap, parallaxTexCoords).rgb;
            float metallic = texture(metallicMap, parallaxTexCoords).r;
            float roughness = texture(roughnessMap, parallaxTexCoords).r * roughnessScale;
            vec3 N = getNormalFromMap();


            // calculate reflectance at normal incidence; if dia-electric (like plastic) use F0
            // of 0.04 and if it's a metal, use the albedo color as F0 (metallic workflow)
            vec3 F0 = vec3(0.04);
            F0 = mix(F0, albedo, metallic);

            // Reflectance equation
            vec3 Lo = vec3(0.0);
            for (int i = 0; i < 4; ++i) {
                // calculate per-light radiance
                vec3 L = normalize(lightPositions[i] - WorldPos);
                vec3 H = normalize(viewDir + L);
                float distance = length(lightPositions[i] - WorldPos);
                float attenuation = 1.0 / (distance * distance);
                vec3 radiance = lightColors[i] * attenuation;

                // Cook-Torrance BRDF
                float NDF = DistributionGGX(N, H, roughness);
                float G = GeometrySmith(N, viewDir, L, roughness);
                vec3 F = fresnelSchlick(max(dot(H, viewDir), 0.0), F0);

                // kS is equal to Fresnel
                vec3 kS = F;
                // for energy conservation, the diffuse and specular light can't
                // be above 1.0 (unless the surface emits light); to preserve this
                // relationship the diffuse component (kD) should equal 1.0 - kS.
                vec3 kD = vec3(1.0) - kS;
                // multiply kD by the inverse metalness such that only non-metals
                // have diffuse lighting, or a linear blend if partly metal (pure metals
                // have no diffuse light).
                kD *= 1.0 - metallic;

                // scale light by NdotL
                float NdotL = max(dot(N, L), 0.0);

                vec3 specular = (NDF * G * F) / (4.0 * max(dot(N, viewDir), 0.0) * NdotL + 0.001);
                Lo += (kD * albedo / PI + specular) * radiance * NdotL;
            }

            vec3 ambient = vec3(0.03) * albedo;  // Simple ambient light
            vec3 color = ambient + Lo;

            // HDR tonemapping
            color = color / (color + vec3(1.0));
            // gamma correct
            color = pow(color, vec3(1.0/exposure));

            fragColour = vec4(color, 1.0);
            //fragColour = vec4(albedo, 1.0);
            //fragColour = vec4(N * 0.5 + 0.5, 1.0);
            //fragColour = vec4(Lo, 1.0);
        }

        '''
        # Compile shader program
        prog = ctx.program(vertex_shader=vertex_shader, fragment_shader=fragment_shader)
        if not prog:
            print(f"Error compiling shader: {prog}")
        # Define vertices for a cube
        vertex_data = np.array([
            # Positions          Normals         UVs
            -1.0, -1.0, 0.0,    0.0, 0.0, 1.0,    0.0, 0.0,
             1.0, -1.0, 0.0,    0.0, 0.0, 1.0,    1.0, 0.0,
            -1.0,  1.0, 0.0,    0.0, 0.0, 1.0,    0.0, 1.0,
             1.0,  1.0, 0.0,    0.0, 0.0, 1.0,    1.0, 1.0,
        ], dtype='f4')

        # Create buffer
        vbo = ctx.buffer(vertex_data.tobytes())
        vao = ctx.vertex_array(prog,[(vbo, '3f 3f 2f', 'position', 'normal', 'uv')],)
        
        # Texture repeat and rotation
        prog['textureRotation'].write(self.texture_rotation.tobytes())
        prog['textureRepeat'].value = self.texture_repeat

        # Shader program uniforms
        prog['modelMatrix'].write(self.model_matrix.astype('f4').tobytes())
        prog['modelViewMatrix'].write(self.model_view.astype('f4').tobytes())
        prog['projectionMatrix'].write(self.projection_matrix.astype('f4').tobytes())
        prog['normalMatrix'].write(self.normal_matrix.astype('f4').tobytes())
        prog['textureRotation'].write(self.texture_rotation.astype('f4').tobytes())
        prog['textureRepeat'].write(self.texture_repeat.tobytes())
        prog['roughnessScale'].value = 0.8
        prog['exposure'].value = 2.2
        #prog['heightScale'].value = 0.01  # Adjust this value for subtle parallax effect
        prog['heightScale'].value = 0.002

        # Light positions and intensities
        """light_positions = [
            [0.0, 2.0, 2.0],  # Light 1
            [-2.0, 2.0, 2.0],  # Light 2
            [2.0, 2.0, -2.0],  # Light 3
            [0.0, 2.0, -2.0]   # Light 4
        ]
        light_colors = [
            [300.0, 300.0, 300.0],  # Light 1 Power
            [300.0, 300.0, 300.0],  # Light 2 Power
            [300.0, 300.0, 300.0],  # Light 3 Power
            [300.0, 300.0, 300.0]   # Light 4 Power
        ]"""
        light_positions = [
            [2.0, 2.0, 2.0],   # Light 1
            [-2.0, 2.0, 2.0],  # Light 2
            [2.0, -2.0, 2.0],  # Light 3
            [2.0, 2.0, -2.0]   # Light 4
        ]
        """light_colors = [
            [150.0, 150.0, 150.0],  # Light 1 Power
            [150.0, 150.0, 150.0],  # Light 2 Power
            [150.0, 150.0, 150.0],  # Light 3 Power
            [150.0, 150.0, 150.0]   # Light 4 Power
        ]"""
        """light_colors = [
            [50.0, 50.0, 50.0],
            [50.0, 50.0, 50.0],
            [50.0, 50.0, 50.0],
            [50.0, 50.0, 50.0]
        ]"""
        light_colors = [
            [10., 10., 10.],
            [10., 10., 10.],
            [10., 10., 10.],
            [10., 10., 10.]
        ]


        prog['lightPositions'].write(np.array(light_positions, dtype='f4').tobytes())
        prog['lightColors'].write(np.array(light_colors, dtype='f4').tobytes())


        # Create framebuffer for offscreen rendering
        fbo = ctx.framebuffer(
            color_attachments=[ctx.texture((512, 512), 3)],
        )
        fbo.use()
        ctx.clear(0.1, 0.1, 0.1)
        vao.render(moderngl.TRIANGLE_STRIP)

        data = fbo.read(components=3)
        image = Image.frombytes('RGB', fbo.size, data)
        image = image.transpose(Image.FLIP_TOP_BOTTOM)  # Flip because OpenGL origin is bottom-left
        self.render = image
        ctx.release()
        return image
