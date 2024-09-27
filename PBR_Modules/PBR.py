#PBR class definition
from curses import OK
import moderngl
from PIL import Image
import numpy as np
tile_maps_keys = ['basecolor','diffuse','displacement','height','metallic','normal','opeacity','roughness','specular','blend_mask']

class PBR:
    def __init__(self, render = None, tile_maps = None):
        self.render = render
        self.tile_maps = {}
        if isinstance(tile_maps, dict):
            for tile in tile_maps_keys:
                if tile in tile_maps:
                    self.tile_maps[tile] = tile_maps[tile]
                else:
                    self.tile_maps[tile] = None
        self.model_matrix = np.eye(4, dtype=np.float32)  # Identity matrix by default
        self.view_matrix = self.look_at(np.array([0.0, 2.0, 5.0], dtype=np.float32),  # Camera position
                                        np.array([0.0, 0.0, 0.0], dtype=np.float32),  # Look at the origin
                                        np.array([0.0, 1.0, 0.0], dtype=np.float32))  # Up vector
        self.projection_matrix = self.perspective(np.radians(45.0), 1.0, 0.1, 100.0)  # Perspective projection
        self.texture_rotation = np.eye(4, dtype=np.float32)  # No texture rotation by default
        self.texture_repeat = (1.0, 1.0)  # Default to no repeat
        self.camera_position = np.array([0.0, 2.0, 5.0], dtype=np.float32)  # Default camera position - slightly above and in front
    def look_at(self, eye, target, up):
        f = np.linalg.norm(target - eye)
        f = (target - eye) / f
        s = np.cross(f, up)
        s = s / np.linalg.norm(s)
        u = np.cross(s, f)
        look_at_matrix = np.array([
            [s[0], u[0], -f[0], 0],
            [s[1], u[1], -f[1], 0],
            [s[2], u[2], -f[2], 0],
            [-np.dot(s, eye), -np.dot(u, eye), np.dot(f, eye), 1]
        ], dtype=np.float32)
        return look_at_matrix

    def perspective(self, fovy, aspect, near, far):
        f = 1.0 / np.tan(fovy / 2)
        return np.array([
            [f / aspect, 0, 0, 0],
            [0, f, 0, 0],
            [0, 0, (far + near) / (near - far), (2 * far * near) / (near - far)],
            [0, 0, -1, 0]
        ], dtype=np.float32)
    def set_matrices(self, model, view, projection):
        self.model_matrix = model
        self.view_matrix = view
        self.projection_matrix = projection
    def set_camera_position(self, cp):
        camera_position = cp
    def is_tile_maps_empty(self):
        # Returns True when tile_maps is an empty dict, or when all values are None/False.
        return self.tile_maps == {} or (not any(self.tile_maps.values()))
    def to_render(self):
        # Returns A PBR rendered material fit for Material Segmentation
        if not self.render is None:
            return self.render
        if self.is_tile_maps_empty():
            return None
        
        # Initialize ModernGL context
        ctx = moderngl.create_context(standalone=True)
        
        # Load tile maps
        texture_dict = {}
        for tile in self.tile_maps:
            if not self.tile_maps[tile] is None:
                texture_dict[tile] = ctx.texture((512, 512), 4, self.tile_maps[tile].tobytes())
        
        # Load tile maps (assuming self.tile_maps contains PIL Image objects)
        texture_dict = {}
        for key in tile_maps_keys:
            if self.tile_maps.get(key) is not None:
                # Convert PIL Image to bytes and then create texture
                image = self.tile_maps[key]
                texture_dict[key] = ctx.texture(image.size, 4, image.tobytes())
            else:
                # If the texture is missing, create a default grey texture
                texture_dict[key] = ctx.texture((1, 1), 4, np.ones((1, 1, 4), dtype=np.uint8).tobytes()) # Should fit into right size?

            # Bind texture to a texture unit
            texture_dict[key].use(location=tile_maps_keys.index(key))

        #Shader Program         #Need to learn more on shader making.
        #Vertex and Fragment shader - @Credit Jon Macey
        # Based on PBR lectures from https://nccastaff.bournemouth.ac.uk/jmacey/
        vertex_shader = """
        #version 330
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
        """
        fragment_shader = """
            #version 330 core
            precision highp float;
            // Based on https://learnopengl.com/PBR/Theory
            // textures from https://freepbr.com/
            in vec2 TexCoords;
            in vec3 WorldPos;
            in vec3 Normal;

            layout (location =0) out vec4 fragColour;

            // material parameters
            uniform sampler2D albedoMap;
            uniform sampler2D normalMap;
            uniform sampler2D metallicMap;
            uniform sampler2D roughnessMap;
            //uniform sampler2D aoMap;
            uniform float roughnessScale;
            // lights
            uniform vec3 lightPositions[4];
            uniform vec3 lightColors[4];
            uniform float exposure;
            uniform vec3 cameraPosition;

            const float PI = 3.14159265359;
            // ----------------------------------------------------------------------------
            // Easy trick to get tangent-normals to world-space to keep PBR code simplified.
            // Don't worry if you don't get what's going on; you generally want to do normal
            // mapping the usual way for performance anways; I do plan make a note of this
            // technique somewhere later in the normal mapping tutorial.
            vec3 getNormalFromMap()
            {
                vec3 tangentNormal = texture(normalMap, TexCoords).xyz * 2.0 - 1.0;

                vec3 Q1  = dFdx(WorldPos);
                vec3 Q2  = dFdy(WorldPos);
                vec2 st1 = dFdx(TexCoords);
                vec2 st2 = dFdy(TexCoords);

                vec3 N   = normalize(Normal);
                vec3 T  = normalize(Q1*st2.t - Q2*st1.t);
                vec3 B  = -normalize(cross(N, T));
                mat3 TBN = mat3(T, B, N);

                return normalize(TBN * tangentNormal);
            }
            // ----------------------------------------------------------------------------
            float DistributionGGX(vec3 N, vec3 H, float roughness)
            {
                float a = roughness*roughness;
                float a2 = a*a;
                float NdotH = max(dot(N, H), 0.0);
                float NdotH2 = NdotH*NdotH;

                float nom   = a2;
                float denom = (NdotH2 * (a2 - 1.0) + 1.0);
                denom = PI * denom * denom;

                return nom / denom;
            }
            // ----------------------------------------------------------------------------
            float GeometrySchlickGGX(float NdotV, float roughness)
            {
                float r = (roughness + 1.0);
                float k = (r*r) / 8.0;

                float nom   = NdotV;
                float denom = NdotV * (1.0 - k) + k;

                return nom / denom;
            }
            // ----------------------------------------------------------------------------
            float GeometrySmith(vec3 N, vec3 V, vec3 L, float roughness)
            {
                float NdotV = max(dot(N, V), 0.0);
                float NdotL = max(dot(N, L), 0.0);
                float ggx2 = GeometrySchlickGGX(NdotV, roughness);
                float ggx1 = GeometrySchlickGGX(NdotL, roughness);

                return ggx1 * ggx2;
            }
            // ----------------------------------------------------------------------------
            vec3 fresnelSchlick(float cosTheta, vec3 F0)
            {
                return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
            }
            // ----------------------------------------------------------------------------
            void main()
            {
                vec3 albedo     = pow(texture(albedoMap, TexCoords).rgb, vec3(2.2));
                float metallic  = texture(metallicMap, TexCoords).r;
                float roughness = texture(roughnessMap, TexCoords).r*roughnessScale;
                //float ao        = texture(aoMap, TexCoords).r;
                // Assume AO is fully lit (1.0) since we don't have an AO map
                float ao = 1.0;

                vec3 N = getNormalFromMap();
                vec3 V = normalize(cameraPosition - WorldPos);

                // calculate reflectance at normal incidence; if dia-electric (like plastic) use F0
                // of 0.04 and if it's a metal, use the albedo color as F0 (metallic workflow)
                vec3 F0 = vec3(0.04);
                F0 = mix(F0, albedo, metallic);

                // reflectance equation
                vec3 Lo = vec3(0.0);
                for(int i = 0; i < 4; ++i)
                {
                    // calculate per-light radiance
                    vec3 L = normalize(lightPositions[i] - WorldPos);
                    vec3 H = normalize(V + L);
                    float distance = length(lightPositions[i] - WorldPos);
                    float attenuation = 1.0 / (distance * distance);
                    vec3 radiance = lightColors[i] * attenuation;

                    // Cook-Torrance BRDF
                    float NDF = DistributionGGX(N, H, roughness);
                    float G   = GeometrySmith(N, V, L, roughness);
                    vec3 F    = fresnelSchlick(max(dot(H, V), 0.0), F0);

                    vec3 nominator    = NDF * G * F;
                    float denominator = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.001; // 0.001 to prevent divide by zero.
                    vec3 specular = nominator / denominator;

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

                    // add to outgoing radiance Lo
                    Lo += (kD * albedo / PI + specular) * radiance * NdotL;  // note that we already multiplied the BRDF by the Fresnel (kS) so we won't multiply by kS again
                }

                // ambient lighting (note that the next IBL tutorial will replace
                // this ambient lighting with environment lighting).
                vec3 ambient = vec3(0.03) * albedo * ao;

                vec3 color = ambient + Lo;

                // HDR tonemapping
                color = color / (color + vec3(1.0));
                // gamma correct
                color = pow(color, vec3(1.0/exposure));

                fragColour = vec4(color, 1.0);
            }

            """
        prog = ctx.program(vertex_shader=vertex_shader, fragment_shader=fragment_shader)
        
        # Define vertices for a cube
        vertices = np.asarray([
            # Positions          Normals             Texture Coords(u,v)
            # Back face
            -1.0, -1.0, -1.0,    0.0,  0.0, -1.0,    0.0, 0.0,
             1.0, -1.0, -1.0,    0.0,  0.0, -1.0,    1.0, 0.0,
             1.0,  1.0, -1.0,    0.0,  0.0, -1.0,    1.0, 1.0,
             1.0,  1.0, -1.0,    0.0,  0.0, -1.0,    1.0, 1.0,
            -1.0,  1.0, -1.0,    0.0,  0.0, -1.0,    0.0, 1.0,
            -1.0, -1.0, -1.0,    0.0,  0.0, -1.0,    0.0, 0.0,

            # Front face
            -1.0, -1.0,  1.0,    0.0,  0.0,  1.0,    0.0, 0.0,
             1.0, -1.0,  1.0,    0.0,  0.0,  1.0,    1.0, 0.0,
             1.0,  1.0,  1.0,    0.0,  0.0,  1.0,    1.0, 1.0,
             1.0,  1.0,  1.0,    0.0,  0.0,  1.0,    1.0, 1.0,
            -1.0,  1.0,  1.0,    0.0,  0.0,  1.0,    0.0, 1.0,
            -1.0, -1.0,  1.0,    0.0,  0.0,  1.0,    0.0, 0.0,

            # Left face
            -1.0,  1.0,  1.0,   -1.0,  0.0,  0.0,    1.0, 0.0,
            -1.0,  1.0, -1.0,   -1.0,  0.0,  0.0,    1.0, 1.0,
            -1.0, -1.0, -1.0,   -1.0,  0.0,  0.0,    0.0, 1.0,
            -1.0, -1.0, -1.0,   -1.0,  0.0,  0.0,    0.0, 1.0,
            -1.0, -1.0,  1.0,   -1.0,  0.0,  0.0,    0.0, 0.0,
            -1.0,  1.0,  1.0,   -1.0,  0.0,  0.0,    1.0, 0.0,

            # Right face
             1.0,  1.0,  1.0,    1.0,  0.0,  0.0,    1.0, 0.0,
             1.0,  1.0, -1.0,    1.0,  0.0,  0.0,    1.0, 1.0,
             1.0, -1.0, -1.0,    1.0,  0.0,  0.0,    0.0, 1.0,
             1.0, -1.0, -1.0,    1.0,  0.0,  0.0,    0.0, 1.0,
             1.0, -1.0,  1.0,    1.0,  0.0,  0.0,    0.0, 0.0,
             1.0,  1.0,  1.0,    1.0,  0.0,  0.0,    1.0, 0.0,

            # Bottom face
            -1.0, -1.0, -1.0,    0.0, -1.0,  0.0,    0.0, 0.0,
             1.0, -1.0, -1.0,    0.0, -1.0,  0.0,    1.0, 0.0,
             1.0, -1.0,  1.0,    0.0, -1.0,  0.0,    1.0, 1.0,
             1.0, -1.0,  1.0,    0.0, -1.0,  0.0,    1.0, 1.0,
            -1.0, -1.0,  1.0,    0.0, -1.0,  0.0,    0.0, 1.0,
            -1.0, -1.0, -1.0,    0.0, -1.0,  0.0,    0.0, 0.0,

            # Top face
            -1.0,  1.0, -1.0,    0.0,  1.0,  0.0,    0.0, 0.0,
             1.0,  1.0, -1.0,    0.0,  1.0,  0.0,    1.0, 0.0,
             1.0,  1.0,  1.0,    0.0,  1.0,  0.0,    1.0, 1.0,
             1.0,  1.0,  1.0,    0.0,  1.0,  0.0,    1.0, 1.0,
            -1.0,  1.0,  1.0,    0.0,  1.0,  0.0,    0.0, 1.0,
            -1.0,  1.0, -1.0,    0.0,  1.0,  0.0,    0.0, 0.0
        ], dtype='f4')

        vbo = ctx.buffer(vertices.tobytes())
        
        # Set matrices and other uniforms
        prog['modelMatrix'].write(self.model_matrix.tobytes())
        model_view_matrix = np.dot(self.view_matrix, self.model_matrix)
        prog['modelViewMatrix'].write(model_view_matrix.tobytes())
        prog['projectionMatrix'].write(self.projection_matrix.tobytes())
        
        # Normal matrix is the inverse transpose of the upper-left 3x3 submatrix
        normal_matrix = np.linalg.inv(model_view_matrix[:3, :3]).T
        prog['normalMatrix'].write(normal_matrix.tobytes())

        
        # Texture repeat and rotation
        prog['textureRotation'].write(self.texture_rotation.tobytes())
        prog['textureRepeat'].value = self.texture_repeat

        light_positions = np.array([
            [10.0, 10.0, 10.0],
            [0.0, 10.0, 0.0],
            [-10.0, -10.0, 10.0],
            [10.0, -10.0, 0.0]
        ], dtype='f4')

        light_colors = np.array([
            [300.0, 300.0, 300.0],
            [300.0, 0.0, 0.0],
            [0.0, 300.0, 0.0],
            [0.0, 0.0, 300.0]
        ], dtype='f4')

        prog['lightPositions'].write(light_positions.tobytes())  # Pass light positions
        prog['lightColors'].write(light_colors.tobytes())  # Pass light colors

        prog['cameraPosition'].write(self.camera_position.tobytes())  # Pass camera position (3D vector)

        prog['roughnessScale'].value = 1.0  # Set roughness scale
        prog['exposure'].value = 1.0  # Set exposure

        # Binding textures
        texture_dict['basecolor'].use(0)  # Albedo
        texture_dict['normal'].use(1)     # Normal
        texture_dict['metallic'].use(2)   # Metallic
        texture_dict['roughness'].use(3)  # Roughness

        vao = ctx.vertex_array(prog, vbo, "position", "normal", "uv")
        fbo = ctx.framebuffer(color_attachments=[ctx.texture((512, 512), 3)])
        fbo.use()
        fbo.clear(0.0, 0.0, 0.0, 1.0)
        vao.render()
        image = Image.frombytes("RGB",fbo.size, fbo.color_attachments[0].read(),"raw", "RGB", 0, -1) #PIL Image Object.
        #result = #image in the wanted format either PIL or numpy array.
        """if result_ok:
            self.render = result
            return result
        self.render = None
        return None"""
        self.render = image
        return image
