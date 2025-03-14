#version 330 core
out vec4 FragColor;

in vec3 FragPos;
in vec3 Normal;

uniform vec3 lightPos;
uniform vec3 lightColor;
uniform vec3 objectColor;

void main() {
    // Ambient lighting (ensures cubes are never fully dark)
     float ambientStrength = 1.0;  // Super bright
    vec3 ambient = ambientStrength * lightColor;

    // Diffuse lighting (light reflecting off surface)
    vec3 norm = normalize(Normal);   
    vec3 lightDir = normalize(lightPos - FragPos);
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * lightColor;

    // Final color calculation
    vec3 result = (ambient + diffuse) * objectColor;
    FragColor = vec4(result, 1.0);
}
