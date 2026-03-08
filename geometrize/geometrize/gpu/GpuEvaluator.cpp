#include "GpuEvaluator.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "../bitmap/bitmap.h"
#include "../shape/circle.h"
#include "../shape/ellipse.h"
#include "../shape/rectangle.h"
#include "../shape/rotatedellipse.h"
#include "../shape/rotatedrectangle.h"
#include "../shape/shape.h"
#include "../shape/triangle.h"
#include "../state.h"

#if defined(GEOMETRIZE_ENABLE_OPENGL_COMPUTE) && defined(__has_include)
#if __has_include(<GL/glew.h>) && __has_include(<GL/gl.h>)
#define GEOMETRIZE_GPU_BACKEND_OPENGL 1
#include <GL/glew.h>
#include <GL/gl.h>
#endif
#endif

#ifndef GEOMETRIZE_GPU_BACKEND_OPENGL
#define GEOMETRIZE_GPU_BACKEND_OPENGL 0
#endif

namespace
{

struct alignas(16) PackedCandidateShape
{
    std::int32_t type = 0;
    std::int32_t alpha = 0;
    std::int32_t reserved0 = 0;
    std::int32_t reserved1 = 0;
    float p0[4] = {0.0F, 0.0F, 0.0F, 0.0F};
    float p1[4] = {0.0F, 0.0F, 0.0F, 0.0F};
};

struct alignas(16) PackedColor
{
    float r = 0.0F;
    float g = 0.0F;
    float b = 0.0F;
    float a = 0.0F;
};

struct alignas(16) ReductionResult
{
    std::uint32_t bestIndex = 0U;
    float bestScore = 0.0F;
    std::uint32_t reserved0 = 0U;
    std::uint32_t reserved1 = 0U;
};

static std::uint8_t clampToU8(const float value)
{
    if(value <= 0.0F) {
        return 0U;
    }
    if(value >= 255.0F) {
        return 255U;
    }
    return static_cast<std::uint8_t>(value + 0.5F);
}

static std::string readShaderFile(const std::string& fileName)
{
    const std::array<std::string, 5U> paths = {{
        "shaders/" + fileName,
        "../shaders/" + fileName,
        "../../shaders/" + fileName,
        "../../../shaders/" + fileName,
        "../../../../shaders/" + fileName
    }};

    for(const std::string& path : paths) {
        std::ifstream stream(path, std::ios::in | std::ios::binary);
        if(!stream.good()) {
            continue;
        }

        std::ostringstream buffer;
        buffer << stream.rdbuf();
        return buffer.str();
    }

    return {};
}

static bool packShape(const geometrize::Shape& shape, const std::uint8_t alpha, PackedCandidateShape& packed)
{
    packed = PackedCandidateShape{};
    packed.type = static_cast<std::int32_t>(shape.getType());
    packed.alpha = static_cast<std::int32_t>(alpha);

    switch(shape.getType()) {
    case geometrize::ShapeTypes::RECTANGLE: {
        const auto* rectangle = dynamic_cast<const geometrize::Rectangle*>(&shape);
        if(!rectangle) {
            return false;
        }
        packed.p0[0] = rectangle->m_x1;
        packed.p0[1] = rectangle->m_y1;
        packed.p0[2] = rectangle->m_x2;
        packed.p0[3] = rectangle->m_y2;
        return true;
    }
    case geometrize::ShapeTypes::ROTATED_RECTANGLE: {
        const auto* rectangle = dynamic_cast<const geometrize::RotatedRectangle*>(&shape);
        if(!rectangle) {
            return false;
        }
        packed.p0[0] = rectangle->m_x1;
        packed.p0[1] = rectangle->m_y1;
        packed.p0[2] = rectangle->m_x2;
        packed.p0[3] = rectangle->m_y2;
        packed.p1[0] = rectangle->m_angle;
        return true;
    }
    case geometrize::ShapeTypes::TRIANGLE: {
        const auto* triangle = dynamic_cast<const geometrize::Triangle*>(&shape);
        if(!triangle) {
            return false;
        }
        packed.p0[0] = triangle->m_x1;
        packed.p0[1] = triangle->m_y1;
        packed.p0[2] = triangle->m_x2;
        packed.p0[3] = triangle->m_y2;
        packed.p1[0] = triangle->m_x3;
        packed.p1[1] = triangle->m_y3;
        return true;
    }
    case geometrize::ShapeTypes::ELLIPSE: {
        const auto* ellipse = dynamic_cast<const geometrize::Ellipse*>(&shape);
        if(!ellipse) {
            return false;
        }
        packed.p0[0] = ellipse->m_x;
        packed.p0[1] = ellipse->m_y;
        packed.p0[2] = ellipse->m_rx;
        packed.p0[3] = ellipse->m_ry;
        return true;
    }
    case geometrize::ShapeTypes::ROTATED_ELLIPSE: {
        const auto* ellipse = dynamic_cast<const geometrize::RotatedEllipse*>(&shape);
        if(!ellipse) {
            return false;
        }
        packed.p0[0] = ellipse->m_x;
        packed.p0[1] = ellipse->m_y;
        packed.p0[2] = ellipse->m_rx;
        packed.p0[3] = ellipse->m_ry;
        packed.p1[0] = ellipse->m_angle;
        return true;
    }
    case geometrize::ShapeTypes::CIRCLE: {
        const auto* circle = dynamic_cast<const geometrize::Circle*>(&shape);
        if(!circle) {
            return false;
        }
        packed.p0[0] = circle->m_x;
        packed.p0[1] = circle->m_y;
        packed.p0[2] = circle->m_r;
        return true;
    }
    default:
        return false;
    }
}

} // namespace

namespace geometrize
{
namespace gpu
{

struct GpuEvaluator::Impl
{
    bool available = false;
    std::string status = "GPU evaluator is disabled at build time.";

#if GEOMETRIZE_GPU_BACKEND_OPENGL
    bool initialized = false;
    std::uint32_t textureWidth = 0U;
    std::uint32_t textureHeight = 0U;
    GLuint evaluateProgram = 0U;
    GLuint reduceProgram = 0U;
    GLuint targetTexture = 0U;
    GLuint currentTexture = 0U;
    GLuint shapeBuffer = 0U;
    GLuint scoreBuffer = 0U;
    GLuint colorBuffer = 0U;
    GLuint reductionBuffer = 0U;
#endif
};

#if GEOMETRIZE_GPU_BACKEND_OPENGL

namespace
{

static const char* fallbackEvaluateShader = R"GLSL(
#version 430
layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

struct CandidateShape {
    int type;
    int alpha;
    int reserved0;
    int reserved1;
    vec4 p0;
    vec4 p1;
};

layout(std430, binding = 0) readonly buffer CandidateBuffer {
    CandidateShape shapes[];
};
layout(std430, binding = 1) writeonly buffer ScoreBuffer {
    float scores[];
};
layout(std430, binding = 2) writeonly buffer ColorBuffer {
    vec4 colors[];
};

layout(binding = 0, rgba8) uniform readonly image2D targetImage;
layout(binding = 1, rgba8) uniform readonly image2D currentImage;

uniform int uWidth;
uniform int uHeight;
uniform int uCandidateCount;
uniform float uBaseTotal;

shared vec4 sColorAccum[64];
shared float sDeltaAccum[64];
shared vec4 sBestColor;

float sqr(float v)
{
    return v * v;
}

bool pointInTriangle(vec2 p, vec2 a, vec2 b, vec2 c)
{
    vec2 v0 = c - a;
    vec2 v1 = b - a;
    vec2 v2 = p - a;

    float dot00 = dot(v0, v0);
    float dot01 = dot(v0, v1);
    float dot02 = dot(v0, v2);
    float dot11 = dot(v1, v1);
    float dot12 = dot(v1, v2);

    float denom = dot00 * dot11 - dot01 * dot01;
    if(abs(denom) < 1e-6) {
        return false;
    }
    float inv = 1.0 / denom;
    float u = (dot11 * dot02 - dot01 * dot12) * inv;
    float v = (dot00 * dot12 - dot01 * dot02) * inv;
    return (u >= 0.0 && v >= 0.0 && (u + v) <= 1.0);
}

bool insideShape(const CandidateShape shape, vec2 p)
{
    if(shape.type == 1) {
        float x1 = min(shape.p0.x, shape.p0.z);
        float x2 = max(shape.p0.x, shape.p0.z);
        float y1 = min(shape.p0.y, shape.p0.w);
        float y2 = max(shape.p0.y, shape.p0.w);
        return p.x >= x1 && p.x <= x2 && p.y >= y1 && p.y <= y2;
    }
    if(shape.type == 2) {
        float x1 = min(shape.p0.x, shape.p0.z);
        float x2 = max(shape.p0.x, shape.p0.z);
        float y1 = min(shape.p0.y, shape.p0.w);
        float y2 = max(shape.p0.y, shape.p0.w);
        vec2 center = vec2((x1 + x2) * 0.5, (y1 + y2) * 0.5);
        float radians = shape.p1.x * 3.14159265 / 180.0;
        float c = cos(-radians);
        float s = sin(-radians);
        vec2 d = p - center;
        vec2 pr = vec2(d.x * c - d.y * s, d.x * s + d.y * c) + center;
        return pr.x >= x1 && pr.x <= x2 && pr.y >= y1 && pr.y <= y2;
    }
    if(shape.type == 4) {
        vec2 a = vec2(shape.p0.x, shape.p0.y);
        vec2 b = vec2(shape.p0.z, shape.p0.w);
        vec2 c = vec2(shape.p1.x, shape.p1.y);
        return pointInTriangle(p, a, b, c);
    }
    if(shape.type == 8) {
        float rx = max(shape.p0.z, 1e-4);
        float ry = max(shape.p0.w, 1e-4);
        vec2 d = p - vec2(shape.p0.x, shape.p0.y);
        return (d.x * d.x) / (rx * rx) + (d.y * d.y) / (ry * ry) <= 1.0;
    }
    if(shape.type == 16) {
        float rx = max(shape.p0.z, 1e-4);
        float ry = max(shape.p0.w, 1e-4);
        float radians = shape.p1.x * 3.14159265 / 180.0;
        float c = cos(-radians);
        float s = sin(-radians);
        vec2 d = p - vec2(shape.p0.x, shape.p0.y);
        vec2 pr = vec2(d.x * c - d.y * s, d.x * s + d.y * c);
        return (pr.x * pr.x) / (rx * rx) + (pr.y * pr.y) / (ry * ry) <= 1.0;
    }
    if(shape.type == 32) {
        float r = max(shape.p0.z, 1e-4);
        vec2 d = p - vec2(shape.p0.x, shape.p0.y);
        return dot(d, d) <= r * r;
    }
    return false;
}

vec4 blendColor(vec4 destination, vec4 source)
{
    float a = source.a;
    vec4 outColor;
    outColor.rgb = destination.rgb * (1.0 - a) + source.rgb * a;
    outColor.a = destination.a * (1.0 - a) + a;
    return outColor;
}

void main()
{
    uint candidateIndex = gl_WorkGroupID.x;
    uint localId = gl_LocalInvocationID.x;
    if(candidateIndex >= uint(uCandidateCount)) {
        return;
    }

    CandidateShape shape = shapes[candidateIndex];
    float alpha = clamp(float(shape.alpha) / 255.0, 0.0, 1.0);
    float colorScale = (shape.alpha > 0) ? (257.0 * 255.0 / float(shape.alpha)) : 0.0;

    float sumR = 0.0;
    float sumG = 0.0;
    float sumB = 0.0;
    float count = 0.0;

    uint pixelCount = uint(uWidth * uHeight);
    for(uint pixel = localId; pixel < pixelCount; pixel += 64U) {
        int x = int(pixel % uint(uWidth));
        int y = int(pixel / uint(uWidth));
        vec2 point = vec2(float(x), float(y));
        if(!insideShape(shape, point)) {
            continue;
        }

        vec4 targetColor = imageLoad(targetImage, ivec2(x, y)) * 255.0;
        vec4 currentColor = imageLoad(currentImage, ivec2(x, y)) * 255.0;
        sumR += ((targetColor.r - currentColor.r) * colorScale + currentColor.r * 257.0);
        sumG += ((targetColor.g - currentColor.g) * colorScale + currentColor.g * 257.0);
        sumB += ((targetColor.b - currentColor.b) * colorScale + currentColor.b * 257.0);
        count += 1.0;
    }

    sColorAccum[localId] = vec4(sumR, sumG, sumB, count);
    barrier();

    for(uint stride = 32U; stride > 0U; stride >>= 1U) {
        if(localId < stride) {
            sColorAccum[localId] += sColorAccum[localId + stride];
        }
        barrier();
    }

    if(localId == 0U) {
        vec4 accum = sColorAccum[0];
        if(accum.w <= 0.0) {
            sBestColor = vec4(0.0, 0.0, 0.0, alpha);
        } else {
            float rr = clamp(floor((accum.x / accum.w) / 256.0), 0.0, 255.0);
            float gg = clamp(floor((accum.y / accum.w) / 256.0), 0.0, 255.0);
            float bb = clamp(floor((accum.z / accum.w) / 256.0), 0.0, 255.0);
            sBestColor = vec4(rr / 255.0, gg / 255.0, bb / 255.0, alpha);
        }
    }
    barrier();

    float delta = 0.0;
    for(uint pixel = localId; pixel < pixelCount; pixel += 64U) {
        int x = int(pixel % uint(uWidth));
        int y = int(pixel / uint(uWidth));
        vec2 point = vec2(float(x), float(y));
        if(!insideShape(shape, point)) {
            continue;
        }

        vec4 targetColor = imageLoad(targetImage, ivec2(x, y)) * 255.0;
        vec4 beforeColor = imageLoad(currentImage, ivec2(x, y)) * 255.0;
        vec4 afterColor = blendColor(beforeColor / 255.0, sBestColor) * 255.0;

        float beforeError = sqr(targetColor.r - beforeColor.r) +
            sqr(targetColor.g - beforeColor.g) +
            sqr(targetColor.b - beforeColor.b) +
            sqr(targetColor.a - beforeColor.a);

        float afterError = sqr(targetColor.r - afterColor.r) +
            sqr(targetColor.g - afterColor.g) +
            sqr(targetColor.b - afterColor.b) +
            sqr(targetColor.a - afterColor.a);

        delta += afterError - beforeError;
    }

    sDeltaAccum[localId] = delta;
    barrier();

    for(uint stride = 32U; stride > 0U; stride >>= 1U) {
        if(localId < stride) {
            sDeltaAccum[localId] += sDeltaAccum[localId + stride];
        }
        barrier();
    }

    if(localId == 0U) {
        float rgbaCount = float(uWidth * uHeight * 4);
        float total = max(0.0, uBaseTotal + sDeltaAccum[0]);
        scores[candidateIndex] = sqrt(total / rgbaCount) / 255.0;
        colors[candidateIndex] = sBestColor;
    }
}
)GLSL";

static const char* fallbackReduceShader = R"GLSL(
#version 430
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(std430, binding = 0) readonly buffer ScoreBuffer {
    float scores[];
};

struct ReductionResult {
    uint bestIndex;
    float bestScore;
    uint reserved0;
    uint reserved1;
};

layout(std430, binding = 1) writeonly buffer ResultBuffer {
    ReductionResult result;
};

uniform uint uCandidateCount;

shared float sBestScore[256];
shared uint sBestIndex[256];

void main()
{
    uint tid = gl_LocalInvocationID.x;

    float bestScore = 1e30;
    uint bestIndex = 0U;

    for(uint i = tid; i < uCandidateCount; i += 256U) {
        float score = scores[i];
        if(score < bestScore) {
            bestScore = score;
            bestIndex = i;
        }
    }

    sBestScore[tid] = bestScore;
    sBestIndex[tid] = bestIndex;
    barrier();

    for(uint stride = 128U; stride > 0U; stride >>= 1U) {
        if(tid < stride) {
            float otherScore = sBestScore[tid + stride];
            uint otherIndex = sBestIndex[tid + stride];
            if(otherScore < sBestScore[tid]) {
                sBestScore[tid] = otherScore;
                sBestIndex[tid] = otherIndex;
            }
        }
        barrier();
    }

    if(tid == 0U) {
        result.bestIndex = sBestIndex[0];
        result.bestScore = sBestScore[0];
    }
}
)GLSL";

static std::string shaderOrFallback(const std::string& fileName, const char* fallbackSource)
{
    const std::string shader = readShaderFile(fileName);
    if(!shader.empty()) {
        return shader;
    }
    return fallbackSource;
}

static bool checkCompileStatus(const GLuint shader, std::string& error)
{
    GLint compiled = 0;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &compiled);
    if(compiled == GL_TRUE) {
        return true;
    }

    GLint logLength = 0;
    glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &logLength);
    std::vector<char> log(logLength > 1 ? static_cast<std::size_t>(logLength) : 1U, '\0');
    if(logLength > 0) {
        glGetShaderInfoLog(shader, logLength, nullptr, log.data());
    }
    error = std::string(log.data());
    return false;
}

static bool checkProgramStatus(const GLuint program, std::string& error)
{
    GLint linked = 0;
    glGetProgramiv(program, GL_LINK_STATUS, &linked);
    if(linked == GL_TRUE) {
        return true;
    }

    GLint logLength = 0;
    glGetProgramiv(program, GL_INFO_LOG_LENGTH, &logLength);
    std::vector<char> log(logLength > 1 ? static_cast<std::size_t>(logLength) : 1U, '\0');
    if(logLength > 0) {
        glGetProgramInfoLog(program, logLength, nullptr, log.data());
    }
    error = std::string(log.data());
    return false;
}

static GLuint createComputeProgram(const std::string& source, std::string& error)
{
    const GLuint shader = glCreateShader(GL_COMPUTE_SHADER);
    const GLchar* const src = source.c_str();
    glShaderSource(shader, 1, &src, nullptr);
    glCompileShader(shader);

    if(!checkCompileStatus(shader, error)) {
        glDeleteShader(shader);
        return 0U;
    }

    const GLuint program = glCreateProgram();
    glAttachShader(program, shader);
    glLinkProgram(program);
    glDeleteShader(shader);

    if(!checkProgramStatus(program, error)) {
        glDeleteProgram(program);
        return 0U;
    }

    return program;
}

} // namespace

#endif // GEOMETRIZE_GPU_BACKEND_OPENGL

GpuEvaluator::GpuEvaluator() : m_impl(new Impl())
{
#if GEOMETRIZE_GPU_BACKEND_OPENGL
    m_impl->status = "GPU evaluator is waiting for an active OpenGL 4.3 context.";
#else
    m_impl->status = "GPU evaluator unavailable. Build with GEOMETRIZE_ENABLE_OPENGL_COMPUTE and OpenGL/GLEW.";
#endif
}

GpuEvaluator::~GpuEvaluator()
{
#if GEOMETRIZE_GPU_BACKEND_OPENGL
    if(m_impl) {
        if(m_impl->evaluateProgram != 0U) {
            glDeleteProgram(m_impl->evaluateProgram);
            m_impl->evaluateProgram = 0U;
        }
        if(m_impl->reduceProgram != 0U) {
            glDeleteProgram(m_impl->reduceProgram);
            m_impl->reduceProgram = 0U;
        }
        if(m_impl->targetTexture != 0U) {
            glDeleteTextures(1, &m_impl->targetTexture);
            m_impl->targetTexture = 0U;
        }
        if(m_impl->currentTexture != 0U) {
            glDeleteTextures(1, &m_impl->currentTexture);
            m_impl->currentTexture = 0U;
        }
        if(m_impl->shapeBuffer != 0U) {
            glDeleteBuffers(1, &m_impl->shapeBuffer);
            m_impl->shapeBuffer = 0U;
        }
        if(m_impl->scoreBuffer != 0U) {
            glDeleteBuffers(1, &m_impl->scoreBuffer);
            m_impl->scoreBuffer = 0U;
        }
        if(m_impl->colorBuffer != 0U) {
            glDeleteBuffers(1, &m_impl->colorBuffer);
            m_impl->colorBuffer = 0U;
        }
        if(m_impl->reductionBuffer != 0U) {
            glDeleteBuffers(1, &m_impl->reductionBuffer);
            m_impl->reductionBuffer = 0U;
        }
    }
#endif
    delete m_impl;
    m_impl = nullptr;
}

bool GpuEvaluator::isAvailable() const
{
    return m_impl && m_impl->available;
}

const std::string& GpuEvaluator::statusMessage() const
{
    return m_impl->status;
}

bool GpuEvaluator::supportsShapeType(const geometrize::Shape& shape) const
{
    PackedCandidateShape packed;
    return packShape(shape, 255U, packed);
}

bool GpuEvaluator::evaluateBestCandidate(
    const geometrize::Bitmap& target,
    const geometrize::Bitmap& current,
    const std::vector<geometrize::State>& candidates,
    const double lastScore,
    geometrize::gpu::BatchEvaluationResult& result)
{
#if !GEOMETRIZE_GPU_BACKEND_OPENGL
    (void)target;
    (void)current;
    (void)candidates;
    (void)lastScore;
    (void)result;
    m_impl->available = false;
    return false;
#else
    if(candidates.empty()) {
        m_impl->available = false;
        m_impl->status = "GPU evaluator failed: candidate batch is empty.";
        return false;
    }
    if(target.getWidth() != current.getWidth() || target.getHeight() != current.getHeight()) {
        m_impl->available = false;
        m_impl->status = "GPU evaluator failed: target/current bitmap dimensions do not match.";
        return false;
    }

    if(glGetString(GL_VERSION) == nullptr) {
        m_impl->available = false;
        m_impl->status = "GPU evaluator failed: no current OpenGL context.";
        return false;
    }

    if(!m_impl->initialized) {
        const GLenum glewInitResult = glewInit();
        glGetError();
        if(glewInitResult != GLEW_OK) {
            m_impl->available = false;
            m_impl->status = "GPU evaluator failed: could not initialize GLEW.";
            return false;
        }
        if(!GLEW_VERSION_4_3) {
            m_impl->available = false;
            m_impl->status = "GPU evaluator failed: OpenGL 4.3 compute shaders are unavailable.";
            return false;
        }

        std::string shaderError;
        m_impl->evaluateProgram = createComputeProgram(shaderOrFallback("evaluate_shapes.comp", fallbackEvaluateShader), shaderError);
        if(m_impl->evaluateProgram == 0U) {
            m_impl->available = false;
            m_impl->status = "GPU evaluator failed: evaluate_shapes.comp compile/link error: " + shaderError;
            return false;
        }

        m_impl->reduceProgram = createComputeProgram(shaderOrFallback("error_reduce.comp", fallbackReduceShader), shaderError);
        if(m_impl->reduceProgram == 0U) {
            m_impl->available = false;
            m_impl->status = "GPU evaluator failed: error_reduce.comp compile/link error: " + shaderError;
            return false;
        }

        glGenTextures(1, &m_impl->targetTexture);
        glGenTextures(1, &m_impl->currentTexture);
        glGenBuffers(1, &m_impl->shapeBuffer);
        glGenBuffers(1, &m_impl->scoreBuffer);
        glGenBuffers(1, &m_impl->colorBuffer);
        glGenBuffers(1, &m_impl->reductionBuffer);

        m_impl->initialized = true;
    }

    std::vector<PackedCandidateShape> packedCandidates;
    packedCandidates.reserve(candidates.size());
    for(const geometrize::State& state : candidates) {
        if(!state.m_shape) {
            m_impl->status = "GPU evaluator failed: encountered candidate with no shape.";
            return false;
        }

        PackedCandidateShape packed;
        if(!packShape(*state.m_shape, state.m_alpha, packed)) {
            m_impl->status = "GPU evaluator fallback: encountered unsupported shape type.";
            return false;
        }
        packedCandidates.push_back(packed);
    }

    const auto uploadTexture = [&](const GLuint texture, const geometrize::Bitmap& bitmap, const bool sizeChanged) {
        glBindTexture(GL_TEXTURE_2D, texture);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

        const std::uint32_t width = bitmap.getWidth();
        const std::uint32_t height = bitmap.getHeight();
        const auto& data = bitmap.getDataRef();
        if(sizeChanged) {
            glTexImage2D(
                GL_TEXTURE_2D,
                0,
                GL_RGBA8,
                static_cast<GLsizei>(width),
                static_cast<GLsizei>(height),
                0,
                GL_RGBA,
                GL_UNSIGNED_BYTE,
                data.data());
        } else {
            glTexSubImage2D(
                GL_TEXTURE_2D,
                0,
                0,
                0,
                static_cast<GLsizei>(width),
                static_cast<GLsizei>(height),
                GL_RGBA,
                GL_UNSIGNED_BYTE,
                data.data());
        }
    };

    const bool textureSizeChanged = m_impl->textureWidth != target.getWidth() || m_impl->textureHeight != target.getHeight();
    uploadTexture(m_impl->targetTexture, target, textureSizeChanged);
    uploadTexture(m_impl->currentTexture, current, textureSizeChanged);
    m_impl->textureWidth = target.getWidth();
    m_impl->textureHeight = target.getHeight();

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_impl->shapeBuffer);
    glBufferData(
        GL_SHADER_STORAGE_BUFFER,
        static_cast<GLsizeiptr>(packedCandidates.size() * sizeof(PackedCandidateShape)),
        packedCandidates.data(),
        GL_DYNAMIC_DRAW);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_impl->scoreBuffer);
    glBufferData(
        GL_SHADER_STORAGE_BUFFER,
        static_cast<GLsizeiptr>(packedCandidates.size() * sizeof(float)),
        nullptr,
        GL_DYNAMIC_DRAW);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_impl->colorBuffer);
    glBufferData(
        GL_SHADER_STORAGE_BUFFER,
        static_cast<GLsizeiptr>(packedCandidates.size() * sizeof(PackedColor)),
        nullptr,
        GL_DYNAMIC_DRAW);

    ReductionResult defaultResult;
    defaultResult.bestIndex = 0U;
    defaultResult.bestScore = std::numeric_limits<float>::max();
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_impl->reductionBuffer);
    glBufferData(
        GL_SHADER_STORAGE_BUFFER,
        sizeof(ReductionResult),
        &defaultResult,
        GL_DYNAMIC_DRAW);

    const float rgbaCount = static_cast<float>(target.getWidth() * target.getHeight() * 4U);
    const float score255 = static_cast<float>(lastScore * 255.0);
    const float baseTotal = score255 * score255 * rgbaCount;

    glUseProgram(m_impl->evaluateProgram);
    glUniform1i(glGetUniformLocation(m_impl->evaluateProgram, "uWidth"), static_cast<GLint>(target.getWidth()));
    glUniform1i(glGetUniformLocation(m_impl->evaluateProgram, "uHeight"), static_cast<GLint>(target.getHeight()));
    glUniform1i(glGetUniformLocation(m_impl->evaluateProgram, "uCandidateCount"), static_cast<GLint>(packedCandidates.size()));
    glUniform1f(glGetUniformLocation(m_impl->evaluateProgram, "uBaseTotal"), baseTotal);

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, m_impl->shapeBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, m_impl->scoreBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, m_impl->colorBuffer);
    glBindImageTexture(0, m_impl->targetTexture, 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA8);
    glBindImageTexture(1, m_impl->currentTexture, 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA8);
    glDispatchCompute(static_cast<GLuint>(packedCandidates.size()), 1U, 1U);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

    glUseProgram(m_impl->reduceProgram);
    glUniform1ui(glGetUniformLocation(m_impl->reduceProgram, "uCandidateCount"), static_cast<GLuint>(packedCandidates.size()));
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, m_impl->scoreBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, m_impl->reductionBuffer);
    glDispatchCompute(1U, 1U, 1U);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_BUFFER_UPDATE_BARRIER_BIT);

    ReductionResult reduced;
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_impl->reductionBuffer);
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(ReductionResult), &reduced);
    if(reduced.bestIndex >= packedCandidates.size()) {
        m_impl->status = "GPU evaluator failed: reduction returned invalid candidate index.";
        return false;
    }

    PackedColor bestColor;
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_impl->colorBuffer);
    glGetBufferSubData(
        GL_SHADER_STORAGE_BUFFER,
        static_cast<GLintptr>(reduced.bestIndex * sizeof(PackedColor)),
        sizeof(PackedColor),
        &bestColor);

    result.bestIndex = reduced.bestIndex;
    result.bestScore = reduced.bestScore;
    result.bestColor = geometrize::rgba{
        clampToU8(bestColor.r * 255.0F),
        clampToU8(bestColor.g * 255.0F),
        clampToU8(bestColor.b * 255.0F),
        clampToU8(bestColor.a * 255.0F)
    };

    m_impl->available = true;
    m_impl->status = "GPU evaluator active (OpenGL compute).";
    return true;
#endif
}

}
}

