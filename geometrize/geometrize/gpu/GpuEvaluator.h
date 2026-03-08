#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "../bitmap/rgba.h"

namespace geometrize
{
class Bitmap;
class Shape;
class State;
}

namespace geometrize
{
namespace gpu
{

/**
 * @brief Encapsulates GPU evaluation output for a candidate batch.
 */
struct BatchEvaluationResult
{
    std::size_t bestIndex = 0U;
    double bestScore = 0.0;
    geometrize::rgba bestColor{0U, 0U, 0U, 0U};
};

/**
 * @brief Evaluates candidate states on GPU using OpenGL compute shaders.
 *
 * Memory layout:
 * - target/current image data are uploaded as RGBA8 textures.
 * - candidate shape descriptors are uploaded as std430 SSBO records.
 * - shader writes candidate scores + candidate colors to SSBOs.
 * - a reduction shader writes only the best index/score to a result SSBO.
 */
class GpuEvaluator
{
public:
    GpuEvaluator();
    ~GpuEvaluator();
    GpuEvaluator(const GpuEvaluator&) = delete;
    GpuEvaluator& operator=(const GpuEvaluator&) = delete;

    /**
     * @brief isAvailable True when the OpenGL compute pipeline is ready.
     */
    bool isAvailable() const;

    /**
     * @brief statusMessage Human-readable status/error text.
     */
    const std::string& statusMessage() const;

    /**
     * @brief supportsShapeType True when a shape can be packed for shader evaluation.
     */
    bool supportsShapeType(const geometrize::Shape& shape) const;

    /**
     * @brief evaluateBestCandidate Evaluates a candidate batch and returns the best entry.
     * @return True when successful. False indicates caller should use CPU fallback.
     */
    bool evaluateBestCandidate(
        const geometrize::Bitmap& target,
        const geometrize::Bitmap& current,
        const std::vector<geometrize::State>& candidates,
        double lastScore,
        geometrize::gpu::BatchEvaluationResult& result);

private:
    struct Impl;
    Impl* m_impl;
};

}
}

