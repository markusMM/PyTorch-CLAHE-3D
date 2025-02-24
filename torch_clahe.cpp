#include <cmath>
#include <torch/torch.h>
#include <torch/extension.h>

// for indexing
using namespace torch::indexing;

torch::Tensor compute_clahe(
    torch::Tensor &input, 
    torch::Tensor &mask, 
    float clipLimit = 0.2, 
    int nbins = 256, 
    int overlap = 0,
    bool adjust_edges = true,
    bool block_norm = false
){
    // Initialization
    int hmax = nbins - 1;
    input = input.to(torch::kFloat);

    // Dimensions
    int64_t batchSize = input.size(0);
    int64_t numChannels = input.size(1);
    int64_t numBlocks = input.size(2);
    auto blockSize = input[0][0][0].sizes();
    
    // Gaussian weights for smoothing
    int smooth_len;
    double sigma;
    torch::Tensor gaussianWeights;
    if (adjust_edges){
        if (overlap > 0){
            sigma = overlap / 3.0;
            smooth_len = overlap;
        } else {
            sigma = blockSize[0] / 9.0;
            smooth_len = (int) std::round(blockSize[0] / 3);
        }
        // smoothing weights left and right
        gaussianWeights = torch::arange(0, smooth_len).pow(2);
        gaussianWeights = gaussianWeights.div_(-2.0 * sigma * sigma);
        gaussianWeights = gaussianWeights.exp();
        gaussianWeights = gaussianWeights.div_(gaussianWeights.max());
    }

    // Result tensor with same size as input
    auto result = torch::zeros_like(input);

    bool masked = (mask.sizes() == input.sizes());

    for (int64_t b = 0; b < batchSize; ++b) {
        for (int64_t c = 0; c < numChannels; ++c) {
            for (int64_t i = 0; i < numBlocks; ++i) {

                // Get current block and apply CLAHE
                auto block = input.index({b, c, i});
                
                if (masked) {
                    auto maskBlock = mask.index({b, c, i});
                    if (maskBlock.sum().item<float>() == 0) continue;
                    block = maskBlock * block;
                }

                // Normalize block
                if (block_norm){
                    auto minVal = block.min().item<float>();
                    auto maxVal = block.max().item<float>();
                    if (maxVal != minVal) {
                        block = (block - minVal) * (hmax / (maxVal - minVal));
                    }
                }

                // Compute histogram and apply CLAHE transformations (omitted for brevity)
                auto hist = torch::histc(block.view(-1), nbins, 0, hmax).to(torch::kFloat);
                if ((clipLimit > 0) && (clipLimit < 1)) {
                    auto cdf = hist.cumsum(0);
                    auto cmax = cdf.index({-1}).item<float>();
                    cdf = cdf / cmax;
                    torch::Tensor clipLimitTensor = torch::ones_like(cdf).fill_(clipLimit);
                    torch::Tensor excess = (cdf - clipLimitTensor).clamp_min(0);
                    cdf = torch::min(cdf, clipLimitTensor);
                    auto numExcessBins = (excess > 0).sum().item<int>();
                    torch::Tensor excessPerBin = excess.sum() / numExcessBins;
                    hist = hist + excessPerBin;
                }
                hist = hist / hist.max() * 255.0;
                auto index = block.view(-1).to(torch::kInt);
                auto equalizedBlock = torch::index_select(hist, 0, index);
                block = equalizedBlock.reshape(block.sizes()) / hmax;

                if (adjust_edges) {
                    // Define smooth ranges
                    std::vector<std::tuple<int64_t, int64_t, int64_t>> smooth_ranges = {
                        {blockSize[0], smooth_len, 0}, 
                        {blockSize[1], smooth_len, 1},
                        {blockSize[2], smooth_len, 2}
                    };

                    // Iterate over the dimensions (x, y, z)
                    for (auto& [size, smooth_len, dim] : smooth_ranges) {
                        if (smooth_len > 0) {
                            // Prepare slices
                            std::vector<torch::indexing::TensorIndex> slices(3, Slice());
                            
                            // Smooth the smoothing regions
                            slices[dim] = Slice(0, smooth_len);
                            block.index_put_({slices[0], slices[1], slices[2]}, 
                                block.index(slices) * torch::flip(gaussianWeights, {0}));
                            slices[dim] = Slice(-smooth_len, None);
                            block.index_put_({slices[0], slices[1], slices[2]}, 
                                block.index(slices) * gaussianWeights);
                        }
                    }
                }

                // Store equalized block in result
                auto targetBlock = result.index({b, c, i});
                targetBlock.copy_(block);
            }
        }
    }

    return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("compute_clahe", &compute_clahe, "CLAHE with overlap and interpolation");
}