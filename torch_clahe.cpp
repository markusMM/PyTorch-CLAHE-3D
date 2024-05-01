#include <torch/torch.h>
#include <torch/extension.h>

// for indexing
using namespace torch::indexing;

torch::Tensor compute_clahe(torch::Tensor &input, torch::Tensor &mask, float clipLimit = 0.2)
{
    // Ensure the input tensor is contiguous
    input = input.contiguous();

    // Get the device type of the input tensor
    torch::Device device = input.device();

    // Get the number of blocks, channels, and dimensions
    int64_t batchSize = input.size(0);
    int64_t numChannels = input.size(1);
    int64_t numBlocks = input.size(2);

    // Get whther to use mask
    bool masked = true;
    if (mask.sizes() != input.sizes())
    {
        masked = false;
    }

    // Iterate over each block
    for (int64_t b = 0; b < batchSize; ++b)
    {
        for (int64_t c = 0; c < numChannels; ++c)
        {
            for (int64_t i = 0; i < numBlocks; ++i)
            {
                // Check if the mask is provided and focused CLAHE is requested
                if (masked)
                {
                    // Get the current mask block
                    torch::Tensor maskBlock = mask[b][c][i];

                    if (maskBlock.sum().item<float>() == 0)
                    {
                        // Skip processing if the block is completely masked out
                        continue;
                    }
                    else if (maskBlock.sum().item<float>() < 1)
                    {
                        // Apply the mask to the input block
                        // if block is not completely masked in
                        input[b][c][i] *= maskBlock;
                    }
                }

                // Get the current block
                torch::Tensor block = input[b][c][i];

                // Move block to the same device as input tensor
                torch::Tensor blockContiguous = block.to(device);

                // Get the dimensions of the block
                int64_t xSize = blockContiguous.size(0);
                int64_t ySize = blockContiguous.size(1);
                int64_t zSize = blockContiguous.size(2);

                // Normalize the block to the range [0, 255]
                blockContiguous = (blockContiguous - blockContiguous.min()) * (255.0 / (blockContiguous.max() - blockContiguous.min()));

                // Compute the histogram of the block
                torch::Tensor hist = torch::histc(blockContiguous.view(-1), 256, 0, 255).to(torch::kFloat);

                // Clip the CDF to the specified limit
                if ((clipLimit > 0) && (clipLimit < 1))
                {
                    // Compute the cumulative distribution function (CDF) of the histogram
                    torch::Tensor cdf = hist.cumsum(0);

                    // Normalize the CDF to the range [0, 1]
                    cdf = cdf / cdf[-1];

                    // Compute the excess distribution above the clip limit
                    torch::Tensor clipLimitTensor = torch::ones_like(cdf).fill_(clipLimit);
                    torch::Tensor excess = (cdf - clipLimitTensor).to(torch::kFloat);

                    // clipping
                    cdf = torch::min(cdf, clipLimitTensor);

                    // Redistribute the excess among the histogram bins
                    torch::Tensor numExcessBins = (excess > 0).to(torch::kFloat).sum();
                    torch::Tensor excessPerBin = excess.sum() / numExcessBins;
                    hist = hist + excessPerBin;
                }

                // Scale the histogram to the range [0, 255]
                hist = hist / hist.max() * 255.0;

                // Compute the equalized block using the modified histogram
                torch::Tensor index = blockContiguous.view(-1).to(torch::kInt);
                torch::Tensor equalizedBlock = torch::index_select(hist, 0, index);

                // Copy back to the tensor
                input[b][c][i] = equalizedBlock.reshape({xSize, ySize, zSize}).to(torch::kFloat) / 255.0;
            }
        }
    }
    return input;
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("compute_clahe", &compute_clahe, "Compute CLAHE inplace");
};
