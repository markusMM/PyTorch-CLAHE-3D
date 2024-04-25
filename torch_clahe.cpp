#include <torch/extension.h>
#include <opencv2/opencv.hpp>

void compute_clahe(torch::Tensor &input, torch::Tensor &mask, int blockSize, float clipLimit)
{
    // Ensure the input tensor is contiguous
    input = input.contiguous();

    // Get the device type of the input tensor
    torch::Device device = input.device();

    // Get the number of blocks, channels, and dimensions
    int64_t batchSize = input.size(0);
    int64_t numChannels = input.size(1);
    int64_t numBlocks = input.size(2);
    int64_t blockHeight = blockSize;
    int64_t blockWidth = blockSize;
    int64_t blockDepth = blockSize;

    // Iterate over each block
    for (int64_t b = 0; b < batchSize; ++b)
    {
        for (int64_t c = 0; c < numChannels; ++c)
        {
            for (int64_t i = 0; i < numBlocks; ++i)
            {
                // Check if the mask is provided and focused CLAHE is requested
                if (!mask.is_empty() && clipLimit < 0)
                {
                    // Get the current mask block
                    torch::Tensor maskBlock = mask[b][i];

                    // Apply the mask to the input block
                    input[b][c][i] *= maskBlock;
                }

                // Get the current block
                torch::Tensor block = input[b][c][i];

                // Move block to the same device as input tensor
                block = block.to(device);

                // Convert block to CPU tensor for OpenCV
                torch::Tensor blockCPU = block.to(torch::kCPU);

                // Convert block to OpenCV Mat
                cv::Mat blockMat(blockHeight, blockWidth, CV_32F, blockCPU.data_ptr<float>());

                // Convert block to 8-bit unsigned integer
                blockMat.convertTo(blockMat, CV_8U, 255.0);

                // Apply CLAHE
                cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
                clahe->setClipLimit(clipLimit);
                clahe->apply(blockMat, blockMat);

                // Convert block back to float
                blockMat.convertTo(blockMat, CV_32F, 1.0 / 255.0);

                // Copy back to the tensor
                memcpy(blockCPU.data_ptr<float>(), blockMat.data, blockMat.total() * sizeof(float));

                // Move block back to the original device
                block = blockCPU.to(device);
            }
        }
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("compute_clahe", &compute_clahe, "Compute CLAHE inplace");
}
