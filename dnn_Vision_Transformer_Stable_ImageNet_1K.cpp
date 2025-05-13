/**
 * @file dnn_Vision_Transformer_StableImageNet.cpp
 * @brief Comparative training of Vision Transformer (ViT) vs ResNet-34 on Stable ImageNet-1K
 *
 * This program implements:
 * - Vision Transformer architecture (ViT) with configurable parameters
 * - ResNet-34 baseline model
 * - Multi-GPU training with data augmentation
 * - Model serialization/deserialization
 *
 * Key features:
 * - Supports both supervised classification
 * - Implements proper signal handling for clean shutdown
 * - Automatic recovery from training interruptions
 */

#include <dlib/cmd_line_parser.h>
#include <dlib/data_io.h>
#include <dlib/dnn.h>
#include <dlib/global_optimization.h>
#include <dlib/gui_widgets.h>
#include <dlib/svm_threaded.h>

using namespace std;
using namespace dlib;

#include <dlib/dnn.h>
#include <sstream>
#include <csignal>
#include <random>
#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

#define __USE_VIT__ 1

/**
 * @struct imagenet_dataset
 * @brief Container for ImageNet-1K dataset with images and labels
 *
 * @var images          RGB pixel matrices (224x224 by default)
 * @var labels          Textual class labels
 * @var numeric_labels  Numerical class indices (0-999)
 */
struct imagenet_dataset
{
    std::vector<matrix<rgb_pixel>> images;
    std::vector<std::string> labels;
    std::vector<unsigned long> numeric_labels;
};

/**
 * @brief Loads and splits Stable ImageNet-1K dataset
 *
 * @param dataset_file    Path to serialized dataset file
 * @param training_images Output container for training images
 * @param training_labels Output container for training labels
 * @param testing_images  Output container for test images
 * @param testing_labels  Output container for test labels
 * @param test_fraction   Fraction of data to use for testing (default: 5%)
 *
 * @throws serialization_error If dataset file is invalid
 *
 * Note: Automatically shuffles data before splitting
 */
void load_stable_imagenet_1k(
    const std::string& dataset_file,
    std::vector<matrix<rgb_pixel>>& training_images,
    std::vector<unsigned long>& training_labels,
    std::vector<matrix<rgb_pixel>>& testing_images,
    std::vector<unsigned long>& testing_labels,
    double test_fraction = 0.05
)
{
    imagenet_dataset dataset;
    deserialize(dataset_file) >> dataset.images >> dataset.labels >> dataset.numeric_labels;

    // Create indices for shuffling
    std::vector<size_t> indices(dataset.images.size());
    for (size_t i = 0; i < indices.size(); ++i) indices[i] = i;

    // Shuffle indices for random train/test split
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(indices.begin(), indices.end(), g);

    size_t split_point = static_cast<size_t>(dataset.images.size() * (1.0 - test_fraction));

    // Reserve space for efficiency
    training_images.clear();
    training_labels.clear();
    testing_images.clear();
    testing_labels.clear();
    training_images.reserve(split_point);
    training_labels.reserve(split_point);
    testing_images.reserve(dataset.images.size() - split_point);
    testing_labels.reserve(dataset.images.size() - split_point);

    // Split into training and testing sets
    for (size_t i = 0; i < indices.size(); ++i)
    {
        size_t idx = indices[i];

        if (i < split_point)
        {
            training_images.push_back(dataset.images[idx]);
            training_labels.push_back(dataset.numeric_labels[idx]);
        }
        else
        {
            testing_images.push_back(dataset.images[idx]);
            testing_labels.push_back(dataset.numeric_labels[idx]);
        }
    }
}

/*!
    This module implements a complete Vision Transformer (ViT) architecture with
    support for both standard and Mixture-of-Experts (MoE) variants.

    KEY COMPONENTS:
    1. Patch Processing:
       - Image splitting and linear embedding
       - Learned position embeddings (RoPE)
       - Sequence format conversion

    2. Transformer Core:
       - Multi-head self-attention with RoPE
       - Configurable feed-forward networks:
         * Standard FFN (dense)
         * MoE variant with expert routing
       - Residual connections and layer normalization

    3. Architectural Features:
       - Rotary Positional Embeddings (RoPE)
       - Attention scaling for stability
       - Configurable dropout and activation
       - Training/inference mode switching

    UNIQUE ASPECTS:
    - Pure attention-based vision processing
    - Support for both dense and sparse (MoE) FFNs
    - Memory-efficient RoPE implementation
    - Flexible configuration system
    - Full dlib framework integration
*/
void DBG_INFO(std::string dbg_msg) {
    if (!dbg_msg.empty()) cout << dbg_msg << endl;
}
void DBG_INFO(std::string dbg_msg, const tensor& t, const bool display_t = false, int S = 10, int K = 5, int R = 8, int C = 8) {
    if (!dbg_msg.empty()) {
        cout << dbg_msg << "num_samples=" << t.num_samples() << ", k=" << t.k() << ", nr=" << t.nr() << ", nc=" << t.nc() << endl;
        if (display_t) {
            S = std::min(K, static_cast<int>(t.num_samples()));
            K = std::min(K, static_cast<int>(t.k()));
            R = std::min(R, static_cast<int>(t.nr()));
            C = std::min(C, static_cast<int>(t.nc()));
            for (int s = 0; s < t.num_samples(); ++s) {
                cout << "[";
                for (int k = 0; k < t.k(); ++k) {
                    cout << "[\t";
                    for (int r = 0; r < t.nr(); ++r) {
                        for (int c = 0; c < t.nc(); ++c) {
                            if (c < C) cout << setw(8) << fixed << setprecision(3) << t.host()[tensor_index(t, s, k, r, c)] << " ";
                            else if (c == C) {
                                cout << "...";
                                break;
                            }
                        }
                        if (r < R) cout << endl << "\t";
                        else if (r == R) {
                            cout << endl << "(...)" << endl;
                            break;
                        }
                    }
                    cout << "]";
                }
                if (s < S) cout << "]" << endl;
                if (s == (S - 1)) break;
            }
        }
    }
}

class display_tensor_ {
public:
    display_tensor_() {}
    template <typename SUBNET> void setup(const SUBNET& /* sub */) {}

    template <typename SUBNET> void forward(const SUBNET& sub, resizable_tensor& output) {
        auto& prev = sub.get_output();
        output.copy_size(prev);
        tt::copy_tensor(false, output, 0, prev, 0, prev.k());
        DBG_INFO("display_tensor.forward: ", output, false);
    }
    template <typename SUBNET> void backward(const tensor& gradient_input, SUBNET& sub, tensor& /*params_grad*/) {
        auto& prev = sub.get_gradient_input();
        tt::copy_tensor(true, prev, 0, gradient_input, 0, gradient_input.k());
    }

    const tensor& get_layer_params() const { return params; }
    tensor& get_layer_params() { return params; }

    friend void serialize(const display_tensor_& /* item */, std::ostream& out) {}
    friend void deserialize(display_tensor_& /* item */, std::istream& in) {}

    friend std::ostream& operator<<(std::ostream& out, const display_tensor_& /* item */) {
        out << "display_tensor";
        return out;
    }
    friend void to_xml(const display_tensor_& /* item */, std::ostream& out) {
        out << "<display_tensor />\n";
    }
private:
    dlib::resizable_tensor params; // unused
};
template <typename SUBNET> using display_tensor = add_layer<display_tensor_, SUBNET>;
#ifdef __USE_VIT__
namespace vit
{
    /*!
        Rotary Positional Embedding Layer (RoPE)
        Implements positional encoding via rotation of query/key vectors
        and handles both forward and backward passes.
    */
    class rotary_positional_embedding_ {
    public:
        explicit rotary_positional_embedding_(void) {
        }

        template <typename SUBNET>
        void setup(const SUBNET& sub) {
            // Precompute the rotation angles and their trigonometric values
            seq_len = sub.get_output().nr();
            d_head = sub.get_output().nc();
            compute_rotation_angles();
            precompute_trigonometric_values();
        }

        template <typename SUBNET>
        void forward(const SUBNET& sub, resizable_tensor& output) {
            const tensor& input = sub.get_output();
            output.copy_size(input);
            tt::copy_tensor(false, output, 0, input, 0, input.k());

            // Apply rotary embedding to the output
            apply_rotary_embedding(output);
        }

        template <typename SUBNET>
        void backward(
            const tensor& gradient_input,
            SUBNET& sub,
            tensor& params_grad
        ) {
            tensor& prev = sub.get_gradient_input();
            resizable_tensor grad_output;
            grad_output.copy_size(gradient_input);
            tt::copy_tensor(false, grad_output, 0, gradient_input, 0, gradient_input.k());

            // Apply the inverse rotation to the gradient (transpose of the rotation matrix)
            apply_rotary_embedding(grad_output, true);
            tt::copy_tensor(true, prev, 0, grad_output, 0, grad_output.k());
        }

        const tensor& get_layer_params() const { return params; }
        tensor& get_layer_params() { return params; }

        friend void serialize(const rotary_positional_embedding_& item, std::ostream& out) {
            serialize("rotary_positional_embedding_", out);
            serialize(item.seq_len, out);
            serialize(item.d_head, out);
            serialize(item.angles, out);
            serialize(item.cos_values, out);
            serialize(item.sin_values, out);
        }

        friend void deserialize(rotary_positional_embedding_& item, std::istream& in) {
            std::string version;
            deserialize(version, in);
            if (version != "rotary_positional_embedding_")
                throw serialization_error("Unexpected version found while deserializing rotary_positional_embedding_.");
            deserialize(item.seq_len, in);
            deserialize(item.d_head, in);
            deserialize(item.angles, in);
            deserialize(item.cos_values, in);
            deserialize(item.sin_values, in);
        }

        friend std::ostream& operator<<(std::ostream& out, const rotary_positional_embedding_& item) {
            out << "rotary_positional_embedding";
            out << " (d_head=" << item.d_head << ", seq_len=" << item.seq_len << ")";
            return out;
        }

        friend void to_xml(const rotary_positional_embedding_& item, std::ostream& out)
        {
            out << "<rotary_positional_embedding"
                << " d_head='" << item.d_head << "'"
                << " seq_len='" << item.seq_len << "'"
                << "/>\n";
        }

    protected:
        void compute_rotation_angles() {
            // Following the original RoPE paper formulation
            const float base = 10000.0f;
            const long half_dim = d_head / 2;
            angles.set_size(seq_len, half_dim);

            for (long pos = 0; pos < seq_len; ++pos) {
                for (long i = 0; i < half_dim; ++i) {
                    float exponent = -2.0f * (i + 0.5f) / d_head;
                    float inv_freq = std::pow(base, exponent);
                    angles(pos, i) = pos * std::pow(base, exponent);
                }
            }
        }

        void precompute_trigonometric_values() {
            // Precompute cos and sin for all angles
            cos_values.set_size(angles.nr(), angles.nc());
            sin_values.set_size(angles.nr(), angles.nc());

            for (long i = 0; i < angles.size(); ++i) {
                cos_values(i) = std::cos(angles(i));
                sin_values(i) = std::sin(angles(i));
            }
        }

        template <typename tensor_type>
        void apply_rotary_embedding(
            tensor_type& x,
            bool is_backward = false
        ) const {
            const long batch_size = x.num_samples();
            const long num_heads = x.k();
            const long seq_length = x.nr();
            const long dim = x.nc();
            const bool is_odd = (dim % 2 != 0);
            const long rot_dim = is_odd ? dim - 1 : dim;

            DLIB_CASSERT(dim == d_head, "Input dimension must match d_head param");
            DLIB_CASSERT(seq_length == seq_len, "Sequence length must match seq_len param");

            auto* ptr = x.host();
            const long stride = seq_length * dim;

            for (long n = 0; n < batch_size; ++n) {
                for (long h = 0; h < num_heads; ++h) {
                    auto* x_ptr = ptr + (n * num_heads + h) * stride;

                    for (long pos = 0; pos < seq_length; ++pos) {
                        const float* cos = &cos_values(pos, 0);
                        const float* sin = &sin_values(pos, 0);

                        for (long i = 0; i < rot_dim; i += 2) {
                            const float x0 = x_ptr[pos * dim + i];
                            const float x1 = x_ptr[pos * dim + i + 1];

                            if (!is_backward) {
                                x_ptr[pos * dim + i] = x0 * cos[i / 2] - x1 * sin[i / 2];
                                x_ptr[pos * dim + i + 1] = x0 * sin[i / 2] + x1 * cos[i / 2];
                            }
                            else {
                                x_ptr[pos * dim + i] = x0 * cos[i / 2] + x1 * sin[i / 2];
                                x_ptr[pos * dim + i + 1] = -x0 * sin[i / 2] + x1 * cos[i / 2];
                            }
                        }
                    }
                }
            }
        }

    private:
        long seq_len, d_head;       // Sequence length and dimension of each head
        matrix<float> angles;       // Precomputed rotation angles (seq_len x d_head/2)
        matrix<float> cos_values;   // Precomputed cosine values
        matrix<float> sin_values;   // Precomputed sine values
        resizable_tensor params;    // Empty tensor (no learnable parameters)
    };

    // Helper to easily add RoPE to a network
    template <typename SUBNET>
    using rope = add_layer<rotary_positional_embedding_, SUBNET>;

    /*!
        WHAT THIS OBJECT REPRESENTS
            This object represents a custom layer that rearranges convolutional
            feature maps into a sequence format suitable for transformer processing.

            Input dimensions:  (batch_size, channels, height, width)
            Output dimensions: (batch_size, 1, height*width, channels)
    !*/
    class patches_to_sequence_
    {
    public:
        patches_to_sequence_() = default;
        patches_to_sequence_(const patches_to_sequence_& item) = default;
        patches_to_sequence_& operator= (const patches_to_sequence_& item) = default;

        template <typename SUBNET>
        void setup(const SUBNET& /*sub*/)
        {
            // No specific setup needed
        }

        template <typename SUBNET>
        void forward(
            const SUBNET& sub,
            resizable_tensor& output
        )
        {
            const tensor& input = sub.get_output();

            // Input dimensions: (batch_size, D, h_patches, w_patches)
            const long batch_size = input.num_samples();
            const long dims = input.k();        // Embedding dimension
            const long h_patches = input.nr();  // Height in patches (H/P)
            const long w_patches = input.nc();  // Width in patches (W/P)
            const long num_patches = h_patches * w_patches;

            // Resize output: (batch_size, 1, num_patches, dims)
            output.set_size(batch_size, 1, num_patches, dims);

            // Reorganize data
            for (long n = 0; n < batch_size; ++n) {
                long patch_idx = 0;
                for (long h = 0; h < h_patches; ++h) {
                    for (long w = 0; w < w_patches; ++w) {
                        for (long d = 0; d < dims; ++d) {
                            output.host()[(n * num_patches + patch_idx) * dims + d] =
                                input.host()[((n * dims + d) * h_patches + h) * w_patches + w];
                        }
                        ++patch_idx;
                    }
                }
            }
        }

        template <typename SUBNET>
        void backward(
            const tensor& gradient_input,
            SUBNET& sub,
            tensor& /*params_grad*/
        )
        {
            tensor& grad_output = sub.get_gradient_input();

            // Dimensions
            const long batch_size = grad_output.num_samples();
            const long dims = grad_output.k();
            const long h_patches = grad_output.nr();
            const long w_patches = grad_output.nc();
            const long num_patches = h_patches * w_patches;

            // Backpropagate gradient - accumulate instead of resetting
            for (long n = 0; n < batch_size; ++n) {
                long patch_idx = 0;
                for (long h = 0; h < h_patches; ++h) {
                    for (long w = 0; w < w_patches; ++w) {
                        for (long d = 0; d < dims; ++d) {
                            grad_output.host()[((n * dims + d) * h_patches + h) * w_patches + w] +=
                                gradient_input.host()[(n * num_patches + patch_idx) * dims + d];
                        }
                        ++patch_idx;
                    }
                }
            }
        }

        const tensor& get_layer_params() const { return params; }
        tensor& get_layer_params() { return params; }

        friend void serialize(const patches_to_sequence_& /*item*/, std::ostream& out)
        {
            serialize("patches_to_sequence_", out);
        }

        friend void deserialize(patches_to_sequence_& /*item*/, std::istream& in)
        {
            std::string version;
            deserialize(version, in);
            if (version != "patches_to_sequence_")
                throw serialization_error("Unexpected version found while deserializing patches_to_sequence_.");
        }

        friend std::ostream& operator<<(std::ostream& out, const patches_to_sequence_& /*item*/)
        {
            out << "patches_to_sequence";
            return out;
        }

        friend void to_xml(const patches_to_sequence_& /*item*/, std::ostream& out)
        {
            out << "<patches_to_sequence />\n";
        }

    private:
        resizable_tensor params; // Empty as this layer has no trainable parameters
    };

    template <typename SUBNET>
    using patches_to_sequence = add_layer<patches_to_sequence_, SUBNET>;

    /*!
        WHAT THIS OBJECT REPRESENTS
            This layer scales inputs by 1/sqrt(d_k) as required in attention mechanisms
            to prevent extremely small gradients when input dimensionality is large.
    !*/
    template <long d_k_>
    class scale_weights_ : public multiply_ {
    public:
        explicit scale_weights_() : multiply_(1.0f / std::sqrt(static_cast<float>(d_k_))) {}
    };

    template <long d_k, typename SUBNET>
    using scale_weights = add_layer<scale_weights_<d_k>, SUBNET>;

    // Attention mechanism component extractors
    template <long seq_len, long d_model, long num_heads, typename SUBNET>
    using query = reshape_to<num_heads, seq_len, d_model / num_heads, linear_no_bias<d_model, SUBNET>>;

    template <long seq_len, long d_model, long num_heads, typename SUBNET>
    using key = reshape_to<num_heads, seq_len, d_model / num_heads, linear_no_bias<d_model, SUBNET>>;

    template <long seq_len, long d_model, long num_heads, typename SUBNET>
    using value = reshape_to<num_heads, seq_len, d_model / num_heads, linear_no_bias<d_model, SUBNET>>;

    /*!
        This layer implements multi-head self-attention as described in
        "Attention Is All You Need" (Vaswani et al., 2017).

        Template parameters:
            - ACT: Activation function type
            - DO: Dropout layer type for regularization
            - d_model: Model dimension (must be divisible by num_heads)
            - num_heads: Number of attention heads
    !*/
    template <template <typename> class ACT, template <typename> class DO,
        long seq_len, long d_model, long num_heads, typename SUBNET>
    using multihead_attention =
        rms_norm<add_prev1<
        DO<linear_no_bias<d_model, reshape_to<1, seq_len, d_model,
        multm_prev2<softmaxm<tril_mask<
        scale_weights<d_model / num_heads,
        multm_prev3<
        // Apply RoPE to queries & keys
        rope<query<seq_len, d_model, num_heads, skip1<
        tag3<transpose<
        rope<key<seq_len, d_model, num_heads, skip1<
        tag2<value<seq_len, d_model, num_heads,
        tag1<SUBNET>>>>>>>>>>>>>>>>>>>>>;

    /*!
        Mixture-of-Experts components:
    !*/
    // Expert router that selects between N experts
    // Uses softmax over scaled logits for probability distribution
    template <long num_experts, typename SUBNET>
    using moe_router = softmax<fc<num_experts, SUBNET>>;

    // Single expert network (same structure as standard FFN)
    // Typically multiple experts run in parallel
    template <template <typename> class ACT, template <typename> class DO,
        long d_model, typename SUBNET>
    using expert = DO<linear<d_model, ACT<DO<linear<d_model * 4, SUBNET>>>>>;

    // Combines expert outputs using router probabilities
    // Performs weighted sum of experts with residual connection
    template <template <typename> class ACT, template <typename> class DO,
        long d_model, typename SUBNET>
    using weighted_sum_of_experts = add_prev<itag3,
        mult_prev<itag1, extract<0, 1, 1, 1, skip6<         // Expert 1
        itag1<expert<ACT, DO, d_model, iskip<
		itag3<mult_prev<itag2, extract<1, 1, 1, 1, skip6<   // Expert 2
        itag2<expert<ACT, DO, d_model,
        itag0<SUBNET>>>>>>>>>>>>>>;

    // Complete MoE feed-forward layer
    template <template <typename> class ACT, template <typename> class DO,
        long d_model, typename SUBNET>
    using moe_feed_forward =
        rms_norm<add_prev5<
		weighted_sum_of_experts<ACT, DO, d_model, skip5<
        tag6<moe_router<2,
        tag5<SUBNET>>>>>>>;

    /*!
        Standard feed-forward network (FFN) for transformers.
        Expands to 4*d_model then projects back to d_model.
        Includes activation, dropout and residual connection.
    !*/
    template <template <typename> class ACT, template <typename> class DO,
        long d_model, typename SUBNET>
    using feed_forward =
        rms_norm<add_prev5<
        DO<linear<d_model,
        ACT<DO<linear<d_model * 4,
        tag5<SUBNET>>>>>>>>;

    /*!
        This defines a standard transformer encoder block with self-attention
        followed by a feed-forward network, each with residual connections.

        Template parameters:
            - ACT: Activation function type
            - DO: Dropout layer type for regularization
            - seq_len: Sequence length (number of tokens/patches)
            - d_model: Model dimension
            - num_heads: Number of attention heads
    !*/
    template <template <typename> class ACT, template <typename> class DO,
        long seq_len, long d_model, long num_heads, typename SUBNET>
    using transformer_block_std =
        feed_forward<ACT, DO, d_model,
        multihead_attention<ACT, DO, seq_len, d_model, num_heads, SUBNET>>;
    template <template <typename> class ACT, template <typename> class DO,
        long seq_len, long d_model, long num_heads, typename SUBNET>
    using transformer_block_moe =
        moe_feed_forward<ACT, DO, d_model,
        multihead_attention<ACT, DO, seq_len, d_model, num_heads, SUBNET>>;

    /*!
        This layer projects image patches into an embedding space and adds
        learned position embeddings to maintain spatial information.

        Template parameters:
            - d_model: Embedding dimension
            - patch_size: Size of image patches (assumed square)
    !*/
    template <long d_model, long patch_size, typename SUBNET>
    using patch_embedding = layer_norm<sig<linear<d_model, patches_to_sequence<con<d_model, patch_size, patch_size, patch_size, patch_size, SUBNET>>>>>;

    /*!
        WHAT THIS OBJECT REPRESENTS
            This object defines a configurable Vision Transformer backbone architecture.
            It does not include a classification head or task-specific output layers,
            making it suitable as a feature extractor for various downstream tasks.

            A Vision Transformer (ViT) processes images by:
            1) Dividing the image into fixed-size patches
            2) Linearly embedding each patch
            3) Processing the sequence with transformer encoder blocks

        Template parameters:
            - image_size: Input image size (assumed square)
            - patch_size: Size of image patches (assumed square)
            - num_layers: Number of transformer encoder blocks
            - num_heads: Number of attention heads per block
            - embedding_dim: Dimension of token embeddings
            - activation_func: Activation function for feed-forward networks
            - dropout_policy: Dropout layer type for regularization
    !*/
    template <
        long image_size = 224,
        long patch_size = 16,
        long num_layers = 12,
        long num_heads = 8,
        long embedding_dim = 768,
        bool use_moe = false,
        template <typename> class activation_func = gelu,
        template <typename> class dropout_policy = dropout_10
    >
    struct vit_config {
        // Core model parameters
        static constexpr long IMAGE_SIZE = image_size;
        static constexpr long PATCH_SIZE = patch_size;
        static constexpr long NUM_LAYERS = num_layers;
        static constexpr long NUM_HEADS = num_heads;
        static constexpr long EMBEDDING_DIM = embedding_dim;
        static constexpr long NUM_PATCHES = (IMAGE_SIZE / PATCH_SIZE) * (IMAGE_SIZE / PATCH_SIZE);
        static constexpr bool USE_MOE = use_moe;

        // Runtime configuration validation
        struct validation {
            static_assert(IMAGE_SIZE > 0, "Image size must be positive");
            static_assert(PATCH_SIZE > 0, "Patch size must be positive");
            static_assert(IMAGE_SIZE % PATCH_SIZE == 0, "Image size must be divisible by patch size");
            static_assert(NUM_LAYERS > 0, "Number of layers must be positive");
            static_assert(NUM_HEADS > 0, "Number of attention heads must be positive");
            static_assert(EMBEDDING_DIM % NUM_HEADS == 0, "Embedding dimension must be divisible by number of heads");
        };

        // Network component definitions
        template <typename SUBNET>
        using t_output = avg_pool_everything<activation_func<dropout_policy<fc<EMBEDDING_DIM, SUBNET>>>>;

        template <typename SUBNET>
        using i_output = avg_pool_everything<activation_func<multiply<fc<EMBEDDING_DIM, SUBNET>>>>;

        template <typename SUBNET>
        using t_transformer_block = std::conditional_t<USE_MOE,
            transformer_block_moe<activation_func, dropout_policy, NUM_PATCHES, EMBEDDING_DIM, NUM_HEADS, SUBNET>,
            transformer_block_std<activation_func, dropout_policy, NUM_PATCHES, EMBEDDING_DIM, NUM_HEADS, SUBNET>>;

        template <typename SUBNET>
        using i_transformer_block = std::conditional_t<USE_MOE,
            transformer_block_moe<activation_func, multiply, NUM_PATCHES, EMBEDDING_DIM, NUM_HEADS, SUBNET>,
            transformer_block_std<activation_func, multiply, NUM_PATCHES, EMBEDDING_DIM, NUM_HEADS, SUBNET>>;

        // Complete network definition (training vs inference mode)
        template<bool is_training, typename INPUT_LAYER>
        using network_type = std::conditional_t<is_training,
            t_output<repeat<NUM_LAYERS, t_transformer_block,
            patch_embedding<EMBEDDING_DIM, PATCH_SIZE, INPUT_LAYER>>>,
            i_output<repeat<NUM_LAYERS, i_transformer_block,
            patch_embedding<EMBEDDING_DIM, PATCH_SIZE, INPUT_LAYER>>>>;

        // Model information utilities
        struct model_info {
            static std::string describe() {
                std::stringstream ss;
                ss << "Vision Transformer configuration:\n"
                    << "- image size: " << IMAGE_SIZE << "x" << IMAGE_SIZE << "\n"
                    << "- patch size: " << PATCH_SIZE << "x" << PATCH_SIZE << "\n"
                    << "- number of patches: " << NUM_PATCHES << "\n"
                    << "- layers: " << NUM_LAYERS << "\n"
                    << "- attention heads: " << NUM_HEADS << "\n"
                    << "- MoE enabled: " << (USE_MOE ? "true" : "false") << "\n"
                    << "- embedding dimension: " << EMBEDDING_DIM;
                return ss.str();
            }
        };
    };
}
#else
namespace resnet
{
    using namespace dlib;
    // BN is bn_con or affine layer
    template<template<typename> class BN>
    struct def
    {
        // the resnet basic block, where BN is bn_con or affine
        template<long num_filters, int stride, typename SUBNET>
        using basicblock = BN<con<num_filters, 3, 3, 1, 1,
            relu<BN<con<num_filters, 3, 3, stride, stride, SUBNET>>>>>;

        // the resnet bottleneck block
        template<long num_filters, int stride, typename SUBNET>
        using bottleneck = BN<con<4 * num_filters, 1, 1, 1, 1,
            relu<BN<con<num_filters, 3, 3, stride, stride,
            relu<BN<con<num_filters, 1, 1, 1, 1, SUBNET>>>>>>>>;

        // the resnet residual, where BLOCK is either basicblock or bottleneck
        template<template<long, int, typename> class BLOCK, long num_filters, typename SUBNET>
        using residual = add_prev1<BLOCK<num_filters, 1, tag1<SUBNET>>>;

        // a resnet residual that does subsampling on both paths
        template<template<long, int, typename> class BLOCK, long num_filters, typename SUBNET>
        using residual_down = add_prev2<avg_pool<2, 2, 2, 2,
            skip1<tag2<BLOCK<num_filters, 2,
            tag1<SUBNET>>>>>>;

        // residual block with optional downsampling
        template<
            template<template<long, int, typename> class, long, typename> class RESIDUAL,
            template<long, int, typename> class BLOCK,
            long num_filters,
            typename SUBNET
        >
        using residual_block = relu<RESIDUAL<BLOCK, num_filters, SUBNET>>;

        template<long num_filters, typename SUBNET>
        using resbasicblock_down = residual_block<residual_down, basicblock, num_filters, SUBNET>;
        template<long num_filters, typename SUBNET>
        using resbottleneck_down = residual_block<residual_down, bottleneck, num_filters, SUBNET>;

        // some definitions to allow the use of the repeat layer
        template<typename SUBNET> using resbasicblock_512 = residual_block<residual, basicblock, 512, SUBNET>;
        template<typename SUBNET> using resbasicblock_256 = residual_block<residual, basicblock, 256, SUBNET>;
        template<typename SUBNET> using resbasicblock_128 = residual_block<residual, basicblock, 128, SUBNET>;
        template<typename SUBNET> using resbasicblock_64 = residual_block<residual, basicblock, 64, SUBNET>;
        template<typename SUBNET> using resbottleneck_512 = residual_block<residual, bottleneck, 512, SUBNET>;
        template<typename SUBNET> using resbottleneck_256 = residual_block<residual, bottleneck, 256, SUBNET>;
        template<typename SUBNET> using resbottleneck_128 = residual_block<residual, bottleneck, 128, SUBNET>;
        template<typename SUBNET> using resbottleneck_64 = residual_block<residual, bottleneck, 64, SUBNET>;

        // common processing for standard resnet inputs
        template<typename INPUT>
        using input_processing = max_pool<3, 3, 2, 2, relu<BN<con<64, 7, 7, 2, 2, INPUT>>>>;

        // the resnet backbone with basicblocks
        template<long nb_512, long nb_256, long nb_128, long nb_64, typename INPUT>
        using backbone_basicblock =
            repeat<nb_512, resbasicblock_512, resbasicblock_down<512,
            repeat<nb_256, resbasicblock_256, resbasicblock_down<256,
            repeat<nb_128, resbasicblock_128, resbasicblock_down<128,
            repeat<nb_64, resbasicblock_64, input_processing<INPUT>>>>>>>>;

        // the resnet backbone with bottlenecks
        template<long nb_512, long nb_256, long nb_128, long nb_64, typename INPUT>
        using backbone_bottleneck =
            repeat<nb_512, resbottleneck_512, resbottleneck_down<512,
            repeat<nb_256, resbottleneck_256, resbottleneck_down<256,
            repeat<nb_128, resbottleneck_128, resbottleneck_down<128,
            repeat<nb_64, resbottleneck_64, input_processing<INPUT>>>>>>>>;

        // the backbones for the classic architectures
        template<typename INPUT> using backbone_34 = backbone_basicblock<2, 5, 3, 3, INPUT>;
    };
}
#endif // __USE_VIT__

namespace model
{
#ifdef __USE_VIT__
	// Define a Vision Transformer configuration for Stable ImageNet-1K
    using imagenet_vit_config = vit::vit_config<
        224,        // Image size
        16,         // Patch size 
        4,          // Transformer layers
        6,          // Attention heads
        192,        // Embedding dimension
		true,       // Use Mixture-of-Experts
        gelu,       // Use GELU activation
        dropout_10  // Use 10% dropout
    >;
    using train = loss_multiclass_log<fc<1000, imagenet_vit_config::network_type<true, input_rgb_image_sized<224>>>>;
    using feats = loss_multiclass_log<fc<1000, imagenet_vit_config::network_type<false, input_rgb_image_sized<224>>>>;
#else
    template<template<typename> class BN>
    using net_type = loss_multiclass_log<fc<1000, avg_pool_everything<
        typename resnet::def<BN>::template backbone_34<
        input_rgb_image
        >>>>;

    using train = net_type<bn_con>;
    using feats = net_type<affine>;
#endif
}

// Define a cross-platform signal handling system
namespace {
    std::atomic<bool> g_terminate_flag(false);

#ifdef _WIN32
    // Windows-specific handler
    BOOL WINAPI console_ctrl_handler(DWORD ctrl_type) {
        if (ctrl_type == CTRL_C_EVENT) {
            g_terminate_flag.store(true);
            cout << "\nCtrl+C detected, cleaning up and closing the program..." << endl;
            return TRUE;
        }
        return FALSE;
    }
#else
    // Unix/Linux/macOS handler
    void signal_handler(int signal) {
        if (signal == SIGINT) {
            g_terminate_flag.store(true);
            cout << "\nCtrl+C detected, cleaning up and closing the program..." << endl;
        }
    }
#endif

    // Setup the interrupt handler based on platform
    void setup_interrupt_handler() {
#ifdef _WIN32
        if (!SetConsoleCtrlHandler(console_ctrl_handler, TRUE)) {
            cerr << "ERROR: Could not set control handler" << endl;
        }
#else
        struct sigaction sa;
        sa.sa_handler = signal_handler;
        sigemptyset(&sa.sa_mask);
        sa.sa_flags = 0;
        sigaction(SIGINT, &sa, NULL);
#endif
    }
}

// Data augmentation helper that creates random crops with various transformations
rectangle make_random_cropping_rect(
    const matrix<rgb_pixel>& img,
    dlib::rand& rnd
)
{
    // figure out what rectangle we want to crop from the image
    double mins = 0.466666666, maxs = 0.875;
    auto scale = mins + rnd.get_random_double() * (maxs - mins);
    auto size = scale * std::min(img.nr(), img.nc());
    rectangle rect(size, size);
    // randomly shift the box around
    point offset(rnd.get_random_32bit_number() % (img.nc() - rect.width()),
        rnd.get_random_32bit_number() % (img.nr() - rect.height()));
    return move_rect(rect, offset);
}

void randomly_crop_image(
    const matrix<rgb_pixel>& img,
    matrix<rgb_pixel>& crop,
    dlib::rand& rnd
)
{
    auto rect = make_random_cropping_rect(img, rnd);

    // now crop it out as a 224x224 image.
    extract_image_chip(img, chip_details(rect, chip_dims(224, 224)), crop);
    // Also randomly flip the image
    if (rnd.get_random_double() > 0.5) crop = fliplr(crop);
    // And then randomly adjust the colors.
    apply_random_color_offset(crop, rnd);
}

void randomly_crop_images(
    const matrix<rgb_pixel>& img,
    dlib::array<matrix<rgb_pixel>>& crops,
    dlib::rand& rnd,
    long num_crops
)
{
    std::vector<chip_details> dets;
    for (long i = 0; i < num_crops; ++i)
    {
        auto rect = make_random_cropping_rect(img, rnd);
        dets.push_back(chip_details(rect, chip_dims(224, 224)));
    }
    extract_image_chips(img, dets, crops);

    for (auto&& img : crops)
    {
        // Also randomly flip the image
        if (rnd.get_random_double() > 0.5) img = fliplr(img);
        // And then randomly adjust the colors.
        apply_random_color_offset(img, rnd);
    }
}

int main(const int argc, const char** argv)
try
{
    // Setup interrupt handling for clean termination
    setup_interrupt_handler();

    // The default settings are fine for the example already.
    command_line_parser parser;
    parser.add_option("batch", "set the mini batch size per GPU (default: 48)", 1);
    parser.add_option("learning-rate", "set the initial learning rate", 1);
    parser.add_option("min-learning-rate", "set the min learning rate (default: 1e-5)", 1);
    parser.add_option("num-gpus", "number of GPUs (default: 1)", 1);
    parser.set_group_name("Help Options");
    parser.add_option("h", "alias for --help");
    parser.add_option("help", "display this message and exit");
    parser.parse(argc, argv);

    if (parser.number_of_arguments() < 1 || parser.option("h") || parser.option("help"))
    {
        cout << "This example needs the Stable ImageNet-1K dataset to run." << endl;
        cout << "You can build it from https://github.com/Cydral/Dlib-ImageNet-Datasets." << endl;
        parser.print_options();
        return EXIT_SUCCESS;
    }
    
    const size_t num_gpus = get_option(parser, "num-gpus", 1);
    const size_t batch_size = get_option(parser, "batch", 48) * num_gpus;
#ifdef __USE_VIT__
    const double learning_rate = get_option(parser, "learning-rate", 1e-3);
#else
    const double learning_rate = get_option(parser, "learning-rate", 1e-1);
#endif
    const double weight_decay = 1e-4;
    const double momentum = 0.9;
    const double min_learning_rate = get_option(parser, "min-learning-rate", 1e-5);

    // Print the configuration of our Vision Transformer
#ifdef __USE_VIT__
    cout << "Using " << model::imagenet_vit_config::model_info::describe() << endl;
#endif

    // Load the Stable ImageNet-1K dataset into memory
    std::vector<matrix<rgb_pixel>> training_images, testing_images;
    std::vector<unsigned long> training_labels, testing_labels;
    load_stable_imagenet_1k(parser[0], training_images, training_labels, testing_images, testing_labels);
    dlib::rand rnd;
    std::vector<int> gpus(num_gpus);
    std::iota(gpus.begin(), gpus.end(), 0);

	model::train net;
    dnn_trainer<model::train> trainer(net, sgd(weight_decay, momentum));
    trainer.be_verbose();
    trainer.set_learning_rate(learning_rate);
    trainer.set_mini_batch_size(batch_size);
#ifdef __USE_VIT__
    trainer.set_synchronization_file("vit_stable_imagenet_trainer.sync", std::chrono::minutes(15));
#else
    trainer.set_synchronization_file("resnet_stable_imagenet_trainer.sync", std::chrono::minutes(15));
#endif
    trainer.set_iterations_without_progress_threshold(20000);
    set_all_bn_running_stats_window_sizes(net, 1000);
    disable_duplicative_biases(net);
    cout << trainer << endl;

    std::vector<matrix<rgb_pixel>> samples;
    std::vector<unsigned long> labels;

    dlib::pipe<std::pair<matrix<rgb_pixel>, unsigned long>> data(200);
    auto f = [&data, &training_images, &training_labels](time_t seed)
        {
            dlib::rand rnd(time(0) + seed);
            std::pair<matrix<rgb_pixel>, unsigned long> temp;
            unsigned long idx;
            while (data.is_enabled()) {
                idx = rnd.get_random_32bit_number() % training_images.size();
                temp.first = training_images[rnd.get_random_32bit_number() % training_images.size()];
                randomly_crop_image(training_images[idx], temp.first, rnd);
				temp.second = training_labels[idx];
                data.enqueue(temp);
            }
        };
    std::thread data_loader1([f]() { f(1); });
    std::thread data_loader2([f]() { f(2); });
    std::thread data_loader3([f]() { f(3); });
    std::thread data_loader4([f]() { f(4); });

#ifdef __USE_VIT__
    if (!file_exists("vit_stable_imagenet_trainer.sync") &&
        file_exists("vit-s-16_stable_imagenet_1k.dat")) deserialize("vit-s-16_stable_imagenet_1k.dat") >> net;
#else
    if (!file_exists("resnet_stable_imagenet_trainer.sync") &&
        file_exists("resnet34_stable_imagenet_1k.dat")) deserialize("resnet34_stable_imagenet_1k.dat") >> net;
#endif

    while (trainer.get_learning_rate() >= min_learning_rate && !g_terminate_flag.load())
    {
        samples.clear();
        labels.clear();

        std::pair<matrix<rgb_pixel>, unsigned long> temp;
        while (samples.size() < batch_size)
        {
            data.dequeue(temp);
            samples.push_back(std::move(temp.first));
            labels.push_back(temp.second);
        }

        trainer.train_one_step(samples, labels);
    }
    data.disable();
    data_loader1.join();
    data_loader2.join();
    data_loader3.join();
    data_loader4.join();

    trainer.get_net();
    net.clean();
    cout << "saving network" << endl;
#ifdef __USE_VIT__
    serialize("vit-s-16_stable_imagenet_1k.dat") << net;
#else
    serialize("resnet34_stable_imagenet_1k.dat") << net;
#endif

    // If training was interrupted by Ctrl+C, check if we should exit
    if (g_terminate_flag.load()) {
        cout << "training interrupted by user. Exiting..." << endl;
        return EXIT_FAILURE;
    }

    // Create a softmax version of the network for probability outputs
    cout << "\nEvaluating model on test dataset..." << endl;
    softmax<model::feats::subnet_type> snet;
    snet.subnet() = net.subnet();

    int num_right_top5 = 0;
    int num_wrong_top5 = 0;
    int num_right_top1 = 0;
    int num_wrong_top1 = 0;

    // Process each test image
    for (size_t i = 0; i < testing_images.size(); ++i)
    {
        dlib::array<matrix<rgb_pixel>> crops;

        // Generate 16 random crops for test-time augmentation
        const int num_crops = 16;
        randomly_crop_images(testing_images[i], crops, rnd, num_crops);

        // Get averaged predictions across all crops
        matrix<float, 1, 1000> p = sum_rows(mat(snet(crops.begin(), crops.end()))) / num_crops;

        // Top-1 accuracy check
        if (index_of_max(p) == testing_labels[i])
            ++num_right_top1;
        else
            ++num_wrong_top1;

        // Top-5 accuracy check
        bool found_match = false;
        matrix<float, 1, 1000> p_copy = p; // Create a copy to modify
        for (int k = 0; k < 5; ++k)
        {
            long predicted_label = index_of_max(p_copy);
            p_copy(predicted_label) = 0; // Zero out the max to get next highest

            if (predicted_label == testing_labels[i])
            {
                found_match = true;
                break;
            }
        }

        if (found_match)
            ++num_right_top5;
        else
            ++num_wrong_top5;

        // Progress reporting
        if ((i + 1) % 100 == 0)
        {
            cout << "Processed " << (i + 1) << " of " << testing_images.size()
                << " test images" << endl;
        }
    }

    // Final accuracy reporting
    cout << "\nTest Results:" << endl;
    cout << "Top-1 Accuracy: " << num_right_top1 / double(num_right_top1 + num_wrong_top1)
        << " (" << num_right_top1 << "/" << (num_right_top1 + num_wrong_top1) << ")" << endl;
    cout << "Top-5 Accuracy: " << num_right_top5 / double(num_right_top5 + num_wrong_top5)
        << " (" << num_right_top5 << "/" << (num_right_top5 + num_wrong_top5) << ")" << endl;

    return EXIT_SUCCESS;
}
catch (const exception& e)
{
    cout << e.what() << endl;
    return EXIT_FAILURE;
}