// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*
    This is an example illustrating the use of the deep learning tools from the dlib C++
    Library. In this example program we are going to show how one can train a Vision Transformer (ViT)
    neural network using an unsupervised loss function.

    To train the unsupervised loss, we will use the self-supervised learning (SSL) method
    called Barlow Twins, introduced in this paper:
    "Barlow Twins: Self-Supervised Learning via Redundancy Reduction" by Jure Zbontar,
    Li Jing, Ishan Misra, Yann LeCun, Stéphane Deny.

    For this example, we're using the Vision Transformer (ViT) architecture, which
    was introduced in the paper "An Image is Worth 16x16 Words: Transformers for
    Image Recognition at Scale" by Alexei A. Dosovitskiy, et al. ViT divides images
    into fixed-size patches, processes them with a standard transformer encoder, and
    has shown comparable or better performance than traditional CNNs like ResNet.

    We will train our ViT on the CIFAR-10 dataset which contains relatively small
    images (32x32 pixels). Due to the small image size, we use a simplified ViT with
    fewer layers and heads than would be used for larger images.
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
#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

#define __USE_VIT__ 1

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
            static_assert(IMAGE_SIZE% PATCH_SIZE == 0, "Image size must be divisible by patch size");
            static_assert(NUM_LAYERS > 0, "Number of layers must be positive");
            static_assert(NUM_HEADS > 0, "Number of attention heads must be positive");
            static_assert(EMBEDDING_DIM% NUM_HEADS == 0, "Embedding dimension must be divisible by number of heads");
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
// A custom definition of ResNet50 with a downsampling factor of 8 instead of 32.
// It is essentially the original ResNet50, but without the max pooling and a
// convolutional layer with a stride of 1 instead of 2 at the input.
namespace resnet50
{
    using namespace dlib;
    template <template <typename> class BN>
    struct def
    {
        template <long N, int K, int S, typename SUBNET>
        using conv = add_layer<con_<N, K, K, S, S, K / 2, K / 2>, SUBNET>;

        template<long N, int S, typename SUBNET>
        using bottleneck = BN<conv<4 * N, 1, 1, relu<BN<conv<N, 3, S, relu<BN<conv<N, 1, 1, SUBNET>>>>>>>>;

        template <long N, typename SUBNET>
        using residual = add_prev1<bottleneck<N, 1, tag1<SUBNET>>>;

        template <typename SUBNET> using res_512 = relu<residual<512, SUBNET>>;
        template <typename SUBNET> using res_256 = relu<residual<256, SUBNET>>;
        template <typename SUBNET> using res_128 = relu<residual<128, SUBNET>>;
        template <typename SUBNET> using res_64 = relu<residual<64, SUBNET>>;

        template <long N, int S, typename SUBNET>
        using transition = add_prev2<BN<conv<4 * N, 1, S, skip1<tag2<bottleneck<N, S, tag1<SUBNET>>>>>>>;

        template <typename INPUT>
        using backbone = avg_pool_everything<
            repeat<2, res_512, transition<512, 2,
            repeat<5, res_256, transition<256, 2,
            repeat<3, res_128, transition<128, 2,
            repeat<2, res_64, transition<64, 1,
            relu<BN<conv<64, 3, 1, INPUT>>>>>>>>>>>>;
    };
};
#endif // __USE_VIT__

// This model namespace contains the definitions for:
// - SSL model using the Barlow Twins loss, a projector head and an input_rgb_image_pair.
// - A feature extractor model using the loss_metric (to get the outputs) and an input_rgb_image.
namespace model
{
    // Projector network for the Barlow Twins method
    template <typename SUBNET> using projector = fc<128, relu<bn_fc<fc<512, SUBNET>>>>;

#ifdef __USE_VIT__
    // Define a lightweight Vision Transformer configuration for CIFAR-10
    using cifar10_vit_config = vit::vit_config<
        32,         // Image size (CIFAR-10 images are 32x32)
        4,          // Patch size 
        4,          // Transformer layers
        8,          // Attention heads
        144,        // Embedding dimension
		true,       // Use Mixture-of-Experts (MoE)
        gelu,       // Use GELU activation
        dropout_10  // Use 10% dropout
    >;
    // Complete training model with Barlow Twins loss
    using train = loss_barlow_twins<projector<cifar10_vit_config::network_type<true, input_rgb_image_pair>>>;
    // Feature extraction model for inference
    using feats = loss_metric<cifar10_vit_config::network_type<false, input_rgb_image>>;
#else
    using train = loss_barlow_twins<projector<resnet50::def<bn_con>::backbone<input_rgb_image_pair>>>;
    using feats = loss_metric<resnet50::def<affine>::backbone<input_rgb_image>>;
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
    const matrix<rgb_pixel>& image,
    dlib::rand& rnd
)
{
    const double mins = 7. / 15.;
    const double maxs = 7. / 8.;
    const auto scale = rnd.get_double_in_range(mins, maxs);
    const auto size = scale * std::min(image.nr(), image.nc());
    const rectangle rect(size, size);
    const point offset(rnd.get_random_32bit_number() % (image.nc() - rect.width()),
        rnd.get_random_32bit_number() % (image.nr() - rect.height()));
    return move_rect(rect, offset);
}

// Apply various data augmentations to create different views of the same image
matrix<rgb_pixel> augment(
    const matrix<rgb_pixel>& image,
    const bool prime,
    dlib::rand& rnd
)
{
    matrix<rgb_pixel> crop;
    // blur
    matrix<rgb_pixel> blurred;
    const double sigma = rnd.get_double_in_range(0.1, 1.1);
    if (!prime || (prime && rnd.get_random_double() < 0.1))
    {
        const auto rect = gaussian_blur(image, blurred, sigma);
        extract_image_chip(blurred, rect, crop);
        blurred = crop;
    }
    else
    {
        blurred = image;
    }

    // randomly crop
    const auto rect = make_random_cropping_rect(image, rnd);
    extract_image_chip(blurred, chip_details(rect, chip_dims(32, 32)), crop);

    // image left-right flip
    if (rnd.get_random_double() < 0.5)
        flip_image_left_right(crop);

    // color augmentation
    if (rnd.get_random_double() < 0.8)
        disturb_colors(crop, rnd, 0.5, 0.5);

    // grayscale
    if (rnd.get_random_double() < 0.2)
    {
        matrix<unsigned char> gray;
        assign_image(gray, crop);
        assign_image(crop, gray);
    }

    // solarize
    if (prime && rnd.get_random_double() < 0.2)
    {
        for (auto& p : crop)
        {
            if (p.red > 128)
                p.red = 255 - p.red;
            if (p.green > 128)
                p.green = 255 - p.green;
            if (p.blue > 128)
                p.blue = 255 - p.blue;
        }
    }
    return crop;
}

int main(const int argc, const char** argv)
try
{    
    // Setup interrupt handling for clean termination
    setup_interrupt_handler();

    // The default settings are fine for the example already.
    command_line_parser parser;
    parser.add_option("batch", "set the mini batch size per GPU (default: 64)", 1);
    parser.add_option("dims", "set the projector dimensions (default: 128)", 1);
    parser.add_option("lambda", "penalize off-diagonal terms (default: 1/dims)", 1);
    parser.add_option("learning-rate", "set the initial learning rate (default: 1e-3)", 1);
    parser.add_option("min-learning-rate", "set the min learning rate (default: 1e-5)", 1);
    parser.add_option("num-gpus", "number of GPUs (default: 1)", 1);
    parser.set_group_name("Help Options");
    parser.add_option("h", "alias for --help");
    parser.add_option("help", "display this message and exit");
    parser.parse(argc, argv);

    if (parser.number_of_arguments() < 1 || parser.option("h") || parser.option("help"))
    {
        cout << "This example needs the CIFAR-10 dataset to run." << endl;
        cout << "You can get CIFAR-10 from https://www.cs.toronto.edu/~kriz/cifar.html" << endl;
        cout << "Download the binary version the dataset, decompress it, and put the 6" << endl;
        cout << "bin files in a folder.  Then give that folder as input to this program." << endl;
        parser.print_options();
        return EXIT_SUCCESS;
    }

    const size_t num_gpus = get_option(parser, "num-gpus", 1);
    const size_t batch_size = get_option(parser, "batch", 64) * num_gpus;
    const long dims = get_option(parser, "dims", 128);
    const double lambda = get_option(parser, "lambda", 1.0 / dims);
    const double learning_rate = get_option(parser, "learning-rate", 1e-3);
    const double min_learning_rate = get_option(parser, "min-learning-rate", 1e-5);

    // Print the configuration of our Vision Transformer
#ifdef __USE_VIT__
    cout << "Using " << model::cifar10_vit_config::model_info::describe() << endl;
#endif

    // Load the CIFAR-10 dataset into memory
    std::vector<matrix<rgb_pixel>> training_images, testing_images;
    std::vector<unsigned long> training_labels, testing_labels;
    load_cifar_10_dataset(parser[0], training_images, training_labels, testing_images, testing_labels);

    // Initialize the model with the specified projector dimensions and lambda
    // According to literature, lambda = 1/dims works well on CIFAR-10
    model::train net((loss_barlow_twins_(lambda)));
    layer<1>(net).layer_details().set_num_outputs(dims);
    disable_duplicative_biases(net);
    dlib::rand rnd;
    std::vector<int> gpus(num_gpus);
    std::iota(gpus.begin(), gpus.end(), 0);

    // Train the feature extractor using the Barlow Twins method
    {
        dnn_trainer<model::train, adam> trainer(net, adam(1e-6, 0.9, 0.999), gpus);
        trainer.set_mini_batch_size(batch_size);
        trainer.set_learning_rate(learning_rate);
        trainer.set_min_learning_rate(min_learning_rate);
        trainer.set_iterations_without_progress_threshold(15000);
#ifdef __USE_VIT__
        trainer.set_synchronization_file("vit_barlow_twins_sync", std::chrono::minutes(30));
#else
        trainer.set_synchronization_file("resnet_barlow_twins_sync", std::chrono::minutes(30));
#endif
        trainer.be_verbose();
        cout << trainer << endl;

        // During the training, we will compute the empirical cross-correlation matrix
        // between the features of both versions of the augmented images.
        // This visualization helps to see how well the model is learning.
        resizable_tensor eccm;
        eccm.set_size(dims, dims);
        resizable_tensor za_norm, zb_norm, means, invstds, rms, rvs, gamma, beta;
        const double eps = DEFAULT_BATCH_NORM_EPS;
        gamma.set_size(1, dims);
        beta.set_size(1, dims);
        image_window win;

#ifdef __USE_VIT__
        if (!file_exists("vit_barlow_twins_sync") &&
            file_exists("vit-s-4_self_supervised_cifar_10.dat")) deserialize("vit-s-4_self_supervised_cifar_10.dat") >> net;
#else
        if (!file_exists("resnet_barlow_twins_sync") &&
            file_exists("res50_self_supervised_cifar_10.dat")) deserialize("res50_self_supervised_cifar_10.dat") >> net;
#endif
        std::vector<std::pair<matrix<rgb_pixel>, matrix<rgb_pixel>>> batch;
        while (trainer.get_learning_rate() >= trainer.get_min_learning_rate() && !g_terminate_flag.load())
        {
            batch.clear();
            while (batch.size() < trainer.get_mini_batch_size())
            {
                const auto idx = rnd.get_random_32bit_number() % training_images.size();
                auto image = training_images[idx];
                batch.emplace_back(augment(image, false, rnd), augment(image, true, rnd));
            }
            trainer.train_one_step(batch);

            // Compute the empirical cross-correlation matrix every 100 steps
            // This is for visualization only and not required for training
            if (trainer.get_train_one_step_calls() % 100 == 0)
            {
                // Wait for threaded processing to stop in the trainer.
                trainer.get_net(force_flush_to_disk::no);
                // Get the output from the last fc layer
                const auto& out = net.subnet().get_output();
                // Skip if output is empty (trainer might have cleaned network state)
                if (out.size() == 0)
                    continue;
                // Separate both augmented versions of the images
                alias_tensor split(out.num_samples() / 2, dims);
                auto za = split(out);
                auto zb = split(out, split.size());
                gamma = 1;
                beta = 0;
                // Perform batch normalization on each feature representation, independently
                tt::batch_normalize(eps, za_norm, means, invstds, 1, rms, rvs, za, gamma, beta);
                tt::batch_normalize(eps, zb_norm, means, invstds, 1, rms, rvs, za, gamma, beta);
                // Compute the empirical cross-correlation matrix between the features
                tt::gemm(0, eccm, 1, za_norm, true, zb_norm, false);
                eccm /= batch_size;
                win.set_image(round(abs(mat(eccm)) * 255));
                win.set_title("Barlow Twins step#: " + to_string(trainer.get_train_one_step_calls()));
            }
        }
        trainer.get_net();
        net.clean();
        // After training, we can discard the projector head and just keep the backone
        // to train it or finetune it on other downstream tasks.
#ifdef __USE_VIT__
        serialize("vit-s-4_self_supervised_cifar_10.dat") << net;
        cout << "ViT backbone saved to vit-s-4_self_supervised_cifar_10.dat" << endl;
#else
        serialize("res50_self_supervised_cifar_10.dat") << net;
        cout << "Resnet-50 backbone saved to res50_self_supervised_cifar_10.dat" << endl;
#endif

        // If training was interrupted by Ctrl+C, check if we should exit
        if (g_terminate_flag.load()) {
            cout << "Training interrupted by user. Exiting..." << endl;
            return EXIT_FAILURE;
        }
    }

    // Now, we initialize the feature extractor model with the backbone we have just learned.
    model::feats fnet(layer<5>(net));
    // And we will generate all the features for the training set to train a multiclass SVM
    // classifier.
    std::vector<matrix<float, 0, 1>> features;
    cout << "Extracting features for linear classifier..." << endl;
    features = fnet(training_images, 4 * batch_size);
    vector_normalizer<matrix<float, 0, 1>> normalizer;
    normalizer.train(features);
    for (auto& feature : features)
        feature = normalizer(feature);

    // Find the most appropriate C setting using find_max_global.
    auto cross_validation_score = [&](const double c)
        {
            svm_multiclass_linear_trainer<linear_kernel<matrix<float, 0, 1>>, unsigned long> trainer;
            trainer.set_c(c);
            trainer.set_epsilon(0.01);
            trainer.set_max_iterations(100);
            trainer.set_num_threads(std::thread::hardware_concurrency());
            cout << "C: " << c << endl;
            const auto cm = cross_validate_multiclass_trainer(trainer, features, training_labels, 3);
            const double accuracy = sum(diag(cm)) / sum(cm);
            cout << "cross validation accuracy: " << accuracy << endl;
            cout << "confusion matrix:\n " << cm << endl;
            return accuracy;
        };
    const auto result = find_max_global(cross_validation_score, 1e-3, 1000, max_function_calls(50));
    cout << "Best SVM hyperparameter C: " << result.x(0) << endl;

    // Proceed to train the SVM classifier with the best C.
    svm_multiclass_linear_trainer<linear_kernel<matrix<float, 0, 1>>, unsigned long> trainer;
    trainer.set_num_threads(std::thread::hardware_concurrency());
    trainer.set_c(result.x(0));
    cout << "Training Multiclass SVM on Vision Transformer features..." << endl;
    const auto df = trainer.train(features, training_labels);
#ifdef __USE_VIT__
    serialize("vit_multiclass_svm_cifar_10.dat") << df;
    cout << "SVM classifier saved to vit_multiclass_svm_cifar_10.dat" << endl;
#else
    serialize("res50_multiclass_svm_cifar_10.dat") << df;
    cout << "SVM classifier saved to res50_multiclass_svm_cifar_10.dat" << endl;
#endif

    // Finally, we can compute the accuracy of the model on the CIFAR-10 train and test images.
    auto compute_accuracy = [&fnet, &df, batch_size](
        const std::vector<matrix<float, 0, 1>>& samples,
        const std::vector<unsigned long>& labels
        )
        {
            size_t num_right = 0;
            size_t num_wrong = 0;
            for (size_t i = 0; i < labels.size(); ++i)
            {
                if (labels[i] == df(samples[i]))
                    ++num_right;
                else
                    ++num_wrong;
            }
            cout << "  num right:  " << num_right << endl;
            cout << "  num wrong:  " << num_wrong << endl;
            cout << "  accuracy:   " << num_right / static_cast<double>(num_right + num_wrong) << endl;
            cout << "  error rate: " << num_wrong / static_cast<double>(num_right + num_wrong) << endl;
        };

    // Evaluate on training set
#ifdef __USE_VIT__
    cout << "\nViT training accuracy" << endl;
#else
    cout << "\nResnet-50 training accuracy" << endl;
#endif
    compute_accuracy(features, training_labels);

    // Evaluate on testing set
#ifdef __USE_VIT__
    cout << "\nViT testing accuracy" << endl;
#else
    cout << "\nResnet-50 testing accuracy" << endl;
#endif
    features = fnet(testing_images, 4 * batch_size);
    for (auto& feature : features) feature = normalizer(feature);
    compute_accuracy(features, testing_labels);

    return EXIT_SUCCESS;
}
catch (const exception& e)
{
    cout << e.what() << endl;
    return EXIT_FAILURE;
}