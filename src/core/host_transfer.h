#pragma once

#include "core/arena.h"

#include <cuda_runtime_api.h>

#include <array>
#include <cstddef>
#include <cstdint>

namespace ninfer {

// Bounded two-buffer bridge between pageable session images and device memory. CUDA only sees
// the fixed pinned buffers; the large, byte-bounded cache remains ordinary pageable RAM.
//
// Calls may enqueue any number of 1D or device-pitched/host-contiguous 2D regions. finish() makes
// every submitted copy visible at its destination. The class is single-stream and single-owner;
// the Engine execution mutex supplies that ownership for session save/restore.
class HostTransferStager {
public:
    HostTransferStager(cudaStream_t stream, std::size_t buffer_bytes);
    ~HostTransferStager() noexcept;

    HostTransferStager(const HostTransferStager&)            = delete;
    HostTransferStager& operator=(const HostTransferStager&) = delete;
    HostTransferStager(HostTransferStager&&)                 = delete;
    HostTransferStager& operator=(HostTransferStager&&)      = delete;

    [[nodiscard]] std::size_t buffer_bytes() const noexcept { return buffer_bytes_; }

    void device_to_host(void* host, const void* device, std::size_t bytes);
    void host_to_device(void* device, const void* host, std::size_t bytes);

    // Host rows are tightly packed. Device rows start device_pitch bytes apart.
    void device_to_host_2d(void* host, const void* device, std::size_t device_pitch,
                           std::size_t width, std::size_t height);
    void host_to_device_2d(void* device, std::size_t device_pitch, const void* host,
                           std::size_t width, std::size_t height);

    void finish();

private:
    enum class Direction : std::uint8_t { None, DeviceToHost, HostToDevice };

    struct Slot {
        explicit Slot(std::size_t bytes) : buffer(bytes) {}

        PinnedHostBuffer buffer;
        cudaEvent_t ready        = nullptr;
        void* pageable_target    = nullptr;
        std::size_t copied_bytes = 0;
        bool pending             = false;
    };

    void set_direction(Direction direction);
    Slot& acquire_slot();
    void retire(Slot& slot);
    void enqueue_device_to_host(void* host, const void* device, std::size_t bytes);
    void enqueue_host_to_device(void* device, const void* host, std::size_t bytes);
    void enqueue_device_to_host_2d(void* host, const void* device, std::size_t device_pitch,
                                   std::size_t width, std::size_t height);
    void enqueue_host_to_device_2d(void* device, std::size_t device_pitch, const void* host,
                                   std::size_t width, std::size_t height);

    cudaStream_t stream_      = nullptr;
    std::size_t buffer_bytes_ = 0;
    std::array<Slot, 2> slots_;
    std::size_t next_slot_ = 0;
    Direction direction_   = Direction::None;
};

} // namespace ninfer
