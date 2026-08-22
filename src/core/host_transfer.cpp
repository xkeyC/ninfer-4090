#include "core/host_transfer.h"

#include "core/device.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace ninfer {

HostTransferStager::HostTransferStager(cudaStream_t stream, std::size_t buffer_bytes)
    : stream_(stream), buffer_bytes_(buffer_bytes), slots_{Slot(buffer_bytes), Slot(buffer_bytes)} {
    if (stream == nullptr) { throw std::invalid_argument("HostTransferStager stream is null"); }
    if (buffer_bytes == 0) {
        throw std::invalid_argument("HostTransferStager buffer size must be nonzero");
    }
    CUDA_CHECK(cudaEventCreateWithFlags(&slots_[0].ready, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&slots_[1].ready, cudaEventDisableTiming));
}

HostTransferStager::~HostTransferStager() noexcept {
    finish();
    for (Slot& slot : slots_) {
        if (slot.ready != nullptr) {
            (void)cudaEventDestroy(slot.ready);
            slot.ready = nullptr;
        }
    }
}

void HostTransferStager::set_direction(Direction direction) {
    if (direction_ == direction) { return; }
    finish();
    direction_ = direction;
}

void HostTransferStager::retire(Slot& slot) {
    if (!slot.pending) { return; }
    CUDA_CHECK(cudaEventSynchronize(slot.ready));
    if (direction_ == Direction::DeviceToHost) {
        std::memcpy(slot.pageable_target, slot.buffer.data(), slot.copied_bytes);
    }
    slot.pageable_target = nullptr;
    slot.copied_bytes    = 0;
    slot.pending         = false;
}

HostTransferStager::Slot& HostTransferStager::acquire_slot() {
    Slot& slot = slots_[next_slot_];
    next_slot_ = (next_slot_ + 1) % slots_.size();
    retire(slot);
    return slot;
}

void HostTransferStager::enqueue_device_to_host(void* host, const void* device, std::size_t bytes) {
    Slot& slot = acquire_slot();
    CUDA_CHECK(cudaMemcpyAsync(slot.buffer.data(), device, bytes, cudaMemcpyDeviceToHost, stream_));
    CUDA_CHECK(cudaEventRecord(slot.ready, stream_));
    slot.pageable_target = host;
    slot.copied_bytes    = bytes;
    slot.pending         = true;
}

void HostTransferStager::enqueue_host_to_device(void* device, const void* host, std::size_t bytes) {
    Slot& slot = acquire_slot();
    std::memcpy(slot.buffer.data(), host, bytes);
    CUDA_CHECK(cudaMemcpyAsync(device, slot.buffer.data(), bytes, cudaMemcpyHostToDevice, stream_));
    CUDA_CHECK(cudaEventRecord(slot.ready, stream_));
    slot.pending = true;
}

void HostTransferStager::device_to_host(void* host, const void* device, std::size_t bytes) {
    if (bytes == 0) { return; }
    if (host == nullptr || device == nullptr) {
        throw std::invalid_argument("HostTransferStager 1D source or destination is null");
    }
    set_direction(Direction::DeviceToHost);
    auto* host_cursor         = static_cast<std::uint8_t*>(host);
    const auto* device_cursor = static_cast<const std::uint8_t*>(device);
    while (bytes != 0) {
        const std::size_t chunk = std::min(bytes, buffer_bytes_);
        enqueue_device_to_host(host_cursor, device_cursor, chunk);
        host_cursor += chunk;
        device_cursor += chunk;
        bytes -= chunk;
    }
}

void HostTransferStager::host_to_device(void* device, const void* host, std::size_t bytes) {
    if (bytes == 0) { return; }
    if (host == nullptr || device == nullptr) {
        throw std::invalid_argument("HostTransferStager 1D source or destination is null");
    }
    set_direction(Direction::HostToDevice);
    auto* device_cursor     = static_cast<std::uint8_t*>(device);
    const auto* host_cursor = static_cast<const std::uint8_t*>(host);
    while (bytes != 0) {
        const std::size_t chunk = std::min(bytes, buffer_bytes_);
        enqueue_host_to_device(device_cursor, host_cursor, chunk);
        device_cursor += chunk;
        host_cursor += chunk;
        bytes -= chunk;
    }
}

void HostTransferStager::enqueue_device_to_host_2d(void* host, const void* device,
                                                   std::size_t device_pitch, std::size_t width,
                                                   std::size_t height) {
    Slot& slot = acquire_slot();
    CUDA_CHECK(cudaMemcpy2DAsync(slot.buffer.data(), width, device, device_pitch, width, height,
                                 cudaMemcpyDeviceToHost, stream_));
    CUDA_CHECK(cudaEventRecord(slot.ready, stream_));
    slot.pageable_target = host;
    slot.copied_bytes    = width * height;
    slot.pending         = true;
}

void HostTransferStager::enqueue_host_to_device_2d(void* device, std::size_t device_pitch,
                                                   const void* host, std::size_t width,
                                                   std::size_t height) {
    Slot& slot              = acquire_slot();
    const std::size_t bytes = width * height;
    std::memcpy(slot.buffer.data(), host, bytes);
    CUDA_CHECK(cudaMemcpy2DAsync(device, device_pitch, slot.buffer.data(), width, width, height,
                                 cudaMemcpyHostToDevice, stream_));
    CUDA_CHECK(cudaEventRecord(slot.ready, stream_));
    slot.pending = true;
}

void HostTransferStager::device_to_host_2d(void* host, const void* device, std::size_t device_pitch,
                                           std::size_t width, std::size_t height) {
    if (width == 0 || height == 0) { return; }
    if (host == nullptr || device == nullptr || device_pitch < width) {
        throw std::invalid_argument("HostTransferStager 2D geometry is invalid");
    }
    if (width > buffer_bytes_) {
        throw std::invalid_argument("HostTransferStager 2D row exceeds its staging buffer");
    }
    set_direction(Direction::DeviceToHost);
    auto* host_cursor                = static_cast<std::uint8_t*>(host);
    const auto* device_cursor        = static_cast<const std::uint8_t*>(device);
    const std::size_t rows_per_chunk = buffer_bytes_ / width;
    while (height != 0) {
        const std::size_t rows = std::min(height, rows_per_chunk);
        enqueue_device_to_host_2d(host_cursor, device_cursor, device_pitch, width, rows);
        host_cursor += width * rows;
        device_cursor += device_pitch * rows;
        height -= rows;
    }
}

void HostTransferStager::host_to_device_2d(void* device, std::size_t device_pitch, const void* host,
                                           std::size_t width, std::size_t height) {
    if (width == 0 || height == 0) { return; }
    if (host == nullptr || device == nullptr || device_pitch < width) {
        throw std::invalid_argument("HostTransferStager 2D geometry is invalid");
    }
    if (width > buffer_bytes_) {
        throw std::invalid_argument("HostTransferStager 2D row exceeds its staging buffer");
    }
    set_direction(Direction::HostToDevice);
    auto* device_cursor              = static_cast<std::uint8_t*>(device);
    const auto* host_cursor          = static_cast<const std::uint8_t*>(host);
    const std::size_t rows_per_chunk = buffer_bytes_ / width;
    while (height != 0) {
        const std::size_t rows = std::min(height, rows_per_chunk);
        enqueue_host_to_device_2d(device_cursor, device_pitch, host_cursor, width, rows);
        device_cursor += device_pitch * rows;
        host_cursor += width * rows;
        height -= rows;
    }
}

void HostTransferStager::finish() {
    for (Slot& slot : slots_) { retire(slot); }
    direction_ = Direction::None;
    next_slot_ = 0;
}

} // namespace ninfer
