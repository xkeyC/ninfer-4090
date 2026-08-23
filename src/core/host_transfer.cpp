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
        if (slot.target_pitch == slot.row_bytes) {
            std::memcpy(slot.pageable_target, slot.buffer.data(), slot.copied_bytes);
        } else {
            auto* target = static_cast<std::uint8_t*>(slot.pageable_target);
            const auto* source = static_cast<const std::uint8_t*>(slot.buffer.data());
            for (std::size_t row = 0; row < slot.rows; ++row) {
                std::memcpy(target + row * slot.target_pitch, source + row * slot.row_bytes,
                            slot.row_bytes);
            }
        }
    }
    slot.pageable_target = nullptr;
    slot.copied_bytes    = 0;
    slot.target_pitch    = 0;
    slot.row_bytes       = 0;
    slot.rows            = 0;
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
    slot.target_pitch    = bytes;
    slot.row_bytes       = bytes;
    slot.rows            = 1;
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
    slot.target_pitch    = width;
    slot.row_bytes       = width;
    slot.rows            = height;
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

void HostTransferStager::device_to_host_strided(void* host, std::size_t host_pitch,
                                                const void* device, std::size_t row_bytes,
                                                std::size_t rows) {
    if (row_bytes == 0 || rows == 0) { return; }
    if (host == nullptr || device == nullptr || host_pitch < row_bytes) {
        throw std::invalid_argument("HostTransferStager strided destination is invalid");
    }
    if (row_bytes > buffer_bytes_) {
        throw std::invalid_argument("HostTransferStager strided row exceeds its staging buffer");
    }
    set_direction(Direction::DeviceToHost);
    auto* host_cursor                = static_cast<std::uint8_t*>(host);
    const auto* device_cursor        = static_cast<const std::uint8_t*>(device);
    const std::size_t rows_per_chunk = buffer_bytes_ / row_bytes;
    while (rows != 0) {
        const std::size_t chunk_rows = std::min(rows, rows_per_chunk);
        Slot& slot                   = acquire_slot();
        const std::size_t bytes      = chunk_rows * row_bytes;
        CUDA_CHECK(cudaMemcpyAsync(slot.buffer.data(), device_cursor, bytes,
                                   cudaMemcpyDeviceToHost, stream_));
        CUDA_CHECK(cudaEventRecord(slot.ready, stream_));
        slot.pageable_target = host_cursor;
        slot.copied_bytes    = bytes;
        slot.target_pitch    = host_pitch;
        slot.row_bytes       = row_bytes;
        slot.rows            = chunk_rows;
        slot.pending         = true;
        host_cursor += chunk_rows * host_pitch;
        device_cursor += bytes;
        rows -= chunk_rows;
    }
}

void HostTransferStager::host_fragments_to_device(
    void* device, std::span<const std::span<const std::uint8_t>> fragments,
    std::size_t fragment_offset, std::size_t fragment_bytes) {
    if (fragments.empty() || fragment_bytes == 0) { return; }
    if (device == nullptr || fragment_bytes > buffer_bytes_) {
        throw std::invalid_argument("HostTransferStager fragmented source is invalid");
    }
    for (const auto fragment : fragments) {
        if (fragment_offset > fragment.size() ||
            fragment_bytes > fragment.size() - fragment_offset) {
            throw std::invalid_argument("HostTransferStager fragment is truncated");
        }
    }
    set_direction(Direction::HostToDevice);
    auto* device_cursor              = static_cast<std::uint8_t*>(device);
    const std::size_t rows_per_chunk = buffer_bytes_ / fragment_bytes;
    std::size_t begin                = 0;
    while (begin < fragments.size()) {
        const std::size_t rows = std::min(rows_per_chunk, fragments.size() - begin);
        Slot& slot             = acquire_slot();
        auto* target           = static_cast<std::uint8_t*>(slot.buffer.data());
        for (std::size_t row = 0; row < rows; ++row) {
            const auto fragment = fragments[begin + row];
            std::memcpy(target + row * fragment_bytes, fragment.data() + fragment_offset,
                        fragment_bytes);
        }
        const std::size_t bytes = rows * fragment_bytes;
        CUDA_CHECK(cudaMemcpyAsync(device_cursor, slot.buffer.data(), bytes,
                                   cudaMemcpyHostToDevice, stream_));
        CUDA_CHECK(cudaEventRecord(slot.ready, stream_));
        slot.pending = true;
        device_cursor += bytes;
        begin += rows;
    }
}

void HostTransferStager::finish() {
    for (Slot& slot : slots_) { retire(slot); }
    direction_ = Direction::None;
    next_slot_ = 0;
}

} // namespace ninfer
