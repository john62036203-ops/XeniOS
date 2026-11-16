/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2022 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/apu/audio_system.h"

#include <atomic>
#include <chrono>
#include <limits>
#include <thread>
#include <dlfcn.h>
#include <mach/mach.h>
#include <pthread.h>
#include <cstring>
#include "xenia/cpu/ppc/ppc_context.h"

#include "xenia/apu/apu_flags.h"
#include "xenia/apu/audio_driver.h"
#include "xenia/apu/nop_audio_driver.h"
#include "xenia/apu/xma_decoder.h"
#include "xenia/base/assert.h"
#include "xenia/base/byte_stream.h"
#include "xenia/base/clock.h"
#include "xenia/base/logging.h"
#include "xenia/base/math.h"
#include "xenia/base/platform.h"
#include "xenia/base/profiling.h"
#include "xenia/base/ring_buffer.h"
#include "xenia/base/string_buffer.h"
#include "xenia/base/threading.h"
#include "xenia/cpu/thread_state.h"
#include "xenia/kernel/kernel_state.h"
#include "xenia/kernel/guest_scheduler.h"
namespace xe::kernel {
KernelState* kernel_state();
}

// As with normal Microsoft, there are like twelve different ways to access
// the audio APIs. Early games use XMA*() methods almost exclusively to touch
// decoders. Later games use XAudio*() and direct memory writes to the XMA
// structures (as opposed to the XMA* calls), meaning that we have to support
// both.
//
// For ease of implementation, most audio related processing is handled in
// AudioSystem, and the functions here call off to it.
// The XMA*() functions just manipulate the audio system in the guest context
// and let the normal AudioSystem handling take it, to prevent duplicate
// implementations. They can be found in xboxkrnl_audio_xma.cc

DEFINE_uint32(apu_max_queued_frames, 8,
              "Allows changing max buffered audio frames to reduce audio "
              "delay. Lowering this value might cause performance issues. "
              "Value range: [4-64]",
              "APU");
UPDATE_from_uint32(apu_max_queued_frames, 2024, 8, 31, 20, 64);
#if XE_PLATFORM_IOS
// XeniOS 2.0.1's iOS default.
UPDATE_from_uint32(apu_max_queued_frames, 2026, 11, 3, 14, 32);
#endif  // XE_PLATFORM_IOS

// fork diagnostic: what the audio worker is doing, logged from a side thread
// every 2 s so a worker stuck inside the guest callback is still seen.
namespace {
std::atomic<uint64_t> g_ad_pumps{0}, g_ad_sem_drops{0}, g_ad_cb_begin{0}, g_ad_cb_end{0};
std::atomic<uint64_t> g_ad_cb_max_us{0}, g_ad_cb_since_us{0}, g_ad_parked{0};
std::atomic<bool> g_ad_started{false};
std::atomic<xe::cpu::ppc::PPCContext*> g_ad_ctx{nullptr};
std::atomic<mach_port_t> g_ad_mach{MACH_PORT_NULL};
void AdDumpHostStack(mach_port_t th, const char* label) {
  if (th == MACH_PORT_NULL || th == mach_thread_self()) return;
  XELOGI("[audio-diag] host stack of {}", label);
#if defined(__aarch64__)
  arm_thread_state64_t st;
  mach_msg_type_number_t count = ARM_THREAD_STATE64_COUNT;
  if (thread_suspend(th) != KERN_SUCCESS) return;
  if (thread_get_state(th, ARM_THREAD_STATE64, (thread_state_t)&st, &count) == KERN_SUCCESS) {
    uint64_t pcs[20];
    int n = 0;
    pcs[n++] = arm_thread_state64_get_pc(st);
    pcs[n++] = arm_thread_state64_get_lr(st);
    uint64_t fp = arm_thread_state64_get_fp(st);
    for (int i = 0; i < 16 && fp && (fp & 7) == 0; ++i) {
      vm_size_t got = 0;
      uint64_t frame[2] = {0, 0};
      if (vm_read_overwrite(mach_task_self(), fp, sizeof(frame), (vm_address_t)frame, &got) !=
              KERN_SUCCESS ||
          got != sizeof(frame)) {
        break;
      }
      if (frame[1]) pcs[n++] = frame[1];
      if (frame[0] <= fp || n >= 20) break;
      fp = frame[0];
    }
    thread_resume(th);
    for (int i = 0; i < n; ++i) {
      Dl_info info = {};
      const uint64_t pc = pcs[i] & 0x0000FFFFFFFFFFFFull;
      if (dladdr(reinterpret_cast<void*>(pc), &info) && info.dli_sname) {
        XELOGI("[audio-diag]   #{} {:x} {}+{:#x}", i, pc, info.dli_sname,
               pc - reinterpret_cast<uint64_t>(info.dli_saddr));
      } else if (info.dli_fbase) {
        XELOGI("[audio-diag]   #{} {:x} {}+{:#x}", i, pc,
               info.dli_fname ? info.dli_fname : "?", pc - reinterpret_cast<uint64_t>(info.dli_fbase));
      } else {
        XELOGI("[audio-diag]   #{} {:x} (JIT or unknown)", i, pc);
      }
    }
    return;
  }
  thread_resume(th);
#endif
}

// When the worker has been inside one guest callback for >1 s: the guest
// registers and the host call stack of the worker, to see what it waits on.
void AdDumpStuck() {
  // fork: once, every fiber and what it waits on, plus recent signals, to find
  // who should set the two events the audio callback waits for.
  static std::atomic<bool> dumped_fibers{false};
  if (!dumped_fibers.exchange(true)) {
    auto* ks = xe::kernel::kernel_state();
    if (ks && xe::kernel::GuestScheduler::enabled() && ks->guest_scheduler()) {
      XELOGW("[audio-diag] fiber dump follows");
      ks->guest_scheduler()->DiagnosticDump();
    }
  }
  if (auto* ctx = g_ad_ctx.load()) {
    XELOGI("[audio-diag] stuck guest lr={:08X} ctr={:08X} r1={:08X} r3={:08X} r4={:08X} r5={:08X}",
           uint32_t(ctx->lr), uint32_t(ctx->ctr), uint32_t(ctx->r[1]), uint32_t(ctx->r[3]),
           uint32_t(ctx->r[4]), uint32_t(ctx->r[5]));
  }
  AdDumpHostStack(g_ad_mach.load(), "Audio Worker");
  // fork: also the host stacks of the guest CPU dispatch threads, to see
  // whether the audio fibers on CPU 4/5 sit in a host wait or a spin.
  static std::atomic<int> cpu_dumps{0};
  if (cpu_dumps.fetch_add(1) < 2) {
    thread_act_array_t threads = nullptr;
    mach_msg_type_number_t n = 0;
    if (task_threads(mach_task_self(), &threads, &n) == KERN_SUCCESS) {
      for (mach_msg_type_number_t i = 0; i < n; ++i) {
        char name[64] = {};
        pthread_t pt = pthread_from_mach_thread_np(threads[i]);
        if (pt) pthread_getname_np(pt, name, sizeof(name));
        if (std::strncmp(name, "Guest CPU", 9) == 0) {
          AdDumpHostStack(threads[i], name);
        }
        mach_port_deallocate(mach_task_self(), threads[i]);
      }
      vm_deallocate(mach_task_self(), (vm_address_t)threads, n * sizeof(thread_act_t));
    }
  }
}
uint64_t AdNowUs() {
  return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
                                   std::chrono::steady_clock::now().time_since_epoch())
                                   .count());
}
void AdStartWatch() {
  if (g_ad_started.exchange(true)) return;
  std::thread([] {
    uint64_t last_end = 0;
    for (int i = 0; i < 300; ++i) {   // ~10 minutes
      std::this_thread::sleep_for(std::chrono::seconds(2));
      const uint64_t b = g_ad_cb_begin.load(), e = g_ad_cb_end.load();
      const uint64_t since = g_ad_cb_since_us.load();
      const uint64_t in_cb_ms = (b > e && since) ? (AdNowUs() - since) / 1000 : 0;
      XELOGI("[audio-diag] pumps={} sem_drops={} callbacks begun={} done={} (+{} in 2 s) "
             "max_cb_ms={} in_callback_now_ms={} parked={} frame={}",
             g_ad_pumps.load(), g_ad_sem_drops.load(), b, e, e - last_end,
             g_ad_cb_max_us.load() / 1000, in_cb_ms, g_ad_parked.load(),
             xe::logging::GetFrameNumber());
      last_end = e;
      static int dumps = 0;
      if (in_cb_ms > 1000 && dumps < 3) {
        ++dumps;
        AdDumpStuck();
      }
    }
  }).detach();
}
}  // namespace

namespace xe {
namespace apu {

AudioSystem::AudioSystem(cpu::Processor* processor)
    : memory_(processor->memory()),
      processor_(processor),
      worker_running_(false) {
  queued_frames_ = std::clamp(cvars::apu_max_queued_frames,
                              static_cast<uint32_t>(kMinimumQueuedFrames),
                              static_cast<uint32_t>(kMaximumQueuedFrames));

  for (size_t i = 0; i < kMaximumClientCount; ++i) {
    client_semaphores_[i] = xe::threading::Semaphore::Create(0, queued_frames_);
  }
  pending_work_event_ = xe::threading::Event::CreateAutoResetEvent(false);
  assert_not_null(pending_work_event_);

  xma_decoder_ = std::make_unique<xe::apu::XmaDecoder>(processor_);

  resume_event_ = xe::threading::Event::CreateAutoResetEvent(false);
  assert_not_null(resume_event_);
}

AudioSystem::~AudioSystem() {
  if (xma_decoder_) {
    xma_decoder_->Shutdown();
  }
}

X_STATUS AudioSystem::Setup(kernel::KernelState* kernel_state) {
  X_STATUS result = xma_decoder_->Setup(kernel_state);
  if (result) {
    return result;
  }

  worker_running_ = true;
  worker_thread_ =
      kernel::object_ref<kernel::XHostThread>(new kernel::XHostThread(
          kernel_state, 128 * 1024, 0,
          [this]() {
            WorkerThreadMain();
            return 0;
          },
          kernel_state->GetSystemProcess()));
  // As we run audio callbacks the debugger must be able to suspend us.
  worker_thread_->set_can_debugger_suspend(true);
  worker_thread_->set_name("Audio Worker");
  worker_thread_->Create();
  // Set high priority for this thread for better pacing.
  worker_thread_->SetPriority(24);
  return X_STATUS_SUCCESS;
}

void AudioSystem::WorkerThreadMain() {
  // Initialize driver and ringbuffer.
  Initialize();

  // The host mixer releases a client's semaphore on its own coarse cadence,
  // but Xenos audio subsystem operates at 5.333ms interval (see
  // xaudio2_audio_driver.cc) Interval scales inversely with guest_time_scalar.
  // We therefore pace pumps to each client's next_pump_us deadline and use
  // the semaphore only as back-pressure: a frame is submitted only if
  // a host output slot is free, otherwise it is dropped (the host queue
  // is full, so it is already well buffered).
  while (worker_running_) {
    const uint64_t now = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now().time_since_epoch())
            .count());

    size_t client_index = kMaximumClientCount;
    uint64_t earliest_pump_us = std::numeric_limits<uint64_t>::max();
    {
      std::lock_guard<std::mutex> clients_lock(clients_mutex_);

      for (size_t i = 0; i < kMaximumClientCount; ++i) {
        if (!clients_[i].in_use ||
            clients_[i].next_pump_us >= earliest_pump_us) {
          continue;
        }
        earliest_pump_us = clients_[i].next_pump_us;
        client_index = i;
      }

      if (client_index != kMaximumClientCount) {
        const double scalar = xe::Clock::guest_time_scalar();
        const uint64_t min_us =
            scalar > 0.0 ? static_cast<uint64_t>(kAudioPumpInterval / scalar)
                         : kAudioPumpInterval;
        // Counted from the deadline, so a late pump is made up and the host
        // queue absorbs the stall, until it falls further behind than the
        // queue holds.
        uint64_t next_pump_us = earliest_pump_us + min_us;
        if (now > earliest_pump_us + min_us * queued_frames_) {
          next_pump_us = now + min_us;
        }
        clients_[client_index].next_pump_us = next_pump_us;
      }
    }

    // No clients yet: park until one registers or we're told to stop.
    if (client_index == kMaximumClientCount) {
      g_ad_parked.fetch_add(1, std::memory_order_relaxed);
      xe::threading::Wait(pending_work_event_.get(), true);
      WaitWhilePaused();
      continue;
    }

    // Pace to kAudioIntervalSlack ahead of the deadline.
    const uint64_t wake_target_us = earliest_pump_us > kAudioIntervalSlack
                                        ? earliest_pump_us - kAudioIntervalSlack
                                        : 0;
    if (wake_target_us > now) {
      const std::chrono::milliseconds timeout((wake_target_us - now) / 1000);
      auto result =
          xe::threading::Wait(pending_work_event_.get(), true, timeout);
      if (result == xe::threading::WaitResult::kSuccess) {
        WaitWhilePaused();
        continue;
      }

      const uint64_t now_precise = static_cast<uint64_t>(
          std::chrono::duration_cast<std::chrono::microseconds>(
              std::chrono::steady_clock::now().time_since_epoch())
              .count());
      if (wake_target_us > now_precise) {
        xe::threading::NanoSleepPrecise((wake_target_us - now_precise) * 1000);
      }
    }

    // Submit only if the host has a free output slot; otherwise drop (the host
    // queue is full, so it is already well buffered).
    if (xe::threading::Wait(client_semaphores_[client_index].get(), false,
                            std::chrono::milliseconds(0)) !=
        xe::threading::WaitResult::kSuccess) {
      g_ad_sem_drops.fetch_add(1, std::memory_order_relaxed);
      continue;
    }
    g_ad_pumps.fetch_add(1, std::memory_order_relaxed);

    // UnregisterClient waits on this after clearing in_use.
    std::lock_guard<std::mutex> cb_lk(clients_[client_index].callback_mutex);

    uint32_t client_callback = 0;
    uint32_t client_callback_arg = 0;
    {
      std::lock_guard<std::mutex> clients_lock(clients_mutex_);
      if (clients_[client_index].in_use) {
        client_callback = clients_[client_index].callback;
        client_callback_arg = clients_[client_index].wrapped_callback_arg;
      }
    }

    if (client_callback) {
      SCOPE_profile_cpu_i("apu", "xe::apu::AudioSystem->client_callback");
      uint64_t args[] = {client_callback_arg};
      const uint64_t cb_t0 = AdNowUs();
      g_ad_cb_since_us.store(cb_t0, std::memory_order_relaxed);
      if (!g_ad_ctx.load(std::memory_order_relaxed)) {
        g_ad_ctx.store(worker_thread_->thread_state()->context());
        g_ad_mach.store(mach_thread_self());
      }
      g_ad_cb_begin.fetch_add(1, std::memory_order_relaxed);
      processor_->Execute(worker_thread_->thread_state(), client_callback, args,
                          xe::countof(args));
      const uint64_t cb_us = AdNowUs() - cb_t0;
      uint64_t prev = g_ad_cb_max_us.load(std::memory_order_relaxed);
      while (cb_us > prev && !g_ad_cb_max_us.compare_exchange_weak(prev, cb_us)) {
      }
      g_ad_cb_end.fetch_add(1, std::memory_order_relaxed);
    }
  }
  worker_running_ = false;

  // TODO(benvanik): call module API to kill?
}

int AudioSystem::FindFreeClient() {
  for (int i = 0; i < kMaximumClientCount; i++) {
    auto& client = clients_[i];
    if (!client.in_use) {
      return i;
    }
  }

  return -1;
}

void AudioSystem::Initialize() {}

void AudioSystem::WaitWhilePaused() {
  // A loop, not a single wait: a Resume that came while the worker was still in
  // a callback (the Pause timed out) leaves resume_event_ set, and the next
  // Pause must not fall through it.
  while (paused_ && worker_running_) {
    pause_fence_.Signal();
    xe::threading::Wait(resume_event_.get(), false);
  }
}

void AudioSystem::StopWorker() {
  worker_running_ = false;
  pending_work_event_->Set();
  // A paused worker waits for the resume.
  resume_event_->Set();
  if (worker_thread_) {
    worker_thread_->Wait(0, 0, 0, nullptr);
    worker_thread_.reset();
  }
}

void AudioSystem::Shutdown() {
  StopWorker();

  // Unregister all active clients to shut down their audio drivers before
  // the semaphores are destroyed with this AudioSystem.
  uint32_t wrapped_callback_args[kMaximumClientCount] = {};
  {
    std::lock_guard<std::mutex> clients_lock(clients_mutex_);
    for (size_t i = 0; i < kMaximumClientCount; ++i) {
      if (clients_[i].in_use) {
        DestroyDriver(clients_[i].driver);
        wrapped_callback_args[i] = clients_[i].wrapped_callback_arg;
        clients_[i].driver = nullptr;
        clients_[i].callback = 0;
        clients_[i].callback_arg = 0;
        clients_[i].wrapped_callback_arg = 0;
        clients_[i].in_use = false;
      }
    }
  }
  // Outside clients_mutex_, as the heap takes the global lock.
  for (uint32_t wrapped_callback_arg : wrapped_callback_args) {
    if (wrapped_callback_arg) {
      memory()->SystemHeapFree(wrapped_callback_arg);
    }
  }
}

X_STATUS AudioSystem::RegisterClient(uint32_t callback, uint32_t callback_arg,
                                     size_t* out_index) {
  // Outside clients_mutex_, as the heap takes the global lock.
  uint32_t ptr = memory()->SystemHeapAlloc(0x4);
  xe::store_and_swap<uint32_t>(memory()->TranslateVirtual(ptr), callback_arg);

  std::unique_lock<std::mutex> clients_lock(clients_mutex_);

  auto index = FindFreeClient();
  assert_true(index >= 0);
  if (index < 0) {
    clients_lock.unlock();
    memory()->SystemHeapFree(ptr);
    return X_STATUS_UNSUCCESSFUL;
  }

  auto client_semaphore = client_semaphores_[index].get();
  auto ret = client_semaphore->Release(queued_frames_, nullptr);
  assert_true(ret);

  AudioDriver* driver = nullptr;
  auto result = CreateClientDriver(index, client_semaphore, &driver);
  if (XFAILED(result)) {
    XELOGE("AudioSystem::RegisterClient: CreateDriver failed for index={}",
           index);
    clients_lock.unlock();
    memory()->SystemHeapFree(ptr);
    return result;
  }
  assert_not_null(driver);
  XELOGI(
      "AudioSystem::RegisterClient: driver created for index={}, driver={:p}",
      index, (void*)driver);

  clients_[index].driver = driver;
  clients_[index].callback = callback;
  clients_[index].callback_arg = callback_arg;
  clients_[index].wrapped_callback_arg = ptr;
  clients_[index].in_use = true;
  clients_[index].next_pump_us = 0;
  clients_[index].frames_submitted.store(0);
  clients_[index].frames_processed.store(0);
  clients_[index].frames_dropped.store(0);

  // Wake the worker so it re-scans and starts pacing this client immediately.
  pending_work_event_->Set();

  XELOGI("AudioSystem::RegisterClient: client {} registered successfully",
         index);
  AdStartWatch();

  if (out_index) {
    *out_index = index;
  }

  return X_STATUS_SUCCESS;
}

void AudioSystem::SubmitFrame(size_t index, float* samples) {
  SCOPE_profile_cpu_f("apu");

  std::lock_guard<std::mutex> clients_lock(clients_mutex_);
  assert_true(index < kMaximumClientCount);
  if (index >= kMaximumClientCount || !clients_[index].in_use ||
      !clients_[index].driver) {
    XELOGW(
        "SubmitFrame called for invalid/unregistered client index {} "
        "(in_use={}, driver={:p})",
        index, index < kMaximumClientCount ? clients_[index].in_use : false,
        index < kMaximumClientCount ? (void*)clients_[index].driver : nullptr);

    // Submit silence instead of dropping the frame to maintain the callback
    // chain.  If we don't submit anything, the audio driver's OnBufferEnd
    // callback will never fire, causing the semaphore to leak.
    if (index < kMaximumClientCount && clients_[index].driver) {
      static float silence[apu::AudioDriver::kFrameSamplesMax] = {0};
      clients_[index].frames_dropped++;
      (clients_[index].driver)->SubmitFrame(silence);
    } else if (index < kMaximumClientCount) {
      // Tick the semaphore so the worker doesn't stall on a dead client.
      client_semaphores_[index]->Release(1, nullptr);
    }
    return;
  }
  clients_[index].frames_submitted++;
  clients_[index].frames_processed++;
  (clients_[index].driver)->SubmitFrame(samples);
}

bool AudioSystem::GetClientPerformance(size_t index,
                                       ClientPerformance* out_perf) {
  if (index >= kMaximumClientCount || !out_perf) {
    return false;
  }

  if (!clients_[index].in_use) {
    return false;
  }

  out_perf->frames_submitted = clients_[index].frames_submitted.load();
  out_perf->frames_processed = clients_[index].frames_processed.load();
  out_perf->frames_dropped = clients_[index].frames_dropped.load();
  return true;
}

void AudioSystem::UnregisterClient(size_t index) {
  SCOPE_profile_cpu_f("apu");

  assert_true(index < kMaximumClientCount);
  AudioDriver* driver_to_destroy;
  {
    std::lock_guard<std::mutex> clients_lock(clients_mutex_);
    XELOGI(
        "AudioSystem::UnregisterClient: index={}, driver={:p}", index,
        index < kMaximumClientCount ? (void*)clients_[index].driver : nullptr);
    driver_to_destroy = clients_[index].driver;
    // Leak wrapped_callback_arg: in-flight callback may hold this pointer.
    clients_[index].driver = nullptr;
    clients_[index].callback = 0;
    clients_[index].callback_arg = 0;
    clients_[index].wrapped_callback_arg = 0;
    clients_[index].in_use = false;
    clients_[index].next_pump_us = 0;
    clients_[index].frames_submitted.store(0);
    clients_[index].frames_processed.store(0);
    clients_[index].frames_dropped.store(0);
  }

  // Wait for any in-flight callback; can't hold clients_mutex_ (callback
  // re-enters).
  {
    std::lock_guard<std::mutex> lk(clients_[index].callback_mutex);
  }

  DestroyDriver(driver_to_destroy);

  // Drain the semaphore of its count.
  auto client_semaphore = client_semaphores_[index].get();
  xe::threading::WaitResult wait_result;
  do {
    wait_result = xe::threading::Wait(client_semaphore, false,
                                      std::chrono::milliseconds(0));
  } while (wait_result == xe::threading::WaitResult::kSuccess);
  assert_true(wait_result == xe::threading::WaitResult::kTimeout);
}

bool AudioSystem::Save(ByteStream* stream) {
  stream->Write(kAudioSaveSignature);

  // Count the number of used clients first.
  // Any gaps should be handled gracefully.
  uint32_t used_clients = 0;
  for (int i = 0; i < kMaximumClientCount; i++) {
    if (clients_[i].in_use) {
      used_clients++;
    }
  }

  stream->Write(used_clients);
  for (uint32_t i = 0; i < kMaximumClientCount; i++) {
    auto& client = clients_[i];
    if (!client.in_use) {
      continue;
    }

    stream->Write(i);
    stream->Write(client.callback);
    stream->Write(client.callback_arg);
    stream->Write(client.wrapped_callback_arg);
  }

  return true;
}

bool AudioSystem::Restore(ByteStream* stream) {
  if (stream->Read<uint32_t>() != kAudioSaveSignature) {
    XELOGE("AudioSystem::Restore - Invalid magic value!");
    return false;
  }

  uint32_t num_clients = stream->Read<uint32_t>();
  for (uint32_t i = 0; i < num_clients; i++) {
    auto id = stream->Read<uint32_t>();
    assert_true(id < kMaximumClientCount);

    auto& client = clients_[id];

    // Reset the semaphore and recreate the driver ourselves.
    if (client.driver) {
      UnregisterClient(id);
    }

    client.callback = stream->Read<uint32_t>();
    client.callback_arg = stream->Read<uint32_t>();
    client.wrapped_callback_arg = stream->Read<uint32_t>();

    client.next_pump_us = 0;
    client.in_use = true;

    auto client_semaphore = client_semaphores_[id].get();
    auto ret = client_semaphore->Release(queued_frames_, nullptr);
    assert_true(ret);

    AudioDriver* driver = nullptr;
    auto status = CreateClientDriver(id, client_semaphore, &driver);
    if (XFAILED(status)) {
      XELOGE(
          "AudioSystem::Restore - Call to CreateDriver failed with status "
          "{:08X}",
          status);
      return false;
    }

    assert_not_null(driver);
    client.driver = driver;
  }

  return true;
}

void AudioSystem::Pause() {
  if (paused_) {
    return;
  }
  // A pause that timed out can leave the fence signaled; drop that first.
  pause_fence_.TryWait();
  paused_ = true;

  // Kind of a hack, but it works.
  pending_work_event_->Set();
  // Bounded: the worker signals only between callbacks, and a callback blocked
  // on a guest lock must not wedge the caller.
  bool parked = false;
  for (int i = 0; i < 2000 && !(parked = pause_fence_.TryWait()); ++i) {
    xe::threading::Sleep(std::chrono::milliseconds(1));
  }
  if (!parked) {
    XELOGW("AudioSystem: the worker didn't reach its pause point in 2 s");
  }

  xma_decoder_->Pause();
}

void AudioSystem::Resume() {
  if (!paused_) {
    return;
  }
  paused_ = false;

  resume_event_->Set();

  xma_decoder_->Resume();
}

X_STATUS AudioSystem::CreateClientDriver(size_t index,
                                         xe::threading::Semaphore* semaphore,
                                         AudioDriver** out_driver) {
  *out_driver = nullptr;
  X_STATUS result = CreateDriver(index, semaphore, out_driver);
  if (result == X_STATUS_SUCCESS && *out_driver) {
    return result;
  }

  if (*out_driver) {
    DestroyDriver(*out_driver);
  }
  // The audio worker paces registered clients; release their slots immediately.
  *out_driver = new nop::NopAudioDriver(semaphore);
  XELOGW("APU: {} failed to create a driver for client {}; continuing silently",
         name(), index);
  return X_STATUS_SUCCESS;
}

void AudioSystem::DestroyDriver(AudioDriver* driver) {
  if (driver) {
    driver->Shutdown();
    delete driver;
  }
}

}  // namespace apu
}  // namespace xe
